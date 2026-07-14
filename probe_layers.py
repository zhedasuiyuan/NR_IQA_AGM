"""Layer-wise probing of SigLIP features for NR-IQA.

For every hidden layer of the vision encoder we pool its ``[B, N, D]`` tokens to
a vector and fit a *frozen-backbone* Ridge probe to predict MOS, then report
test SRCC/PLCC. This answers the paper's Part-1 question: *where* in the network
does IQA-relevant information live, and does the optimal depth differ by
distortion type (the KADID breakdown).

Two pooling variants are probed side by side:

  * ``mean``   -- mean over tokens, then Ridge. The conservative linear probe:
                  it measures *linearly decodable* signal at each layer.
  * ``native`` -- the backbone's own pooler (SigLIP2's learned attention-pooling
                  head), then Ridge. A *fixed* attention-pool probe. NOTE: the
                  head was trained on the final layer, so it is biased toward the
                  top -- read it as "what the deployed pooler can still recover",
                  not as a clean per-layer learned attention probe. A PE-style
                  *learned* attention probe (a trainable query pooling fit per
                  layer) is the natural phase-2 extension.

Fast first cut: no training loop -- Ridge is closed-form, alpha is picked on the
val split, and per-distortion curves reuse the same fitted model.

Example:
    python probe_layers.py --dataset KADID10K --batch_size 8
    python probe_layers.py --dataset KonIQ_10K --max_images 2000   # quick smoke
"""

from __future__ import annotations

import argparse
import copy
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoProcessor

from configs.default import MODEL_CONFIG, _make_dataset_paths
from dataset import build_splits
from models.multi_layer_fusion import (
    backbone_hidden_size,
    backbone_num_hidden_layers,
    extract_token_features,
    native_pool,
)

# Per-dataset distortion-type groupings, so the per-distortion figure is
# interpretable. The type id is the ``TT`` field of the distorted filename
# (``Ixx_TT_LL`` for KADID, ``ixx_TT_L`` for TID). Groupings follow distortion
# families and are a defensible starting point -- edit to taste for the paper.
DISTORTION_GROUPS = {
    "KADID10K": {  # 25 types -> KADID's 7 official super-categories
        "Blurs": [1, 2, 3],
        "Color": [4, 5, 6, 7, 8],
        "Compression": [9, 10],
        "Noise": [11, 12, 13, 14, 15],
        "Brightness": [16, 17, 18],
        "Spatial": [19, 20, 21, 22, 23],
        "Sharpness/Contrast": [24, 25],
    },
    "TID2013": {  # 24 types grouped by distortion family
        "Noise": [1, 2, 3, 4, 5, 6, 7, 19, 20, 21, 22],
        "Blur": [8, 9],
        "Compression": [10, 11, 12, 13],
        "Spatial": [14, 15, 24],
        "Intensity/Contrast": [16, 17],
        "Color": [18, 23],
    },
}


def _groupings(groups_map, test_types, which):
    """Ordered breakdown items per requested granularity.

    Returns ``{granularity: [(label, group, type_ids), ...]}``. ``group`` breaks
    down by distortion family; ``type`` breaks down by each distortion id present
    in the test split (tagged with its family). ``both`` returns both.
    """
    type2group = {t: g for g, ts in groups_map.items() for t in ts}
    out = {}
    if which in ("group", "both"):
        out["group"] = [(g, g, ts) for g, ts in groups_map.items()]
    if which in ("type", "both"):
        present = sorted(set(int(t) for t in test_types))
        out["type"] = [(t, type2group.get(t, "?"), [t]) for t in present]
    return out


def _to_pixel_values(processor, images, model, device):
    """Preprocess a batch of ``[B, C, H, W]`` uint8-range images exactly as
    ``train.py`` does, and move to the model's device/dtype."""
    pv = processor(images=images, return_tensors="pt")["pixel_values"]
    return pv.to(device=device, dtype=next(model.parameters()).dtype)


# ---------------------------------------------------------------------------
# Feature extraction: pooled vector per layer, per pooling variant
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_pooled(model, processor, dataset, layer_indices, device, batch_size):
    """Return ``(mean_pool, native_pool, scores)``.

    ``mean_pool`` / ``native_pool`` are ``[N, L, D]`` float32 arrays (L layers),
    ``scores`` is ``[N]``. Order matches the dataset (``shuffle=False``), so the
    caller can align external per-item metadata (e.g. KADID distortion type).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    mean_chunks, native_chunks, score_chunks = [], [], []
    for batch in loader:
        pixel_values = _to_pixel_values(processor, batch["image"], model, device)
        feats, _ = extract_token_features(model, pixel_values, layer_indices)
        # feats: list of L tensors [B, N, D]
        mean_pooled = torch.stack([f.mean(dim=1) for f in feats], dim=1)          # [B, L, D]
        native_pooled = torch.stack([native_pool(model, f) for f in feats], dim=1)  # [B, L, D]

        mean_chunks.append(mean_pooled.float().cpu())
        native_chunks.append(native_pooled.float().cpu())
        score_chunks.append(batch["score"].float())

    return (
        torch.cat(mean_chunks).numpy(),
        torch.cat(native_chunks).numpy(),
        torch.cat(score_chunks).numpy(),
    )


# ---------------------------------------------------------------------------
# Ridge probe (closed-form, alpha picked on val by SRCC)
# ---------------------------------------------------------------------------
def ridge_probe(Xtr, ytr, Xva, yva, Xte, yte, alphas=(1e-2, 1e-1, 1, 10, 100, 1000)):
    """Fit standardized Ridge on train, select alpha on val SRCC, score on test.

    Returns ``(test_srcc, test_plcc, test_preds)``.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr = (Xtr - mu) / sd
    Zva = (Xva - mu) / sd
    Zte = (Xte - mu) / sd
    ybar = ytr.mean()
    yc = ytr - ybar
    G = Ztr.T @ Ztr
    Zty = Ztr.T @ yc
    I = np.eye(G.shape[0])

    best = (-2.0, None)  # (val_srcc, w)
    for a in alphas:
        w = np.linalg.solve(G + a * I, Zty)
        val_pred = Zva @ w + ybar
        srcc = spearmanr(val_pred, yva)[0]
        if srcc > best[0]:
            best = (srcc, w)

    w = best[1]
    te_pred = Zte @ w + ybar
    return spearmanr(te_pred, yte)[0], pearsonr(te_pred, yte)[0], te_pred


# ---------------------------------------------------------------------------
# Learned attention probe (PE-style): a trainable single-query attention-pool
# head fit per layer. Tokens are re-extracted on the fly (no_grad, frozen
# backbone), so nothing large is cached. Run on a few target layers, not all H.
# ---------------------------------------------------------------------------
class AttnPool(nn.Module):
    """One learned query attends over a layer's tokens, then a linear predicts MOS."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * dim ** -0.5)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 1)

    def forward(self, tokens):  # tokens: [B, N, D]
        q = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(q, tokens, tokens)  # [B, 1, D]
        return self.fc(self.norm(pooled[:, 0])).squeeze(-1)  # [B]


@torch.no_grad()
def _attn_eval(probes, model, processor, dataset, layers, device, batch_size):
    """Return ``{layer: (srcc, plcc, preds)}`` for a set of trained probes."""
    for p in probes.values():
        p.eval()
    preds = {l: [] for l in layers}
    ys = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4):
        pv = _to_pixel_values(processor, batch["image"], model, device)
        feats, _ = extract_token_features(model, pv, layers)
        for l, f in zip(layers, feats):
            preds[l].append(probes[str(l)](f.float()).cpu())
        ys.append(batch["score"].float())
    y = torch.cat(ys).numpy()
    out = {}
    for l in layers:
        pr = torch.cat(preds[l]).numpy()
        out[l] = (spearmanr(pr, y)[0], pearsonr(pr, y)[0], pr)
    return out


def attention_probe(model, processor, train_ds, val_ds, test_ds, layers, dim, device,
                    batch_size, epochs, lr, num_heads, max_train):
    """Train one :class:`AttnPool` per target layer (heads share each backbone
    forward), early-stop each head on its own val SRCC, and score on test.

    Returns ``{layer: (test_srcc, test_plcc, test_preds)}``.
    """
    tr = train_ds
    if max_train and len(train_ds) > max_train:
        tr = Subset(train_ds, list(range(max_train)))  # deterministic head subset

    probes = nn.ModuleDict({str(l): AttnPool(dim, num_heads) for l in layers}).to(device).float()
    opt = torch.optim.Adam(probes.parameters(), lr=lr, weight_decay=1e-4)
    best_val = {l: -2.0 for l in layers}
    best_state = {l: None for l in layers}

    for ep in range(epochs):
        for p in probes.values():
            p.train()
        for batch in DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=4):
            pv = _to_pixel_values(processor, batch["image"], model, device)
            with torch.no_grad():
                feats, _ = extract_token_features(model, pv, layers)  # frozen backbone
            y = batch["score"].to(device).float()
            opt.zero_grad()
            loss = sum(F.mse_loss(probes[str(l)](f.float()), y) for l, f in zip(layers, feats))
            loss.backward()
            opt.step()

        val = _attn_eval(probes, model, processor, val_ds, layers, device, batch_size)
        for l in layers:
            if val[l][0] > best_val[l]:
                best_val[l] = val[l][0]
                best_state[l] = copy.deepcopy(probes[str(l)].state_dict())
        print(f"  [attn] epoch {ep + 1}/{epochs} val SRCC "
              + " ".join(f"L{l}={val[l][0]:.3f}" for l in layers))

    for l in layers:
        if best_state[l] is not None:
            probes[str(l)].load_state_dict(best_state[l])
    return _attn_eval(probes, model, processor, test_ds, layers, device, batch_size)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, default="KADID10K",
                   help="within-dataset id: KADID10K, KonIQ_10K, SPAQ, CLIVE, ...")
    p.add_argument("--data_dir", type=str, default="./Dataset")
    p.add_argument("--model_id", type=str, default=MODEL_CONFIG["model_id"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="probe_out")
    p.add_argument("--max_images", type=int, default=None,
                   help="cap images per split for a quick smoke test")
    p.add_argument("--breakdown", choices=["group", "type", "both"], default="both",
                   help="synthetic-set distortion breakdown granularity (KADID/TID)")
    # learned attention probe (PE-style); runs in addition to the linear probe.
    p.add_argument("--attention", action="store_true",
                   help="also fit a learned attention-pool probe on selected layers")
    p.add_argument("--attn_layers", type=str, default="",
                   help="comma-separated layers for the attention probe; "
                        "default = all layers (heads share one backbone forward)")
    p.add_argument("--attn_topk", type=int, default=0,
                   help="restrict to the top-k linear-probe layers + last layer; 0 = all layers")
    p.add_argument("--attn_epochs", type=int, default=20)
    p.add_argument("--attn_lr", type=float, default=1e-3)
    p.add_argument("--attn_heads", type=int, default=8)
    p.add_argument("--attn_max_train", type=int, default=4000,
                   help="cap training images for the attention probe (runtime); 0 = all")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = _make_dataset_paths(args.data_dir)

    print(f"Loading {args.model_id} ...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(args.device).eval()

    H = backbone_num_hidden_layers(model)
    layer_indices = list(range(1, H + 1))  # block outputs 1..H (0 = patch embeddings, skipped)
    print(f"Probing {H} layers on '{args.dataset}' (seed {args.seed}) ...")

    train_ds, val_ds, test_ds = build_splits(args.dataset, paths, args.seed)

    # Synthetic sets (KADID/TID): per-item distortion type of the *test* split
    # (order matches extraction). Parsed from the ``..._TT_..`` filename field.
    groups_map = DISTORTION_GROUPS.get(args.dataset)
    test_types = None
    if groups_map is not None:
        full, idx_te = test_ds.dataset, test_ds.indices
        test_types = np.array([int(full.data.iloc[i]["dist_img"].split("_")[1]) for i in idx_te])

    if args.max_images is not None:
        from torch.utils.data import Subset
        cap = lambda ds: Subset(ds, list(range(min(args.max_images, len(ds)))))
        train_ds, val_ds, test_ds = cap(train_ds), cap(val_ds), cap(test_ds)
        if test_types is not None:
            test_types = test_types[: len(test_ds)]

    print("Extracting features (train/val/test) ...")
    tr_mean, tr_nat, ytr = extract_pooled(model, processor, train_ds, layer_indices, args.device, args.batch_size)
    va_mean, va_nat, yva = extract_pooled(model, processor, val_ds, layer_indices, args.device, args.batch_size)
    te_mean, te_nat, yte = extract_pooled(model, processor, test_ds, layer_indices, args.device, args.batch_size)

    # ---- per-layer probe, both pooling variants ----
    variants = {"mean": (tr_mean, va_mean, te_mean), "native": (tr_nat, va_nat, te_nat)}
    rows, te_preds = [], {}  # te_preds[(variant, layer)] = predictions (for KADID breakdown)
    for name, (Xtr, Xva, Xte) in variants.items():
        for li, layer in enumerate(layer_indices):
            srcc, plcc, pred = ridge_probe(Xtr[:, li], ytr, Xva[:, li], yva, Xte[:, li], yte)
            rows.append({"pooling": name, "layer": layer, "srcc": srcc, "plcc": plcc})
            te_preds[(name, layer)] = pred
            print(f"  [{name:6s}] layer {layer:2d}/{H}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")

    # ---- learned attention probe on selected layers (optional) ----
    if args.attention:
        if args.attn_layers:
            attn_layers = sorted({int(x) for x in args.attn_layers.split(",")})
        elif args.attn_topk > 0:
            mean_srcc = {r["layer"]: r["srcc"] for r in rows if r["pooling"] == "mean"}
            top = sorted(mean_srcc, key=lambda l: -mean_srcc[l])[: args.attn_topk]
            attn_layers = sorted(set(top) | {H})  # top-k plus the final layer
        else:
            attn_layers = list(layer_indices)  # all layers (default)
        print(f"Learned attention probe on layers {attn_layers} "
              f"({args.attn_epochs} epochs, lr {args.attn_lr}) ...")
        res = attention_probe(
            model, processor, train_ds, val_ds, test_ds, attn_layers,
            backbone_hidden_size(model), args.device, args.batch_size,
            args.attn_epochs, args.attn_lr, args.attn_heads,
            args.attn_max_train or None,
        )
        for layer in attn_layers:
            srcc, plcc, pred = res[layer]
            rows.append({"pooling": "attention", "layer": layer, "srcc": srcc, "plcc": plcc})
            te_preds[("attention", layer)] = pred
            print(f"  [attn  ] layer {layer:2d}/{H}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")

    overall_csv = os.path.join(args.out_dir, f"{args.dataset}_layerwise.csv")
    with open(overall_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pooling", "layer", "srcc", "plcc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {overall_csv}")

    # ---- synthetic sets (KADID/TID): per-distortion SRCC at each layer ----
    # ``by_group`` is the readable headline figure; ``by_type`` is the fine detail.
    # (For the group CSV, the ``label`` column equals ``group`` by construction.)
    groupings = _groupings(groups_map, test_types, args.breakdown) if test_types is not None else {}
    for gran, items in groupings.items():
        path = os.path.join(args.out_dir, f"{args.dataset}_by_{gran}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["pooling", "layer", "label", "group", "srcc", "n"])
            writer.writeheader()
            for (name, layer), pred in te_preds.items():
                for label, group, type_ids in items:
                    mask = np.isin(test_types, type_ids)
                    if mask.sum() < 10:
                        continue
                    writer.writerow({"pooling": name, "layer": layer, "label": label,
                                     "group": group, "srcc": spearmanr(pred[mask], yte[mask])[0],
                                     "n": int(mask.sum())})
        print(f"Wrote {path}")

    _try_plot(args, rows, te_preds, test_types, yte, layer_indices, groupings)


def _try_plot(args, rows, te_preds, test_types, yte, layer_indices, groupings):
    """Best-effort matplotlib figures; never blocks CSV output."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping plots: {e})")
        return

    # overall per-layer SRCC, both pooling variants
    plt.figure(figsize=(8, 5))
    for name in ("mean", "native"):
        xs = [r["layer"] for r in rows if r["pooling"] == name]
        ys = [r["srcc"] for r in rows if r["pooling"] == name]
        plt.plot(xs, ys, marker="o", label=f"{name}-pool")
    attn = sorted((r["layer"], r["srcc"]) for r in rows if r["pooling"] == "attention")
    if attn:
        axs, ays = zip(*attn)
        if len(axs) == len(layer_indices):  # full curve -> line
            plt.plot(axs, ays, marker="o", color="crimson", label="attention (learned)")
        else:  # a few selected layers -> markers
            plt.scatter(axs, ays, marker="s", s=70, color="crimson", zorder=5,
                        label="attention (learned)")
    plt.xlabel("layer"); plt.ylabel("test SRCC"); plt.legend()
    plt.title(f"{args.dataset}: layer-wise IQA probe")
    plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(args.out_dir, f"{args.dataset}_layerwise.png")
    plt.savefig(out, dpi=150); print(f"Wrote {out}")

    # Synthetic-set heatmaps: distortion (group and/or type) x layer, mean-pool.
    for gran, items in groupings.items():
        labels = [str(lbl) for lbl, _, _ in items]
        mat = np.full((len(items), len(layer_indices)), np.nan)
        for ri, (_, _, type_ids) in enumerate(items):
            mask = np.isin(test_types, type_ids)
            if mask.sum() < 10:
                continue
            for cj, layer in enumerate(layer_indices):
                mat[ri, cj] = spearmanr(te_preds[("mean", layer)][mask], yte[mask])[0]
        plt.figure(figsize=(11, max(3.0, 0.35 * len(items))))
        im = plt.imshow(mat, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="SRCC")
        plt.yticks(range(len(labels)), labels)
        plt.xticks(range(len(layer_indices)), layer_indices)
        plt.xlabel("layer")
        plt.title(f"{args.dataset}: SRCC by distortion {gran} (mean-pool)")
        plt.tight_layout()
        out = os.path.join(args.out_dir, f"{args.dataset}_heatmap_{gran}.png")
        plt.savefig(out, dpi=150); print(f"Wrote {out}")


if __name__ == "__main__":
    main()
