"""Per-layer summarization-bottleneck study for multi-layer IQA readout.

Question (Part-2 of the paper): ALF summarizes each layer as CLS+AP (one/two
vectors) before fusing across layers, and *acknowledges* this "spatial averaging
may neglect fine-grained spatial details". Does WIDENING that per-layer bottleneck
recover IQA-relevant local-distortion detail?

Everything downstream of the summarizer is identical across arms -- a single
learned-query PMA fuses the per-layer summary tokens, then a linear predicts MOS.
Only the SUMMARIZER changes:

The only thing that changes across arms is how the PATCH grid is summarized --
CLS rides along in every arm (``--keep_cls``, default on) as a constant add-on,
so the variable under study is purely the patch-summary WIDTH:

  * ``ap``   -- mean over patch tokens -> 1 patch vector/layer (width 1). With
                CLS kept, ``ap`` == ALF (CLS + AP): the baseline to beat.
  * ``pma``  -- learned k-query attention pool -> k patch vectors/layer (Set
                Transformer PMA_k). ``--width`` = k.
  * ``tome`` -- parameter-free bipartite token merging -> r patch vectors/layer
                (Bolya et al., ToMe). ``--width`` = r, content-adaptive.

So each layer's summary is ``[CLS?] + <width> patch tokens``. Leading CLS/register
tokens are auto-detected and excluded from the patch summarizer (SigLIP2 has none
-> CLS prepend is a no-op and ``ap`` == mean).

The frozen backbone forwards ONCE; all layers' tokens are cached as an fp16
memmap keyed by (dataset, seed) and REUSED across summarizer configs, so a whole
sweep costs one backbone pass. Read this before train.py: it decides whether the
method leg exists before committing to a LoRA sweep.

Example:
    # build cache once, then sweep configs off it (cache kept by default):
    python bottleneck_probe.py --dataset KADID10K --summarizer mean  --cache_dir /data/bneck_cache
    python bottleneck_probe.py --dataset KADID10K --summarizer pma --width 4 --cache_dir /data/bneck_cache
    python bottleneck_probe.py --dataset KADID10K --summarizer tome --width 4 --cache_dir /data/bneck_cache
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

from configs.default import MODEL_CONFIG, _make_dataset_paths
from dataset import build_splits
from models.multi_layer_fusion import (
    _unwrap_backbone,
    _vision_config,
    backbone_hidden_size,
    backbone_num_hidden_layers,
    extract_token_features,
)
from probe_layers import _build_layer_cache, _to_pixel_values


def num_prefix_tokens(model, ntok):
    """How many leading non-patch tokens the hidden states carry (CLS + any
    register tokens). ``ntok - num_patches``; 0 for SigLIP2 (no CLS)."""
    cfg = _vision_config(_unwrap_backbone(model))
    img = cfg.image_size[0] if isinstance(cfg.image_size, (list, tuple)) else cfg.image_size
    n_patches = (img // cfg.patch_size) ** 2
    return max(0, int(ntok) - n_patches)


# ---------------------------------------------------------------------------
# Parameter-free token merging (ToMe, Bolya et al. ICLR'23): bipartite soft
# matching applied repeatedly until a layer's tokens are reduced to ``target``.
# Merges are size-weighted so a merged token stays an unbiased mean of the
# originals it absorbed. No learned parameters -- allocation is content-adaptive.
# ---------------------------------------------------------------------------
def _merge_step(x, size, r):
    """One bipartite-matching merge: remove ``r`` of the alternating-split
    a-tokens into their most-similar b-token. ``x``: [B,N,D], ``size``: [B,N,1]."""
    B, N, D = x.shape
    m = F.normalize(x, dim=-1)
    a, b = m[:, ::2], m[:, 1::2]                          # [B,na,D], [B,nb,D]
    xa, xb = x[:, ::2], x[:, 1::2]
    sa, sb = size[:, ::2], size[:, 1::2]
    scores = a @ b.transpose(-1, -2)                      # [B,na,nb]
    node_max, node_idx = scores.max(dim=-1)              # best b-match per a-token
    edge = node_max.argsort(dim=-1, descending=True)     # most-similar first
    src_i = edge[:, :r]                                   # a-tokens to merge away
    unm_i = edge[:, r:]                                   # a-tokens kept
    dst_i = node_idx.gather(1, src_i)                     # their b targets

    gd = lambda t, idx, c: t.gather(1, idx[..., None].expand(-1, -1, c))
    num = (xb * sb).clone()                               # weighted sum accumulator
    den = sb.clone()
    src_val = gd(xa, src_i, D) * gd(sa, src_i, 1)
    num.scatter_add_(1, dst_i[..., None].expand(-1, -1, D), src_val)
    den.scatter_add_(1, dst_i[..., None].expand(-1, -1, 1), gd(sa, src_i, 1))
    xb_new, sb_new = num / den, den

    x = torch.cat([gd(xa, unm_i, D), xb_new], dim=1)
    size = torch.cat([gd(sa, unm_i, 1), sb_new], dim=1)
    return x, size


def tome_reduce(x, target):
    """Reduce ``x`` [B,N,D] to [B,target,D] by repeated bipartite merging."""
    if x.shape[1] <= target:
        return x
    size = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
    while x.shape[1] > target:
        n = x.shape[1]
        na = (n + 1) // 2                                # max mergeable this step
        x, size = _merge_step(x, size, min(n - target, na))
    return x


# ---------------------------------------------------------------------------
# Readout head: per-layer summarizer -> add layer embedding -> cross-layer PMA(1)
# -> linear. Only the summarizer differs across arms.
# ---------------------------------------------------------------------------
class Readout(nn.Module):
    def __init__(self, kind, dim, L, width, heads=8, n_prefix=0, keep_cls=True):
        super().__init__()
        self.kind, self.width, self.L, self.n_prefix = kind, width, L, n_prefix
        self.keep_cls = keep_cls and n_prefix >= 1        # CLS available and wanted
        self.in_norm = nn.LayerNorm(dim)                 # kill per-layer scale gap
        if kind == "pma":
            self.q = nn.Parameter(torch.randn(L, width, dim) * dim ** -0.5)
            self.summ_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.layer_emb = nn.Parameter(torch.zeros(L, dim))   # per-layer provenance
        self.cross_q = nn.Parameter(torch.randn(1, 1, dim) * dim ** -0.5)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 1)

    def _summarize(self, x):  # x: [B,L,N,D] -> [B, sum_l t_l, D]
        B, L, N, D = x.shape
        x = self.in_norm(x)
        outs = []
        for li in range(L):
            xl = x[:, li]                                # [B,N,D] (incl. prefix)
            patch = xl[:, self.n_prefix:]               # patch tokens only
            if self.kind == "ap":
                s = patch.mean(1, keepdim=True)          # width 1 (== ALF's AP)
            elif self.kind == "pma":
                q = self.q[li].unsqueeze(0).expand(B, -1, -1)
                s, _ = self.summ_attn(q, patch, patch)   # width k
            elif self.kind == "tome":
                s = tome_reduce(patch, self.width)       # width r
            else:
                raise ValueError(self.kind)
            if self.keep_cls:                            # CLS rides along, every arm
                s = torch.cat([xl[:, :1], s], dim=1)
            outs.append(s + self.layer_emb[li])
        return torch.cat(outs, dim=1)

    def forward(self, x):
        tok = self._summarize(x)                         # [B,T,D]
        q = self.cross_q.expand(x.shape[0], -1, -1)
        pooled, _ = self.cross_attn(q, tok, tok)
        return self.fc(self.norm(pooled[:, 0])).squeeze(-1)


# ---------------------------------------------------------------------------
# Cache: forward the frozen backbone once per split, keyed by (dataset, seed),
# reused across summarizer configs.
# ---------------------------------------------------------------------------
def ensure_cache(model, processor, ds, layers, device, bs, cache_root, split):
    """Return ``(memmap[N,L,Ntok,D], y[N])``; build it if not already on disk."""
    os.makedirs(cache_root, exist_ok=True)
    path = os.path.join(cache_root, f"{split}.f16")
    meta = os.path.join(cache_root, f"{split}.json")
    ypath = os.path.join(cache_root, f"{split}_y.npy")
    if os.path.exists(path) and os.path.exists(meta):
        shape = tuple(json.load(open(meta))["shape"])
        return np.memmap(path, dtype=np.float16, mode="r", shape=shape), np.load(ypath)
    print(f"  [cache] building {split} ({len(ds)} imgs x {len(layers)} layers) -> {path}")
    mm, y = _build_layer_cache(model, processor, ds, layers, device, bs, path)
    json.dump({"shape": list(mm.shape)}, open(meta, "w"))
    np.save(ypath, y)
    return mm, y


@torch.no_grad()
def _eval(head, mm, y, device, bs):
    head.eval()
    preds = []
    for i in range(0, mm.shape[0], bs):
        toks = torch.from_numpy(np.ascontiguousarray(mm[i:i + bs]).copy()).to(device).float()
        preds.append(head(toks).cpu())
    pr = torch.cat(preds).numpy()
    return spearmanr(pr, y)[0], pearsonr(pr, y)[0]


def train_head(kind, width, dim, L, tr_mm, ytr, va_mm, yva, te_mm, yte,
               device, bs, epochs, lr, heads, seed, n_prefix=0, keep_cls=True):
    """Train one Readout on the cached tokens; early-stop on val SRCC; return
    ``(test_srcc, test_plcc, best_val_srcc, n_summary_tokens)``."""
    torch.manual_seed(seed)
    head = Readout(kind, dim, L, width, heads, n_prefix, keep_cls).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    N = tr_mm.shape[0]
    rng = np.random.default_rng(seed)
    best_val, best_state = -2.0, None
    for ep in range(epochs):
        head.train()
        order = rng.permutation(N)
        for i in range(0, N, bs):
            bidx = np.sort(order[i:i + bs])              # sorted -> sequential reads
            toks = torch.from_numpy(np.ascontiguousarray(tr_mm[bidx]).copy()).to(device).float()
            y = torch.from_numpy(ytr[bidx]).to(device).float()
            opt.zero_grad()
            loss = F.mse_loss(head(toks), y)
            loss.backward()
            opt.step()
        vs = _eval(head, va_mm, yva, device, bs)[0]
        if vs > best_val:
            best_val, best_state = vs, copy.deepcopy(head.state_dict())
        print(f"    seed{seed} ep {ep + 1:2d}/{epochs} val SRCC={vs:.4f} (best {best_val:.4f})", flush=True)
    if best_state is not None:
        head.load_state_dict(best_state)
    patch_w = 1 if kind == "ap" else width
    T = L * (patch_w + (1 if head.keep_cls else 0))     # +CLS/layer when kept
    ts, tp = _eval(head, te_mm, yte, device, bs)
    return ts, tp, best_val, T


def train_head_live(kind, width, dim, L, model, processor, train_ds, val_ds, test_ds,
                    layers, device, bs, epochs, lr, heads, seed, n_prefix, keep_cls):
    """No-cache counterpart of :func:`train_head`: the frozen backbone forwards on
    the fly each epoch (only the current batch's tokens on GPU, zero disk). Use
    when the on-disk cache is I/O-bound; costs a backbone pass per epoch instead."""
    torch.manual_seed(seed)
    head = Readout(kind, dim, L, width, heads, n_prefix, keep_cls).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)

    def stack(pv):  # [B,C,H,W] -> [B,L,N,D] on device
        with torch.no_grad():
            feats, _ = extract_token_features(model, pv, layers)
        return torch.stack(feats, dim=1).float()

    @torch.no_grad()
    def ev(ds):
        head.eval()
        preds, ys = [], []
        for batch in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4):
            pv = _to_pixel_values(processor, batch["image"], model, device)
            preds.append(head(stack(pv)).cpu())
            ys.append(batch["score"].float())
        pr, y = torch.cat(preds).numpy(), torch.cat(ys).numpy()
        return spearmanr(pr, y)[0], pearsonr(pr, y)[0]

    best_val, best_state = -2.0, None
    for ep in range(epochs):
        head.train()
        for batch in DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4):
            pv = _to_pixel_values(processor, batch["image"], model, device)
            y = batch["score"].to(device).float()
            opt.zero_grad()
            loss = F.mse_loss(head(stack(pv)), y)
            loss.backward()
            opt.step()
        vs = ev(val_ds)[0]
        if vs > best_val:
            best_val, best_state = vs, copy.deepcopy(head.state_dict())
        print(f"    seed{seed} ep {ep + 1:2d}/{epochs} val SRCC={vs:.4f} (best {best_val:.4f})", flush=True)
    if best_state is not None:
        head.load_state_dict(best_state)
    patch_w = 1 if kind == "ap" else width
    T = L * (patch_w + (1 if head.keep_cls else 0))
    ts, tp = ev(test_ds)
    return ts, tp, best_val, T


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, default="KADID10K")
    p.add_argument("--data_dir", type=str, default="./Dataset")
    p.add_argument("--model_id", type=str, default=MODEL_CONFIG["model_id"])
    p.add_argument("--summarizer", choices=["ap", "pma", "tome"], default="ap",
                   help="patch summarizer; 'ap'+keep_cls == ALF baseline")
    p.add_argument("--width", type=int, default=1, help="k (pma) or r (tome); ap ignores it")
    p.add_argument("--keep_cls", action=argparse.BooleanOptionalAction, default=True,
                   help="prepend CLS to every layer summary when present (no-op on SigLIP)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seeds", type=int, default=1, help="retrain over N seeds, report mean+/-std")
    p.add_argument("--seed", type=int, default=42, help="split seed (also base head seed)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache_dir", type=str, default="",
                   help="fp16 all-layer token cache root (e.g. /data/bneck_cache); "
                        "empty -> no cache: forward the backbone on the fly each epoch "
                        "(use when the cache is disk-I/O-bound)")
    p.add_argument("--max_images", type=int, default=None, help="cap all splits (smoke test)")
    p.add_argument("--build_cache_only", action="store_true", help="build cache and exit")
    p.add_argument("--preload", action="store_true",
                   help="load the whole cache into RAM once (memcpy reads instead of "
                        "per-batch disk); use when the cache fits RAM, e.g. CLIP/DINO")
    p.add_argument("--out_csv", type=str, default="bneck_out/bneck_results.csv")
    args = p.parse_args()

    paths = _make_dataset_paths(args.data_dir)
    print(f"Loading {args.model_id} ...")
    try:
        processor = AutoProcessor.from_pretrained(args.model_id)
    except Exception:
        processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(args.device).eval()

    H = backbone_num_hidden_layers(model)
    dim = backbone_hidden_size(model)
    layers = list(range(1, H + 1))
    train_ds, val_ds, test_ds = build_splits(args.dataset, paths, args.seed)
    if args.max_images is not None:
        cap = lambda ds: Subset(ds, list(range(min(args.max_images, len(ds)))))
        train_ds, val_ds, test_ds = cap(train_ds), cap(val_ds), cap(test_ds)

    model_tag = args.model_id.rstrip("/").split("/")[-1]
    common = (dim, H)  # shared head-shape args

    if args.cache_dir:
        # Cache keyed by dataset+seed (+ smoke cap) so configs reuse one backbone pass.
        tag = f"_cap{args.max_images}" if args.max_images is not None else ""
        cache_root = os.path.join(args.cache_dir, f"{args.dataset}_s{args.seed}_{model_tag}{tag}")
        ec = lambda ds, split: ensure_cache(model, processor, ds, layers, args.device,
                                            args.batch_size, cache_root, split)
        tr_mm, ytr = ec(train_ds, "train")
        va_mm, yva = ec(val_ds, "val")
        te_mm, yte = ec(test_ds, "test")
        if args.build_cache_only:
            print(f"Cache ready at {cache_root}")
            return
        if args.preload:  # pull the whole cache into RAM once -> reads are memcpy,
            gb = sum(m.nbytes for m in (tr_mm, va_mm, te_mm)) / 1e9  # not per-batch disk
            print(f"  [preload] loading {gb:.0f} GB cache into RAM ...", flush=True)
            tr_mm, va_mm, te_mm = np.array(tr_mm), np.array(va_mm), np.array(te_mm)
        ntok = tr_mm.shape[2]
        fit = lambda s: train_head(args.summarizer, args.width, *common, tr_mm, ytr,
                                   va_mm, yva, te_mm, yte, args.device, args.batch_size,
                                   args.epochs, args.lr, args.heads, s, n_prefix, args.keep_cls)
    else:
        if args.build_cache_only:
            p.error("--build_cache_only requires --cache_dir")
        # No cache: get Ntok from a single forward, then forward per epoch.
        pv0 = _to_pixel_values(processor, train_ds[0]["image"].unsqueeze(0), model, args.device)
        ntok = extract_token_features(model, pv0, layers[:1])[0][0].shape[1]
        fit = lambda s: train_head_live(args.summarizer, args.width, *common, model,
                                        processor, train_ds, val_ds, test_ds, layers,
                                        args.device, args.batch_size, args.epochs, args.lr,
                                        args.heads, s, n_prefix, args.keep_cls)

    n_prefix = num_prefix_tokens(model, ntok)  # CLS/register tokens, 0 for SigLIP
    cls_on = args.keep_cls and n_prefix >= 1
    mode = "cached" if args.cache_dir else "live (no cache)"
    print(f"[{args.dataset}] {args.summarizer} width={args.width} keep_cls={cls_on} [{mode}]  "
          f"{H} layers, D={dim}, Ntok={ntok}, prefix={n_prefix} "
          f"({'CLS present' if n_prefix else 'no CLS'}), {args.seeds} seed(s), {args.epochs} epochs")
    t0 = time.time()
    results = [fit(args.seed + s) for s in range(args.seeds)]
    srccs = np.array([r[0] for r in results])
    plccs = np.array([r[1] for r in results])
    T = results[0][3]
    print(f"  test SRCC={srccs.mean():.4f}+/-{srccs.std():.4f}  "
          f"PLCC={plccs.mean():.4f}  tokens={T}  ({time.time() - t0:.0f}s)")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    new = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "dataset", "summarizer", "width", "tokens", "test_SRCC",
                        "test_SRCC_std", "test_PLCC", "val_SRCC"])
        w.writerow([model_tag, args.dataset, args.summarizer, args.width, T,
                    f"{srccs.mean():.4f}", f"{srccs.std():.4f}",
                    f"{plccs.mean():.4f}", f"{max(r[2] for r in results):.4f}"])
    print(f"Appended -> {args.out_csv}")


if __name__ == "__main__":
    main()
