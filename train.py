"""
Training script for NR-IQA with Activation-Gated MLP (AGM).

Backbone : SigLIP-2 vision encoder  (PEFT: LoRA or Deep Prompt Tuning)
Head     : MLP3_Gated (learnable activation gating)
Loss     : MSE + pair-wise margin loss

Usage examples
--------------
    # Train on KonIQ-10K with default settings
    python train.py --dataset KonIQ_10K

    # Train on CLIVE, LoRA rank 8, 20 epochs
    python train.py --dataset CLIVE --peft_method LoRA --lora_r 8 --epochs 20

    # Cross-dataset: train on KonIQ-10K, evaluate on CLIVE
    python train.py --dataset KonIQ_10K_CLIVE

    # Resume from a previous run
    python train.py --dataset KonIQ_10K --resume

    # Dry-run (100 train steps per epoch) for debugging
    python train.py --dataset CLIVE --dry_run

Author: Ankit Yadav
"""

import argparse
import copy
import json
import os
import warnings

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from glob import glob
from peft import (
    LoraConfig,
    PromptEncoderConfig,
    TaskType,
    get_peft_model,
)
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

from tracker import Tracker

from configs.default import MODEL_CONFIG, TRAIN_CONFIG, DATASET_PATHS, _make_dataset_paths
from dataset import build_splits
from models import MLP3_Gated, SIGLIPWithMLP, MultiLayerFusion, extract_token_features, native_pool
from models import DualEncoderFusion, extract_trunk, extract_aux_tokens, aux_hidden_size
from models.activations import ParamSigmoid2, ParamLeakyReLU2
from seed import Seed, seed_worker
from util import margin_loss, metric, Overlay, BAD_QUALITY_PROMPT, Text_Template_baseline

warnings.simplefilter(action="ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _clean_old_checkpoints(stage_name: str, max_keep: int):
    for pattern in (f"checkpoints/{stage_name}_step*.pt",
                    f"checkpoints/{stage_name}_step*/"):
        items = sorted(glob(pattern), key=os.path.getctime)
        if len(items) > max_keep:
            for p in items[:-max_keep]:
                os.system(f"rm -rf {p}")


def _db_name(loader):
    """Walk through Subset wrappers to find the underlying ``db_name``."""
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return getattr(ds, "db_name", "unknown")


def _fusion_diagnostics(fusion_module, layer_indices):
    """Per-epoch fusion internals for tracking (gate trajectory, layer weights).

    Confirms whether a step-0-identical fusion actually turns on: if
    ``fusion/gate`` (adaptive) or ``fusion/out_proj_norm`` (cross_attention)
    stays ~0, the residual is switched off and the fusion is a no-op.
    """
    if fusion_module is None:
        return {}
    f = fusion_module.fuser
    diags = {}
    if hasattr(f, "gate"):                                  # adaptive
        diags["fusion/gate"] = f.gate.detach().float().mean().item()
    if getattr(f, "alpha", None) is not None:               # adaptive 'static'
        w = (torch.softmax(f.alpha.detach().float(), dim=-1) if f.norm == "softmax"
             else torch.sigmoid(f.alpha.detach().float()))
        for idx, wl in zip(layer_indices, w.tolist()):
            diags[f"fusion/w_layer{idx}"] = wl
    if hasattr(f, "out_proj"):                              # cross_attention
        diags["fusion/out_proj_norm"] = f.out_proj.weight.detach().float().norm().item()
    return diags


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, mlp, processor, dataloader_eval, device, dry_run=False, fusion=None,
            aux_model=None, aux_processor=None, dual_fusion=None):
    """Run inference on *dataloader_eval* and return (results_dict, avg_loss).

    The model, MLP and (optional) fusion are deep-copied so evaluation doesn't
    affect the training-mode state. The frozen ``aux_model`` (dual-encoder mode)
    is used as-is. Prediction is dispatched through a ``predict`` closure so the
    non-dual path keeps the exact ``SIGLIPWithMLP`` behaviour.
    """
    model_copy = copy.deepcopy(model)
    mlp_copy   = copy.deepcopy(mlp)
    base = model_copy.module if hasattr(model_copy, "module") else model_copy
    base = base.float()
    mlp_f = mlp_copy.float()

    if dual_fusion is not None:
        dfz = copy.deepcopy(dual_fusion).float().to(device).eval()
        base = base.to(device).eval()
        mlp_f = mlp_f.to(device).eval()

        def predict(images):
            inputs     = processor(images=images, return_tensors="pt").to(device)
            aux_inputs = aux_processor(images=images, return_tensors="pt").to(device)
            trunk      = extract_trunk(base, inputs["pixel_values"])
            aux_tokens = extract_aux_tokens(aux_model, aux_inputs["pixel_values"])
            feats      = native_pool(base, dfz(trunk, aux_tokens))
            return mlp_f(feats).squeeze(1)
    else:
        fusion_copy = copy.deepcopy(fusion).float() if fusion is not None else None
        combined = SIGLIPWithMLP(
            base_model=base, mlp_head=mlp_f, device=device, fusion=fusion_copy,
        ).to(device).eval()

        def predict(images):
            inputs = processor(images=images, return_tensors="pt").to(device)
            return combined(inputs["pixel_values"])

    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    cnt = 0

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        for batch in tqdm(dataloader_eval, desc="Evaluating"):
            if dry_run:
                cnt += 1
                if cnt >= 32:
                    break

            images   = batch["image"].to(device)
            gt       = batch["score"].to(device)
            preds    = predict(images)

            loss_mse = torch.nn.functional.mse_loss(preds, gt)
            loss_mrg = margin_loss(gt, preds)
            total_loss += (loss_mse + loss_mrg).item()
            n_batches  += 1

            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(gt.float().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    m = metric()
    m.calcuate_srcc(y_true, y_pred)
    m.calculate_plcc(y_true, y_pred)

    avg_loss = total_loss / max(n_batches, 1)
    results  = dict(m.result)
    results["avg_eval_loss"] = float(avg_loss)

    del model_copy, mlp_copy
    torch.cuda.empty_cache()
    return results, avg_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    cfg = {**TRAIN_CONFIG}
    cfg.update({
        "epochs":                       args.epochs,
        "batch_size":                   args.batch_size,
        "learning_rate":                args.lr,
        "backbone_lr":                  (args.backbone_lr if args.backbone_lr is not None
                                         else (TRAIN_CONFIG["full_ft_backbone_lr"]
                                               if args.peft_method == "NA" else args.lr)),
        "weight_decay":                 args.weight_decay,
        "checkpoint_steps":             args.checkpoint_steps,
        "max_checkpoints":              args.max_checkpoints,
        "stage_name":                   args.stage_name,
        "peft_method":                  args.peft_method,
        "gradient_accumulation_steps":  args.grad_accum,
        "do_eval":                      not args.no_eval,
        "eval_epoch_steps":             args.eval_every,
        "lr_scheduler":                 not args.no_scheduler,
        "lr_scheduler_milestones":      list(map(int, args.lr_milestones.split(","))),
        "use_gradient_clip":            args.gradient_clip > 0,
        "gradient_clip":                args.gradient_clip,
        "resume":                       args.resume,
        "dry_run":                      args.dry_run,
        "tracker":                      ("none" if args.no_wandb else args.tracker),
        "project":                      args.project,
    })
    if args.lora_r is not None:
        cfg["lora_config"]["r"]            = args.lora_r
        cfg["lora_config"]["lora_alpha"]   = args.lora_alpha
        cfg["lora_config"]["lora_dropout"] = args.lora_dropout
    if args.lora_targets is not None:
        if args.lora_targets.strip() == "all-linear":
            cfg["lora_config"]["target_modules"] = "all-linear"
        else:
            names = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
            cfg["lora_config"]["target_modules"] = rf"vision_model\..*\.({'|'.join(names)})$"

    dataset_paths = _make_dataset_paths(args.data_dir)

    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])

    # ── Processor ────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(args.model_id)

    # ── Backbone ─────────────────────────────────────────────────────────
    model = AutoModel.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)

    # ── PEFT ─────────────────────────────────────────────────────────────
    if cfg["peft_method"] == "LoRA":
        print(f"Applying LoRA (targets={cfg['lora_config']['target_modules']}) ...")
        lora_cfg = LoraConfig(
            r=cfg["lora_config"]["r"],
            lora_alpha=cfg["lora_config"]["lora_alpha"],
            lora_dropout=cfg["lora_config"]["lora_dropout"],
            target_modules=cfg["lora_config"]["target_modules"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    elif cfg["peft_method"] == "DPT":
        print("Applying Deep Prompt Tuning ...")
        v_cfg = model.vision_model.config
        model.config.vocab_size = model.config.text_config.vocab_size
        dpt_cfg = PromptEncoderConfig(
            peft_type="P_TUNING",
            task_type=TaskType.FEATURE_EXTRACTION,
            num_layers=v_cfg.num_hidden_layers,
            num_virtual_tokens=cfg["dpt_config"]["no_learnable_tokens"],
            encoder_reparameterization_type="MLP",
            token_dim=v_cfg.hidden_size,
            num_transformer_submodules=v_cfg.num_hidden_layers,
            num_attention_heads=v_cfg.num_attention_heads,
        )
        model = get_peft_model(model, dpt_cfg)
        model.print_trainable_parameters()
    else:
        print("Full fine-tuning (no PEFT adapter).")
        model.requires_grad_(True)

    # ── MLP head ─────────────────────────────────────────────────────────
    mlp = MLP3_Gated(input_dim=args.mlp_input_dim).to(device).to(torch.bfloat16)
    with torch.no_grad():
        for m in mlp.modules():
            if isinstance(m, (ParamSigmoid2, ParamLeakyReLU2)):
                m.float()

    mlp.requires_grad_(True)

    # ── Multi-layer fusion (optional) ────────────────────────────────────
    fusion = None
    if args.fusion_type != "none":
        explicit_layers = None
        if args.fusion_layers is not None:
            explicit_layers = [int(x) for x in args.fusion_layers.split(",") if x.strip()]
        fusion = MultiLayerFusion.from_backbone(
            model,
            fusion_type=args.fusion_type,
            stride=args.fusion_stride,
            layer_indices=explicit_layers,
            first_n=args.fusion_first_n,
            last_n=args.fusion_last_n,
            adaptive_conditioning=args.adaptive_conditioning,
            adaptive_norm=args.adaptive_norm,
            fusion_num_heads=args.fusion_num_heads,
            fusion_gate_init=args.fusion_gate_init,
            dropout=args.fusion_dropout,
            alf_use_cls=args.alf_use_cls,
            fusion_query_layer=args.fusion_query_layer,
            summarizer=args.summarizer,
            summary_width=args.summary_width,
        ).to(device).to(torch.bfloat16)
        fusion.requires_grad_(True)
        print(f"Multi-layer fusion: type={args.fusion_type} "
              f"layers(hidden_states idx)={fusion.layer_indices} "
              f"conditioning={args.adaptive_conditioning} norm={args.adaptive_norm} "
              f"num_heads={args.fusion_num_heads} dropout={args.fusion_dropout} "
              f"gate_init={args.fusion_gate_init}")

    # ── Dual-encoder cross-attention fusion (optional) ───────────────────
    # A frozen auxiliary encoder (e.g. DINOv2/DINOv3) provides key/value tokens;
    # the SigLIP trunk queries them. Enabled by --aux_model_id (HF id or a local
    # directory, e.g. DINOv3 from disk). Mutually exclusive with --fusion_type.
    aux_model = aux_processor = dual_fusion = None
    if args.aux_model_id is not None:
        if args.fusion_type != "none":
            raise ValueError(
                "--aux_model_id (dual encoder) and --fusion_type (intra-encoder "
                "multi-layer fusion) are mutually exclusive; enable one at a time."
            )
        print(f"Dual encoder: loading frozen aux backbone '{args.aux_model_id}' ...")
        aux_model = AutoModel.from_pretrained(
            args.aux_model_id, torch_dtype=torch.bfloat16,
            trust_remote_code=args.aux_trust_remote_code,
        ).to(device)
        aux_model.requires_grad_(False)
        aux_model.eval()

        proc_id = args.aux_processor_id or args.aux_model_id
        try:
            aux_processor = AutoProcessor.from_pretrained(
                proc_id, trust_remote_code=args.aux_trust_remote_code)
        except Exception:
            aux_processor = AutoImageProcessor.from_pretrained(
                proc_id, trust_remote_code=args.aux_trust_remote_code)

        dual_fusion = DualEncoderFusion(
            q_dim=args.mlp_input_dim,
            kv_dim=aux_hidden_size(aux_model),
            aux_model_id=args.aux_model_id,
            num_heads=args.aux_num_heads,
            dropout=args.aux_dropout,
            gate_init=args.aux_gate_init,
        ).to(device).to(torch.bfloat16)
        dual_fusion.requires_grad_(True)
        print(f"Dual encoder: q_dim={args.mlp_input_dim} "
              f"kv_dim={aux_hidden_size(aux_model)} num_heads={args.aux_num_heads} "
              f"dropout={args.aux_dropout} gate_init={args.aux_gate_init}")

    # ── Dataset / DataLoader ─────────────────────────────────────────────
    # val drives best-checkpoint selection; test is held out for final reporting.
    train_ds, val_ds, test_ds = build_splits(args.dataset, dataset_paths, Seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True, worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        drop_last=False, worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        drop_last=False, worker_init_fn=seed_worker,
    )

    # ── Optimizer / Scheduler ────────────────────────────────────────────
    # Two param groups: the backbone (group 0) and the head — MLP + optional
    # fusion (group 1). For LoRA/DPT both default to --lr; for full FT the
    # backbone uses the paper's conservative 5e-6 while the head stays at --lr.
    head_params = list(mlp.parameters())
    if fusion is not None:
        head_params += list(fusion.parameters())
    if dual_fusion is not None:
        head_params += list(dual_fusion.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": list(model.parameters()), "lr": cfg["backbone_lr"]},
            {"params": head_params,              "lr": cfg["learning_rate"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    print(f"Optimizer LRs — backbone: {cfg['backbone_lr']:.2e}  head: {cfg['learning_rate']:.2e}")
    scheduler = MultiStepLR(
        optimizer,
        milestones=cfg["lr_scheduler_milestones"],
        gamma=0.2,
    )

    # ── Resume state ─────────────────────────────────────────────────────
    resume_path     = f"resume_state/{cfg['stage_name']}_latest.pt"
    global_step     = 0
    start_epoch     = 0
    best_eval_loss  = float("inf")
    patience_ctr    = 0
    best_val_SRCC   = float("-inf")
    best_val_PLCC   = float("-inf")
    best_test_SRCC  = float("-inf")
    best_test_PLCC  = float("-inf")
    best_epoch      = 0
    ckpt            = None

    if cfg["resume"] and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        best_eval_loss = ckpt["best_eval_loss"]
        patience_ctr   = ckpt["patience_counter"]
        start_epoch    = ckpt["epoch"]
        global_step    = ckpt["global_step"]
        best_val_SRCC  = ckpt.get("best_val_SRCC", best_val_SRCC)
        best_val_PLCC  = ckpt.get("best_val_PLCC", best_val_PLCC)
        best_test_SRCC = ckpt.get("best_test_SRCC", best_test_SRCC)
        best_test_PLCC = ckpt.get("best_test_PLCC", best_test_PLCC)
        best_epoch     = ckpt.get("best_epoch", best_epoch)
        print(f"Resuming from {resume_path} (epoch {start_epoch}, step {global_step})")
    else:
        print("Starting training from scratch.")

    model.train()
    mlp.train()
    if fusion is not None:
        fusion.train()
    if dual_fusion is not None:
        dual_fusion.train()

    # The tapped indices are needed in the training loop; capture them before
    # `prepare` wraps `fusion` (the wrapper hides plain attributes).
    fusion_layer_indices = fusion.layer_indices if fusion is not None else None
    # ALF returns the pooled [B, D] vector directly (skip native_pool).
    fusion_returns_pooled = getattr(fusion, "returns_pooled", False) if fusion is not None else False

    # ── Accelerator prepare ──────────────────────────────────────────────
    # fusion and dual_fusion are mutually exclusive (enforced at build time).
    if fusion is not None:
        model, mlp, fusion, optimizer, scheduler, train_loader = accelerator.prepare(
            model, mlp, fusion, optimizer, scheduler, train_loader,
        )
    elif dual_fusion is not None:
        model, mlp, dual_fusion, optimizer, scheduler, train_loader = accelerator.prepare(
            model, mlp, dual_fusion, optimizer, scheduler, train_loader,
        )
    else:
        model, mlp, optimizer, scheduler, train_loader = accelerator.prepare(
            model, mlp, optimizer, scheduler, train_loader,
        )

    if ckpt is not None:
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        mlp.load_state_dict(ckpt["mlp_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if fusion is not None and ckpt.get("fusion_state_dict") is not None:
            accelerator.unwrap_model(fusion).load_state_dict(ckpt["fusion_state_dict"])
        if dual_fusion is not None and ckpt.get("dual_fusion_state_dict") is not None:
            accelerator.unwrap_model(dual_fusion).load_state_dict(ckpt["dual_fusion_state_dict"])

    # ── Experiment tracker (Aim by default) ──────────────────────────────
    tracker = Tracker(
        cfg["tracker"],
        project=cfg["project"],
        name=f"{cfg['stage_name']}_{args.dataset}",
        config=cfg,
        enabled=accelerator.is_main_process,
    )
    tracker.log_code(__file__)

    train_db = _db_name(train_loader)
    val_db   = _db_name(val_loader)
    test_db  = _db_name(test_loader)
    stage    = cfg["stage_name"]

    def _eval(loader):
        return evaluate(
            model, mlp, processor, loader, device, dry_run=cfg["dry_run"],
            fusion=(accelerator.unwrap_model(fusion) if fusion is not None else None),
            aux_model=aux_model, aux_processor=aux_processor,
            dual_fusion=(accelerator.unwrap_model(dual_fusion)
                         if dual_fusion is not None else None),
        )

    # ── Epoch loop ───────────────────────────────────────────────────────
    for epoch in tqdm(range(start_epoch, cfg["epochs"]), desc="Epochs",
                      total=cfg["epochs"], initial=start_epoch):
        dry_cnt = 0
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            total_loss = 0.0
            accum_loss = 0.0
            optimizer.zero_grad()

            for batch in tqdm(train_loader, desc="Training"):
                if cfg["dry_run"]:
                    dry_cnt += 1
                    if dry_cnt >= 100:
                        break

                images = batch["image"].to(device)
                inputs = processor(images=images, return_tensors="pt").to(model.device)

                if dual_fusion is not None:
                    aux_inputs = aux_processor(images=images, return_tensors="pt").to(device)
                    trunk = extract_trunk(model, inputs["pixel_values"])
                    with torch.no_grad():
                        aux_tokens = extract_aux_tokens(aux_model, aux_inputs["pixel_values"])
                    features = native_pool(model, dual_fusion(trunk, aux_tokens))
                elif fusion is not None:
                    feats, trunk = extract_token_features(
                        model, inputs["pixel_values"], fusion_layer_indices
                    )
                    fused = fusion(feats, trunk)
                    features = fused if fusion_returns_pooled else native_pool(model, fused)
                else:
                    try:
                        features = model.module.get_image_features(**inputs)
                    except Exception:
                        features = model.get_image_features(**inputs)

                score = mlp(features)
                loss_mse    = torch.nn.functional.mse_loss(score.squeeze(1), batch["score"].to(device))
                loss_margin = margin_loss(batch["score"].to(device), score.squeeze(1))
                loss = (loss_mse + loss_margin) / cfg["gradient_accumulation_steps"]

                accelerator.backward(loss)
                accum_loss += loss.item()
                total_loss += loss.item() * cfg["gradient_accumulation_steps"]
                global_step += 1

                # Gate weight logging
                if tracker.enabled:
                    try:
                        w_mean = torch.sigmoid(mlp.module.act1.g).mean().item()
                    except AttributeError:
                        w_mean = torch.sigmoid(mlp.act1.g).mean().item()
                    tracker.log({"gate/w_mean": w_mean}, step=global_step)

                if global_step % cfg["gradient_accumulation_steps"] == 0:
                    if cfg["use_gradient_clip"]:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    optimizer.step()
                    optimizer.zero_grad()
                    tqdm.write(f"Step {global_step} — accum loss: {accum_loss:.4f}")
                    if tracker.enabled:
                        tracker.log({f"{stage}_accumulated_loss": accum_loss}, step=global_step)
                    accum_loss = 0.0

                # Periodic checkpoint
                if global_step % cfg["checkpoint_steps"] == 0 and accelerator.is_main_process:
                    os.makedirs("checkpoints", exist_ok=True)
                    ckpt_dir = f"checkpoints/{stage}_step_train_{train_db}_Test{test_db}_{global_step}/"
                    accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                    torch.save(mlp.state_dict(), f"{ckpt_dir}/mlp.pt")
                    if fusion is not None:
                        accelerator.unwrap_model(fusion).save(f"{ckpt_dir}/fusion.pt")
                    if dual_fusion is not None:
                        accelerator.unwrap_model(dual_fusion).save(f"{ckpt_dir}/dual_fusion.pt")
                    _clean_old_checkpoints(stage, cfg["max_checkpoints"])

                # Per-step logging
                if tracker.enabled:
                    tracker.log({
                        f"{stage}_loss":  loss.item(),
                        f"{stage}_epoch": epoch + 1,
                        f"{stage}_step":  global_step,
                        f"{stage}_lr":    optimizer.param_groups[0]["lr"],
                    }, step=global_step)

            # End of epoch
            avg = total_loss / max(len(train_loader), 1)
            if tracker.enabled:
                tracker.log({f"{stage}_total_loss": avg, f"{stage}_epoch": epoch + 1}, step=global_step)
            print(f"Epoch {epoch+1}/{cfg['epochs']} — avg loss: {avg:.4f}")

            if cfg["lr_scheduler"]:
                scheduler.step()

            # ── Validation (drives best-checkpoint selection) ────────────
            improved = False
            if cfg["do_eval"] and epoch % cfg["eval_epoch_steps"] == 0:
                val_results, val_loss = _eval(val_loader)
                val_SRCC, val_PLCC = val_results["SRCC"], val_results["PLCC"]
                if val_SRCC > best_val_SRCC:
                    best_val_SRCC, best_val_PLCC, best_epoch = val_SRCC, val_PLCC, epoch + 1
                    improved = True
                    # Test the just-selected model. Test never drives selection.
                    test_results, _ = _eval(test_loader)
                    best_test_SRCC, best_test_PLCC = test_results["SRCC"], test_results["PLCC"]
                if accelerator.is_main_process:
                    tqdm.write(f"  Epoch {epoch+1} — val SRCC: {val_SRCC:.4f}  val PLCC: {val_PLCC:.4f}  "
                               f"val_loss: {val_loss:.4f}  | best val SRCC: {best_val_SRCC:.4f} "
                               f"@ epoch {best_epoch}  (test SRCC: {best_test_SRCC:.4f} "
                               f"PLCC: {best_test_PLCC:.4f})")
                    log = {
                        f"{stage}_val_SRCC": val_SRCC,
                        f"{stage}_val_PLCC": val_PLCC,
                        f"{stage}_val_loss": val_loss,
                        f"{stage}_best_val_SRCC": best_val_SRCC,
                        f"{stage}_best_val_PLCC": best_val_PLCC,
                    }
                    if improved:
                        log[f"{stage}_test_SRCC"] = best_test_SRCC
                        log[f"{stage}_test_PLCC"] = best_test_PLCC
                    if fusion is not None:
                        log.update(_fusion_diagnostics(
                            accelerator.unwrap_model(fusion), fusion_layer_indices))
                    if dual_fusion is not None:
                        log.update(_fusion_diagnostics(
                            accelerator.unwrap_model(dual_fusion), None))
                    tracker.log(log, step=global_step)

            # ── Best checkpoint (selected on validation) ──────────────────
            if accelerator.is_main_process and improved:
                os.makedirs("best_checkpoints", exist_ok=True)
                best_dir = f"best_checkpoints/{stage}_train_{train_db}_test_{test_db}"
                accelerator.unwrap_model(model).save_pretrained(best_dir)
                torch.save(mlp.state_dict(), f"{best_dir}/mlp.pt")
                if fusion is not None:
                    accelerator.unwrap_model(fusion).save(f"{best_dir}/fusion.pt")
                if dual_fusion is not None:
                    accelerator.unwrap_model(dual_fusion).save(f"{best_dir}/dual_fusion.pt")
                tqdm.write(f"  [Best] val SRCC={best_val_SRCC:.4f} "
                           f"(test SRCC={best_test_SRCC:.4f}) saved to {best_dir}")

            # ── Resume state at epoch end ─────────────────────────────────
            if accelerator.is_main_process:
                os.makedirs("resume_state", exist_ok=True)
                torch.save({
                    "epoch":              epoch + 1,
                    "global_step":        global_step,
                    "model_state_dict":   accelerator.unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_eval_loss":     best_eval_loss,
                    "patience_counter":   patience_ctr,
                    "mlp_state_dict":     mlp.state_dict(),
                    "best_val_SRCC":      best_val_SRCC,
                    "best_val_PLCC":      best_val_PLCC,
                    "best_test_SRCC":     best_test_SRCC,
                    "best_test_PLCC":     best_test_PLCC,
                    "best_epoch":         best_epoch,
                    "fusion_state_dict":  (accelerator.unwrap_model(fusion).state_dict()
                                           if fusion is not None else None),
                    "dual_fusion_state_dict": (accelerator.unwrap_model(dual_fusion).state_dict()
                                               if dual_fusion is not None else None),
                }, resume_path)

    # ── Save final checkpoint + report the val-selected best ─────────────
    if accelerator.is_main_process:
        os.makedirs("checkpoints", exist_ok=True)
        final_dir = f"checkpoints/{stage}_final_train_{train_db}_Test{test_db}/"
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        torch.save(mlp.state_dict(), f"{final_dir}/mlp.pt")
        if fusion is not None:
            accelerator.unwrap_model(fusion).save(f"{final_dir}/fusion.pt")
        if dual_fusion is not None:
            accelerator.unwrap_model(dual_fusion).save(f"{final_dir}/dual_fusion.pt")

        if cfg["do_eval"]:
            results = {
                "dataset":    args.dataset,
                "train_db":   train_db,
                "val_db":     val_db,
                "test_db":    test_db,
                "best_epoch": int(best_epoch),
                "val_SRCC":   float(best_val_SRCC),
                "val_PLCC":   float(best_val_PLCC),
                "test_SRCC":  float(best_test_SRCC),
                "test_PLCC":  float(best_test_PLCC),
            }
            os.makedirs("results", exist_ok=True)
            res_path = f"results/results_{stage}_Train_{train_db}_Test_{test_db}.json"
            with open(res_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {res_path}  (best epoch {best_epoch}: "
                  f"val SRCC={best_val_SRCC:.4f}, test SRCC={best_test_SRCC:.4f})")

    tracker.finish()

    accelerator.free_memory()
    torch.cuda.empty_cache()
    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train NR-IQA AGM model")

    # Dataset
    p.add_argument("--dataset", type=str, required=True,
                   help="Dataset id: CLIVE, KonIQ_10K, SPAQ, KADID10K, FLIVE, "
                        "AGIQA3K, AGIQA1K, KonIQ_10K_CLIVE, CLIVE_KonIQ_10K")
    p.add_argument("--data_dir", type=str, default="./Dataset",
                   help="Root directory containing all dataset folders")

    # Model
    p.add_argument("--model_id", type=str, default=MODEL_CONFIG["model_id"])
    p.add_argument("--mlp_input_dim", type=int, default=MODEL_CONFIG["mlp_input_dim"])

    # Multi-layer fusion
    p.add_argument("--fusion_type", type=str, default="none",
                   choices=["none", "mls", "soft_mls", "adaptive", "cross_attention",
                            "alf", "summary"],
                   help="Multi-layer feature fusion before the MLP head. "
                        "'none' = vanilla single-layer get_image_features; "
                        "'mls' = RAE-V2 multi-layer sum (hard replace); "
                        "'soft_mls' = trunk + alpha*MLS residual (keeps the trunk; "
                        "alpha init = --fusion_gate_init); "
                        "'adaptive' = learned weighted residual (step-0 identical); "
                        "'cross_attention' = trunk queries the tapped layers' tokens, "
                        "residual via zero-init out_proj (step-0 identical); "
                        "'alf' = attentive fusion of per-layer CLS+AP summaries; "
                        "'summary' = widened per-layer summary (--summarizer ap/pma/tome, "
                        "--summary_width) then attentive cross-layer fusion -- ALF's "
                        "power at a fraction of cross_attention's K/V tokens.")
    p.add_argument("--fusion_stride", type=int, default=4,
                   help="Tap every Nth block, right-anchored on the last block "
                        "(stride=1 taps all layers). Resolved per-backbone from depth.")
    p.add_argument("--fusion_layers", type=str, default=None,
                   help="Explicit hidden_states block indices to tap, comma-separated "
                        "(e.g. '3,7,11,27'; valid range 1..num_hidden_layers, 0 is the "
                        "patch embedding). Overrides --fusion_stride when set.")
    p.add_argument("--fusion_first_n", type=int, default=None,
                   help="Tap the first N transformer blocks (hidden_states 1..N). Overrides "
                        "--fusion_stride; mutually exclusive with --fusion_layers/--fusion_last_n.")
    p.add_argument("--fusion_last_n", type=int, default=None,
                   help="Tap the last N transformer blocks (hidden_states H-N+1..H). Overrides "
                        "--fusion_stride; mutually exclusive with --fusion_layers/--fusion_first_n.")
    p.add_argument("--adaptive_conditioning", type=str, default="static",
                   choices=["uniform", "static", "image"],
                   help="Adaptive weight source: 'uniform' (fixed 1/L baseline), "
                        "'static' (learned [L] vector), 'image' (MLP of pooled trunk).")
    p.add_argument("--adaptive_norm", type=str, default="softmax",
                   choices=["softmax", "sigmoid"],
                   help="Adaptive weight normalisation.")
    p.add_argument("--fusion_num_heads", type=int, default=8,
                   help="Number of heads for --fusion_type cross_attention "
                        "(must divide the backbone hidden size).")
    p.add_argument("--fusion_dropout", type=float, default=0.0,
                   help="Dropout inside the fusion module (cross_attention / "
                        "adaptive image conditioning).")
    p.add_argument("--fusion_query_layer", type=int, default=None,
                   help="cross_attention only: use this tapped hidden_states layer index "
                        "as the attention QUERY instead of the final-layer trunk (must be "
                        "one of --fusion_layers). Tests querying from a quality-rich "
                        "intermediate layer rather than the semantically-invariant output.")
    p.add_argument("--alf_use_cls", action="store_true",
                   help="alf/summary: keep the CLS token in each per-layer summary "
                        "(faithful for CLS-bearing backbones like CLIP/DINO). Default "
                        "AP/patch-only (SigLIP2 has no CLS).")
    p.add_argument("--summarizer", type=str, default="pma", choices=["ap", "pma", "tome"],
                   help="--fusion_type summary: per-layer patch summarizer. 'ap' = mean "
                        "(== ALF), 'pma' = learned k-query attention pool, 'tome' = "
                        "parameter-free bipartite token merge.")
    p.add_argument("--summary_width", type=int, default=4,
                   help="--fusion_type summary: tokens per layer for pma (k) / tome (r); "
                        "'ap' ignores it.")
    p.add_argument("--fusion_gate_init", type=float, default=0.0,
                   help="Warm-start the fusion residual. 0.0 = step-0 identical (residual "
                        "off); a small positive value (e.g. 0.1) turns it on at init so the "
                        "fusion isn't born switched off. Applies to adaptive (gate value) "
                        "and cross_attention (out_proj init scale).")

    # Dual encoder (SigLIP trunk queries a frozen aux encoder's tokens)
    p.add_argument("--aux_model_id", type=str, default=None,
                   help="Auxiliary vision encoder for dual-encoder cross-attention fusion: "
                        "a HuggingFace id or a local directory (e.g. DINOv3 from disk). "
                        "When set, the frozen aux encoder supplies key/value tokens and the "
                        "SigLIP trunk queries them (step-0 identical via zero-init out_proj). "
                        "Mutually exclusive with --fusion_type. Recommended: a DINOv3 ViT-L "
                        "checkpoint dir, or 'facebook/dinov2-large'.")
    p.add_argument("--aux_processor_id", type=str, default=None,
                   help="Image processor id/path for the aux encoder (default: --aux_model_id).")
    p.add_argument("--aux_trust_remote_code", action="store_true",
                   help="Pass trust_remote_code=True when loading the aux encoder/processor "
                        "(needed for DINOv3 loaded from disk).")
    p.add_argument("--aux_num_heads", type=int, default=8,
                   help="Heads for the dual-encoder cross-attention (must divide --mlp_input_dim).")
    p.add_argument("--aux_dropout", type=float, default=0.0,
                   help="Dropout inside the dual-encoder cross-attention block.")
    p.add_argument("--aux_gate_init", type=float, default=0.1,
                   help="Warm-start the dual-encoder residual (out_proj init scale). "
                        "Default 0.1 turns the residual on at init so the cross-attention "
                        "block isn't born switched off (0.0 = step-0 identical, residual off).")

    # PEFT
    p.add_argument("--peft_method", type=str, default=TRAIN_CONFIG["peft_method"],
                   choices=["LoRA", "DPT", "NA"])
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str, default=None,
                   help="LoRA target projections on the vision tower, comma-separated "
                        "(e.g. 'q_proj,v_proj' or 'q_proj,k_proj,v_proj,out_proj'); "
                        "or 'all-linear' for every Linear. Default: q_proj,k_proj.")

    # Training
    p.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    p.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    p.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"],
                   help="Head (MLP + fusion) LR; also the backbone LR for LoRA/DPT")
    p.add_argument("--backbone_lr", type=float, default=None,
                   help="Backbone LR. Default: equals --lr, except full FT "
                        "(--peft_method NA) where it defaults to 5e-6 (paper recipe).")
    p.add_argument("--weight_decay", type=float, default=TRAIN_CONFIG["weight_decay"])
    p.add_argument("--grad_accum", type=int, default=TRAIN_CONFIG["gradient_accumulation_steps"])
    p.add_argument("--gradient_clip", type=float, default=0.0,
                   help="Max gradient norm; 0 disables clipping")

    # Scheduler
    p.add_argument("--no_scheduler", action="store_true")
    p.add_argument("--lr_milestones", type=str, default="30,35",
                   help="Comma-separated epoch milestones for MultiStepLR")

    # Checkpointing / resuming
    p.add_argument("--checkpoint_steps", type=int, default=TRAIN_CONFIG["checkpoint_steps"])
    p.add_argument("--max_checkpoints", type=int, default=TRAIN_CONFIG["max_checkpoints"])
    p.add_argument("--stage_name", type=str, default=f"AGM_seed{Seed}")
    p.add_argument("--resume", action="store_true")

    # Eval
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--eval_every", type=int, default=TRAIN_CONFIG["eval_epoch_steps"],
                   help="Run evaluation every N epochs")

    # Logging
    p.add_argument("--tracker", type=str, default=TRAIN_CONFIG["tracker"],
                   choices=["aim", "wandb", "none"],
                   help="Experiment tracker backend (default: aim)")
    p.add_argument("--project", type=str, default=TRAIN_CONFIG["project"],
                   help="Tracker project / experiment name")
    p.add_argument("--wandb_project", dest="project", type=str,
                   default=argparse.SUPPRESS,
                   help="[deprecated] alias for --project")
    p.add_argument("--no_wandb", action="store_true",
                   help="[deprecated] alias for --tracker none")

    # Debug
    p.add_argument("--dry_run", action="store_true",
                   help="Only run 100 train batches and 32 eval batches per epoch")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
