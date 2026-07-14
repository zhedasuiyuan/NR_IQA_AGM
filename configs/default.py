"""
Default configuration dictionaries.

Override individual keys via CLI flags in train.py / eval.py.
Author: Ankit Yadav
"""

MODEL_CONFIG = {
    "model_id": "google/siglip2-so400m-patch16-512",
    "mlp_input_dim": 1152,
    "mlp_hidden_dim": 512,
}

TRAIN_CONFIG = {
    "epochs": 15,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "full_ft_backbone_lr": 5e-6,         # backbone LR for full FT (peft_method=NA); head stays at learning_rate
    "weight_decay": 0,
    "checkpoint_steps": 5000,
    "max_checkpoints": 5,
    "stage_name": "stage1",

    "lora_config": {
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        # q+k on the vision tower only (matches the original repo; the text tower is
        # unused by get_image_features). Override via --lora_targets, e.g.
        # "q_proj,v_proj", "q_proj,k_proj,v_proj,out_proj", or "all-linear".
        "target_modules": r"vision_model\..*\.(q_proj|k_proj)$",
    },
    "dpt_config": {
        "no_learnable_tokens": 200,
    },
    "peft_method": "LoRA",               # "LoRA", "DPT", or "NA" (full FT)

    "tracker": "aim",                    # "aim" (default), "wandb", or "none"
    "project": "NR_IQA_AGM",             # tracker project / experiment name
    "gradient_accumulation_steps": 6,
    "do_eval": True,
    "eval_epoch_steps": 1,

    "lr_scheduler": True,
    "lr_scheduler_milestones": [30, 35],
    "lr_warmup_ratio": 0.1,

    "early_stopping": False,
    "patience": 3,

    "use_gradient_clip": False,
    "gradient_clip": 1.0,

    "resume": False,
    "dry_run": False,
}


def _make_dataset_paths(base: str) -> dict:
    """Build per-dataset root paths from a single *base* directory."""
    return {
        "KonIQ_10K":  f"{base}/KonIQ_10K",
        "CLIVE":      f"{base}/CLIVE/ChallengeDB_release",
        "SPAQ":       f"{base}/SPAQ",
        "KADID10K":   f"{base}/KADID-10K",
        "TID2013":    f"{base}/TID2013",
        "FLIVE":      f"{base}/FLIVE",
        "AGIQA3K":    f"{base}/AGIQA-3k",
        "AGIQA1K":    f"{base}/AGIQA-1k",
    }


DATASET_PATHS = _make_dataset_paths("./Dataset")
