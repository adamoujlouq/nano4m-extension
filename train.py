import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.optim as optim
import yaml
import wandb

from data.multimodal import create_multimodal_masked_dataloader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_dataloader(cfg: dict, split: str):
    dcfg = cfg["data"]
    return create_multimodal_masked_dataloader(
        root_dir=dcfg["root_dir"],
        split=split,
        modalities=dcfg["modalities"],
        vocab_sizes=dcfg["vocab_sizes"],
        max_seq_lens=dcfg["max_seq_lens"],
        input_alphas=dcfg["input_alphas"],
        target_alphas=dcfg["target_alphas"],
        input_tokens_range=dcfg["input_tokens_range"],
        target_tokens_range=dcfg["target_tokens_range"],
        overlap_vocab=True,
        overlap_posembs=True,
        sample_from_k_augmentations=dcfg.get("sample_from_k_augmentations", 10),
        text_tokenizer_path=dcfg.get("text_tokenizer_path", "gpt2"),
        text_max_length=dcfg.get("text_max_length", 256),
        batch_size=cfg["training"]["batch_size"],
        infinite=True,
        num_workers=4,
        pin_memory=True,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        distributed=False,
    )


def build_model(cfg: dict, device: torch.device):
    from models.fourm import FourM

    dcfg = cfg["data"]
    model = FourM(
        enc_tokens_read_key="enc_tokens",
        dec_tokens_read_key="dec_tokens",
        enc_modalities_read_key="enc_modalities",
        dec_modalities_read_key="dec_modalities",
        enc_positions_read_key="enc_positions",
        dec_positions_read_key="dec_positions",
        enc_pad_mask_read_key="enc_pad_mask",
        dec_pad_mask_read_key="dec_pad_mask",
        modalities=dcfg["modalities"],
        vocab_sizes=dcfg["vocab_sizes"],
        max_seq_lens=dcfg["max_seq_lens"],
        **cfg["model"],
    )
    return model.to(device)


def init_wandb(cfg: dict, seed: int, run_name: str, model: torch.nn.Module):
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled", True):
        print("W&B logging disabled by config.")
        return None

    tcfg = cfg["training"]
    run = wandb.init(
        project=wcfg.get("project", "COM304_nano4M"),
        entity=wcfg.get("entity", None),
        mode=wcfg.get("mode", "online"),
        name=run_name,
        group=wcfg.get("group", None),
        tags=wcfg.get("tags", []),
        notes=wcfg.get("notes", None),
        config={
            "seed": seed,
            "run_name": run_name,
            "data": cfg["data"],
            "model": cfg["model"],
            "training": tcfg,
        },
    )

    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")

    if wcfg.get("watch_model", False):
        wandb.watch(
            model,
            log=wcfg.get("watch_log", "gradients"),
            log_freq=wcfg.get("watch_log_freq", tcfg["log_every"]),
        )

    if wandb.run is not None:
        print(f"W&B run: {wandb.run.url}")

    return run


def train(cfg: dict, seed: int, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {seed} | Run: {run_name}")

    set_seed(seed)
    model = build_model(cfg, device)
    print(f"Params: {model.get_num_params() / 1e6:.1f}M")

    tcfg = cfg["training"]
    wandb_run = init_wandb(cfg, seed, run_name, model)

    optimizer = optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )

    os.makedirs(f"results/{run_name}", exist_ok=True)
    log_path = f"results/{run_name}/metrics.jsonl"

    print("Building dataloaders...")
    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    model.train()
    for step in range(1, tcfg["num_steps"] + 1):
        lr = get_cosine_lr(step, tcfg["num_steps"], tcfg["warmup_steps"], tcfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = next(train_loader)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        loss, modality_losses = model(batch)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optimizer.step()

        if step % tcfg["log_every"] == 0:
            # Validation loss
            model.eval()
            with torch.no_grad():
                val_batch = next(val_loader)
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                val_loss, val_modality_losses = model(val_batch)
            model.train()

            log = {
                "step": step,
                "train_loss": loss.item(),
                "val_loss": val_loss.item(),
                "lr": lr,
                "grad_norm": grad_norm.item(),
                **{f"train/{k}": v.item() for k, v in modality_losses.items()},
                **{f"val/{k}": v.item() for k, v in val_modality_losses.items()},
            }

            if wandb_run is not None:
                wandb.log(log)

            print(
                f"[{step:>6}] train={loss.item():.4f} val={val_loss.item():.4f} "
                f"grad_norm={grad_norm.item():.4f} lr={lr:.2e}"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(log) + "\n")

        if step % tcfg["save_every"] == 0:
            ckpt_path = f"results/{run_name}/ckpt_step{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    if wandb_run is not None:
        wandb.summary["final_step"] = tcfg["num_steps"]
        wandb.finish()
    print(f"Training done. Logs: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.run_name is None:
        enabled_arch_changes = [
            name
            for name, enabled in (
                ("swiglu", cfg["model"].get("use_swiglu", False)),
                ("rope", cfg["model"].get("use_rope", False)),
                ("qknorm", cfg["model"].get("use_qk_norm", False)),
            )
            if enabled
        ]
        tag = "_".join(enabled_arch_changes) if enabled_arch_changes else "baseline"
        args.run_name = f"{tag}_seed{args.seed}"

    train(cfg, seed=args.seed, run_name=args.run_name)
