import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.optim as optim
import yaml


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


def build_model(cfg: dict, device: torch.device):
    from models.fourm import FourM

    modalities = ["rgb", "depth", "normals", "captions"]
    vocab_sizes = [1024, 1024, 1024, 1024]
    max_seq_lens = [196, 196, 196, 64]

    model = FourM(
        enc_tokens_read_key="enc_tokens",
        dec_tokens_read_key="dec_tokens",
        enc_modalities_read_key="enc_modalities",
        dec_modalities_read_key="dec_modalities",
        enc_positions_read_key="enc_positions",
        dec_positions_read_key="dec_positions",
        enc_pad_mask_read_key="enc_pad_mask",
        dec_pad_mask_read_key="dec_pad_mask",
        modalities=modalities,
        vocab_sizes=vocab_sizes,
        max_seq_lens=max_seq_lens,
        **cfg["model"],
    )
    return model.to(device)


def fake_batch(device: torch.device, batch_size: int = 4):
    """
    Génère un batch factice pour tester le training loop.
    À remplacer par ton vrai DataLoader quand il sera prêt.
    """
    modalities = ["rgb", "depth", "normals", "captions"]
    vocab_sizes = [1024, 1024, 1024, 1024]
    max_seq_lens = [196, 196, 196, 64]

    B = batch_size
    enc_len = 128
    dec_len = 64

    num_mods = len(modalities)

    enc_tokens = torch.randint(0, 1024, (B, enc_len), device=device)
    enc_modalities = torch.randint(0, num_mods, (B, enc_len), device=device)
    enc_positions = torch.arange(enc_len, device=device).unsqueeze(0).expand(B, -1)
    enc_pad_mask = torch.ones(B, enc_len, dtype=torch.bool, device=device)

    dec_tokens = torch.randint(0, 1024, (B, dec_len), device=device)
    dec_modalities = torch.randint(0, num_mods, (B, dec_len), device=device)
    dec_positions = torch.arange(dec_len, device=device).unsqueeze(0).expand(B, -1)
    dec_pad_mask = torch.ones(B, dec_len, dtype=torch.bool, device=device)

    return {
        "enc_tokens": enc_tokens,
        "enc_modalities": enc_modalities,
        "enc_positions": enc_positions,
        "enc_pad_mask": enc_pad_mask,
        "dec_tokens": dec_tokens,
        "dec_modalities": dec_modalities,
        "dec_positions": dec_positions,
        "dec_pad_mask": dec_pad_mask,
    }


def train(cfg: dict, seed: int, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {seed} | Run: {run_name}")

    set_seed(seed)
    model = build_model(cfg, device)
    print(f"Params: {model.get_num_params() / 1e6:.1f}M")

    tcfg = cfg["training"]
    optimizer = optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )

    os.makedirs(f"results/{run_name}", exist_ok=True)
    log_path = f"results/{run_name}/metrics.jsonl"

    model.train()
    for step in range(1, tcfg["num_steps"] + 1):
        # Mise à jour du learning rate (cosine warmup)
        lr = get_cosine_lr(step, tcfg["num_steps"], tcfg["warmup_steps"], tcfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = fake_batch(device, batch_size=tcfg["batch_size"] // 16 or 4)

        optimizer.zero_grad()
        loss, modality_losses = model(batch)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optimizer.step()

        if step % tcfg["log_every"] == 0:
            log = {
                "step": step,
                "loss": loss.item(),
                "lr": lr,
                "grad_norm": grad_norm.item(),
                **{f"val_loss_{k}": v.item() for k, v in modality_losses.items()},
            }
            print(
                f"[{step:>6}] loss={loss.item():.4f} "
                f"rgb={modality_losses.get('rgb', torch.tensor(0)).item():.4f} "
                f"depth={modality_losses.get('depth', torch.tensor(0)).item():.4f} "
                f"grad_norm={grad_norm.item():.4f} lr={lr:.2e}"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(log) + "\n")

        if step % tcfg["save_every"] == 0:
            ckpt_path = f"results/{run_name}/ckpt_step{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    print(f"Training done. Logs: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.run_name is None:
        tag = "swiglu" if cfg["model"].get("use_swiglu") else "baseline"
        args.run_name = f"{tag}_seed{args.seed}"

    train(cfg, seed=args.seed, run_name=args.run_name)