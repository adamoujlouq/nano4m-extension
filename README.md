# nano4M architecture extensions

This repository contains the COM-304 nano4M extension experiments.

Implemented variants:

- `configs/baseline.yaml`: original nano4M baseline.
- `configs/swiglu.yaml`: week 1 variant, replacing the GELU MLP by SwiGLU.
- `configs/rope.yaml`: week 2 variant, adding RoPE to encoder and decoder self-attention only.
- `configs/qknorm.yaml`: week 3 variant, adding QK-Norm to self-attention and cross-attention.
- `configs/all.yaml`: combined variant with SwiGLU, RoPE, and QK-Norm.

## Setup

Create or activate your Python environment, then install the runtime packages used by the training code:

```bash
pip install torch numpy einops pyyaml wandb transformers tokenizers timm matplotlib
```

On Kuma, make sure the dataset path in the config points to the shared CLEVR tokens:

```yaml
data:
  root_dir: /work/com-304/datasets/clevr_com_304/
```

If you use a local copy of the dataset, edit `data.root_dir` in the config before launching.

## W&B logging

Training logs to Weights & Biases by default under:

```text
entity: adam-oujlouq
project: COM304_nano4M
```

Before launching on a new machine, authenticate once:

```bash
wandb login
```

Each run prints its W&B URL at startup. The dashboard receives the global losses, per-modality losses, learning rate, and gradient norm, so curves are available live during training.

## Week 2: RoPE run

RoPE is controlled by these model flags:

```yaml
model:
  use_rope: true
  rope_base: 10000.0
```

Manual training command:

```bash
python train.py --config configs/rope.yaml --seed 42 --run_name rope_seed42
```

If you do not pass `--run_name`, `train.py` will automatically name the run from the enabled architecture flags, e.g. `rope_seed42`.

For offline W&B logging on a machine without an online W&B session, use:

```bash
WANDB_MODE=offline python train.py --config configs/rope.yaml --seed 42 --run_name rope_seed42
```

The training script writes checkpoints and JSONL metrics under:

```text
results/<run_name>/
```

## Week 3: QK-Norm run

QK-Norm is controlled by:

```yaml
model:
  use_qk_norm: true
```

Manual training command without W&B:

```bash
python train.py --config configs/qknorm.yaml --seed 42 --run_name qknorm_seed42
```

## Combined run

The combined config enables all three architecture changes and keeps W&B disabled:

```bash
python train.py --config configs/all.yaml --seed 42 --run_name swiglu_rope_qknorm_seed42
```

## Notes on the RoPE implementation

RoPE is applied only inside self-attention layers, to the query/key tensors of the encoder and decoder. Cross-attention is left non-rotary, and it still receives the existing sinusoidal position embeddings so decoder target positions and encoder token positions remain visible to the cross-modal prediction path.

## Notes on the QK-Norm implementation

QK-Norm applies a learned per-head LayerNorm to the query and key tensors before the attention dot product. It is enabled for encoder self-attention, decoder self-attention, and encoder-decoder cross-attention.
