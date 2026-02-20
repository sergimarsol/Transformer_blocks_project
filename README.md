# CSE 256 — PA2: Transformer Encoder & Decoder

## Project Structure

```
PA2_code/
├── main.py              # Entry point — run any part of the assignment
├── transformer.py       # Transformer Encoder, Decoder, and all attention variants
├── tokenizer.py         # SimpleTokenizer (character/word-level)
├── dataset.py           # PyTorch Datasets for classification and language modeling
├── utilities.py         # Sanity-check & attention-map visualization
├── speechesdataset/     # Training and test data (speeches + labels)
└── plot_scripts/        # Plotting scripts for generating figures
    ├── plot_part1.py
    ├── plot_part2.py
    └── plot_part3.py
```

## Requirements

- Python 3.8+
- PyTorch (tested with 2.x)
- matplotlib

```bash
pip install torch matplotlib
```

## How to Run

All commands should be executed from within the `PA2_code/` directory:

```bash
cd PA2_code
```

### Part 1 — Encoder + Classifier

Trains a Transformer Encoder jointly with a feedforward classifier on the speeches classification task. Reports per-epoch train/test accuracy and runs attention-map sanity checks.

```bash
python main.py part1
```

### Part 2 — Decoder Pretraining (Language Modeling)

Trains a GPT-like Transformer Decoder with full causal attention on the combined training speeches. Reports training perplexity every 100 iterations and final test perplexity for each politician (Obama, W. Bush, H. Bush).

```bash
python main.py part2
```

### Part 3 — Architectural Explorations

Trains the decoder with the baseline (full attention) plus one or more architectural variants, then prints a comparison table of test perplexities.

**Run all explorations (default):**

```bash
python main.py part3
```

**Run only specific methods** (any combination of `window`, `alibi`, `block_sparse`):

```bash
python main.py part3 --methods window alibi
python main.py part3 --methods block_sparse
python main.py part3 --methods alibi block_sparse
```

| Method | Description |
|---|---|
| `window` | Local sliding-window attention (W=3, W=5) |
| `alibi` | ALiBi positional biases — no positional embeddings |
| `block_sparse` | Blockwise sparse attention (B=4, B=8) |

## Hyperparameters

All defaults are shared across every part:

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| `batch_size` | 16 | `n_embd` | 64 |
| `block_size` | 32 | `n_head` | 2 |
| `learning_rate` | 1e-3 | `n_layer` | 4 |
| `n_hidden` | 100 | `epochs_CLS` | 15 |
| `max_iters` | 500 | `eval_iters` | 200 |

- **Encoder + Classifier params:** 577,107
- **Decoder params:** 864,011 (identical for all attention variants)

## Implementation Details

### Part 1: Transformer Encoder + Feedforward Classifier

- Multi-head self-attention (implemented from scratch), feedforward, residual connections + LayerNorm
- Learnable positional embeddings
- Padding-mask-aware mean pooling → 1-hidden-layer classifier (64 → 100 → 3)

### Part 2: Transformer Decoder (GPT-like)

- Causal (masked) multi-head self-attention — no future-token attention
- Token + positional embeddings → 4 decoder layers → LayerNorm → linear head
- Cross-entropy loss computed internally; perplexity = exp(mean loss)

### Part 3: Architectural Explorations

All variants modify **only** the attention pattern in the decoder; everything else (FFN, residuals, LayerNorm, embeddings, training loop) is unchanged.

- **Local Window Attention** — Restricts each token to attend to at most *W* preceding tokens via a distance-based mask.
- **ALiBi** — Replaces positional embeddings with per-head linear distance biases added to attention scores. Slopes follow a geometric schedule: $slope_h = 2^{-8h/H}$. The positional embedding layer is retained but unused, preserving the parameter count.
- **Blockwise Sparse Attention** — Partitions the sequence into fixed-size blocks. Each token attends only within its own block and the immediately previous block, combined with the causal mask.

## Results

Quick comparison of test perplexity:

| Configuration | Obama | W.Bush | H.Bush |
|---|---|---|---|
| Full attention | 395.51 | 488.96 | 430.65 |
| Window (W=5) | 377.81 | 454.91 | 399.03 |
| Block Sparse (B=8) | 393.90 | 475.48 | 425.53 |
| **ALiBi** | **354.39** | **450.14** | **385.70** |

**ALiBi** achieves the lowest perplexity across all speakers (~8–10% improvement over the baseline).

## Plotting Scripts

Three standalone scripts in `plot_scripts/` generate the figures used in the report:

| Script | Description | Output |
|---|---|---|
| `plot_part1.py` | Epoch vs Train/Test Accuracy (Encoder + Classifier) | `part1_accuracy.png` |
| `plot_part2.py` | Iteration vs Training Perplexity (Decoder LM) | `part2_perplexity.png` |
| `plot_part3.py` | Grouped bar chart comparing test perplexity across all attention variants | `part3_comparison.png` |

```bash
python plot_scripts/plot_part1.py
python plot_scripts/plot_part2.py
python plot_scripts/plot_part3.py
```
