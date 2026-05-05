# Baby Logic Language Model

An autoregressive language model trained on child-directed speech (CHILDES), optionally pre-trained on Dyck/parentheses sequences or POS-tag data. Evaluates on the Crain & Nakayama syntactic test and the BLiMP benchmark.

---

## Prerequisites

Two raw data files must exist before running anything:

| File | Used by |
|------|---------|
| `./data/childes.train` | All CHILDES-based tasks |
| `./pre-predata/shuff_dyck/dyck_sequences.txt` | Paren/Dyck pre-training only |

C4 (used for POS pre-training) is streamed directly from HuggingFace — no manual download needed.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Experimental Conditions

There are 5 experimental conditions. Each includes NTP (next-token prediction on CHILDES) as a training stage. The two pre-training conditions require an intermediate checkpoint to be produced first — see [Running Experiments](#running-experiments).

| Condition | Key | Description |
|-----------|-----|-------------|
| NTP only | `next_word` | Baseline — next-token prediction on CHILDES |
| POS → NTP | `pos_then_next_word` | Pre-train on C4 POS tags, fine-tune on CHILDES |
| Paren → NTP | `paren_then_next_word` | Pre-train on Dyck sequences, fine-tune on CHILDES |
| NTP → NSP | `next_word_then_nsp` | Train on CHILDES, fine-tune on next-sentence prediction |
| NTP → NUP | `next_word_then_nup` | Train on CHILDES, fine-tune on next-utterance prediction |

---

## Pipeline Overview

Data prep scripts run once to build datasets on disk. `main.py` then reads those datasets to train and evaluate — it does not call any data prep scripts itself.

```
Step 1  format.py          →  data/nt_text.txt, nsp_text.jsonl, nup_text.jsonl
Step 2  make_paren.py      →  pre-predata/tokenized_paren/, tokenizers/paren_tokenizer
Step 3  pos_data.py        →  data/pos_dataset/, tokenizers/pos_tokenizer
Step 4  dataprep.py        →  data/base/{nt,nsp,nup}_dataset
        dataprep.py --paren → data/paren/nt_dataset
Step 5  main.py            →  trained models, training_results.csv
```

Steps 2–4 (paren) are only needed for paren experiments.
Steps 3–4 (pos) are only needed for POS experiments.

---

## Step-by-Step

### Step 1 — Format raw CHILDES data

Always run this first. Produces the three raw text files used by `dataprep.py`.

```bash
python scripts/format.py
```

Outputs:
- `./data/nt_text.txt` — plain text for next-token training
- `./data/nsp_text.jsonl` — consecutive sentence pairs
- `./data/nup_text.jsonl` — consecutive utterance pairs

---

### Step 2 — Paren pre-training data *(skip if not doing a paren experiment)*

Converts raw Dyck sequences to special-token format and builds the paren tokenizer.

```bash
python scripts/make_paren.py           # process all lines
python scripts/make_paren.py 50000     # limit to 50k lines (quick test)
```

Outputs:
- `./pre-predata/tokenized_paren/tokenized_paren.txt`
- `./tokenizers/paren_tokenizer`

---

### Step 3 — POS data *(skip if not doing a POS experiment)*

Streams C4, converts text to POS tags, builds the POS tokenizer and dataset.

```bash
python scripts/pos_data.py             # create tokenizer + dataset
python scripts/pos_data.py --no-tok    # reuse existing tokenizer
```

Outputs:
- `./tokenizers/pos_tokenizer`
- `./data/pos_dataset`

---

### Step 4 — Tokenize and save CHILDES datasets

Reads the files from Step 1 and saves HuggingFace datasets to disk.

```bash
python scripts/dataprep.py             # base Pythia tokenizer (always run)
python scripts/dataprep.py --paren     # paren tokenizer (paren experiments only)
```

Outputs (base tokenizer):
- `./data/base/nt_dataset`
- `./data/base/nsp_dataset`
- `./data/base/nup_dataset`

Outputs (paren tokenizer):
- `./data/paren/nt_dataset`

---

### Step 5 — Train

```bash
python main.py --tasks <task1> [task2 ...] --epochs <n> [--pretrain-tokens <n>] [--runs <n>] [--tag <label>]
```

| Argument | Description |
|----------|-------------|
| `--tasks` | Ordered list of task keys to run (see conditions table above) |
| `--epochs` | `num_train_epochs` applied to every task in the run |
| `--pretrain-tokens` | Token budget for pre-training stages; auto-computes `train_truncation` |
| `--runs` | Number of times to repeat the full task sequence (default: `1`). Each run uses its index as the random seed, so results are statistically independent. |
| `--tag` | Optional label written to every row of `training_results.csv` for grouping runs (e.g. `pilot`, `final`). |

Each task trains the model, evaluates it (CN + BLiMP), and appends a row to `training_results.csv`. Tasks run sequentially; GPU memory is freed between them.

---

## Running Experiments

### CHILDES baseline

```bash
python main.py --tasks next_word --epochs 1
```

### Post-training conditions (NSP / NUP)

These warm-start from the NTP checkpoint (`./models/pythia/nt-model`), so run `next_word` first.

```bash
python main.py --tasks next_word next_word_then_nsp next_word_then_nup --epochs 1
```

### Pre-training conditions (POS / Paren)

Pre-training tasks (`pos_pretrain`, `paren_pretrain`) must precede their fine-tuning stage in `--tasks`. Pass `--pretrain-tokens` to match the pre-training token budget to CHILDES.

```bash
# POS pre-training → NTP fine-tune
python main.py --tasks pos_pretrain pos_then_next_word \
               --epochs 1 --pretrain-tokens 5000000

# Paren pre-training → NTP fine-tune
python main.py --tasks paren_pretrain paren_then_next_word \
               --epochs 1 --pretrain-tokens 5000000
```

#### Token budget for pre-training

Pre-training datasets are much larger than CHILDES. Use `--pretrain-tokens` to cap them:

- **paren_pretrain** — examples are exactly 512 tokens: `train_truncation = pretrain_tokens // 512`
- **pos_pretrain** — examples are variable length (≤ 512): `main.py` computes the average automatically

Set `--pretrain-tokens` to match the total token count of your CHILDES NTP training run for a fair comparison.

---

## Configuration

All configuration lives in `scripts/config.py`.

**`TrainingConfig`** — optimizer hyperparameters shared across all tasks (learning rate, batch size, scheduler, etc.).

**`TaskConfig`** — one entry per task. Key fields:

| Field | Description |
|-------|-------------|
| `num_train_epochs` | Epochs over the training set (overridden by `--epochs` in `main.py`) |
| `train_truncation` | Cap training examples (overridden by `--pretrain-tokens` for pre-train tasks) |
| `model_load_path` | Checkpoint to warm-start from; `None` = random init |
| `use_custom_collator` | `True` for NSP / NUP tasks |

**`PRETRAIN_CONFIGS`** — the two intermediate pre-training stages (`pos_pretrain`, `paren_pretrain`). Run these to produce checkpoints consumed by `pos_then_next_word` and `paren_then_next_word`.

**`TASK_CONFIGS`** — the 5 experimental conditions.

---

## Results

All evaluation results are appended to `training_results.csv`:

| Column | Description |
|--------|-------------|
| `timestamp` | UTC datetime the run completed (ISO 8601) |
| `tag` | Experiment label passed via `--tag` (empty string if omitted) |
| `run` | Run index (1, 2, …); also used as the random seed |
| `task_type` | Task name |
| `base_model` | Base model architecture ID |
| `warmup_from` | Checkpoint the model was initialized from (`random_init` if trained from scratch) |
| `epochs` | Number of training epochs |
| `train_tokens` | Tokens in the training set for one epoch |
| `total_tokens` | Total tokens seen (`train_tokens × epochs`) |
| `learning_rate` | Peak learning rate |
| `batch_size` | Per-device training batch size |
| `CEL` | Cross-entropy loss |
| `perplexity` | Exp of eval loss |
| `CN` | Crain & Nakayama syntactic evaluation |
| `BLiMP` | Average BLiMP suite accuracy |

> **Note:** If you have a `training_results.csv` from before these columns were added, delete or rename it before running — new rows use a different header and will not align with old ones.
