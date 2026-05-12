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

There are two experiment suites (10M and 100M tokens), each with 5 conditions. Every condition includes an NTP stage on CHILDES. Pre-training conditions require an intermediate checkpoint first — see [Running Experiments](#running-experiments).

### 10M suite (≈10M tokens per condition)

| Condition | Tasks | Description |
|-----------|-------|-------------|
| Baseline NTP | `ntp_10m` | 10M tokens NTP on CHILDES |
| NTP → NSP | `ntp_10m_for_nsp` + `nsp_10m` | 5M NTP then 5M next-sentence prediction |
| NTP → NUP | `ntp_10m_for_nup` + `nup_10m` | 5M NTP then 5M next-utterance prediction |
| Dyck → NTP | `dyck_pretrain` + `dyck_5m_childes` | 5M Dyck pre-train then 5M CHILDES NTP |
| POS → NTP | `pos_pretrain` + `pos_5m_childes` | 5M POS pre-train then 5M CHILDES NTP |

### 100M suite (≈79.2M tokens per condition)

| Condition | Tasks | Description |
|-----------|-------|-------------|
| Baseline NTP | `ntp_100m` | 3 epochs × full CHILDES (79.2M tokens) |
| NTP → NSP | `ntp_100m_for_nsp` + `nsp_100m` | 3 epochs × first half NTP + 3 epochs × second half NSP |
| NTP → NUP | `ntp_100m_for_nup` + `nup_100m` | 3 epochs × first half NTP + 3 epochs × second half NUP |
| Dyck → NTP | `dyck_pretrain_100m` + `dyck_100m_childes` | 39.6M Dyck pre-train + 3 epochs × first half CHILDES |
| POS → NTP | `pos_pretrain_100m` + `pos_100m_childes` | 39.6M POS pre-train + 3 epochs × first half CHILDES |

> The 10M suite also supports the original full-CHILDES conditions (`next_word`, `next_word_then_nsp`, `next_word_then_nup`, `pos_then_next_word`, `paren_then_next_word`) if you want to run without dataset splits.

---

## Pipeline Overview

Data prep scripts run once to build datasets on disk. `main.py` then reads those datasets to train and evaluate — it does not call any data prep scripts itself.

```
Step 1  format.py               →  data/nt_text.txt, nsp_text.jsonl, nup_text.jsonl
Step 2  make_paren.py           →  pre-predata/tokenized_paren/, tokenizers/paren_tokenizer
Step 3  pos_data.py             →  data/pos_dataset/, tokenizers/pos_tokenizer
Step 4  dataprep.py             →  data/base/{nt,nsp,nup}_dataset
        dataprep.py --paren     →  data/paren/nt_dataset
Step 5  make_split_datasets.py  →  data/split/{nt_10m, nt_5m_a, nt_half_a,
                                              nsp_5m_b, nsp_half_b,
                                              nup_5m_b, nup_half_b}
Step 6  main.py                 →  trained models, training_results.csv
```

Steps 2–4 (paren) are only needed for paren experiments.
Steps 3–4 (pos) are only needed for POS experiments.
Step 5 is only needed for the 10M and 100M split-dataset conditions.

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

### Step 5 — Create split datasets *(skip if not doing 10M or 100M conditions)*

Reads `data/nt_text.txt` and `data/childes.train` and creates token-boundary-aligned splits under `data/split/`.

```bash
python scripts/make_split_datasets.py
```

Outputs:
- `./data/split/nt_10m` — first 10M tokens of CHILDES (NTP, 10M baseline)
- `./data/split/nt_5m_a` — first 5M tokens (NTP stage 1 for post-training and pre-training conditions)
- `./data/split/nt_half_a` — first ~13.2M tokens (NTP stage 1 for 100M conditions)
- `./data/split/nsp_5m_b` — NSP pairs from tokens 5M–10M
- `./data/split/nsp_half_b` — NSP pairs from second half of CHILDES
- `./data/split/nup_5m_b` — NUP pairs from tokens 5M–10M
- `./data/split/nup_half_b` — NUP pairs from second half of CHILDES

---

### Step 6 — Train

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

The simplest way to run all experiments is `run.sh`, which covers both suites in order:

```bash
bash run.sh
```

To run individual conditions:

### 10M suite

```bash
# 1. Baseline NTP
python main.py --tasks ntp_10m --epochs 1 --runs 3 --tag "10m"

# 2. Post-training NSP
python main.py --tasks ntp_10m_for_nsp nsp_10m --epochs 1 --runs 3 --tag "10m"

# 3. Post-training NUP
python main.py --tasks ntp_10m_for_nup nup_10m --epochs 1 --runs 3 --tag "10m"

# 4. Dyck pre-training → CHILDES NTP
python main.py --tasks dyck_pretrain dyck_5m_childes \
               --epochs 1 --pretrain-tokens 5000000 --runs 3 --tag "10m"

# 5. POS pre-training → CHILDES NTP
python main.py --tasks pos_pretrain pos_5m_childes \
               --epochs 1 --pretrain-tokens 5000000 --runs 3 --tag "10m"
```

### 100M suite

The 100M tasks use `lock_epochs=True` in config, so their epoch counts (3) are not overridden by `--epochs`. Pass `--epochs 1` only to cap the pre-training stage.

```bash
# 1. Baseline NTP (3 epochs × full CHILDES)
python main.py --tasks ntp_100m --runs 3 --tag "100m"

# 2. Post-training NSP
python main.py --tasks ntp_100m_for_nsp nsp_100m --runs 3 --tag "100m"

# 3. Post-training NUP
python main.py --tasks ntp_100m_for_nup nup_100m --runs 3 --tag "100m"

# 4. Dyck pre-training → CHILDES NTP
python main.py --tasks dyck_pretrain_100m dyck_100m_childes \
               --epochs 1 --pretrain-tokens 39600000 --runs 3 --tag "100m"

# 5. POS pre-training → CHILDES NTP
python main.py --tasks pos_pretrain_100m pos_100m_childes \
               --epochs 1 --pretrain-tokens 39600000 --runs 3 --tag "100m"
```

### Pre-training token budget

`--pretrain-tokens` caps how many tokens are used from the (much larger) pre-training dataset:

- **paren / dyck** — examples are exactly 512 tokens: `train_truncation = pretrain_tokens // 512`
- **pos** — examples are variable length: `main.py` computes the average automatically

Set `--pretrain-tokens` to match the CHILDES token count for a fair comparison (5M for 10M suite, 39.6M for 100M suite).

---

## Configuration

All configuration lives in `scripts/config.py`.

**`TrainingConfig`** — optimizer hyperparameters shared across all tasks (learning rate, batch size, scheduler, etc.).

**`TaskConfig`** — one entry per task. Key fields:

| Field | Description |
|-------|-------------|
| `num_train_epochs` | Epochs over the training set (overridden by `--epochs` unless `lock_epochs=True`) |
| `lock_epochs` | If `True`, `--epochs` does not override `num_train_epochs` (used for 100M conditions) |
| `train_truncation` | Cap training examples (overridden by `--pretrain-tokens` for pre-train tasks) |
| `model_load_path` | Checkpoint to warm-start from; `None` = random init |
| `use_custom_collator` | `True` for NSP / NUP tasks |

**`PRETRAIN_CONFIGS`** — intermediate pre-training stages that produce checkpoints consumed by fine-tuning tasks: `pos_pretrain`, `paren_pretrain` (10M); `dyck_pretrain_100m`, `pos_pretrain_100m` (100M).

**`TASK_CONFIGS`** — all experimental conditions: the original 5 full-CHILDES conditions plus the 10M and 100M split-dataset conditions.

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
