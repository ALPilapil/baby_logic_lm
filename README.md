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

Dependencies are managed with [Poetry](https://python-poetry.org/) (requires Python >=3.10,<3.14 — torch 2.8.0 has no wheels for 3.14+).

```bash
poetry install
```

This creates a `.venv` in the project root and installs the `baby_logic_lm` package (under `src/`) in editable mode. Run any command from this README/CLAUDE.md by prefixing it with `poetry run`, e.g. `poetry run python -m baby_logic_lm.cli.pipeline --tasks ntp_10m --epochs 1`, or activate the environment first with `poetry env activate` (or `poetry shell` if the shell plugin is installed).

CUDA runtime packages (`nvidia-cu12-*`, `triton`) are marked Linux-only in `pyproject.toml` and are skipped automatically on macOS; they install automatically on a Linux/CUDA training box.

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

Data prep scripts run once to build datasets on disk. `baby_logic_lm.cli.pipeline` then reads those datasets to train and evaluate — it does not call any data prep scripts itself.

```
Step 1  format.py               →  data/nt_text.txt, nsp_text.jsonl, nup_text.jsonl
Step 2  make_paren.py           →  pre-predata/tokenized_paren/, tokenizers/paren_tokenizer
Step 3  pos_data.py             →  data/pos_dataset/, tokenizers/pos_tokenizer
Step 4  dataprep.py             →  data/base/{nt,nsp,nup}_dataset
        dataprep.py --paren     →  data/paren/nt_dataset
Step 5  make_split_datasets.py  →  data/split/{nt_10m, nt_5m_a, nt_half_a,
                                              nsp_5m_b, nsp_half_b,
                                              nup_5m_b, nup_half_b}
Step 6  cli.pipeline            →  trained models, training_results.csv
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
python -m baby_logic_lm.cli.pipeline --tasks <task1> [task2 ...] --epochs <n> [--pretrain-tokens <n>] [--runs <n>] [--tag <label>]
```

| Argument | Description |
|----------|-------------|
| `--tasks` | Ordered list of task keys to run (see conditions table above; keys are the YAML file stems under `configs/task/`) |
| `--epochs` | `num_train_epochs` applied to every task in the run |
| `--pretrain-tokens` | Token budget for pre-training stages; auto-computes `train_truncation` |
| `--runs` | Number of times to repeat the full task sequence (default: `1`). Each run uses its index as the random seed, so results are statistically independent. |
| `--tag` | Optional label written to every row of `training_results.csv` for grouping runs (e.g. `pilot`, `final`). |

Each task trains the model, evaluates it (CN + BLiMP), and appends a row to `training_results.csv`. Tasks run sequentially; GPU memory is freed between them. Training metrics and final CN/BLiMP/perplexity are also logged to Weights & Biases (set `wandb.mode=offline` or `disabled` via a Hydra override, or edit `configs/wandb/default.yaml`, to disable cloud syncing).

For a single task, `baby_logic_lm.cli.train` is a plain Hydra entry point that also supports arbitrary config overrides and genuine multirun sweeps (not available via `cli.pipeline`, whose `--tasks` sequences are stateful/checkpoint-chained rather than independent trials):

```bash
python -m baby_logic_lm.cli.train task=ntp_10m training.num_train_epochs=5
python -m baby_logic_lm.cli.train -m task=ntp_10m,ntp_100m training.learning_rate=1e-4,2.5e-4
```

---

## Running Experiments

Before kicking off a real (multi-hour) suite, run the smoke test to catch crashes and confirm CEL/perplexity/CN/BLiMP log correctly to `training_results.csv` and your wandb account:

```bash
bash scripts/smoke_test.sh
```

It trains 3 tiny fixtures (`configs/task/smoke_eval_pretrain.yaml`, `smoke_eval_finetune.yaml`, `smoke_split_phase.yaml`) covering checkpoint chaining, real CN/BLiMP scoring, and the split-phase/`token_limit` training path used by every 100M-suite condition — writes real, `smoke_check`-tagged rows/runs to your CSV and wandb project for you to inspect, verifies them automatically via `scripts/check_smoke_results.py`, and cleans up the disposable model checkpoints it creates.

The simplest way to run all experiments is `run_train.sh` (training) and `run_eval.sh` (re-evaluation), which each cover both suites in order:

```bash
bash run_train.sh
bash run_eval.sh
```

To run individual conditions:

### 10M suite

```bash
# 1. Baseline NTP
python -m baby_logic_lm.cli.pipeline --tasks ntp_10m --epochs 1 --runs 3 --tag "10m"

# 2. Post-training NSP
python -m baby_logic_lm.cli.pipeline --tasks ntp_10m_for_nsp nsp_10m --epochs 1 --runs 3 --tag "10m"

# 3. Post-training NUP
python -m baby_logic_lm.cli.pipeline --tasks ntp_10m_for_nup nup_10m --epochs 1 --runs 3 --tag "10m"

# 4. Dyck pre-training → CHILDES NTP
python -m baby_logic_lm.cli.pipeline --tasks dyck_pretrain dyck_5m_childes \
               --epochs 1 --pretrain-tokens 5000000 --runs 3 --tag "10m"

# 5. POS pre-training → CHILDES NTP
python -m baby_logic_lm.cli.pipeline --tasks pos_pretrain pos_5m_childes \
               --epochs 1 --pretrain-tokens 5000000 --runs 3 --tag "10m"
```

### 100M suite

The 100M tasks set `lock_epochs: true` in their config, so their epoch counts (3) are not overridden by `--epochs`. Pass `--epochs 1` only to cap the pre-training stage.

```bash
# 1. Baseline NTP (3 epochs × full CHILDES)
python -m baby_logic_lm.cli.pipeline --tasks ntp_100m --runs 3 --tag "100m"

# 2. Post-training NSP
python -m baby_logic_lm.cli.pipeline --tasks ntp_100m_for_nsp nsp_100m --runs 3 --tag "100m"

# 3. Post-training NUP
python -m baby_logic_lm.cli.pipeline --tasks ntp_100m_for_nup nup_100m --runs 3 --tag "100m"

# 4. Dyck pre-training → CHILDES NTP
python -m baby_logic_lm.cli.pipeline --tasks dyck_pretrain_100m dyck_100m_childes \
               --epochs 1 --pretrain-tokens 39600000 --runs 3 --tag "100m"

# 5. POS pre-training → CHILDES NTP
python -m baby_logic_lm.cli.pipeline --tasks pos_pretrain_100m pos_100m_childes \
               --epochs 1 --pretrain-tokens 39600000 --runs 3 --tag "100m"
```

### Pre-training token budget

`--pretrain-tokens` caps how many tokens are used from the (much larger) pre-training dataset:

- **paren / dyck** — examples are exactly 512 tokens: `train_truncation = pretrain_tokens // 512`
- **pos** — examples are variable length: `cli.pipeline` computes the average automatically

Set `--pretrain-tokens` to match the CHILDES token count for a fair comparison (5M for 10M suite, 39.6M for 100M suite).

---

## Configuration

All configuration lives under `configs/`, composed via [Hydra](https://hydra.cc) against the structured schema in `src/baby_logic_lm/config_schema.py`.

**`configs/model/pythia_160m.yaml`** (`ModelConfig`) — the GPTNeoX architecture, pinned locally (hidden size, layers, heads, etc.) instead of fetched implicitly from the HF Hub at train time.

**`configs/training/default.yaml`** (`TrainingConfig`) — optimizer hyperparameters shared across all tasks (learning rate, batch size, scheduler, etc.).

**`configs/wandb/default.yaml`** (`WandbConfig`) — Weights & Biases project/entity/mode.

**`configs/task/*.yaml`** (`TaskConfig`) — one file per task. Key fields:

| Field | Description |
|-------|-------------|
| `num_train_epochs` | Epochs over the training set (overridden by `--epochs` unless `lock_epochs: true`) |
| `lock_epochs` | If `true`, `--epochs` does not override `num_train_epochs` (used for 100M conditions) |
| `train_truncation` | Cap training examples (overridden by `--pretrain-tokens` for pre-train tasks) |
| `model_load_path` | Checkpoint to warm-start from; `null` = random init |
| `use_custom_collator` | `true` for NSP / NUP tasks |
| `is_pretrain` | `true` for intermediate pre-training stages (`pos_pretrain`, `dyck_pretrain`, `dyck_pretrain_100m`, `pos_pretrain_100m`) whose checkpoints are consumed by fine-tuning tasks |

To add a new experimental condition, add a `configs/task/<name>.yaml` file setting whatever fields differ from `TaskConfig`'s defaults — nothing else needs to change. `configs/task/smoke*.yaml` are cheap dev fixtures (tiny truncation, CN/BLiMP off) for exercising the pipeline end-to-end without real compute; they aren't experimental conditions.

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
