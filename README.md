# Baby Logic Language Model

An autoregressive language model trained on logic and next utterance prediction. Uses CHILDES child-directed speech data, optionally pre-trained on Dyck/parentheses sequences or POS-tag data.

---

## Prerequisites

Two raw data files must exist before running anything:

| File | Used by |
|------|---------|
| `./data/childes.train` | All CHILDES-based tasks (next-token, NSP, NUP) |
| `./pre-predata/shuff_dyck/dyck_sequences.txt` | Paren/Dyck pre-training only |

C4 (used for POS training) is streamed directly from HuggingFace — no manual download needed.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Pipeline Overview

Data prep scripts run once to build datasets on disk. `train.py` then reads from those datasets — it does not call any data prep scripts itself. If a dataset or tokenizer doesn't exist when `train.py` runs, it will crash.

```
format.py          →   nt_text.txt, nsp_text.jsonl, nup_text.jsonl
make_paren.py      →   tokenized_paren.txt, tokenizers/paren_tokenizer
pos_data.py        →   data/pos_dataset, tokenizers/pos_tokenizer
dataprep.py        →   data/nt_dataset, data/nsp_dataset, data/nup_dataset
train.py           →   trained models, training_results.csv
```

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

### Step 4 — Tokenize and save datasets

Reads the files from Step 1 and saves HuggingFace datasets to disk. Datasets are saved to tokenizer-specific subdirectories so running with different tokenizers never overwrites each other.

```bash
python scripts/dataprep.py             # use base Pythia tokenizer
python scripts/dataprep.py --paren     # use paren tokenizer
```

Outputs (base tokenizer):
- `./data/base/nt_dataset`
- `./data/base/nsp_dataset`
- `./data/base/nup_dataset`

Outputs (paren tokenizer):
- `./data/paren/nt_dataset`
- `./data/paren/nsp_dataset`
- `./data/paren/nup_dataset`

---

### Step 5 — Train

Edit `tasks_to_run` in `scripts/train.py` to select which experiments to run, then:

```bash
python scripts/train.py
```

Each task trains the model, evaluates it (CN + BLiMP), and appends a row to `training_results.csv`. Tasks run sequentially; GPU memory is freed between them.

---

## Configuring Experiments

All configuration lives in `scripts/config.py`. There are two things you can change:

**`TrainingConfig`** — hyperparameters that apply to all tasks:

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 2.5e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8
    # ... etc
```

**`TASK_CONFIGS`** — one entry per experiment. To add a new variant, add an entry here:

```python
"my_new_variant": TaskConfig(
    name             = "my_new_variant",
    data_path        = "./data/nt_dataset",
    model_save_path  = "./models/pythia/my-model",
    tokenizer_path   = "./tokenizers/pos_tokenizer",  # omit for base tokenizer
    model_load_path  = "./models/pythia/pos-model",   # omit for fresh init
    use_custom_collator = False,                       # True for NSP/NUP
    train_truncation = None,                           # set int for quick tests
    eval_truncation  = None,
),
```

Then uncomment it in `train.py`:

```python
tasks_to_run = [
    TASK_CONFIGS["my_new_variant"],
]
```

### Two-stage pipelines

Set `model_load_path` on the second task to point at the first task's `model_save_path`. List them in order in `tasks_to_run`:

```python
tasks_to_run = [
    TASK_CONFIGS["pos_pretrain"],        # stage 1: trains and saves pos model
    TASK_CONFIGS["pos_then_next_word"],  # stage 2: loads pos model, fine-tunes
]
```

### Preset experiments

| Name | Description |
|------|-------------|
| `next_word` | Next-token prediction on CHILDES from scratch |
| `pos_pretrain` | POS-tag pre-training on C4 |
| `paren_pretrain` | Dyck/paren pre-training |
| `nsp` | Next-sentence prediction on CHILDES |
| `nup` | Next-utterance prediction on CHILDES |
| `pos_then_next_word` | POS pre-train → next-word fine-tune |
| `paren_then_next_word` | Paren pre-train → next-word fine-tune |
| `next_word_then_nup` | Next-word train → NUP fine-tune |

---

## Results

All evaluation results are appended to `training_results.csv` with columns:

| Column | Description |
|--------|-------------|
| `task_type` | Name of the task from `TaskConfig` |
| `CEL` | Cross-entropy loss |
| `perplexity` | Exp of eval loss |
| `CN` | Crain & Nakayama syntactic evaluation |
| `BLiMP` | Average BLiMP suite accuracy |