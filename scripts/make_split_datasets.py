"""
make_split_datasets.py — creates first/second half splits of CHILDES.

Splits the format.py output at ~5M, ~10M, and ~13.2M (half) token boundaries
and saves 7 new datasets to data/split/.

Run once from the project root before training:

    python scripts/make_split_datasets.py
"""

import json
import os
import re
import sys
import tempfile

sys.path.insert(0, "scripts")

from transformers import AutoTokenizer

from config import BASE_MODEL_ID
from dataprep import make_nt_dataset, make_pair_dataset
from format import make_nup_pairs

SPLIT_DIR = "./data/split"

# Half of the existing NT train set: 51572 examples × 512 tokens = 26,404,864 tokens total.
# Half = 13,202,432 tokens.
HALF_TOKENS = 51572 // 2 * 512


# ── Boundary detection ────────────────────────────────────────────────────────

def find_boundaries(nt_lines, tokenizer):
    """
    Scan nt_text.txt line by line and return the line indices where cumulative
    token count first crosses 5M, 10M, and HALF_TOKENS.
    """
    targets = {5_000_000: "5m", 10_000_000: "10m", HALF_TOKENS: "half"}
    boundaries = {}
    cumulative = 0

    print("Scanning nt_text.txt for token boundaries...")
    for i, line in enumerate(nt_lines):
        if line.strip():
            cumulative += len(
                tokenizer(line.strip(), add_special_tokens=False)["input_ids"]
            )
        for tgt, key in list(targets.items()):
            if cumulative >= tgt and key not in boundaries:
                boundaries[key] = i
                print(f"  {key}: line {i:,}  (~{cumulative:,} tokens)")
                del targets[tgt]
        if not targets:
            break

    for key in list(targets.values()):
        boundaries[key] = len(nt_lines)
        print(f"  {key}: end of file (target not reached)")

    return boundaries


# ── NSP helpers ───────────────────────────────────────────────────────────────

def nsp_pairs_from_lines(lines):
    """
    Build NSP pairs from a list of already-cleaned (no speaker tags) text lines.
    Consecutive non-empty lines become (s1, s2) pairs.
    """
    sentences = [l.strip() for l in lines if l.strip()]
    return [{"s1": sentences[i], "s2": sentences[i + 1]}
            for i in range(len(sentences) - 1)]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def write_text(lines, tmpdir, name):
    path = os.path.join(tmpdir, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_jsonl(pairs, tmpdir, name):
    path = os.path.join(tmpdir, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # ── Load source text ──────────────────────────────────────────────────────

    print("Reading data/nt_text.txt ...")
    with open("./data/nt_text.txt", encoding="utf-8") as f:
        nt_lines = f.read().splitlines()
    print(f"  {len(nt_lines):,} lines")

    # Speaker-tagged text — needed for NUP pair generation.
    # Re-derive from childes.train: strip annotations but keep *SPEAKER: tags.
    print("Reading data/childes.train for NUP (speaker-tagged) ...")
    with open("./data/childes.train", encoding="utf-8") as f:
        raw = f.read()
    tagged_lines = []
    for line in raw.strip().split("\n"):
        cleaned = re.sub(r"\[.*?\]", "", line)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            tagged_lines.append(cleaned)
    print(f"  {len(tagged_lines):,} tagged lines")

    # ── Find boundaries ───────────────────────────────────────────────────────

    bnd = find_boundaries(nt_lines, tokenizer)
    b5m, b10m, bhalf = bnd["5m"], bnd["10m"], bnd["half"]

    # Proportional boundaries in tagged_lines (NUP needs speaker tags)
    ratio = len(tagged_lines) / max(len(nt_lines), 1)
    tb5m   = int(b5m   * ratio)
    tb10m  = int(b10m  * ratio)
    tbhalf = int(bhalf * ratio)
    print(f"\nTagged-line boundaries: 5m={tb5m:,}, 10m={tb10m:,}, half={tbhalf:,}")

    # ── Create datasets ───────────────────────────────────────────────────────

    with tempfile.TemporaryDirectory() as tmp:

        # NT datasets (fixed 512-token chunks)
        print("\n── NT ───────────────────────────────────────────────────────────")

        print("Creating data/split/nt_10m ...")
        make_nt_dataset(
            write_text(nt_lines[:b10m], tmp, "nt_10m"),
            f"{SPLIT_DIR}/nt_10m", tokenizer,
        )

        print("Creating data/split/nt_5m_a (first 5M) ...")
        make_nt_dataset(
            write_text(nt_lines[:b5m], tmp, "nt_5m_a"),
            f"{SPLIT_DIR}/nt_5m_a", tokenizer,
        )

        print("Creating data/split/nt_half_a (first ~13.2M) ...")
        make_nt_dataset(
            write_text(nt_lines[:bhalf], tmp, "nt_half_a"),
            f"{SPLIT_DIR}/nt_half_a", tokenizer,
        )

        # NSP datasets (consecutive sentence pairs)
        print("\n── NSP ──────────────────────────────────────────────────────────")

        print("Creating data/split/nsp_5m_b (second 5M) ...")
        pairs = nsp_pairs_from_lines(nt_lines[b5m:b10m])
        make_pair_dataset(
            write_jsonl(pairs, tmp, "nsp_5m_b"),
            f"{SPLIT_DIR}/nsp_5m_b", tokenizer,
        )

        print("Creating data/split/nsp_half_b (second half) ...")
        pairs = nsp_pairs_from_lines(nt_lines[bhalf:])
        make_pair_dataset(
            write_jsonl(pairs, tmp, "nsp_half_b"),
            f"{SPLIT_DIR}/nsp_half_b", tokenizer,
        )

        # NUP datasets (consecutive utterance pairs — needs speaker tags)
        print("\n── NUP ──────────────────────────────────────────────────────────")

        print("Creating data/split/nup_5m_b (second 5M) ...")
        pairs = make_nup_pairs("\n".join(tagged_lines[tb5m:tb10m]))
        make_pair_dataset(
            write_jsonl(pairs, tmp, "nup_5m_b"),
            f"{SPLIT_DIR}/nup_5m_b", tokenizer,
        )

        print("Creating data/split/nup_half_b (second half) ...")
        pairs = make_nup_pairs("\n".join(tagged_lines[tbhalf:]))
        make_pair_dataset(
            write_jsonl(pairs, tmp, "nup_half_b"),
            f"{SPLIT_DIR}/nup_half_b", tokenizer,
        )

    print(f"\nAll 7 split datasets saved to {SPLIT_DIR}/")


if __name__ == "__main__":
    main()
