"""
format.py — converts raw CHILDES data into the three task formats.

Outputs
-------
./data/nt_text.txt       — plain text for next-token training
./data/nsp_text.jsonl    — {"s1": ..., "s2": ...} pairs (sentence-level)
./data/nup_text.jsonl    — {"s1": ..., "s2": ...} pairs (utterance-level)
"""

import json
import re


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_data(text: str, remove_speaker_tags: bool = False) -> str:
    """
    Clean raw CHILDES text.

    Parameters
    ----------
    text               : raw file content
    remove_speaker_tags: if True, strip *SPK: prefixes (needed for next-token
                         data); if False, keep them (needed for NUP parsing)
    """
    lines = text.strip().split("\n")
    cleaned = []

    for line in lines:
        if remove_speaker_tags:
            line = re.sub(r"\*[A-Z]+:\s*", "", line)

        # Always remove CHILDES annotation brackets [...]
        line = re.sub(r"\[.*?\]", "", line)
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            cleaned.append(line)

    return "\n".join(cleaned)


# ── NSP format ────────────────────────────────────────────────────────────────

def make_nsp_pairs(text: str) -> list[dict]:
    """
    Overlapping consecutive-sentence pairs: (s1,s2), (s2,s3), …
    Speaker tags are stripped; bracket annotations already removed by clean_data.
    """
    lines = text.strip().split("\n")
    sentences = []

    for line in lines:
        sentence = re.sub(r"\*[A-Z]+:\s*", "", line).strip()
        if sentence:
            sentences.append(sentence)

    return [{"s1": sentences[i], "s2": sentences[i + 1]}
            for i in range(len(sentences) - 1)]


# ── NUP format ────────────────────────────────────────────────────────────────

def make_nup_pairs(text: str) -> list[dict]:
    """
    Overlapping consecutive-utterance pairs where an utterance is everything
    said by one speaker before the next speaker takes over.
    """
    lines = text.strip().split("\n")
    utterances = []
    current_speaker = None
    current_parts: list[str] = []

    def flush():
        if current_speaker is not None and current_parts:
            utterance = " ".join(current_parts).strip()
            if utterance:
                utterances.append(utterance)

    for line in lines:
        if not line.strip():
            continue

        speaker_match = re.match(r"^\*([A-Z]{2,4}):\s*(.*)", line.strip())

        if speaker_match:
            speaker = speaker_match.group(1)
            content = speaker_match.group(2)

            if speaker != current_speaker:
                flush()
                current_parts = []
                current_speaker = speaker

            content = re.sub(r"\[[^\]]*\]", "", content)
            content = re.sub(r"\s+", " ", content).strip()
            if content:
                current_parts.append(content)

        else:
            # Continuation line (no speaker tag)
            cont = re.sub(r"\[[^\]]*\]", "", line.strip())
            cont = re.sub(r"\s+", " ", cont).strip()
            if cont:
                current_parts.append(cont)

    flush()  # save the last utterance

    return [{"s1": utterances[i], "s2": utterances[i + 1]}
            for i in range(len(utterances) - 1)]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def write_text(text: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_jsonl(pairs: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    raw_path = "./data/childes.train"

    with open(raw_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Next-token: plain text with speaker tags removed
    nt_text = clean_data(raw, remove_speaker_tags=True)
    write_text(nt_text, "./data/nt_text.txt")
    print("Wrote ./data/nt_text.txt")

    # NSP / NUP: keep speaker tags in intermediate form so parsers can use them
    clean_text = clean_data(raw, remove_speaker_tags=False)

    nsp_pairs = make_nsp_pairs(clean_text)
    write_jsonl(nsp_pairs, "./data/nsp_text.jsonl")
    print(f"Wrote ./data/nsp_text.jsonl  ({len(nsp_pairs):,} pairs)")

    nup_pairs = make_nup_pairs(clean_text)
    write_jsonl(nup_pairs, "./data/nup_text.jsonl")
    print(f"Wrote ./data/nup_text.jsonl  ({len(nup_pairs):,} pairs)")


if __name__ == "__main__":
    main()
