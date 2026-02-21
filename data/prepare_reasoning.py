"""
Prepare unified reasoning dataset for Grok fine-tuning.

Merges all downloaded reasoning datasets into a single JSONL
with consistent format: messages[] with <think> reasoning chains.

Tier 1 (main):  grok_code_fast_1000x     — 1017 rows, code+math+reasoning
Tier 2 (supplement):
  - claude45_opus_reasoning_250x          — 250 rows, deep reasoning chains
  - claude37_sonnet_reasoning             — 179 rows, <think> format
  - grok3_reasoning_100x                  — 100 rows, algorithmic reasoning
Tier 3 (extra):
  - gpt45_100x                            — 100 rows, quality responses (no reasoning)
  - grok3_100x                            — 100 rows, general (no reasoning)
  - grok4_brainstorm_200x                 — 192 rows, multi-turn

Usage:
    python prepare_reasoning.py [--include-tier3]
"""

import json
import os
import argparse

DATA_DIR = "/home/ubuntu/grokky.go/data/reasoning"
OUT_DIR = "/home/ubuntu/grokky.go/data"

TIER1 = ["grok_code_fast_1000x.jsonl"]
TIER2 = [
    "claude45_opus_reasoning_250x.jsonl",
    "claude37_sonnet_reasoning.jsonl",
    "grok3_reasoning_100x.jsonl",
]
TIER3 = [
    "gpt45_100x.jsonl",
    "grok3_100x.jsonl",
    "grok4_brainstorm_200x.jsonl",
]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_messages(row):
    """Extract messages from various formats."""
    if "messages" in row:
        return row["messages"]
    if "conversations" in row:
        return row["conversations"]
    if "instruction" in row and "response" in row:
        msgs = [{"role": "user", "content": row["instruction"]}]
        if row.get("input"):
            msgs[0]["content"] += "\n" + row["input"]
        msgs.append({"role": "assistant", "content": row["response"]})
        return msgs
    return None


def has_reasoning(messages):
    """Check if any assistant message has <think> tags."""
    for m in messages:
        if m.get("role") == "assistant" and "<think>" in m.get("content", ""):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-tier3", action="store_true")
    args = parser.parse_args()

    files = TIER1 + TIER2
    if args.include_tier3:
        files += TIER3

    all_messages = []
    stats = {"total": 0, "with_reasoning": 0, "skipped": 0}

    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP (missing): {fname}")
            continue
        rows = load_jsonl(path)
        count = 0
        reasoning_count = 0
        for row in rows:
            msgs = extract_messages(row)
            if msgs is None:
                stats["skipped"] += 1
                continue
            entry = {"messages": msgs, "source": fname.replace(".jsonl", "")}
            if has_reasoning(msgs):
                entry["has_reasoning"] = True
                reasoning_count += 1
            all_messages.append(entry)
            count += 1
        print(f"  {fname}: {count} rows ({reasoning_count} with <think> reasoning)")
        stats["total"] += count
        stats["with_reasoning"] += reasoning_count

    # Save
    out_path = os.path.join(OUT_DIR, "reasoning_combined.jsonl")
    with open(out_path, "w") as f:
        for entry in all_messages:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSaved: {out_path}")
    print(f"  Total: {stats['total']} examples ({stats['with_reasoning']} with reasoning)")
    print(f"  Size: {sz:.1f}MB")
    print(f"  Skipped: {stats['skipped']}")


if __name__ == "__main__":
    main()
