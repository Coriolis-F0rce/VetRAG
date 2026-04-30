#!/usr/bin/env python3
"""
S6: Final Merge and Export
Merge all data sources from S1-S5, deduplicate, and export to training format.
"""

import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set


BASE_DIR = Path(r"D:\Backup\PythonProject2\data_process")
OUTPUT_DIR = BASE_DIR / "final_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data sources
SOURCES = [
    # S1: Merged FAQ data (highest quality, largest)
    (BASE_DIR / "find_faq" / "merged_output" / "s1_merged_all.json", "s1_merged_faq", 1.0),
    # S2: Disease KB -> QA
    (BASE_DIR / "find_diseases" / "s2_disease_qa.json", "s2_disease_qa", 1.0),
    # S3: Expanded topics
    (BASE_DIR / "s3_expanded_output" / "s3_expanded_all.json", "s3_expanded", 1.0),
    # S4: Augmented (if available)
    (BASE_DIR / "s4_augmented_output" / "s4_final.json", "s4_augmented", 1.0),
    # S5: Safety & format QA
    (BASE_DIR / "s5_safety_output" / "s5_safety_qa.json", "s5_safety", 1.0),
]


def compute_hash(entry: Dict) -> str:
    text = (entry.get("instruction", "") + "|" + entry.get("output", "")).strip()
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_entries(path: Path, source: str, weight: float) -> List[Dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        entries = []
        for item in data:
            normalized = {
                "instruction": item.get("instruction", "").strip(),
                "input": item.get("input", "").strip(),
                "output": item.get("output", "").strip(),
            }
            # Merge existing metadata with source info
            meta = dict(item.get("metadata", {}))
            meta["_source"] = source
            meta["_weight"] = weight
            normalized["metadata"] = meta
            entries.append(normalized)
        return entries
    except Exception as e:
        print(f"  [!] Failed to load {path.name}: {e}")
        return []


def main():
    print("=" * 60)
    print("S6: Final Merge and Export")
    print("=" * 60)

    # Load all sources
    all_entries: List[Dict] = []
    source_stats: Dict[str, int] = {}
    seen_hashes: Set[str] = set()

    for path, source, weight in SOURCES:
        entries = load_entries(path, source, weight)
        if entries:
            print(f"\n[*] {source}: {len(entries)} entries")
            source_stats[source] = len(entries)
            all_entries.extend(entries)

    print(f"\n[*] Total entries before dedup: {len(all_entries)}")

    # Dedup by hash
    unique_entries: List[Dict] = []
    dup_count = 0
    for entry in all_entries:
        h = compute_hash(entry)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_entries.append(entry)
        else:
            dup_count += 1

    print(f"[*] Duplicates removed: {dup_count}")
    print(f"[*] Final unique entries: {len(unique_entries)}")

    # Category stats
    categories = defaultdict(int)
    for e in unique_entries:
        cat = e.get("metadata", {}).get("category", "unknown")
        categories[cat] += 1

    print(f"\n[*] By category:")
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
        pct = cnt / len(unique_entries) * 100
        print(f"   {cat}: {cnt} ({pct:.1f}%)")

    # Save final outputs
    print("\n[*] Saving outputs...")

    # 1. Full JSON
    full_path = OUTPUT_DIR / "final_training_data.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(unique_entries, f, ensure_ascii=False, indent=2)
    print(f"   [OK] Full JSON: {len(unique_entries)} entries -> {full_path}")

    # 2. Alpaeca format (instruction + output only)
    alpaca_path = OUTPUT_DIR / "final_training_data_alpaca.jsonl"
    with open(alpaca_path, "w", encoding="utf-8") as f:
        for e in unique_entries:
            alpaca = {
                "instruction": e["instruction"],
                "input": e["input"],
                "output": e["output"],
            }
            f.write(json.dumps(alpaca, ensure_ascii=False) + "\n")
    print(f"   [OK] Alpaca JSONL: {len(unique_entries)} entries -> {alpaca_path}")

    # 3. Summary report
    summary = {
        "total_entries": len(unique_entries),
        "duplicates_removed": dup_count,
        "by_source": source_stats,
        "by_category": dict(categories),
        "sources": [str(p) for p, s, w in SOURCES if p.exists()],
    }
    summary_path = OUTPUT_DIR / "final_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"   [OK] Summary -> {summary_path}")

    print(f"\n{'='*60}")
    print(f"[OK] S6 Done. Final dataset: {len(unique_entries)} entries")
    print(f"   Output dir: {OUTPUT_DIR}")
    print(f"   Main files:")
    print(f"     - final_training_data.json")
    print(f"     - final_training_data_alpaca.jsonl")
    print(f"     - final_summary.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
