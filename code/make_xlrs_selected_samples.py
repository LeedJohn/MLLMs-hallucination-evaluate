import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_from_disk


SEED = 20260903
DATASET_PATH = "/data2/huanjue/datasets/xlrs-lite"
OUTPUT_PATH = "/data2/huanjue/samples/xlrs_selected500_seed20260903.json"

SELECTED_CATEGORIES = [
    "Object properties/Object color",
    "Object properties/Object classification",
    "Object spatial relationship/Object spatial relationship",
    "Counting/Regional counting",
    "Counting/Counting with complex reasoning",
]


def balanced_sample(rows, n, rng):
    by_answer = defaultdict(list)
    for item in rows:
        by_answer[item["answer"]].append(item)
    if len(rows) <= n:
        return sorted(rows, key=lambda x: x["index"])

    answers = sorted(by_answer)
    base = n // len(answers)
    remainder = n % len(answers)
    selected = []
    leftovers = []
    for pos, answer in enumerate(answers):
        bucket = by_answer[answer][:]
        rng.shuffle(bucket)
        take = min(len(bucket), base + (1 if pos < remainder else 0))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])

    if len(selected) < n:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])
    return sorted(selected[:n], key=lambda x: x["index"])


def main():
    rng = random.Random(SEED)
    obj = load_from_disk(DATASET_PATH)
    ds = obj["train"] if hasattr(obj, "keys") else obj
    meta = ds.remove_columns("image")

    rows_by_category = defaultdict(list)
    for dataset_index, row in enumerate(meta):
        category = row["category"]
        if category not in SELECTED_CATEGORIES:
            continue
        rows_by_category[category].append(
            {
                "index": dataset_index,
                "sample_id": str(row["index"]),
                "category": category,
                "answer": row["answer"],
            }
        )

    samples = []
    per_category = {}
    for category in SELECTED_CATEGORIES:
        rows = rows_by_category[category]
        selected = balanced_sample(rows, 100, rng)
        per_category[category] = len(selected)
        samples.extend(selected)

    samples = sorted(samples, key=lambda x: (x["category"], x["index"]))
    payload = {
        "dataset": "xlrs",
        "source": DATASET_PATH,
        "split": "train",
        "seed": SEED,
        "selected_categories": SELECTED_CATEGORIES,
        "total": len(samples),
        "per_category": per_category,
        "samples": samples,
    }

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_path} samples={len(samples)}")
    for category, count in per_category.items():
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()
