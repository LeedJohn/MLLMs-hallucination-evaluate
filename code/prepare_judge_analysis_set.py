import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image


MODELS = ["qwen2_5_vl", "qwen3_vl", "llava_onevision", "gemini_3_1_flash_lite"]
PROMPTS = {
    "pope": ["direct", "cot_light"],
    "mathvista": ["direct", "cot_light_final_first"],
}


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_image(image, path, max_side):
    if isinstance(image, list):
        image = image[0]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")
    image = image.convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    image.save(path, quality=92)


def sample_pope(rows, n, rng):
    by_key = defaultdict(list)
    seen = set()
    for row in rows:
        if row["dataset"] != "pope":
            continue
        if row["model_key"] != MODELS[0] or row["prompt_type"] != PROMPTS["pope"][0]:
            continue
        idx = int(row["sample_index"])
        if idx in seen:
            continue
        seen.add(idx)
        category = row.get("category") or "unknown"
        gold = str(row.get("gold_answer", "")).lower()
        by_key[(category, gold)].append(row)

    selected = []
    keys = sorted(by_key)
    base = n // len(keys)
    remainder = n % len(keys)
    for pos, key in enumerate(keys):
        pool = by_key[key][:]
        rng.shuffle(pool)
        take = base + (1 if pos < remainder else 0)
        selected.extend(pool[:take])

    if len(selected) < n:
        used = {int(row["sample_index"]) for row in selected}
        fallback = [
            row for rows_for_key in by_key.values() for row in rows_for_key
            if int(row["sample_index"]) not in used
        ]
        rng.shuffle(fallback)
        selected.extend(fallback[: n - len(selected)])

    selected = selected[:n]
    selected.sort(key=lambda row: int(row["sample_index"]))
    return selected


def sample_mathvista(rows, n, rng):
    by_key = defaultdict(list)
    seen = set()
    for row in rows:
        if row["dataset"] != "mathvista":
            continue
        if row["model_key"] != MODELS[0] or row["prompt_type"] != PROMPTS["mathvista"][0]:
            continue
        idx = int(row["sample_index"])
        if idx in seen:
            continue
        seen.add(idx)
        answer_type = row.get("answer_type") or "unknown"
        question_type = row.get("question_type") or "unknown"
        by_key[(answer_type, question_type)].append(row)

    selected = []
    keys = sorted(by_key, key=lambda key: len(by_key[key]), reverse=True)
    target_per_key = max(1, n // max(1, min(len(keys), 10)))
    for key in keys:
        pool = by_key[key][:]
        rng.shuffle(pool)
        take = min(target_per_key, len(pool), n - len(selected))
        selected.extend(pool[:take])
        if len(selected) >= n:
            break

    if len(selected) < n:
        used = {int(row["sample_index"]) for row in selected}
        fallback = [
            row for rows_for_key in by_key.values() for row in rows_for_key
            if int(row["sample_index"]) not in used
        ]
        rng.shuffle(fallback)
        selected.extend(fallback[: n - len(selected)])

    selected = selected[:n]
    selected.sort(key=lambda row: int(row["sample_index"]))
    return selected


def build_lookup(rows):
    lookup = {}
    for row in rows:
        if row.get("dataset") not in PROMPTS:
            continue
        key = (
            row.get("dataset"),
            int(row.get("sample_index")),
            row.get("model_key"),
            row.get("prompt_type"),
        )
        lookup[key] = row
    return lookup


def make_source_maps(output_dir, pope_selected, math_selected, max_side):
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    path_map = {}
    meta_map = {}

    pope_ds = load_dataset("/data2/huanjue/datasets/pope", split="test")
    math_ds = load_dataset("/data2/huanjue/datasets/mathvista", split="testmini")
    for dataset, selected, ds in [
        ("pope", pope_selected, pope_ds),
        ("mathvista", math_selected, math_ds),
    ]:
        for row in selected:
            sample_index = int(row["sample_index"])
            source = ds[int(row["dataset_index"])]
            image = source.get("decoded_image") or source.get("image")
            image_path = image_dir / f"{dataset}_{sample_index:04d}.jpg"
            if not image_path.exists():
                save_image(image, image_path, max_side)
            path_map[(dataset, sample_index)] = str(image_path)
            question = source.get("query") or source.get("question") or ""
            meta_map[(dataset, sample_index)] = {
                "question": question,
                "choices": source.get("choices"),
                "gold_answer": source.get("answer"),
            }
    return path_map, meta_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-jsonl", default="/data2/huanjue/results/rule_eval_full_all_v3/full_standardized.jsonl")
    parser.add_argument("--output-dir", default="/data2/huanjue/results/judge_analysis_pope200_math100_seed20260904")
    parser.add_argument("--pope-n", type=int, default=200)
    parser.add_argument("--mathvista-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--image-max-side", type=int, default=900)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.standardized_jsonl)
    lookup = build_lookup(rows)
    pope_selected = sample_pope(rows, args.pope_n, rng)
    math_selected = sample_mathvista(rows, args.mathvista_n, rng)
    image_map, source_meta = make_source_maps(output_dir, pope_selected, math_selected, args.image_max_side)

    sample_records = []
    for dataset, selected in [("pope", pope_selected), ("mathvista", math_selected)]:
        for row in selected:
            sample_records.append({
                "dataset": dataset,
                "sample_index": int(row["sample_index"]),
                "dataset_index": int(row["dataset_index"]),
                "sample_id": row.get("sample_id"),
                "category": row.get("category"),
                "question_type": row.get("question_type"),
                "answer_type": row.get("answer_type"),
                "question": source_meta[(dataset, int(row["sample_index"]))]["question"],
                "choices": source_meta[(dataset, int(row["sample_index"]))]["choices"],
                "gold_answer": row.get("gold_answer") or source_meta[(dataset, int(row["sample_index"]))]["gold_answer"],
                "image_path": image_map[(dataset, int(row["sample_index"]))],
            })

    with (output_dir / "fixed_sample_ids.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": args.seed,
                "pope_n": args.pope_n,
                "mathvista_n": args.mathvista_n,
                "models": MODELS,
                "prompts": PROMPTS,
                "samples": sample_records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    judge_inputs = []
    missing = []
    counter = 1
    for sample in sample_records:
        dataset = sample["dataset"]
        sample_index = int(sample["sample_index"])
        for model_key in MODELS:
            for prompt_type in PROMPTS[dataset]:
                key = (dataset, sample_index, model_key, prompt_type)
                row = lookup.get(key)
                if not row:
                    missing.append(key)
                    continue
                judge_inputs.append({
                    "annotation_id": f"JA{counter:05d}",
                    "dataset": dataset,
                    "sample_index": sample_index,
                    "dataset_index": sample["dataset_index"],
                    "sample_id": sample["sample_id"],
                    "model_key": model_key,
                    "prompt_type": prompt_type,
                    "question": sample.get("question") or row.get("question") or "",
                    "choices": json.dumps(sample.get("choices"), ensure_ascii=False) if sample.get("choices") else "",
                    "gold_answer": row.get("gold_answer"),
                    "model_prediction": row.get("pred_answer"),
                    "is_correct_by_rule": row.get("is_correct"),
                    "raw_output": row.get("raw_output"),
                    "image_path": sample["image_path"],
                    "source_parse_error": row.get("parse_error"),
                    "human_is_hallucination": "",
                    "human_hallucination_type": "",
                    "human_confidence_1_to_3": "",
                })
                counter += 1

    if missing:
        raise RuntimeError(f"Missing model outputs: {missing[:10]} total={len(missing)}")

    input_path = output_dir / "judge_analysis_inputs.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for row in judge_inputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_dir / "judge_analysis_inputs_preview.csv"
    fields = [
        "annotation_id", "dataset", "sample_index", "model_key", "prompt_type",
        "gold_answer", "model_prediction", "is_correct_by_rule", "question",
        "raw_output", "image_path",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in judge_inputs:
            writer.writerow({field: row.get(field, "") for field in fields})

    print("output_dir", output_dir)
    print("sample_count", Counter(row["dataset"] for row in sample_records))
    print("judge_input_count", len(judge_inputs))
    print("by_dataset", Counter(row["dataset"] for row in judge_inputs))
    print("by_model", Counter(row["model_key"] for row in judge_inputs))
    print("by_prompt", Counter(row["prompt_type"] for row in judge_inputs))
    print("pope_gold", Counter(row["gold_answer"] for row in sample_records if row["dataset"] == "pope"))
    print("math_answer_type", Counter(row["answer_type"] for row in sample_records if row["dataset"] == "mathvista"))
    print("wrote", input_path)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
