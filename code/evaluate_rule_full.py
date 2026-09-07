import argparse
import csv
import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


MODEL_PREFIXES = {
    "Qwen2.5-VL-7B-Instruct": "qwen2_5_vl",
    "Qwen3-VL-8B-Instruct": "qwen3_vl",
    "LLaVA-OneVision-Qwen2-7B-OV-HF": "llava_onevision",
}

DATASET_PATHS = {
    "mathvista": "/data2/huanjue/datasets/mathvista",
    "xlrs": "/data2/huanjue/datasets/xlrs-lite",
}


def infer_model_dataset(path):
    name = path.name
    for prefix, model_key in MODEL_PREFIXES.items():
        if name.startswith(prefix + "__"):
            rest = name[len(prefix) + 2 :]
            dataset = rest.split("_pilot__", 1)[0]
            return model_key, prefix, dataset
    raise ValueError(f"Cannot infer model/dataset from filename: {path.name}")


def strip_final_answer(text):
    if text is None:
        return ""
    text = str(text).strip()
    matches = list(re.finditer(r"final\s+answer\s*:?", text, flags=re.IGNORECASE))
    if matches:
        after = text[matches[-1].end() :].strip()
        if after:
            answer_span = re.split(
                r"(?:\r?\n|\b(?:reason|explanation|check)\s*:)",
                after,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            text = answer_span or after
    return text


def normalize_text(text):
    text = strip_final_answer(text).lower().strip()
    text = text.replace("`", "").replace("''", "").replace('"', "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.strip(string.punctuation + " ")
    return text


def normalize_yes_no(text):
    text = normalize_text(text)
    tokens = re.findall(r"[a-z]+", text)
    if not tokens:
        return "unknown"
    if tokens[0] in {"yes", "yeah", "yep"}:
        return "yes"
    if tokens[0] in {"no", "not"}:
        return "no"
    if "yes" in tokens and "no" not in tokens:
        return "yes"
    if "no" in tokens and "yes" not in tokens:
        return "no"
    return "unknown"


def extract_numbers(text):
    text = strip_final_answer(text)
    return re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", text.replace(",", ""))


def to_float(value):
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def normalize_mathvista_prediction(text, answer_type):
    if answer_type in {"integer", "float"}:
        numbers = extract_numbers(text)
        return numbers[-1] if numbers else ""
    text = normalize_text(text)
    # Many multi-choice models return "(A)" or "A."; keep the option letter when it is alone.
    option = re.match(r"^\(?([a-z])\)?(?:[\s\.:,-]|$)", text)
    if option and len(text) <= 8:
        return option.group(1)
    return text


def extract_option_letter(text, num_choices):
    if text is None:
        return None
    raw = str(text).lower()
    cleaned = normalize_text(text)
    if re.fullmatch(r"\(?[a-h]\)?", cleaned):
        letter = cleaned.strip("()")
        return letter if ord(letter) - ord("a") < num_choices else None

    patterns = [
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*\(?([a-h])\)?\b",
        r"\(([a-h])\)",
    ]
    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, raw, flags=re.IGNORECASE):
            candidates.append(match.group(1).lower())
    valid = [x for x in candidates if ord(x) - ord("a") < num_choices]
    return valid[-1] if valid else None


def extract_option_letters(text, num_choices):
    if text is None:
        return ""
    span = strip_final_answer(text)
    cleaned = normalize_text(span).upper()
    valid = {chr(ord("A") + i) for i in range(num_choices)}

    compact = re.sub(r"[^A-H]", "", cleaned)
    if len(compact) == 1 and compact in valid:
        return compact

    letters = re.findall(r"\b([A-H])\b", cleaned)
    letters = [letter for letter in letters if letter in valid]
    unique_letters = sorted(set(letters))
    if len(unique_letters) == 1:
        return unique_letters[0]
    return ""


def normalize_choice_text(choice):
    return normalize_text(str(choice))


def find_gold_option_letter(gold, choices):
    gold_norm = normalize_text(gold)
    for idx, choice in enumerate(choices):
        if normalize_choice_text(choice) == gold_norm:
            return chr(ord("a") + idx)
    return None


def normalize_gold(gold, answer_type):
    if answer_type in {"integer", "float"}:
        value = to_float(gold)
        if value is None:
            return normalize_text(gold)
        if answer_type == "integer":
            return str(int(round(value)))
        return str(value)
    return normalize_text(gold)


def mathvista_match(pred, gold, answer_type):
    if answer_type in {"integer", "float"}:
        p = to_float(pred)
        g = to_float(gold)
        if p is None or g is None:
            return False
        if answer_type == "integer":
            return int(round(p)) == int(round(g))
        return abs(p - g) <= max(1e-3, abs(g) * 1e-3)
    return normalize_text(pred) == normalize_text(gold)


def standardize_record(record, model_key, model_name, dataset, source_row=None):
    prompt_type = record.get("prompt_type")
    raw_output = record.get("raw_output")
    gold = record.get("gold_answer")
    answer_type = record.get("answer_type")

    result = {
        "model_key": model_key,
        "model_name": model_name,
        "dataset": dataset,
        "prompt_type": prompt_type,
        "sample_index": record.get("sample_index"),
        "dataset_index": record.get("dataset_index"),
        "sample_id": record.get("sample_id"),
        "category": record.get("category"),
        "question_type": record.get("question_type"),
        "answer_type": answer_type,
        "gold_answer": gold,
        "raw_output": raw_output,
        "pred_answer": None,
        "pred_option_letter": None,
        "pred_choice": None,
        "gold_option_letter": None,
        "choices": None,
        "is_correct": None,
        "rule_hallucination": None,
        "rule_hallucination_type": None,
        "parse_error": False,
        "source_error": record.get("error"),
        "elapsed_sec": record.get("elapsed_sec"),
    }

    if record.get("error"):
        result["parse_error"] = True
        return result

    if dataset == "pope":
        pred = normalize_yes_no(raw_output)
        gold_norm = normalize_yes_no(gold)
        result["pred_answer"] = pred
        result["gold_answer"] = gold_norm
        result["parse_error"] = pred == "unknown"
        result["is_correct"] = pred == gold_norm
        result["rule_hallucination"] = gold_norm == "no" and pred == "yes"
        result["rule_hallucination_type"] = "object_hallucination" if result["rule_hallucination"] else "none"
        return result

    if dataset == "mathvista":
        choices = source_row.get("choices") if source_row is not None else None
        if choices:
            result["choices"] = list(choices)
            result["gold_option_letter"] = find_gold_option_letter(gold, choices)
            pred_letter = extract_option_letter(raw_output, len(choices))
            if pred_letter is not None:
                choice_idx = ord(pred_letter) - ord("a")
                result["pred_option_letter"] = pred_letter
                result["pred_choice"] = choices[choice_idx]
                pred = normalize_choice_text(choices[choice_idx])
            else:
                pred = normalize_mathvista_prediction(raw_output, answer_type)
        else:
            pred = normalize_mathvista_prediction(raw_output, answer_type)
        gold_norm = normalize_gold(gold, answer_type)
        result["pred_answer"] = pred
        result["gold_answer"] = gold_norm
        result["parse_error"] = pred == ""
        result["is_correct"] = mathvista_match(pred, gold_norm, answer_type)
        result["rule_hallucination"] = None
        result["rule_hallucination_type"] = "not_rule_labeled"
        return result

    if dataset == "xlrs":
        choices = record.get("options")
        if choices:
            result["choices"] = list(choices)
        num_choices = len(choices) if choices else 4
        pred = extract_option_letters(raw_output, num_choices)
        gold_norm = "".join(sorted(set(str(gold).upper().strip())))
        result["pred_answer"] = pred
        result["gold_answer"] = gold_norm
        result["parse_error"] = pred == ""
        result["is_correct"] = pred == gold_norm
        result["rule_hallucination"] = None
        result["rule_hallucination_type"] = "not_rule_labeled"
        return result

    raise ValueError(f"Unsupported dataset: {dataset}")


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else math.nan


def f1_from_counts(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
        return math.nan
    return 2 * precision * recall / (precision + recall)


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["model_key"], record["dataset"], record["prompt_type"])].append(record)

    rows = []
    for (model_key, dataset, prompt_type), items in sorted(groups.items()):
        total = len(items)
        parsed = [x for x in items if not x["parse_error"] and not x["source_error"]]
        correct = sum(x["is_correct"] is True for x in parsed)
        row = {
            "model_key": model_key,
            "dataset": dataset,
            "prompt_type": prompt_type,
            "total": total,
            "parsed": len(parsed),
            "parse_errors": total - len(parsed),
            "accuracy": round(safe_div(correct, len(parsed)), 6) if parsed else "",
            "avg_elapsed_sec": round(sum(float(x["elapsed_sec"] or 0) for x in items) / total, 6) if total else "",
        }

        if dataset == "pope":
            tp = sum(x["pred_answer"] == "yes" and x["gold_answer"] == "yes" for x in parsed)
            fp = sum(x["pred_answer"] == "yes" and x["gold_answer"] == "no" for x in parsed)
            fn = sum(x["pred_answer"] == "no" and x["gold_answer"] == "yes" for x in parsed)
            gold_no = sum(x["gold_answer"] == "no" for x in parsed)
            object_hallucinations = sum(x["rule_hallucination"] is True for x in parsed)
            row.update(
                {
                    "yes_f1": round(f1_from_counts(tp, fp, fn), 6),
                    "object_hallucination_count": object_hallucinations,
                    "object_hallucination_rate_gold_no": round(safe_div(object_hallucinations, gold_no), 6),
                    "rule_hallucination_rate_total": round(safe_div(object_hallucinations, len(parsed)), 6),
                }
            )
        else:
            row.update(
                {
                    "yes_f1": "",
                    "object_hallucination_count": "",
                    "object_hallucination_rate_gold_no": "",
                    "rule_hallucination_rate_total": "",
                }
            )
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_bad_cases(output_dir, records, limit_per_group=50):
    bad_dir = output_dir / "bad_cases"
    bad_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for record in records:
        if record["dataset"] == "pope":
            if record["is_correct"] is False or record["rule_hallucination"] is True:
                grouped[(record["model_key"], record["dataset"], record["prompt_type"])].append(record)
        elif record["dataset"] in {"mathvista", "xlrs"}:
            if record["parse_error"] or record["is_correct"] is False:
                grouped[(record["model_key"], record["dataset"], record["prompt_type"])].append(record)

    manifest = []
    for (model_key, dataset, prompt_type), items in sorted(grouped.items()):
        path = bad_dir / f"{model_key}__{dataset}__{prompt_type}__bad_cases.jsonl"
        if path.exists():
            path.unlink()
        for item in items[:limit_per_group]:
            append_jsonl(path, item)
        manifest.append(
            {
                "model_key": model_key,
                "dataset": dataset,
                "prompt_type": prompt_type,
                "bad_case_count": len(items),
                "saved_count": min(len(items), limit_per_group),
                "path": str(path),
            }
        )
    write_csv(output_dir / "full_bad_case_manifest.csv", manifest)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/data2/huanjue/results/full")
    parser.add_argument("--output-dir", default="/data2/huanjue/results/rule_eval_full")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    standardized_path = output_dir / "full_standardized.jsonl"
    summary_path = output_dir / "full_rule_summary.csv"

    all_records = []
    mathvista_ds = None
    if standardized_path.exists():
        standardized_path.unlink()

    for path in sorted(input_dir.glob("*/*/*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            first_record = None
            for line in handle:
                if line.strip():
                    first_record = json.loads(line)
                    break
        if first_record is None:
            continue

        model_key = first_record.get("model_key") or path.parts[-3]
        model_name = first_record.get("model_name") or model_key
        dataset = first_record.get("dataset")
        if dataset is None:
            dataset_dir = path.parts[-2]
            dataset = "mathvista" if dataset_dir.startswith("mathvista") else dataset_dir

        if dataset == "mathvista" and mathvista_ds is None:
            mathvista_ds = load_dataset(DATASET_PATHS["mathvista"], split="testmini")

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                source_row = None
                if dataset == "mathvista" and mathvista_ds is not None:
                    source_row = mathvista_ds[int(record["dataset_index"])]
                standardized = standardize_record(record, model_key, model_name, dataset, source_row)
                append_jsonl(standardized_path, standardized)
                all_records.append(standardized)

    rows = summarize(all_records)
    write_csv(summary_path, rows)
    write_bad_cases(output_dir, all_records)
    print(f"[write] {standardized_path} records={len(all_records)}")
    print(f"[write] {summary_path} rows={len(rows)}")
    print(f"[write] {output_dir / 'full_bad_case_manifest.csv'}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
