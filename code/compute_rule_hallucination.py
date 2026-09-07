import csv
import json
from collections import defaultdict
from pathlib import Path

MODELS = {"qwen2_5_vl", "qwen3_vl", "llava_onevision", "gemini_3_1_flash_lite"}
DATASET_NAMES = {"mathvista_testmini": "mathvista", "pope": "pope", "xlrs_selected500": "xlrs", "mathvista": "mathvista", "xlrs": "xlrs"}
PROMPT_NAMES = {"cot_light_final_first": "cot", "cot_light": "cot", "direct": "direct"}


def norm_dataset(x):
    return DATASET_NAMES.get(x, x)


def norm_prompt(x):
    return PROMPT_NAMES.get(x, x)


def boolish(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def summarize(rows):
    acc = defaultdict(lambda: {"total": 0, "parsed": 0, "parse_errors": 0, "correct": 0, "mismatch": 0})
    for r in rows:
        model = r.get("model_key")
        if model not in MODELS:
            continue
        dataset = norm_dataset(r.get("dataset"))
        prompt = norm_prompt(r.get("prompt_type"))
        key = (dataset, model, prompt)
        item = acc[key]
        item["total"] += 1
        parse_error = r.get("parse_error") or r.get("source_parse_error")
        is_correct = boolish(r.get("is_correct"))
        if is_correct is None:
            is_correct = boolish(r.get("is_correct_by_rule"))
        if parse_error or is_correct is None:
            item["parse_errors"] += 1
            continue
        item["parsed"] += 1
        if is_correct:
            item["correct"] += 1
        else:
            item["mismatch"] += 1
    return acc


def write_summary(acc, path):
    fields = ["dataset", "model_key", "prompt", "total", "parsed", "parse_errors", "answer_accuracy", "rule_proxy_hallucination_rate_valid", "rule_proxy_hallucination_rate_total"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for dataset, model, prompt in sorted(acc):
            v = acc[(dataset, model, prompt)]
            parsed = v["parsed"]
            total = v["total"]
            mismatch = v["mismatch"]
            w.writerow({
                "dataset": dataset,
                "model_key": model,
                "prompt": prompt,
                "total": total,
                "parsed": parsed,
                "parse_errors": v["parse_errors"],
                "answer_accuracy": round(v["correct"] / parsed, 6) if parsed else "",
                "rule_proxy_hallucination_rate_valid": round(mismatch / parsed, 6) if parsed else "",
                "rule_proxy_hallucination_rate_total": round(mismatch / total, 6) if total else "",
            })

root = Path("/data2/huanjue/results/rule_proxy_hallucination")
root.mkdir(parents=True, exist_ok=True)
full_rows = [json.loads(l) for l in open("/data2/huanjue/results/rule_eval_full_all_v3/full_standardized.jsonl", encoding="utf-8") if l.strip()]
write_summary(summarize(full_rows), root / "rule_proxy_full_summary.csv")

sample_rows = [json.loads(l) for l in open("/data2/huanjue/results/judge_analysis_pope500_math300_seed20260904/judge_analysis_inputs.jsonl", encoding="utf-8") if l.strip()]
write_summary(summarize(sample_rows), root / "rule_proxy_pope500_math300_summary.csv")
print(root / "rule_proxy_full_summary.csv")
print(root / "rule_proxy_pope500_math300_summary.csv")
