import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROMPT_PAIR = {
    "pope": ("direct", "cot_light"),
    "mathvista": ("direct", "cot_light_final_first"),
}


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(value):
    return round(value, 6)


def judge_label(row):
    return str(row.get("judge_is_hallucination", "")).strip().lower()


def judge_type(row):
    return str(row.get("judge_hallucination_type", "")).strip().lower()


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_inputs_outputs(input_rows, output_rows):
    input_by_id = {row["annotation_id"]: row for row in input_rows}
    merged = []
    for output in output_rows:
        source = input_by_id.get(output.get("annotation_id"), {})
        row = dict(source)
        row.update(output)
        merged.append(row)
    return merged


def aggregate_by_model_prompt(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model_key"], row["prompt_type"])].append(row)

    summary = []
    for (dataset, model_key, prompt_type), group in sorted(groups.items()):
        valid = [row for row in group if not row.get("error")]
        labels = Counter(judge_label(row) for row in valid)
        types = Counter(judge_type(row) for row in valid if judge_label(row) == "yes")
        correct = sum(str(row.get("is_correct_by_rule")).lower() == "true" for row in valid)
        total = len(group)
        n = len(valid)
        summary.append({
            "dataset": dataset,
            "model_key": model_key,
            "prompt_type": prompt_type,
            "total": total,
            "valid_judge": n,
            "errors": total - n,
            "answer_accuracy_rule": pct(correct / n) if n else "",
            "judge_hallucination_rate": pct(labels["yes"] / n) if n else "",
            "judge_no_rate": pct(labels["no"] / n) if n else "",
            "judge_uncertain_rate": pct(labels["uncertain"] / n) if n else "",
            "vision_grounding_rate": pct(types["vision-grounding error"] / n) if n else "",
            "reasoning_hallucination_rate": pct(types["reasoning hallucination"] / n) if n else "",
            "factual_inconsistency_rate": pct(types["factual inconsistency"] / n) if n else "",
            "hallucination_type_counts": json.dumps(dict(types), ensure_ascii=False, sort_keys=True),
        })
    return summary


def aggregate_cot_delta(rows):
    by_key = {}
    for row in rows:
        if row.get("error"):
            continue
        by_key[(row["dataset"], int(row["sample_index"]), row["model_key"], row["prompt_type"])] = row

    summary = []
    transitions = []
    for dataset, (direct_prompt, cot_prompt) in PROMPT_PAIR.items():
        model_keys = sorted({row["model_key"] for row in rows if row.get("dataset") == dataset})
        sample_indices = sorted({int(row["sample_index"]) for row in rows if row.get("dataset") == dataset})
        for model_key in model_keys:
            paired = []
            for sample_index in sample_indices:
                direct = by_key.get((dataset, sample_index, model_key, direct_prompt))
                cot = by_key.get((dataset, sample_index, model_key, cot_prompt))
                if direct and cot:
                    paired.append((direct, cot))
            if not paired:
                continue
            direct_yes = sum(judge_label(direct) == "yes" for direct, _ in paired)
            cot_yes = sum(judge_label(cot) == "yes" for _, cot in paired)
            cot_introduced = [
                (direct, cot) for direct, cot in paired
                if judge_label(direct) != "yes" and judge_label(cot) == "yes"
            ]
            cot_fixed = [
                (direct, cot) for direct, cot in paired
                if judge_label(direct) == "yes" and judge_label(cot) != "yes"
            ]
            both_yes = sum(judge_label(direct) == "yes" and judge_label(cot) == "yes" for direct, cot in paired)
            both_no = sum(judge_label(direct) != "yes" and judge_label(cot) != "yes" for direct, cot in paired)
            summary.append({
                "dataset": dataset,
                "model_key": model_key,
                "paired_samples": len(paired),
                "direct_hallucination_rate": pct(direct_yes / len(paired)),
                "cot_hallucination_rate": pct(cot_yes / len(paired)),
                "cot_minus_direct": pct((cot_yes - direct_yes) / len(paired)),
                "cot_introduced_count": len(cot_introduced),
                "cot_fixed_count": len(cot_fixed),
                "both_hallucinated_count": both_yes,
                "both_not_hallucinated_count": both_no,
            })
            for direct, cot in cot_introduced[:8]:
                transitions.append(make_transition_case("cot_introduced_hallucination", direct, cot))
            for direct, cot in cot_fixed[:8]:
                transitions.append(make_transition_case("cot_fixed_hallucination", direct, cot))
    return summary, transitions


def make_transition_case(case_type, direct, cot):
    return {
        "case_type": case_type,
        "dataset": direct["dataset"],
        "sample_index": direct["sample_index"],
        "model_key": direct["model_key"],
        "question": direct.get("question", ""),
        "gold_answer": direct.get("gold_answer", ""),
        "direct_prediction": direct.get("model_prediction", ""),
        "direct_output": direct.get("raw_output", ""),
        "direct_judge_label": direct.get("judge_is_hallucination", ""),
        "direct_judge_type": direct.get("judge_hallucination_type", ""),
        "direct_judge_reason": direct.get("judge_reason", ""),
        "cot_prediction": cot.get("model_prediction", ""),
        "cot_output": cot.get("raw_output", ""),
        "cot_judge_label": cot.get("judge_is_hallucination", ""),
        "cot_judge_type": cot.get("judge_hallucination_type", ""),
        "cot_judge_reason": cot.get("judge_reason", ""),
        "image_path": direct.get("image_path", ""),
    }


def select_failure_cases(rows):
    cases = []
    for row in rows:
        if row.get("error") or judge_label(row) != "yes":
            continue
        cases.append({
            "dataset": row.get("dataset"),
            "model_key": row.get("model_key"),
            "prompt_type": row.get("prompt_type"),
            "sample_index": row.get("sample_index"),
            "question": row.get("question", ""),
            "gold_answer": row.get("gold_answer", ""),
            "model_prediction": row.get("model_prediction", ""),
            "is_correct_by_rule": row.get("is_correct_by_rule", ""),
            "raw_output": row.get("raw_output", ""),
            "judge_hallucination_type": row.get("judge_hallucination_type", ""),
            "judge_confidence": row.get("judge_confidence", ""),
            "judge_reason": row.get("judge_reason", ""),
            "image_path": row.get("image_path", ""),
        })
    cases.sort(key=lambda row: (row["dataset"], row["judge_hallucination_type"], -int(row.get("judge_confidence") or 0)))
    return cases[:80]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="/data2/huanjue/results/judge_analysis_pope200_math100_seed20260904/judge_analysis_inputs.jsonl")
    parser.add_argument("--judge-output-jsonl", default="/data2/huanjue/results/judge_analysis_few_shot_gpt_5_4/zero_shot_judge_outputs.jsonl")
    parser.add_argument("--output-dir", default="/data2/huanjue/results/judge_analysis_few_shot_gpt_5_4")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    input_rows = load_jsonl(args.input_jsonl)
    output_rows = load_jsonl(args.judge_output_jsonl)
    rows = merge_inputs_outputs(input_rows, output_rows)

    write_csv(output_dir / "hallucination_by_dataset_model_prompt.csv", aggregate_by_model_prompt(rows))
    cot_summary, transitions = aggregate_cot_delta(rows)
    write_csv(output_dir / "cot_delta_by_model_dataset.csv", cot_summary)
    write_csv(output_dir / "cot_transition_cases.csv", transitions)
    write_csv(output_dir / "failure_cases_for_report.csv", select_failure_cases(rows))

    print("input_rows", len(input_rows))
    print("judge_outputs", len(output_rows))
    print("merged_rows", len(rows))
    print("wrote", output_dir / "hallucination_by_dataset_model_prompt.csv")
    print("wrote", output_dir / "cot_delta_by_model_dataset.csv")
    print("wrote", output_dir / "cot_transition_cases.csv")
    print("wrote", output_dir / "failure_cases_for_report.csv")


if __name__ == "__main__":
    main()
