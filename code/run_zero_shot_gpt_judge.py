import argparse
import base64
import csv
import io
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI
from PIL import Image


ALLOWED_TYPES = {
    "factual inconsistency",
    "vision-grounding error",
    "reasoning hallucination",
    "none",
    "uncertain",
}


FEW_SHOT_EXAMPLES = """Few-shot examples:

Example 1
Question: Is there a car in the image?
Reference answer: no
Model response: Yes, there is a car in the background.
Judgment:
{
  "is_hallucination": "yes",
  "hallucination_type": "vision-grounding error",
  "confidence": 3,
  "reason": "The model claims that a car is visible although the image does not show one."
}

Example 2
Question: If the chart shows A = 8 and B = 5, which value is larger?
Reference answer: A
Model response: B is larger.
Judgment:
{
  "is_hallucination": "yes",
  "hallucination_type": "reasoning hallucination",
  "confidence": 3,
  "reason": "The model makes an incorrect comparison based on the given values."
}

Example 3
Question: What is the sum of the largest and smallest bars?
Reference answer: 5.50 trillion
Model response: Final answer: 5.50. Reason: The largest bar is 5.25 trillion and the smallest is 0.2474 trillion.
Judgment:
{
  "is_hallucination": "no",
  "hallucination_type": "none",
  "confidence": 3,
  "reason": "The answer is consistent after unit conversion between billion and trillion."
}

Example 4
Question: Which segment is the second largest in the pie chart?
Reference answer: 21%
Model response: The second largest segment is 15%, so the answer is 0.33.
Judgment:
{
  "is_hallucination": "yes",
  "hallucination_type": "vision-grounding error",
  "confidence": 3,
  "reason": "The model misreads the chart segment value even if the final answer may be correct."
}
"""


def image_to_data_url(path, max_side):
    image = Image.open(path).convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def load_records(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def existing_ids(path):
    done = set()
    if not Path(path).exists():
        return done
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("error"):
                done.add(row.get("annotation_id"))
    return done


def append_jsonl(path, record, lock):
    with lock:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def build_prompt(record, prompt_mode):
    choices = record.get("choices") or ""
    if choices:
        choices_block = f"\nChoices: {choices}"
    else:
        choices_block = ""
    examples = ""
    if prompt_mode == "few_shot":
        examples = "\n" + FEW_SHOT_EXAMPLES + "\n"
    return f"""You are a strict hallucination judge for multimodal question answering.

Decide whether the model response contains a hallucination with respect to the image, the question, and the reference answer.

Definitions:
- factual inconsistency: the response introduces external factual claims that conflict with real-world facts or verifiable background knowledge.
- vision-grounding error: the response is inconsistent with visible evidence in the image, including object existence, attributes, counts, chart values, spatial relations, or visible text.
- reasoning hallucination: the response uses incorrect calculation, comparison, option mapping, or multi-step reasoning, even when some visual information is read correctly.
- none: the response is supported by the image/question/reference answer and does not contain hallucinated content.
- uncertain: the visual evidence is too unclear to make a reliable judgment.

Important rules:
- Do not mark hallucination only because the response is verbose.
- If the final answer is correct but the explanation contains unsupported visual evidence, mark hallucination.
- If the reference answer seems inconsistent with clearly visible evidence, judge based on the image and explain briefly.
- Return only valid JSON.
{examples}

Dataset: {record.get("dataset")}
Question: {record.get("question")}{choices_block}
Reference answer: {record.get("gold_answer")}
Model prediction: {record.get("model_prediction")}
Model response: {record.get("raw_output")}

JSON schema:
{{
  "is_hallucination": "yes" | "no" | "uncertain",
  "hallucination_type": "factual inconsistency" | "vision-grounding error" | "reasoning hallucination" | "none" | "uncertain",
  "confidence": 1 | 2 | 3,
  "reason": "one short sentence"
}}"""


def parse_json_response(text):
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)
    data = json.loads(raw)
    is_h = str(data.get("is_hallucination", "")).strip().lower()
    h_type = str(data.get("hallucination_type", "")).strip().lower()
    if is_h not in {"yes", "no", "uncertain"}:
        raise ValueError(f"bad is_hallucination: {is_h}")
    if h_type not in ALLOWED_TYPES:
        raise ValueError(f"bad hallucination_type: {h_type}")
    try:
        confidence = int(data.get("confidence"))
    except Exception as exc:
        raise ValueError("bad confidence") from exc
    if confidence not in {1, 2, 3}:
        raise ValueError(f"bad confidence: {confidence}")
    return {
        "judge_is_hallucination": is_h,
        "judge_hallucination_type": h_type,
        "judge_confidence": confidence,
        "judge_reason": str(data.get("reason", "")).strip(),
    }


def call_judge(client, args, record):
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": build_prompt(record, args.prompt_mode)},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_to_data_url(record["image_path"], args.max_side)},
                            },
                        ],
                    }
                ],
                temperature=0,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            content = response.choices[0].message.content or ""
            parsed = parse_json_response(content)
            return content, parsed, None
        except Exception as exc:
            last_error = repr(exc)
            if attempt >= args.retries:
                return None, None, last_error
            time.sleep(min(20, 2 ** attempt) + random.random())
    return None, None, last_error


def run_one(client, args, record):
    started = time.time()
    content, parsed, error = call_judge(client, args, record)
    result = {
        "annotation_id": record.get("annotation_id"),
        "dataset": record.get("dataset"),
        "model_key": record.get("model_key"),
        "prompt_type": record.get("prompt_type"),
        "sample_index": record.get("sample_index"),
        "human_is_hallucination": record.get("human_is_hallucination"),
        "human_hallucination_type": record.get("human_hallucination_type"),
        "human_confidence_1_to_3": record.get("human_confidence_1_to_3"),
        "judge_model": args.model_name,
        "judge_raw_output": content,
        "error": error,
        "elapsed_sec": round(time.time() - started, 3),
    }
    if parsed:
        result.update(parsed)
    return result


def normalize_binary(value):
    value = str(value).strip().lower()
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


def compute_metrics(rows, dataset=None):
    usable = []
    for row in rows:
        if row.get("error"):
            continue
        if dataset and row.get("dataset") != dataset:
            continue
        truth = normalize_binary(row.get("human_is_hallucination"))
        pred = normalize_binary(row.get("judge_is_hallucination"))
        if truth is None or pred is None:
            continue
        usable.append((row, truth, pred))
    tp = sum(1 for _, truth, pred in usable if truth and pred)
    tn = sum(1 for _, truth, pred in usable if not truth and not pred)
    fp = sum(1 for _, truth, pred in usable if not truth and pred)
    fn = sum(1 for _, truth, pred in usable if truth and not pred)
    total = len(usable)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    true_yes = (tp + fn) / total if total else 0.0
    true_no = (tn + fp) / total if total else 0.0
    pred_yes = (tp + fp) / total if total else 0.0
    pred_no = (tn + fn) / total if total else 0.0
    expected = true_yes * pred_yes + true_no * pred_no
    kappa = (accuracy - expected) / (1 - expected) if abs(1 - expected) > 1e-12 else 0.0

    type_rows = [
        row for row, truth, pred in usable
        if truth and pred and row.get("human_hallucination_type") in ALLOWED_TYPES
    ]
    type_acc = 0.0
    if type_rows:
        type_acc = sum(
            row.get("human_hallucination_type") == row.get("judge_hallucination_type")
            for row in type_rows
        ) / len(type_rows)
    return {
        "dataset": dataset or "all",
        "total_compared": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cohen_kappa": kappa,
        "human_hallucination_rate": true_yes,
        "judge_hallucination_rate": pred_yes,
        "type_accuracy_on_both_positive": type_acc,
        "type_compared": len(type_rows),
    }


def write_summary(rows, output_dir):
    summary = [compute_metrics(rows), compute_metrics(rows, "pope"), compute_metrics(rows, "mathvista")]
    path = Path(output_dir) / "zero_shot_judge_metrics.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    return summary, path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="/data2/huanjue/annotations/human_pope_mathvista_60_seed20260904/human_labels_clean.jsonl")
    parser.add_argument("--output-dir", default="/data2/huanjue/results/judge_zero_shot_gpt_5_4")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://www.rightapi.ai/codex/v1"))
    parser.add_argument("--model-name", default="gpt-5.4")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=260)
    parser.add_argument("--max-side", type=int, default=900)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "zero_shot_judge_outputs.jsonl"

    records = load_records(args.input_jsonl)
    if args.limit:
        records = records[: args.limit]
    done = existing_ids(out_path) if args.resume else set()
    jobs = [record for record in records if record.get("annotation_id") not in done]
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    lock = Lock()
    print(f"[start] records={len(records)} todo={len(jobs)} output={out_path}", flush=True)

    if args.workers <= 1:
        for record in jobs:
            result = run_one(client, args, record)
            append_jsonl(out_path, result, lock)
            print(f"[{'ERR' if result['error'] else 'OK'}] {result['annotation_id']} {result['elapsed_sec']}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, client, args, record): record for record in jobs}
            for future in as_completed(futures):
                result = future.result()
                append_jsonl(out_path, result, lock)
                print(f"[{'ERR' if result['error'] else 'OK'}] {result['annotation_id']} {result['elapsed_sec']}s", flush=True)

    all_outputs = load_records(out_path)
    if args.limit:
        wanted = {record["annotation_id"] for record in records}
        all_outputs = [row for row in all_outputs if row.get("annotation_id") in wanted]
    summary, summary_path = write_summary(all_outputs, output_dir)
    print(f"[write] {summary_path}", flush=True)
    for row in summary:
        print(row, flush=True)


if __name__ == "__main__":
    main()
