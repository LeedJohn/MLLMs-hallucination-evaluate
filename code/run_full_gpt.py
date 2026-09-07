import argparse
import base64
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from datasets import load_dataset, load_from_disk
from openai import OpenAI
from PIL import Image


DATASET_PATHS = {
    "pope": "/data2/huanjue/datasets/pope",
    "mathvista": "/data2/huanjue/datasets/mathvista",
    "xlrs": "/data2/huanjue/datasets/xlrs-lite",
}

DATASET_SPLITS = {
    "pope": "test",
    "mathvista": "testmini",
    "xlrs": "train",
}

DATASET_OUTPUT_NAMES = {
    "pope": "pope",
    "mathvista": "mathvista_testmini",
    "xlrs": "xlrs_selected500",
}

SAMPLE_FILES = {
    "pope": "/data2/huanjue/samples/full_pope_9000.json",
    "mathvista": "/data2/huanjue/samples/full_mathvista_testmini_1000.json",
    "xlrs": "/data2/huanjue/samples/xlrs_selected500_seed20260903.json",
}

DEFAULT_PROMPTS = {
    "pope": ["direct", "cot_light"],
    "mathvista": ["direct", "cot_light_final_first"],
    "xlrs": ["direct", "cot_light_final_first"],
}

def to_rgb_image(image, max_side):
    if isinstance(image, list):
        image = image[0]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image object: {type(image)}")
    image = image.convert("RGB")
    original_size = image.size
    if max_side and max(image.size) > max_side:
        resized = image.copy()
        resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return resized, original_size, resized.size
    return image, original_size, image.size


def image_to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def load_sample_spec(dataset_name):
    with Path(SAMPLE_FILES[dataset_name]).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_image(dataset_name, row):
    if dataset_name == "mathvista":
        return row.get("decoded_image") or row["image"]
    return row["image"]


def build_prompt(dataset_name, row, prompt_type):
    if dataset_name == "pope":
        question = row["question"]
        if prompt_type == "direct":
            prompt = (
                "Answer based only on the image.\n"
                "Output exactly one word: yes or no.\n\n"
                f"Question: {question}"
            )
        elif prompt_type == "cot_light":
            prompt = (
                "Answer based only on the image.\n"
                "Give one brief reason. Then end with exactly: Final answer: yes/no.\n\n"
                f"Question: {question}"
            )
        else:
            raise ValueError(f"Unsupported POPE prompt_type: {prompt_type}")
        return prompt, question, row["answer"], str(row.get("id", row.get("question_id")))

    if dataset_name == "mathvista":
        question = row.get("query") or row["question"]
        if prompt_type == "direct":
            prompt = (
                "Answer based on the image.\n"
                "Output only the final answer. Do not explain.\n\n"
                f"Question: {question}"
            )
        elif prompt_type == "cot_light_final_first":
            prompt = (
                "Answer based on the image.\n"
                "Use exactly two lines.\n"
                "Line 1: Final answer: ...\n"
                "Line 2: Reason: one short sentence.\n\n"
                f"Question: {question}"
            )
        else:
            raise ValueError(f"Unsupported MathVista prompt_type: {prompt_type}")
        return prompt, question, row.get("answer"), str(row.get("pid"))

    if dataset_name == "xlrs":
        question = row["question"]
        options = row["multi-choice options"]
        if isinstance(options, list):
            options_text = "\n".join(str(option) for option in options)
        else:
            options_text = str(options)
        if prompt_type == "direct":
            prompt = (
                "Answer based on the satellite or aerial image.\n"
                "Choose the correct option. Output only one option letter: A, B, C, or D.\n\n"
                f"Question: {question}\n"
                f"Options:\n{options_text}"
            )
        elif prompt_type == "cot_light_final_first":
            prompt = (
                "Answer based on the satellite or aerial image.\n"
                "Use exactly two lines.\n"
                "Line 1: Final answer: one option letter only, A or B or C or D.\n"
                "Line 2: Reason: one short sentence.\n\n"
                f"Question: {question}\n"
                f"Options:\n{options_text}"
            )
        else:
            raise ValueError(f"Unsupported XLRS prompt_type: {prompt_type}")
        return prompt, question, row.get("answer"), str(row.get("index"))

    raise ValueError(f"Unknown dataset: {dataset_name}")


def output_path(output_dir, model_key, dataset_name, prompt_type):
    return Path(output_dir) / model_key / DATASET_OUTPUT_NAMES[dataset_name] / f"{prompt_type}.jsonl"


def existing_success_indices(path):
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error"):
                continue
            done.add(int(row.get("sample_index")))
    return done


def append_jsonl(path, record, lock):
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def build_record(dataset_name, prompt_type, sample_index, sample_meta, row, max_tokens, max_side, model_key, model_name):
    return {
        "prompt_type": prompt_type,
        "sample_index": sample_index,
        "dataset_index": sample_meta["index"],
        "sample_id": sample_meta["sample_id"],
        "model_key": model_key,
        "model_name": model_name,
        "dataset": dataset_name,
        "dataset_split": DATASET_SPLITS[dataset_name],
        "question": None,
        "gold_answer": None,
        "raw_output": None,
        "elapsed_sec": None,
        "image_original_size": None,
        "image_input_size": None,
        "question_type": row.get("question_type") if dataset_name == "mathvista" else None,
        "answer_type": row.get("answer_type") if dataset_name == "mathvista" else ("multi_choice" if dataset_name == "xlrs" else None),
        "category": row.get("category") if dataset_name in {"pope", "xlrs"} else None,
        "options": row.get("multi-choice options") if dataset_name == "xlrs" else None,
        "max_tokens": max_tokens,
        "max_side": max_side,
        "base_url_label": "proaiapi",
        "error": None,
    }


def call_gpt(client, model_name, prompt, data_url, max_tokens, timeout, retries, reasoning_effort):
    last_error = None
    for attempt in range(retries + 1):
        try:
            extra_body = {}
            if reasoning_effort:
                extra_body["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=0,
                timeout=timeout,
                extra_body=extra_body or None,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = repr(exc)
            if attempt >= retries:
                raise
            sleep = min(30, 2 ** attempt) + random.random()
            time.sleep(sleep)
    raise RuntimeError(last_error)


def run_one(client, dataset_name, prompt_type, sample_index, sample_meta, row, args):
    started = time.time()
    record = build_record(
        dataset_name,
        prompt_type,
        sample_index,
        sample_meta,
        row,
        args.max_tokens,
        args.max_side,
        args.model_key,
        args.model_name,
    )
    try:
        prompt, question, gold_answer, _ = build_prompt(dataset_name, row, prompt_type)
        image, original_size, input_size = to_rgb_image(get_image(dataset_name, row), args.max_side)
        record["question"] = question
        record["gold_answer"] = gold_answer
        record["image_original_size"] = list(original_size)
        record["image_input_size"] = list(input_size)
        record["raw_output"] = call_gpt(
            client,
            args.model_name,
            prompt,
            image_to_data_url(image),
            args.max_tokens,
            args.timeout,
            args.retries,
            args.reasoning_effort,
        )
    except Exception as exc:
        record["error"] = repr(exc)
    finally:
        record["elapsed_sec"] = round(time.time() - started, 3)
    return record


def run_one_from_sample(client, ds, dataset_name, prompt_type, sample_index, sample_meta, args):
    row = ds[int(sample_meta["index"])]
    return run_one(client, dataset_name, prompt_type, sample_index, sample_meta, row, args)


def parse_csv(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def load_dataset_split(dataset_name):
    if dataset_name == "xlrs":
        obj = load_from_disk(DATASET_PATHS[dataset_name])
        return obj[DATASET_SPLITS[dataset_name]] if hasattr(obj, "keys") else obj
    return load_dataset(DATASET_PATHS[dataset_name], split=DATASET_SPLITS[dataset_name])


def run(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    for dataset_name in parse_csv(args.datasets):
        ds = load_dataset_split(dataset_name)
        spec = load_sample_spec(dataset_name)
        samples = spec["samples"][: args.limit_samples] if args.limit_samples else spec["samples"]
        prompt_types = parse_csv(args.prompt_types) if args.prompt_types else DEFAULT_PROMPTS[dataset_name]
        print(f"[dataset] {dataset_name} samples={len(samples)} prompts={prompt_types}", flush=True)

        for prompt_type in prompt_types:
            out_path = output_path(args.output_dir, args.model_key, dataset_name, prompt_type)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            done = existing_success_indices(out_path) if args.resume else set()
            jobs = [
                (sample_index, sample_meta)
                for sample_index, sample_meta in enumerate(samples)
                if sample_index not in done
            ]
            lock = Lock()
            print(f"[output] {out_path} done={len(done)} todo={len(jobs)}", flush=True)

            if args.workers <= 1:
                for sample_index, sample_meta in jobs:
                    row = ds[int(sample_meta["index"])]
                    record = run_one(client, dataset_name, prompt_type, sample_index, sample_meta, row, args)
                    append_jsonl(out_path, record, lock)
                    status = "ERR" if record["error"] else "OK"
                    print(f"[{status}] {dataset_name} {prompt_type} #{sample_index} {record['elapsed_sec']}s", flush=True)
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {}
                    for sample_index, sample_meta in jobs:
                        future = executor.submit(
                            run_one_from_sample,
                            client,
                            ds,
                            dataset_name,
                            prompt_type,
                            sample_index,
                            sample_meta,
                            args,
                        )
                        futures[future] = sample_index
                    for future in as_completed(futures):
                        record = future.result()
                        append_jsonl(out_path, record, lock)
                        status = "ERR" if record["error"] else "OK"
                        print(
                            f"[{status}] {dataset_name} {prompt_type} "
                            f"#{record['sample_index']} {record['elapsed_sec']}s",
                            flush=True,
                        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt-4o-mini")
    parser.add_argument("--model-key", default="gpt_4o_mini")
    parser.add_argument("--datasets", default="pope,mathvista")
    parser.add_argument("--prompt-types", default=None)
    parser.add_argument("--output-dir", default="/data2/huanjue/results/full")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://proaiapi.tech/v1"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--reasoning-effort", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
