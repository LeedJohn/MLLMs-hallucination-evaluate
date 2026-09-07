import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from datasets import load_from_disk
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

try:
    from qwen_vl_utils import process_vision_info
except Exception as exc:  # pragma: no cover
    process_vision_info = None
    QWEN_VL_IMPORT_ERROR = exc
else:
    QWEN_VL_IMPORT_ERROR = None


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

MODEL_PATHS = {
    "qwen2_5_vl": "/data2/huanjue/models/Qwen2.5-VL-7B-Instruct",
    "qwen3_vl": "/data2/huanjue/models/Qwen3-VL-8B-Instruct",
    "llava_onevision": "/data2/huanjue/models/LLaVA-OneVision-Qwen2-7B-OV-HF",
}

MODEL_NAMES = {
    "qwen2_5_vl": "Qwen2.5-VL-7B-Instruct",
    "qwen3_vl": "Qwen3-VL-8B-Instruct",
    "llava_onevision": "LLaVA-OneVision-Qwen2-7B-OV-HF",
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


def load_sample_spec(dataset_name):
    path = Path(SAMPLE_FILES[dataset_name])
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_split(dataset_name):
    if dataset_name == "xlrs":
        obj = load_from_disk(DATASET_PATHS[dataset_name])
        return obj["train"] if hasattr(obj, "keys") else obj
    return load_dataset(DATASET_PATHS[dataset_name], split=DATASET_SPLITS[dataset_name])


def get_image(dataset_name, row):
    if dataset_name == "mathvista":
        return row.get("decoded_image") or row["image"]
    return row["image"]


def format_options(options):
    if isinstance(options, list):
        return "\n".join(str(option) for option in options)
    return str(options)


def build_prompt(dataset_name, row, prompt_type):
    if dataset_name == "pope":
        question = row["question"]
        if prompt_type == "direct":
            prompt = f"{question}\nAnswer yes or no."
        elif prompt_type == "cot_light":
            prompt = (
                f"{question}\n"
                "Give a brief reason based on the image. "
                "Then give the final answer as yes or no."
            )
        else:
            raise ValueError(f"Unsupported POPE prompt_type: {prompt_type}")
        return prompt, question, row["answer"], str(row.get("id", row.get("question_id")))

    if dataset_name == "mathvista":
        question = row.get("query") or row["question"]
        if prompt_type == "direct":
            prompt = f"{question}\nGive the final answer only."
        elif prompt_type == "cot_light_final_first":
            prompt = (
                f"{question}\n"
                "First give the final answer in exactly this format: Final answer: ...\n"
                "Then give one short reason based on the image."
            )
        else:
            raise ValueError(f"Unsupported MathVista prompt_type: {prompt_type}")
        return prompt, question, row.get("answer"), str(row.get("pid"))

    if dataset_name == "xlrs":
        question = row["question"]
        options = format_options(row["multi-choice options"])
        if prompt_type == "direct":
            prompt = (
                "Answer based on the satellite or aerial image.\n"
                "Choose the correct option. Output only the option letter, such as A, B, C, or D.\n\n"
                f"Question: {question}\n"
                f"Options:\n{options}"
            )
        elif prompt_type == "cot_light_final_first":
            prompt = (
                "Answer based on the satellite or aerial image.\n"
                "Use exactly two lines.\n"
                "Line 1: Final answer: A/B/C/D\n"
                "Line 2: Reason: one short sentence based on the image.\n\n"
                f"Question: {question}\n"
                f"Options:\n{options}"
            )
        else:
            raise ValueError(f"Unsupported XLRS prompt_type: {prompt_type}")
        return prompt, question, row.get("answer"), str(row.get("index"))

    raise ValueError(f"Unknown dataset: {dataset_name}")


class ModelRunner:
    def __init__(self, model_key, max_pixels, max_memory_per_gpu):
        self.model_key = model_key
        self.model_path = MODEL_PATHS[model_key]
        self.max_pixels = max_pixels
        self.max_memory_per_gpu = max_memory_per_gpu
        self.processor = None
        self.model = None

    def load(self):
        common_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
        }
        model_kwargs = {}
        if self.max_memory_per_gpu and torch.cuda.is_available():
            model_kwargs["max_memory"] = {
                i: self.max_memory_per_gpu for i in range(torch.cuda.device_count())
            }
            model_kwargs["max_memory"]["cpu"] = "96GiB"

        if self.model_key in {"qwen2_5_vl", "qwen3_vl"}:
            if process_vision_info is None:
                raise RuntimeError(f"qwen_vl_utils import failed: {QWEN_VL_IMPORT_ERROR}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=256 * 28 * 28,
                max_pixels=self.max_pixels,
                **common_kwargs,
            )
            model_cls = (
                Qwen3VLForConditionalGeneration
                if self.model_key == "qwen3_vl"
                else Qwen2_5_VLForConditionalGeneration
            )
            self.model = model_cls.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                **model_kwargs,
                **common_kwargs,
            )
            return

        if self.model_key == "llava_onevision":
            self.processor = AutoProcessor.from_pretrained(self.model_path, **common_kwargs)
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                **model_kwargs,
                **common_kwargs,
            )
            return

        raise ValueError(f"Unknown model: {self.model_key}")

    def generate(self, image, prompt, max_new_tokens):
        if self.model_key in {"qwen2_5_vl", "qwen3_vl"}:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                images=image,
                text=text,
                return_tensors="pt",
            ).to(self.model.device, torch.bfloat16)

        prompt_token_count = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated_ids = output[0][prompt_token_count:]
        return self.processor.decode(generated_ids, skip_special_tokens=True).strip()


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


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def build_record(dataset_name, prompt_type, sample_index, sample_meta, row, model_key):
    return {
        "prompt_type": prompt_type,
        "sample_index": sample_index,
        "dataset_index": sample_meta["index"],
        "sample_id": sample_meta["sample_id"],
        "model_key": model_key,
        "model_name": MODEL_NAMES[model_key],
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
        "max_new_tokens": None,
        "max_side": None,
        "max_pixels": None,
        "error": None,
    }


def output_path(output_dir, model_key, dataset_name, prompt_type):
    dataset_dir = DATASET_OUTPUT_NAMES[dataset_name]
    return Path(output_dir) / model_key / dataset_dir / f"{prompt_type}.jsonl"


def parse_csv(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def run(args):
    model_keys = parse_csv(args.models)
    dataset_names = parse_csv(args.datasets)
    for model_key in model_keys:
        if model_key not in MODEL_PATHS:
            raise ValueError(f"Unknown model: {model_key}")
    for dataset_name in dataset_names:
        if dataset_name not in DATASET_PATHS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_key in model_keys:
        runner = ModelRunner(model_key, args.max_pixels, args.max_memory_per_gpu)
        print(f"[load] {MODEL_NAMES[model_key]} from {runner.model_path}", flush=True)
        runner.load()
        print(f"[ready] {MODEL_NAMES[model_key]}", flush=True)

        for dataset_name in dataset_names:
            spec = load_sample_spec(dataset_name)
            ds = load_dataset_split(dataset_name)
            samples = spec["samples"][: args.limit_samples] if args.limit_samples else spec["samples"]
            prompt_types = parse_csv(args.prompt_types) if args.prompt_types else DEFAULT_PROMPTS[dataset_name]
            print(
                f"[dataset] model={model_key} dataset={dataset_name} samples={len(samples)} prompts={prompt_types}",
                flush=True,
            )

            for prompt_type in prompt_types:
                out_path = output_path(output_dir, model_key, dataset_name, prompt_type)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                done = existing_success_indices(out_path) if args.resume else set()
                print(f"[output] {out_path} done={len(done)}", flush=True)

                for sample_index, sample_meta in enumerate(samples):
                    if sample_index in done:
                        continue
                    row = ds[int(sample_meta["index"])]
                    started = time.time()
                    record = build_record(dataset_name, prompt_type, sample_index, sample_meta, row, model_key)
                    record["max_new_tokens"] = args.max_new_tokens
                    record["max_side"] = args.max_side
                    record["max_pixels"] = args.max_pixels
                    try:
                        prompt, question, gold_answer, _ = build_prompt(dataset_name, row, prompt_type)
                        image, original_size, input_size = to_rgb_image(get_image(dataset_name, row), args.max_side)
                        record["question"] = question
                        record["gold_answer"] = gold_answer
                        record["image_original_size"] = list(original_size)
                        record["image_input_size"] = list(input_size)
                        record["raw_output"] = runner.generate(image, prompt, args.max_new_tokens)
                    except Exception as exc:
                        record["error"] = repr(exc)
                    finally:
                        record["elapsed_sec"] = round(time.time() - started, 3)
                        append_jsonl(out_path, record)
                        status = "ERR" if record["error"] else "OK"
                        print(
                            f"[{status}] {model_key} {dataset_name} {prompt_type} "
                            f"#{sample_index} {record['elapsed_sec']}s",
                            flush=True,
                        )

        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen2_5_vl,qwen3_vl,llava_onevision")
    parser.add_argument("--datasets", default="pope,mathvista")
    parser.add_argument("--prompt-types", default=None)
    parser.add_argument("--output-dir", default="/data2/huanjue/results/full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--max-memory-per-gpu", default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
