# Multimodal Hallucination Evaluation

This repository contains the code, selected experimental outputs, fixed sample lists, and human annotations for a study of hallucination in multimodal large language models (MLLMs).

The study evaluates three hallucination types: factual inconsistency, vision-grounding error, and reasoning hallucination. It compares Direct prompting and chain-of-thought (CoT) prompting on POPE, MathVista testmini, and a selected XLRS-Bench-lite subset.

## Models

- Qwen2.5-VL-7B
- Qwen3-VL-8B
- LLaVA-OneVision-7B
- Gemini-3.1-Flash-Lite

## Repository Layout

```text
annotations/  Human annotation files and agreement results.
code/         Inference, sampling, rule-based evaluation, and GPT-judge scripts.
datasets/     Empty placeholder for locally downloaded datasets.
models/       Empty placeholder for locally stored model weights.
results/      Selected per-sample outputs and evaluation results.
samples/      Fixed sample IDs used in the experiments.
```

Raw datasets, model weights, API keys, caches, and complete figure assets are not included in this repository. Dataset and model paths should be configured locally before running the scripts.

## Evaluation

The project uses rule-based answer evaluation, zero-shot GPT judging, and few-shot GPT judging. A human-annotated set is used to measure agreement between automatic detection methods and human judgments.
