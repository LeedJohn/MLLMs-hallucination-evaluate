import json
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    "pope": {
        "path": "/data2/huanjue/datasets/pope",
        "split": "test",
        "output": "/data2/huanjue/samples/full_pope_9000.json",
    },
    "mathvista": {
        "path": "/data2/huanjue/datasets/mathvista",
        "split": "testmini",
        "output": "/data2/huanjue/samples/full_mathvista_testmini_1000.json",
    },
}


def sample_id(dataset_name, row):
    if dataset_name == "pope":
        return str(row.get("id", row.get("question_id")))
    return str(row.get("pid"))


def main():
    for name, cfg in DATASETS.items():
        ds = load_dataset(cfg["path"], split=cfg["split"])
        samples = [
            {"index": idx, "sample_id": sample_id(name, ds[idx])}
            for idx in range(len(ds))
        ]
        payload = {
            "dataset": name,
            "split": cfg["split"],
            "total": len(samples),
            "samples": samples,
        }
        out_path = Path(cfg["output"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[write] {out_path} samples={len(samples)}")


if __name__ == "__main__":
    main()
