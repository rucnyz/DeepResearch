from datasets import load_dataset
import json

#load dataset
ds = load_dataset("gaia-benchmark/GAIA", "2023_all") 

# Convert to Jsonl
with open("gaia.jsonl", "w", encoding="utf-8") as f:
    for item in ds["validation"]:
        # 只提取必要字段
        out = {
            "question": item["Question"],  # GAIA 数据集的字段名称
            "answer": item["Final answer"],  # GAIA 数据集的字段名称
        }

        f.write(json.dumps(out, ensure_ascii=False) + "\n")
