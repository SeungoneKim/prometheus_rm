from datasets import load_dataset
import json
import random
import os
import re
from tqdm import tqdm

data = []
### Benchmark 5: u-Math
init_data = load_dataset("toloka/mu-math", split="test")
idx=0
for item in init_data:
    data.append({
        "idx": f"mu-Math/instance_{idx}",
        "question": item["problem_statement"],
        "answer": item["golden_answer"].strip(),
        "answer_type": "Diverse",
        "human_judgment": item["label"],
        "response": item["model_output"],
        "extracted_answer": ""
    })
    idx += 1
        

print(f"Total items processed: {len(data)}")

with open("mu_math.json", "w") as f:
    json.dump(data, f, indent=4)