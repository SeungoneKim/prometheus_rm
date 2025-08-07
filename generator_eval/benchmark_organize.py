from datasets import load_dataset
import json
import random
import os
import re
from tqdm import tqdm

data = []
instance_num = {
    "Physics": 0,
    "RealMath": 0,
    "TheoremQA": 0,
    "SciBench": 0,
    "u-Math": 0
}

### Benchmark 1: Physics
physics_dir = "./physics"

# Check if the directory exists
if not os.path.exists(physics_dir):
    print(f"Error: Directory '{physics_dir}' does not exist")
    exit(1)

# Get all jsonl files in the directory
jsonl_files = [f for f in os.listdir(physics_dir) if f.endswith('.jsonl')]

if not jsonl_files:
    print(f"No JSONL files found in '{physics_dir}'")
    exit(0)

# Process each jsonl file
for file_name in jsonl_files:
    file_path = os.path.join(physics_dir, file_name)
    print(f"Processing file: {file_path}")
    
    # Open and read the jsonl file
    with open(file_path, 'r') as file:
        # Read line by line since jsonl format has one JSON object per line
        for line_num, line in enumerate(file, 1):
            try:
                # Parse the JSON object
                json_obj = json.loads(line.strip())
                # Here you can process each JSON object
                # For example, print the keys:
                print(f"  Line {line_num}: Contains keys {list(json_obj.keys())}")
            except json.JSONDecodeError:
                print(f"  Error parsing JSON at line {line_num}")

idx =0
for file_name in jsonl_files:
    file_path = os.path.join(physics_dir, file_name)
    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())
                if json_obj["graphs"] == None:
                    if len(json_obj["final_answers"]) >=2:
                        formatted_answers = "\n".join([f"* answer {i+1}: {ans.replace('text{ ','text{')}".strip() for i, ans in enumerate(json_obj["final_answers"])])
                        for trial_idx in range(4):
                            data.append({
                                "idx": f"Physics/instance_{idx}/trial_{trial_idx}",
                                "question": json_obj["questions"],
                                "answer": formatted_answers.strip(),
                                "answer_type": "Diverse",
                                "category": file_name.split("_test")[0]
                            })
                        
                    else:
                        for trial_idx in range(4):
                            data.append({
                                "idx": f"Physics/instance_{idx}/trial_{trial_idx}",
                                "question": json_obj["questions"],
                                "answer": json_obj["final_answers"][0].replace("text{ ","text{").strip(),
                                "answer_type": "Diverse",
                                "category": file_name.split("_test")[0]
                            })
                    idx +=1
                    instance_num["Physics"] += 1
            except json.JSONDecodeError:
                print(f"Error parsing JSON in file {file_name}")

### Benchmark 2: RealMath
for s in ["Math_arXiv","CS_arXiv","Math_StackExchange"]:
    init_data = load_dataset("ethz-spylab/RealMath", split=s)

    for idx,item in enumerate(init_data):    
        answer_ = item["answer"]
        if answer_.startswith("$$") and answer_.endswith("$$"):
            answer_ = answer_[2:-2].strip()
        elif answer_.startswith("$") and answer_.endswith("$"):
            answer_ = answer_[1:-1].strip()

        for trial_idx in range(4):
            data.append({
                "idx": f"RealMath/instance_{idx}/trial_{trial_idx}",
                "question": item["question"],
                "answer": answer_.strip(),
                "answer_type": "Diverse",
                "category": s
            })
        instance_num["RealMath"] += 1

### Benchmark 3: TheoremQA
init_data = load_dataset("TIGER-Lab/TheoremQA", split="test")

idx =0
for item in init_data:
    if item["Picture"] == None:
        for trial_idx in range(4):
            data.append({
                "idx": f"TheoremQA/instance_{idx}/trial_{trial_idx}",
                "question": item["Question"],
                "answer": item["Answer"].strip(),
                "answer_type": "Diverse",
                "category": item["Answer_type"]
            })
        idx +=1
        instance_num["TheoremQA"] += 1

### Benchmark 4: SciBench
init_data = load_dataset("xw27/scibench", split="train")
for idx,item in enumerate(init_data):
    if item["unit"] == "":
        answer_ = item["answer_number"]
    else:
        answer_ = item["answer_number"] + " " + item["unit"]
    
    for trial_idx in range(4):
        data.append({
            "idx": f"SciBench/instance_{idx}/trial_{trial_idx}",
            "question": item["problem_text"],
            "answer": answer_.strip(),
            "answer_type": "Numerical",
            "category": item['source']
        })
    instance_num["SciBench"] += 1


### Benchmark 5: u-Math
init_data = load_dataset("toloka/u-math", split="test")
idx=0
for item in init_data:
    if item["image"] is None:
        for trial_idx in range(4):
            data.append({
                "idx": f"u-Math/instance_{idx}/trial_{trial_idx}",
                "question": item["problem_statement"],
                "answer": item["golden_answer"].split("The final answer: ")[-1].strip(),
                "answer_type": "Diverse",
                "category": item["subject"]
            })
        idx += 1
        instance_num["u-Math"] += 1


print(instance_num)
print(f"Total items processed: {len(data)}")

with open("benchmarks.json", "w") as f:
    json.dump(data, f, indent=4)