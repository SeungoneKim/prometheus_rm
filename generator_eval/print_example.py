import json

with open("./qwen3_4b_think_responses/o3_detailed_few_shot_results.json", "r") as f:
    data1 = json.load(f)

with open("./qwen3_4b_think_responses/o3_concise_zero_shot_results.json", "r") as f:
    data2 = json.load(f)


judgments1 = {}
judgments2 = {}
for d in data1:
    judgments1[d["idx"]] = d["is_it_correct"]

for d in data2:
    judgments2[d["idx"]] = d["is_it_correct"]

for item in data1:
    if item["is_it_correct"] != judgments2.get(item["idx"], False):
        print("Question:")
        print(item["question"])
        print()
        print()
        print("Reference Answer:")
        print(item["answer"])
        print()
        print()
        print("Candidate:")
        print(item["extracted_answer"])
        print()
        print()
        if judgments2[item["idx"]]:
            print2 = "Equivalent"
        else:
            print2= "Not Equivalent"
        if item["is_it_correct"]:
            print1 = "Equivalent"
        else:
            print1 = "Not Equivalent"
        print("Detailed Few-shot:", print1)
        print("Concise Zero-shot:", print2)

        input("Press Enter to continue...")