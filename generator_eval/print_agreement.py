import json

with open("./qwen3_4b_think_responses/o3_detailed_few_shot_results.json", "r") as f:
    data1 = json.load(f)

with open("./qwen3_4b_think_responses/qwen25_14b_concise_zero_shot_results.json", "r") as f:
    data2 = json.load(f)

total =0
agree =0
benchmark_total = {}
benchmark_agree = {}

for item1, item2 in zip(data1, data2):
    benchmark = item1["idx"].split("/")[0]
    if benchmark not in benchmark_total.keys():
        benchmark_total[benchmark] = 0
    if benchmark not in benchmark_agree.keys():
        benchmark_agree[benchmark] = 0
        if item1["is_it_correct"] == item2["is_it_correct"]:
            benchmark_agree[benchmark] += 1
        benchmark_total[benchmark] += 1

for benchmark in benchmark_total.keys():
    agree = benchmark_agree[benchmark]
    total = benchmark_total[benchmark]
    print(f"{benchmark} Agreement: {agree}/{total} ({agree/total:.4f})")