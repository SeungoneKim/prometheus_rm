import json

with open("./qwen3_4b_think_responses/qwen25_14b_detailed_zero_shot_results.json", "r") as f:
    all_items = json.load(f)

benchmark_stats = {}
for item in all_items:
    idx = item.get("idx", "")
    if "/" in idx:
        benchmark = idx.split("/")[0]
    else:
        benchmark = "unknown"
        
    if benchmark not in benchmark_stats:
        benchmark_stats[benchmark] = {"total": 0, "correct": 0}
    
    benchmark_stats[benchmark]["total"] += 1
    if item.get("is_it_correct") == True:
        benchmark_stats[benchmark]["correct"] += 1

print(f"\nBenchmark-specific statistics:")
for benchmark, stats in sorted(benchmark_stats.items()):
    accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    print(f"{benchmark}: {stats['correct']}/{stats['total']} ({accuracy:.4f})")