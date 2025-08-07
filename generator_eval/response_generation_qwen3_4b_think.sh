#!/bin/bash

total_items=17572
num_chunks=10
chunk_size=$((total_items / num_chunks))

# Specify the indices you want to run
selected_indices=(0 1 2 3 4 5 6 7 8 9)  # Change this array as needed

for i in "${selected_indices[@]}"; do
    start_index=$((i * chunk_size))
    if [ $i -eq $((num_chunks - 1)) ]; then
        end_index=$total_items
    else
        end_index=$(((i + 1) * chunk_size))
    fi

    sbatch << EOF
#!/bin/bash

#SBATCH --job-name=[self_motivated_lms]_qwen3_4b_think_response_${i}
#SBATCH --output=qwen3_4b_think_response_${i}.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=1024G
#SBATCH --account=ram
#SBATCH --qos=alignment_shared

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

python3 response_generation_qwen.py \\
    --model_path /datasets/pretrained-llms/Qwen3-4B \\
    --gpu_per_node 2 \\
    --input_file "./benchmarks.json" \\
    --output_file "./qwen3_4b_think_responses/response_${i}.json" \\
    --start_index ${start_index} \\
    --end_index ${end_index} \\
    --temperature 0.6 \\
    --top_p 0.95 \\
    --top_k 20 \\
    --min_p 0 \\
    --max_tokens 32768 \\
    --enable_thinking
EOF

    echo "Submitted job $i with start_index=${start_index} and end_index=${end_index}"
done

