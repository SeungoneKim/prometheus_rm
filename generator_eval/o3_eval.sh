#!/bin/bash
#SBATCH --job-name=[self_motivated_lms]_o3_eval_qwen3_235b_think_detailed_zero_shot
#SBATCH --output=o3_eval_qwen3_235b_think_detailed_zero_shot.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=1024G
#SBATCH --account=ram
#SBATCH --qos=alignment_shared

python3 o3_eval.py \
    --input_file "./qwen3_235b_think_responses/responses.json" \
    --output_file "./qwen3_235b_think_responses/detailed_zero_shot_results.json" \
    --max_tokens 16384