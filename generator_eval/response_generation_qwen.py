import json
import os
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
import math
import time

system_message = """The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively (i.e., <think> reasoning process here </think> <answer> answer here </answer>).
Between <answer> and </answer>, you should be concise and only provide the final prediction without any additional explanations (e.g., <answer> 0.5206 </answer>).
If a question consists of multiple sub-problems and explicitly asks for more than one answer, write all answers inside <answer> and </answer> tags (e.g., <answer> *answer 1: $x$ *answer 2: $$L = \\frac{1}{2} m \\dot{x}^2 \\left(1 + \\frac{4x^2}{a^2}\\right) - \\frac{mgx^2}{a}$$ </answer>)."""

def extract_answer_content(text):
    """
    Extract content between the last <answer> and </answer> tags with validation.
    Returns None if validation fails or no valid answer found.
    If multiple <answer> tags exist, uses the last properly closed one.
    """
    if not text:
        return None

    # Count <answer> and </answer> tags
    answer_open_count = text.count('<answer>')
    answer_close_count = text.count('</answer>')
    
    # Must have at least one opening and closing tag
    if answer_open_count == 0 or answer_close_count == 0:
        return None
    
    # Find the last <answer> tag
    last_answer_start = text.rfind('<answer>')
    if last_answer_start == -1:
        return None
    
    # Find the first </answer> tag that comes after the last <answer> tag
    search_start = last_answer_start + len('<answer>')
    answer_end = text.find('</answer>', search_start)
    
    # If no closing tag found after the last opening tag, it's invalid
    if answer_end == -1:
        return None
    
    # Extract the content (skip the opening tag)
    content = text[search_start:answer_end].strip()
    return content if content else None

def init_llm(model_path, gpu_per_node):
    return LLM(model=model_path, 
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=32768,
        tensor_parallel_size=gpu_per_node,
        enable_prefix_caching=True, 
        enable_chunked_prefill=True,
        swap_space=16,
        max_num_seqs=1024)


def process_benchmarks(model_path, gpu_per_node, input_file, output_file, 
                      temperature, top_p, top_k, min_p, max_tokens, enable_thinking,
                      start_index=None, end_index=None):
    # Load the input file - check if output file exists for resuming
    if os.path.exists(output_file):
        print(f"Output file {output_file} exists. Loading from it for resuming...")
        with open(output_file, "r") as f:
            completed_items = json.load(f)
        
        # Load original data from input file
        with open(input_file, "r") as f:
            all_items = json.load(f)
        
        # Merge completed items back into all_items by matching questions
        completed_dict = {item["idx"]: item for item in completed_items}
        for idx, item in enumerate(all_items):
            if item["idx"] in completed_dict:
                all_items[idx] = completed_dict[item["idx"]]
    else:
        print(f"Loading from input file {input_file}")
        with open(input_file, "r") as f:
            all_items = json.load(f)
    
    # Apply start_index and end_index slicing
    if start_index is not None or end_index is not None:
        items = all_items[start_index:end_index]
        print(f"Processing slice [{start_index}:{end_index}] = {len(items)} items from {input_file}")
    else:
        items = all_items
        print(f"Processing all {len(items)} items from {input_file}")
    
    # Filter items that don't have response and extracted_answer keys (for resuming)
    items_to_process = []
    for i, item in enumerate(items):
        if "response" not in item or "extracted_answer" not in item:
            items_to_process.append((i, item))
    
    print(f"Found {len(items_to_process)} items that need processing")
    
    if not items_to_process:
        print("All items already processed. Exiting.")
        return
    
    # Initialize LLM and tokenizer
    llm = init_llm(model_path, gpu_per_node)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k, 
        min_p=min_p, 
        max_tokens=max_tokens
    )
    
    # Split into 10 sections
    num_sections = 10
    section_size = math.ceil(len(items_to_process) / num_sections)
    
    for section_idx in range(num_sections):
        start_idx = section_idx * section_size
        end_idx = min((section_idx + 1) * section_size, len(items_to_process))
        
        if start_idx >= len(items_to_process):
            break
            
        section_items = items_to_process[start_idx:end_idx]
        print(f"Processing section {section_idx + 1}/{num_sections} with {len(section_items)} items")
        
        # Prepare prompts for this section using chat template
        prompts = []
        for _, item in section_items:
            messages = [
                {"role": "system","content":system_message},
                {"role": "user", "content": item["question"]}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            prompts.append(text)
        
        try:
            # Run vLLM inference for this section
            print(f"Running vLLM inference for section {section_idx + 1}...")
            batch_outputs = llm.generate(prompts, sampling_params)
            
            # Process results and add to items
            for (original_idx, item), output in zip(section_items, batch_outputs):
                response = output.outputs[0].text.strip()
                extracted_answer = extract_answer_content(response)
                
                # Add response and extracted_answer as strings
                item["response"] = response
                item["extracted_answer"] = extracted_answer if extracted_answer else "[FAILED_TO_PROCESS]"
                
                # Update the original item in the items list
                items[original_idx] = item
            
            print(f"Successfully processed section {section_idx + 1}")
            
        except Exception as e:
            print(f"Error processing section {section_idx + 1}: {str(e)}")
            # For failed items in this section, mark them as failed
            for original_idx, item in section_items:
                if "response" not in item:
                    item["response"] = "[FAILED_TO_PROCESS]"
                if "extracted_answer" not in item:
                    item["extracted_answer"] = "[FAILED_TO_PROCESS]"
                items[original_idx] = item
        
        # Save progress after each section with timing
        save_start_time = time.time()
        print(f"Starting to save progress after section {section_idx + 1}...")
        
        # Save progress after each section
        if start_index is not None or end_index is not None:
            # If we're working with a slice, save only the slice
            items_to_save = items
        else:
            # Save all items (both completed and pending)
            items_to_save = items
        
        # Log file info (avoid expensive size calculation)
        print(f"Preparing to save {len(items_to_save)} items to {output_file}...")
        
        # Write with reduced indentation to save space and time
        with open(output_file, "w") as f:
            json.dump(items_to_save, f, indent=2, ensure_ascii=False)
        
        save_end_time = time.time()
        save_duration = save_end_time - save_start_time
        completed_count = sum(1 for item in items_to_save if "response" in item and "extracted_answer" in item)
        print(f"Saved {len(items_to_save)} total items ({completed_count} completed) to {output_file} (section {section_idx + 1} completed)")
        print(f"Save operation took {save_duration:.2f} seconds")
        
        # Log the number of items processed so far
        processed_count = sum(1 for item in items if "response" in item and "extracted_answer" in item)
        print(f"Total items processed so far: {processed_count}/{len(items)}")
    
    # Final save is redundant now since we save after each section
    print(f"Successfully processed {len(items_to_process)} items and saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses for benchmarks using vLLM.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--gpu_per_node", type=int, default=1, help="Number of GPUs per node.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p for sampling.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k for sampling.")
    parser.add_argument("--min_p", type=float, default=0.0, help="Min-p for sampling.")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum tokens to generate.")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking mode in chat template.")
    parser.add_argument("--start_index", type=int, help="Start index for data slicing.")
    parser.add_argument("--end_index", type=int, help="End index for data slicing.")
    
    args = parser.parse_args()
    
    process_benchmarks(args.model_path, args.gpu_per_node, args.input_file, args.output_file,
                      args.temperature, args.top_p, args.top_k, args.min_p, args.max_tokens, args.enable_thinking,
                      args.start_index, args.end_index)
