import json
import os
import re
from typing import Optional

VERIFIER_PASS_TAG = "Final Judgment: Yes"

BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\\[{}]|\{(?:[^{}]|\\[{}]|\{(?:[^{}]|\\[{}]|\{[^{}]*\})*\})*\})*)\}", re.IGNORECASE)

STOP_WORDS = ["</s>", "<|im_end|>", "<|endoftext|>"]

def extract_last_final_answer_optimized(text: str) -> Optional[str]:
    """
    Optimized version: Final answer patterns can be simplified safely.
    """
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<\n]|<[^<])*?)(?:\n|$)",
        r"Final answer:\s*((?:[^<\n]|<[^<])*?)(?:\n|$)",
        r"Final Answer is:\s*((?:[^<\n]|<[^<])*?)(?:\n|$)",
        r"The answer is:\s*((?:[^<\n]|<[^<])*?)(?:\n|$)",
        r"Answer:\s*((?:[^<\n]|<[^<])*?)(?:\n|$)",
        r"Solution:\s*((?:[^<\n]|<[^<])*?)(?:\n|$)",
        r"The solution is:\s*((?:[^<\n]|<[^<])*?)(?:\n|$)",
        r"### Final Answer:\s*((?:[^<\n]|<[^<])*?)(?:\n|$)",
    ]
    
    last_match = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[:-len(stop_word)].strip()
    
    return last_match

def extract_last_boxed_accurate(text: str) -> Optional[str]:
    """
    Keep original accuracy but use pre-compiled regex for speed.
    """
    matches = list(BOXED_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1)
    return None

def extract_solution_fast_accurate(solution_str: str) -> Optional[str]:
    """
    Fast but maintains original accuracy.
    """
    if not solution_str:
        return None
        
    # Try boxed answer first (keeps original accuracy)
    boxed_answer = extract_last_boxed_accurate(solution_str)
    if boxed_answer:
        return boxed_answer
    
    # Fall back to final answer patterns (safe to optimize)
    return extract_last_final_answer_optimized(solution_str)



def merge_json_files(directory, prefix, output_file_name):
    # Initialize an empty list to store all the data
    merged_data = []
    
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a JSON file and starts with the prefix
        if filename.startswith(prefix) and filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    # Ensure the data is a list of dictionaries
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        print(f"Warning: {filename} does not contain a list of dictionaries. Skipping.")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    final_data = []
    count_successful =0
    count_boxed = 0
    count_failed = 0
    for item in merged_data:
        if item["extracted_answer"] == "[FAILED_TO_PROCESS]":
            retried_extracted_answer = extract_solution_fast_accurate(item["response"])
            if retried_extracted_answer is None:
                count_failed += 1
            else:
                item["extracted_answer"] = retried_extracted_answer
                count_boxed += 1
        else:
            count_successful += 1
        final_data.append(item)

    
    # Write the merged data to the output file
    with open(output_file_name, 'w') as outfile:
        json.dump(final_data, outfile, indent=4)
    
    # Print the number of instances in the output file
    print(f"Successfully merged {len(merged_data)} instances into {output_file_name}")
    print(f"Successful extractions: {count_successful}")
    print(f"Boxed extractions: {count_boxed}")
    print(f"Failed extractions: {count_failed}")

if __name__ == "__main__":
    # Example usage
    directory = "./qwen3_14b_think_responses"
    prefix = "response_"
    output_file_name = "./qwen3_14b_think_responses/responses.json"

    merge_json_files(directory, prefix, output_file_name)
