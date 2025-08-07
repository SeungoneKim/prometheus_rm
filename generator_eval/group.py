import json
import os


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
    
    # Write the merged data to the output file
    with open(output_file_name, 'w') as outfile:
        json.dump(merged_data, outfile, indent=4)
    
    # Print the number of instances in the output file
    print(f"Successfully merged {len(merged_data)} instances into {output_file_name}")

if __name__ == "__main__":
    # Example usage
    directory = "./qwen3_14b_think_responses"
    prefix = "response_"
    output_file_name = "./qwen3_14b_think_responses/responses.json"

    merge_json_files(directory, prefix, output_file_name)
