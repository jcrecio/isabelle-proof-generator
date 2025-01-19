"""
Converts a JSON file to JSON Lines (JSONL) format.
Args:
    input_file (str): The path to the input JSON file.
    output_file (str): The path to the output JSONL file.
The function reads the input JSON file line by line, parses each line as JSON,
and writes it to the output file in JSONL format, where each JSON object is 
written on a new line.
"""

import json
import sys


def json_to_jsonl(input_file, output_file):
    with open(input_file, "r") as infile:
        json_data = json.load(infile)
        problems = json_data.get("proofs")
        with open(output_file, "w") as outfile:
            for problem in problems:
                json.dump(problem, outfile)
                outfile.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_jsonl.py input_file.json output_file.jsonl")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        json_to_jsonl(input_file, output_file)
