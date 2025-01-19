"""
Converts a JSON file containing proofs into a JSONL (JSON Lines) file.
Args:
    input_file (str): The path to the input JSON file.
    output_file (str): The path to the output JSONL file.
The input JSON file is expected to have a structure like:
{
    "proofs": [
        {"id": 1, "content": "proof content 1"},
        {"id": 2, "content": "proof content 2"},
        ...
    ]
}
Each proof in the "proofs" list will be written as a separate line in the output JSONL file.
"""

import json
import sys


def json_to_jsonl(input_file, output_file):
    with open(input_file, "r") as infile:
        with open(output_file, "w") as outfile:
            data = json.load(infile)
            for proof in data.get("proofs", []):
                json.dump({"proof": proof}, outfile)
                outfile.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_jsonl.py input_file.json output_file.jsonl")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        json_to_jsonl(input_file, output_file)
