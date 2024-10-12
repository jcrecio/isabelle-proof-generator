'''
Converts a JSON Lines (JSONL) file to a JSON file.
Args:
    input_file (str): The path to the input JSONL file.
    output_file (str): The path to the output JSON file.
The function reads the input JSONL file line by line, converts each line 
from JSONL format to a JSON object, and writes these objects to the output 
JSON file as a JSON array.
'''

import json
import sys

def jsonl_to_json(input_file, output_file):
    with open(input_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            outfile.write('[')
            for line in infile:
                json_data = json.loads(line)
                json.dump(json_data, outfile)
                outfile.write(',')
            outfile.write(']')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jsonl_to_json.py input_file.jsonl output_file.json")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        jsonl_to_json(input_file, output_file)