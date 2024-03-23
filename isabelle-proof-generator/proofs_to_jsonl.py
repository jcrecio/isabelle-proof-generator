import sys
import json

def json_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            data = json.load(infile)
            for proof in data.get("proofs", []):
                json.dump({"proof": proof}, outfile)
                outfile.write('\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_to_jsonl.py input_file.json output_file.jsonl")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        json_to_jsonl(input_file, output_file)