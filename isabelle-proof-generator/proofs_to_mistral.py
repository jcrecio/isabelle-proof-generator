import json
import sys

def jsonl_to_json(input_file, output_file):
    with open(input_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            for line in infile:
                json_data = json.loads(line)
                problem = json_data.get('proof')
                lemma = problem.get('lemma')
                proof = problem.get('proof')
                json.dump({'proof': f'An user requests an Isabelle/HOL proof completion to an AI Assistant proof generator for a current problem, lemma or theorem.[INST]{lemma}[/INST]{proof}'}, outfile)
                outfile.write('\n')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jsonl_to_json.py input_file.jsonl output_file.json")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        jsonl_to_json(input_file, output_file)