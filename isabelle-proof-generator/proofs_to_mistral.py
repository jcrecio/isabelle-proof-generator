import json
import sys

PROMPT_TEMPLATE = 'You are now an specialized agent to infer proofs for theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding theorem statement or lemma.[INST]Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement}[/INST]{proof}'

def map_prompt(theorem_statement, proof):
    return PROMPT_TEMPLATE.replace('{theorem_statement}', theorem_statement).replace('{proof}', proof)

def proofs_jsonl_to_mistral(input_file, output_file):
    with open(input_file, 'r') as infile:
        json_data = json.load(infile)
        problems = json_data.get('proofs')
        with open(output_file, 'w') as outfile:
            for problem in problems:
                theorem_statement = problem.get('theorem_statement')
                proof = problem.get('proof')
                json.dump({'proof': map_prompt(theorem_statement, proof)}, outfile)
                outfile.write('\n')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python proofs_jsonl_to_mistral.py input_file.jsonl output_file.json")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        proofs_jsonl_to_mistral(input_file, output_file)