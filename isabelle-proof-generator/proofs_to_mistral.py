import json
import sys

PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT = 'You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma.[INST]Given the problem context "{context}". Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement}[/INST]{proof}'
PROMPT_TEMPLATE_QUESTION_ANSWER = 'You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma.[INST]Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement}[/INST]{proof}'
NO_CONTEXT = 'NO_CONTEXT'

def map_prompt(context, theorem_statement, proof):
    if context == NO_CONTEXT: 
        return PROMPT_TEMPLATE_QUESTION_ANSWER.replace('{theorem_statement}', theorem_statement).replace('{proof}', proof)
    
    return PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT \
        .replace('{context}', context) \
        .replace('{theorem_statement}', theorem_statement) \
        .replace('{proof}', proof)

def proofs_jsonl_to_mistral(input_file, output_file):
    with open(input_file, 'r') as infile:
        json_data = json.load(infile)
        problems = json_data.get('proofs')
        with open(output_file, 'w') as outfile:
            for problem in problems:
                context = problem.get('context')
                theorem_statement = problem.get('theorem_statement')
                proof = problem.get('proof')
                json.dump({'proof': map_prompt(context, theorem_statement, proof)}, outfile)
                outfile.write('\n')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python proofs_to_mistral.py input_file.json output_file.jsonl")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        proofs_jsonl_to_mistral(input_file, output_file)