"""
This module provides functionality to transform proofs from a JSON file into a specific format for Mistral.
Functions:
    map_prompt(context, theorem_statement, proof):
        Maps the given context, theorem statement, and proof to a formatted prompt string.
    proofs_jsonl_to_mistral(input_file, output_file):
        Reads proofs from an input JSON file and writes them to an output JSONL file in the Mistral format.
Usage:
    Run this script from the command line with the input and output file paths as arguments:
    python proofs_to_mistral.py input_file.json output_file.jsonl
Constants:
    PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT (str): Template for prompts with context.
    PROMPT_TEMPLATE_QUESTION_ANSWER (str): Template for prompts without context.
    NO_CONTEXT (str): Constant indicating no context is provided.
"""

import sys
import json

PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT = '<s>You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step.[INST]Given the problem context "{context}". Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement}[/INST]{proof}</s>'
PROMPT_TEMPLATE_QUESTION_ANSWER = "<s>You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step.[INST]Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement}[/INST]{proof}</s>"
NO_CONTEXT = "NO_CONTEXT"


def map_prompt(context, theorem_statement, proof):
    if context == NO_CONTEXT:
        return PROMPT_TEMPLATE_QUESTION_ANSWER.replace(
            "{theorem_statement}", theorem_statement
        ).replace("{proof}", proof)

    return (
        PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT.replace("{context}", context)
        .replace("{theorem_statement}", theorem_statement)
        .replace("{proof}", proof)
    )


def proofs_jsonl_to_mistral(input_file, output_file):
    with open(input_file, "r") as infile:
        json_data = json.load(infile)
        problems = json_data.get("proofs")
        with open(output_file, "w") as outfile:
            for problem in problems:
                context = problem.get("context")
                theorem_statement = problem.get("theorem_statement")
                proof = problem.get("proof")
                json.dump(
                    {"text": map_prompt(context, theorem_statement, proof)}, outfile
                )
                outfile.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python proofs_to_mistral.py input_file.json output_file.jsonl")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        proofs_jsonl_to_mistral(input_file, output_file)
