'''
This script is used to run the model to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL.
Usage:
    python isabelle-proof-generator/stages/3_run_model.py <model_name> <mode_to_run>
    model_name: The name of the model to use for inference.
    mode_to_run: The mode to run the model. It can be 1 or 2.
        1: Run the model using the generate method.
        2: Run the model using the TextStreamer.
'''

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import sys

PROMPT_TEMPLATE_QUESTION_ANSWER = 'You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step.'
PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT = 'You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step.'

def infer_proof(context, theorem_statement, mode_to_run):
    runtimeFlag = "cuda:0"
    system_prompt = PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT if context else PROMPT_TEMPLATE_QUESTION_ANSWER
    B_INST, E_INST = f"[INST]Given the problem context {context}, " if context else "[INST]", "[/INST]"

    prompt = f"{system_prompt}{B_INST}Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement.strip()}\n{E_INST}"

    if mode_to_run == 1:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_length=100, num_return_sequences=1, temperature=0.7)

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(generated_text)

    elif mode_to_run == 2:
        inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generated_text = model.generate(**inputs, streamer=streamer, max_new_tokens=200)
        print(generated_text)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
print(f"Using device: {device}")

model_name = sys.argv[1]
mode_to_run = sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)

while(True):
    context = input("Please enter the context for the problem (or leave it empty if no context), or write EXIT to quit:")
    if context == "EXIT":
        break
    theorem_statement = input("Please enter the theorem statement you want to infer a proof for, or write EXIT to quit:")
    if theorem_statement == "EXIT":
        break

    proof = infer_proof(context, theorem_statement, mode_to_run, device)
    print('Inferred proof:\n')
    print(proof)