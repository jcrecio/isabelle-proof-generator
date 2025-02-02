"""
This script is responsible for running the model in offline mode.

It is a simple script that:
    1. In prompt execution = full, it takes in a prompt and generates a response using the model.
    2. In prompt execution = instruct, it takes in a context and a theorem statement and generates a proof for the theorem statement using the model.
    
The model is loaded from the adapter path and the base model path.

It uses the env variables OFFLINE_BASE_MODEL and OFFLINE_MODEL to load the base model and the adapter path respectively.

Usage: python stages/3_run_model_offline.py requested_device prompt_mode
"""

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import sys
import os
import torch
from dotenv import load_dotenv

load_dotenv()

base_model_name = os.getenv("OFFLINE_BASE_MODEL")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, device_map="auto", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

adapter_path = os.path.abspath(os.getenv("OFFLINE_MODEL"))
model = PeftModel.from_pretrained(base_model, adapter_path)

# Optional: Combine adapter weights with base model for faster inference
# model = model.merge_and_unload()


def generate_text(prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_length=max_length, temperature=0.7, num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


PROMPT_TEMPLATE_QUESTION_ANSWER = "You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step."
PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT = "You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step."

"""
This function is used to stream the generated text from the model.
"""


def stream(fullprompt, device, initial_max_tokens=200, continuation_tokens=100):
    inputs = tokenizer([fullprompt], return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generated_text = ""
    while True:
        # outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=initial_max_tokens)
        outputs = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=initial_max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_text += tokenizer.decode(outputs[0], skip_special_tokens=True)

        if outputs[0][-1] == tokenizer.eos_token_id:
            break

        # If we did not reach the EOS token, continue generating
        inputs = tokenizer([generated_text], return_tensors="pt").to(device)
        initial_max_tokens = continuation_tokens

    return generated_text


"""
This function is used to infer a proof for a given theorem statement.
It works as follows:
- It composes the full prompt with the theorem statement and the context.
"""


def infer_proof(context, theorem_statement, device):
    system_prompt = (
        PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT
        if context
        else PROMPT_TEMPLATE_QUESTION_ANSWER
    )
    B_INST, E_INST = (
        f"[INST]Given the problem context {context}, " if context else "[INST]"
    ), "[/INST]"

    fullprompt = f"{system_prompt}{B_INST}Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement.strip()}\n{E_INST}"
    print("Full prompt:\n")
    print(fullprompt)
    print("Infering proof...\n")
    stream(fullprompt, device)


device = sys.argv[1]
prompt_mode = sys.argv[2]

device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

while True:
    if prompt_mode == "instruct":
        context = input(
            "Please enter the context for the problem (or leave it empty if no context), or write EXIT to quit:"
        )
        if context == "EXIT":
            break
        theorem_statement = input(
            "Please enter the theorem statement you want to infer a proof for, or write EXIT to quit:"
        )
        if theorem_statement == "EXIT":
            break

        proof = infer_proof(context, theorem_statement, device)
        print("Inferred proof:\n")
        print(proof)
    elif prompt_mode == "full":
        prompt = input("Enter your prompt here, or write EXIT to quit:")
        if prompt == "EXIT":
            break
        response = generate_text(prompt)
        print(response)
