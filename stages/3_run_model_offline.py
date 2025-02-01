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
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import torch


base_model_name = os.getenv("OFFLINE_BASE_MODEL")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, device_map="auto", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

adapter_path = os.getenv("OFFLINE_MODEL")

model = PeftModel.from_pretrained(base_model, adapter_path)

# Optional: Combine adapter weights with base model for faster inference
# model = model.merge_and_unload()


def generate_text(prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_length=max_length, temperature=0.7, num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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

        proof = generate_text(context, theorem_statement, device)
        print("Inferred proof:\n")
        print(proof)
    elif prompt_mode == "full":
        prompt = input("Enter your prompt here, or write EXIT to quit:")
        if prompt == "EXIT":
            break
        response = generate_text(prompt)
        print(response)
