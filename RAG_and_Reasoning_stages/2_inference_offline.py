import os
import sys
from huggingface_hub import login
from datasets import load_dataset
from unsloth import is_bfloat16_supported, FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import PeftModel

hf_token = os.getenv("HF_TOKEN")
login(hf_token)

max_seq_length = 2048
dtype = None
load_in_4bit = True

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are now an specialized agent to infer proofs for problems, theorem statements and lemmas written in Isabelle/HOL.
Infer a proof for the following Isabelle/HOL theorem statement.

### Context:
{}

### Theorem statement:
{}

### Proof:
<think>{}"""


base_model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B"
offline_model_path = "jcrecio/risamath-v0.1"

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = PeftModel.from_pretrained(base_model, offline_model_path)


def infer_proof(context, theorem_statement, device):
    inputs = tokenizer(
        [prompt_style.format(context, theorem_statement, "")], return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=4096,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    return response


def generate_text(prompt):
    inputs = tokenizer(
        [prompt_style.format("NO CONTEXT", prompt, "")], return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=4096,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    return response


prompt_mode = sys.argv[1]
FastLanguageModel.for_inference(model)

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

        proof = infer_proof(context, theorem_statement, "cuda")
        print("Inferred proof:\n")
        print(proof)
    elif prompt_mode == "full":
        prompt = input("Enter your prompt here, or write EXIT to quit:")
        if prompt == "EXIT":
            break
        response = generate_text(prompt)
        print(response)
