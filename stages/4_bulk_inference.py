"""
This script is used to run the model to infer proofs for problems, theorem statements, or lemmas written in Isabelle/HOL.

Author: Juan Carlos Recio Abad
    - device mode: The device to use for inference. It can be cpu, cuda, half, or low.
Functions:
    stream(fullprompt, device, model, tokenizer, initial_max_tokens=200, continuation_tokens=100):
        Generates text based on the given prompt using the specified model and tokenizer.
    infer_proof(context, theorem_statement, device, model, tokenizer):
        Infers a proof for the given theorem statement using the specified model and tokenizer.
    main():
        Main function to execute the script. It initializes the model and tokenizer, and infers proofs for predefined problems.
This script is used to run the model to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL.

Usage:
    python isabelle-proof-generator/stages/4_bulk_inference.py <model_name> <device mode>
    - model_name: The name of the model to use for inference.
    - device mode: The device to use for inference. It can be cpu, cuda, half or low.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import sys

PROMPT_TEMPLATE_QUESTION_ANSWER = "You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step."
PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT = "You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step."

"""
This function generates text based on the given prompt using the specified model and tokenizer.
"""


def stream(
    fullprompt,
    device,
    model,
    tokenizer,
    initial_max_tokens=200,
    continuation_tokens=100,
):
    # Ensure model is on the correct device
    model = model.to(device)

    # Create input tensors on the same device as the model
    inputs = tokenizer([fullprompt], return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generated_text = ""
    while True:
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
This function generates a proof for the given theorem statement using the specified model and tokenizer.
"""


def infer_proof(context, theorem_statement, device, model, tokenizer):
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
    print("Inferring proof...\n")
    return stream(fullprompt, device, model, tokenizer)


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_name> <device_mode>")
        sys.exit(1)

    model_name = sys.argv[1]
    requested_device = sys.argv[2]

    device = select_device(requested_device)

    clear_cache(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = get_initialized_model(model_name, requested_device, device)

    problems = [
        {
            "context": """Here it is an Isabelle/HOL theory that demonstrates several basic concepts:
1. add_zero_right: Shows that adding 0 to any number on the right gives the same number
2. distrib_left: Demonstrates the distributive property of multiplication over addition
3. less_than_zero: Shows that any natural number is less than its successor
4. add_increases: Proves that adding a non-zero number to another number increases it
5. append_nil: Shows that appending an empty list to any list gives the original list""",
            "theorem_statement": '''
theorem distrib_left: "a * (b + c) = a * b + a * c"
by simp
theorem less_than_zero: "⋀n::nat. n < Suc n"
by simp
theorem add_increases: "⋀a b::nat. a ≠ 0 ⟹ b < a + b"''',
        }
    ]

    for problem in problems:
        context = problem["context"]
        theorem_statement = problem["theorem_statement"]
        proof = infer_proof(context, theorem_statement, device, model, tokenizer)
        print("Inferred proof:\n")
        print(proof)


def get_initialized_model(model_name, requested_device, device):
    if requested_device == "low":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if requested_device == "half":
            model = model.half()
        model = model.to(device)
    return model


def clear_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)


def select_device(requested_device):
    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device in ["cuda", "half"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        print(f"Unknown device mode '{requested_device}', defaulting to CPU")

    print(f"Using device: {device}")
    return device


if __name__ == "__main__":
    main()
