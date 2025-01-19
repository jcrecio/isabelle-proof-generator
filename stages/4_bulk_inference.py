"""
This script is used to run the model to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL.
Usage:
    python isabelle-proof-generator/stages/4_bulk_inference.py <model_name> <device mode> <number of problems to solve>
    - model_name: The name of the model to use for inference.
    - device mode: The device to use for inference. It can be cpu, cuda, half or low.
    - number of problems to solve: up to how many problems to do bulk inferene
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from threading import Thread

from datasets import load_dataset
import torch
import sys
import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROMPT_TEMPLATE_QUESTION_ANSWER = "You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step."
PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT = "You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step."


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_gpu_memory_info():
    if torch.cuda.is_available():
        gpu = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(gpu) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu) / 1024**3
        total = torch.cuda.get_device_properties(gpu).total_memory / 1024**3
        free = total - allocated
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free of {total:.2f}GB total"

    return "GPU not available"


class RealTimeStreamer(TextIteratorStreamer):
    """Custom streamer that prints tokens in real-time"""

    def on_finalized_text(self, text: str, stream_end: bool = False):
        print(text, end="", flush=True)


"""
This function is used to generate a response from the model using a streaming approach.
The parameters are used as follows:
- fullprompt: The full prompt to use for generation.
- device: The device to use for inference.
- model: The model to use for inference.
- tokenizer: The tokenizer to use for inference.
- initial_max_tokens: The initial number of tokens to generate, this is used for the first iteration and then it is set to continuation_tokens.
- continuation_tokens: The number of tokens to generate in each iteration after the first one.
- max_generation_tokens: The maximum number of tokens to generate in total.
- max_iterations: The maximum number of iterations to perform.

"""


def stream(
    fullprompt,
    device,
    model,
    tokenizer,
    initial_max_tokens=200,
    continuation_tokens=100,
    max_generation_tokens=8192,
    max_iterations=50,
):
    try:
        tokens_to_generate = initial_max_tokens
        model = model.to(device)

        print(get_gpu_memory_info())

        with torch.cuda.amp.autocast(enabled=True):
            inputs = tokenizer(
                [fullprompt], return_tensors="pt", truncation=True, max_length=2048
            ).to(device)

            streamer = RealTimeStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            accumulated_tokens = tokens_to_generate
            generated_text = ""
            iteration_count = 0

            while accumulated_tokens < max_generation_tokens:
                if iteration_count >= max_iterations:
                    print("\nMaximum iterations reached, stopping generation.")
                    break

                iteration_count += 1
                accumulated_tokens += tokens_to_generate

                generation_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=tokens_to_generate,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    use_cache=True,
                )

                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()

                generated_chunk = ""
                for new_text in streamer:
                    generated_chunk += new_text

                thread.join()
                generated_text += generated_chunk

                if tokenizer.eos_token_id in inputs["input_ids"][0]:
                    break

                if (
                    generated_chunk.strip()
                    in generated_text[: -len(generated_chunk)].strip()
                ):
                    print("\nRepetitive sequence detected, stopping generation.")
                    break

                clear_gpu_memory()

                inputs = tokenizer(
                    [generated_text],
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(device)
                tokens_to_generate = continuation_tokens

        return generated_text

    except RuntimeError as e:
        if "out of memory" in str(e):
            clear_gpu_memory()
            print("WARNING: Out of memory. Attempting to recover...")
            if device.type == "cuda":
                print("Moving model to CPU and retrying with reduced batch size...")
                return stream(
                    fullprompt,
                    torch.device("cpu"),
                    model.to("cpu"),
                    tokenizer,
                    initial_max_tokens=100,
                    continuation_tokens=50,
                )
        raise e


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
    print("\nFull prompt length:", len(fullprompt))
    print("Inferring proof...\n")
    return stream(fullprompt, device, model, tokenizer)


def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_name> <device_mode> <number of problems>")
        sys.exit(1)

    model_name = sys.argv[1]
    requested_device = sys.argv[2]
    number_of_problems = sys.argv[3]

    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device in ["cuda", "half"]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.set_per_process_memory_fraction(0.8)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print(f"Unknown device mode '{requested_device}', defaulting to CPU")

    print(f"Using device: {device}")
    clear_gpu_memory()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        if requested_device == "low":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory={0: "35GB"},
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
            if requested_device == "half":
                model = model.half()
            model = model.to(device)

    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with reduced precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )

    dataset = load_dataset("jcrecio/AFP_Cot_Contextualized_Proofs")
    problems_keys = list(dataset.keys())[0]
    problems = [
        dict(row) for row in dataset[problems_keys].select(range(number_of_problems))
    ]

    for problem in problems:
        context = problem["context"]
        theorem_statement = problem["theorem_statement"]
        try:
            print("Problem to solve:\n")
            print(theorem_statement)
            print()
            proof = infer_proof(context, theorem_statement, device, model, tokenizer)
            print("Inferred proof:\n")
            print(proof)
        except Exception as e:
            print(f"Error during inference: {e}")
            clear_gpu_memory()


if __name__ == "__main__":
    main()
