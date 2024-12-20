'''
This script is used to run the model to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL.
Usage:
    python isabelle-proof-generator/stages/4_bulk_inference.py <model_name> <device mode>
    - model_name: The name of the model to use for inference.
    - device mode: The device to use for inference. It can be cpu, cuda, half or low.
'''

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import sys
import gc
import os

# Set environment variable for expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

PROMPT_TEMPLATE_QUESTION_ANSWER = 'You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step.'
PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT = 'You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step.'

def clear_gpu_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_gpu_memory_info():
    """Get GPU memory information"""
    if torch.cuda.is_available():
        gpu = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(gpu) / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved(gpu) / 1024**3
        total = torch.cuda.get_device_properties(gpu).total_memory / 1024**3
        free = total - allocated
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free of {total:.2f}GB total"
    return "GPU not available"

def stream(fullprompt, device, model, tokenizer, initial_max_tokens=200, continuation_tokens=100):
    """Stream generated text with memory management"""
    try:
        # Move model to device and optimize memory
        model = model.to(device)
        
        # Print memory status before generation
        print(get_gpu_memory_info())
        
        # Create input tensors efficiently
        with torch.cuda.amp.autocast(enabled=True):
            inputs = tokenizer([fullprompt], return_tensors="pt", truncation=True, max_length=2048).to(device)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generated_text = ""
            while True:
                outputs = model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=initial_max_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    use_cache=True
                )

                generated_text += tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if outputs[0][-1] == tokenizer.eos_token_id:
                    break
                
                # Clear memory between generations
                del outputs
                clear_gpu_memory()
                
                inputs = tokenizer([generated_text], return_tensors="pt", truncation=True, max_length=2048).to(device)
                initial_max_tokens = continuation_tokens

        return generated_text
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            clear_gpu_memory()
            print("WARNING: Out of memory. Attempting to recover...")
            if device.type == "cuda":
                # Try to recover by moving to CPU
                print("Moving model to CPU and retrying with reduced batch size...")
                return stream(fullprompt, torch.device("cpu"), model.to("cpu"), tokenizer, 
                            initial_max_tokens=100, continuation_tokens=50)
        raise e

def infer_proof(context, theorem_statement, device, model, tokenizer):
    system_prompt = PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT if context else PROMPT_TEMPLATE_QUESTION_ANSWER
    B_INST, E_INST = f"[INST]Given the problem context {context}, " if context else "[INST]", "[/INST]"

    fullprompt = f"{system_prompt}{B_INST}Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement.strip()}\n{E_INST}"
    print("\nFull prompt length:", len(fullprompt))
    print("Inferring proof...\n")
    return stream(fullprompt, device, model, tokenizer)

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_name> <device_mode>")
        sys.exit(1)

    model_name = sys.argv[1]
    requested_device = sys.argv[2]

    # Set device with memory optimization
    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device in ["cuda", "half"]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print(f"Unknown device mode '{requested_device}', defaulting to CPU")

    print(f"Using device: {device}")
    clear_gpu_memory()

    # Initialize tokenizer with efficient settings
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize model with memory optimization
    try:
        if requested_device == "low":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory={0: "35GB"}  # Limit GPU memory usage
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            if requested_device == "half":
                model = model.half()
            model = model.to(device)

    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with reduced precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    problems = [
        {'context': '''Here it is an Isabelle/HOL theory that demonstrates several basic concepts.:
1. add_zero_right: Shows that adding 0 to any number on the right gives the same number
2. distrib_left: Demonstrates the distributive property of multiplication over addition
3. less_than_zero: Shows that any natural number is less than its successor
4. add_increases: Proves that adding a non-zero number to another number increases it
5. append_nil: Shows that appending an empty list to any list gives the original list''',
         'theorem_statement': '''
theorem distrib_left: "a * (b + c) = a * b + a * c"
by simp
theorem less_than_zero: "⋀n::nat. n < Suc n"
by simp
theorem add_increases: "⋀a b::nat. a ≠ 0 ⟹ b < a + b"'''}
    ]

    for problem in problems:
        context = problem['context']
        theorem_statement = problem['theorem_statement']
        try:
            proof = infer_proof(context, theorem_statement, device, model, tokenizer)
            print('Inferred proof:\n')
            print(proof)
        except Exception as e:
            print(f"Error during inference: {e}")
            clear_gpu_memory()

if __name__ == "__main__":
    main()