from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import  PeftModel
import os, torch

PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT = 'You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step.'

torch.cuda.empty_cache()
model = os.getenv('MODEL_TO_USE')

base_model_reload = AutoModelForCausalLM.from_pretrained(
    model, 
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map= {"": 0})
model = PeftModel.from_pretrained(base_model_reload, new_model)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def infer_proof(theorem_statement):
    runtimeFlag = "cuda:0"
    system_prompt = PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT
    B_INST, E_INST = "[INST]", "[/INST]"

    prompt = f"{system_prompt}{B_INST}{'Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement}'.strip()}\n{E_INST}"

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=200)

while(True):
    theorem_statement = input("Please enter the theorem statement you want to infer a proof for or write EXIT to quit: ")
    if theorem_statement == "EXIT":
        break

    proof = infer_proof(theorem_statement)
    print('Inferred proof:\n')
    print(proof)