from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import  PeftModel
import os, torch
from huggingface_hub import HfApi

'''
Merge the base model with the finetuned model (delta weights) and push it to the Hub
'''
def merge_model_and_push(base_model, new_model, tokenizer):
    base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model, 
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map= {"": 0})
    
    model = PeftModel.from_pretrained(base_model_reload, new_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    local_model_path = f"jcrecio/local/isamath-v0.1"
    local_tokenizer_path = "jcrecio/local/isamath-tokenizer-v0.1"
    repo_id = "jcrecio/isamath-v0.1"

    model = model.merge_and_unload()
    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_tokenizer_path)

    api = HfApi()

    api.upload_folder(
        folder_path=local_model_path,
        repo_id=repo_id,
        repo_type="model",
    )

    api.upload_folder(
        folder_path=local_tokenizer_path,
        repo_id=repo_id,
        repo_type="model",
    )

torch.cuda.empty_cache()

base_model = os.getenv('MODEL_TO_USE')
new_model = os.getenv('NEW_MODEL')
tokenizer = os.getenv('TOKENIZER')

merge_model_and_push(base_model, new_model, tokenizer)