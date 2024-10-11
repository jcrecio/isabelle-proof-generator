from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import  PeftModel
import os, torch

'''
Merge the base model with the finetuned model (delta weights)
'''
def merge_model(base_model, new_model):
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

    model = model.merge_and_unload()
    model.save_pretrained("jcrecio/isamath-v0.1")
    tokenizer.save_pretrained("jcrecio/isamath-tokenizer-v0.1")

    model.push_to_hub(new_model, use_temp_dir=False)
    tokenizer.push_to_hub(new_model, use_temp_dir=False)

torch.cuda.empty_cache()
base_model = os.getenv('MODEL_TO_USE')
new_model = os.getenv('NEW_MODEL')

merge_model(base_model, new_model)