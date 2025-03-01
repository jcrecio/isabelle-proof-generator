from unsloth import FastLanguageModel
from peft import PeftModel
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
WITH_CONTEXT = os.getenv("WITH_CONTEXT")

login(hf_token)

base_model_path = "unsloth/DeepSeek-R1-Distill-Llama-8B"
lora_model_path = (
    "jcrecio/Remath-v0.1-c" if WITH_CONTEXT == "True" else "jcrecio/Remath-v0.1"
)

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_path, max_seq_length=4096, dtype="bfloat16", load_in_4bit=True
)

model = PeftModel.from_pretrained(base_model, lora_model_path)
model = model.merge_and_unload()

repo_id = lora_model_path

merged_model_path = (
    "jcrecio/Remath-v0.1-c-merged"
    if WITH_CONTEXT == "True"
    else "jcrecio/Remath-v0.1-merged"
)
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
model.save_pretrained_merged(
    merged_model_path,
    tokenizer,
    save_method="merged_16bit",
)

model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
model.push_to_hub_merged(merged_model_path, tokenizer, save_method="merged_16bit")
