from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import notebook_login

from dotenv import load_dotenv
load_dotenv()

base_model = "mistralai/Mistral-7B-v0.1" #bn22/Mistral-7B-Instruct-v0.1-sharded
dataset_name, new_model = "jcrecio/afp", "jcrecio/afp_7B"

dataset = load_dataset(dataset_name, split="train", token=os.getenv("HF_TOKEN"))
dataset["chat_sample"][0]

# bitsandbytes = "^x.y.z"  # Replace with the desired version range (e.g., "~2.0")
# transformers = { git = "https://github.com/huggingface/transformers.git", rev = "v4.23.1" }  # Specify Git URL and version (optional)
# peft = { git = "https://github.com/huggingface/peft.git", rev = "v1.7.0" }  # Specify Git URL and version (optional)
# accelerate = { git = "https://github.com/huggingface/accelerate.git", rev = "v4.8.0" }  # Specify Git URL and version (optional)
# trl = "^x.y.z"  # Replace with the desired version range
# xformers = "^x.y.z"  # Replace with the desired version range
# wandb = "^x.y.z"  # Replace with the desired version range
# datasets = "^x.y.z"  # Replace with the desired version range
# einops = "^x.y.z"  # Replace with the desired version range
# sentencepiece 