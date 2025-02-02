"""
This file finetunes a model on a dataset of proofs.

In order to run it, you need to set the following environment variables:
- MODEL_TO_USE: The name of the model to use for fine-tuning. (Example: mistralai/Mathstral-7B-v0.1)
- NEW_MODEL: The name of the new model to be created after fine-tuning. (Example: jcrecio/isamath-v0.1)
- TRAINING_FILE: The path to the training file to use for fine-tuning.
- DATASET: The name of the dataset to use for fine-tuning. (Example: jcrecio/AFP_Cot_Contextualized_Proofs)
- FINETUNE_CONFIG_FILE: The path to the configuration file to use for fine-tuning.
- WANDB_TOKEN: Your Weights & Biases API token to store the finetune process data.
- The configuration file for the finetune hyperparameters should be in a JSON file.

After setting the environment variables, you can run the script with the following command:
> python stages/1_finetune.py
"""

from datetime import datetime
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import os
import torch
import wandb
from datasets import load_dataset
from trl import SFTTrainer

from dotenv import load_dotenv


def read_config(filename):
    with open(filename, "r") as file:
        config = json.load(file)
    return config


load_dotenv()

config = read_config(os.getenv("FINETUNE_CONFIG_FILE"))
base_model = os.getenv("MODEL_TO_USE")
new_model = os.getenv("NEW_MODEL")
dataset_name = os.getenv("DATASET")
wandb_token = os.getenv("WANDB_TOKEN")
train_file = os.getenv("TRAINING_FILE")

dataset = load_dataset(dataset_name, data_files={"train": train_file}, split="train")
dataset = load_dataset(dataset_name, split="train")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=bnb_config, device_map={"": 0}
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

wandb.login(key=wandb_token)
run = wandb.init(
    project=f"{datetime.now().strftime("%H:%M:%S")} Mathstral 7B => FINETUNE",
    job_type="training",
    anonymous="allow",
)

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=config["r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    bias=config["bias"],
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
)
model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    optim=config["optim"],
    save_steps=config["save_steps"],
    logging_steps=config["logging_steps"],
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    fp16=config["fp16"],
    bf16=config["bf16"],
    max_grad_norm=config["max_grad_norm"],
    max_steps=config["max_steps"],
    warmup_ratio=config["warmup_ratio"],
    group_by_length=config["group_by_length"],
    lr_scheduler_type=config["lr_scheduler_type"],
    report_to=config["report_to"],
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

trainer.model.save_pretrained(new_model)
wandb.finish()
model.config.use_cache = True
model.eval()
