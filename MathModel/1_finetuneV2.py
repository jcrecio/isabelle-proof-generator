# python 1_finetune.py <base_model> <new_model_name>

import os
import sys
import json
import wandb
from huggingface_hub import login
from datasets import load_dataset
from unsloth import is_bfloat16_supported, FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from dotenv import load_dotenv

base_model = sys.argv[1]
new_model_name = sys.argv[2]
new_model_local = f"""jcrecio/{new_model_name}"""

load_dotenv()

wandb_token = os.getenv("WANDB_TOKEN")
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

wandb.login(key=wandb_token)
run = wandb.init(
    project=new_model_name,
    job_type="training",
    anonymous="allow",
)


max_seq_length = 2048
dtype = None
load_in_4bit = True


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=hf_token,
)

train_prompt_style = {
    "messages": [
        {
            "role": "system",
            "content": "You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL.",
        },
        {
            "role": "user",
            "content": "Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement}",
        },
        {"role": "assistant", "content": "{proof}"},
    ]
}

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    formatted_prompts = []
    for theorem, proof in zip(examples["theorem_statement"], examples["proof"]):

        conversation = [
            {
                "role": "system",
                "content": "You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL. When you answer, please do it ONLY with the valid Isabelle/HOL proof.",
            },
            {
                "role": "user",
                "content": f"Infer a proof for the following Isabelle/HOL theorem statement: {theorem}",
            },
            {"role": "assistant", "content": proof},
        ]

        formatted_text = ""
        for message in conversation:
            if message["role"] == "system":
                formatted_text += f"<s>[INST] {message['content']} [/INST]"
            elif message["role"] == "user":
                formatted_text += f"[INST] {message['content']} [/INST]"
            else:
                formatted_text += f"{message['content']}</s>"

        formatted_prompts.append(formatted_text)

    return {"text": formatted_prompts}


DATASET_FILE = "afp_extractions.jsonl"

dataset = load_dataset(
    "jcrecio/AFP_Theories",
    data_files=DATASET_FILE,
    split="train",
    trust_remote_code=True,
)

dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        # Use num_train_epochs = 1, warmup_ratio for full training runs!
        num_train_epochs=5,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()

model.save_pretrained(new_model_local)
tokenizer.save_pretrained(new_model_local)

model.save_pretrained_merged(
    new_model_local,
    tokenizer,
    save_method="merged_16bit",
)
