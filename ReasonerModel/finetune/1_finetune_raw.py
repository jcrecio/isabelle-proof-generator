import wandb
import os
from huggingface_hub import login
from datasets import load_dataset
from unsloth import is_bfloat16_supported, FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from dotenv import load_dotenv

load_dotenv()

wandb_token = os.getenv("WANDB_TOKEN")

hf_token = os.getenv("HF_TOKEN")
login(hf_token)

new_model_local = "jcrecio/Remath-v0.1-raw"
wandb.login(key=wandb_token)
run = wandb.init(
    project="Remath-v0.1-raw",
    job_type="training",
    anonymous="allow",
)


max_seq_length = 2048
dtype = None
load_in_4bit = True


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=hf_token,
)

train_prompt_style = """
### Isabelle/HOL Theorem statement:
{}
### Proof:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    theorem_statements = examples["theorem_statement"]
    proofs = examples["proof"]
    texts = []
    for theorem_statement, proof in zip(theorem_statements, proofs):
        text = train_prompt_style.format(theorem_statement, proof) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


DATASET_FILE = "afp_extractions_reasoning.jsonl"

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
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
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
