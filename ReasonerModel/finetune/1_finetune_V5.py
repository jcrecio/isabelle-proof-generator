import json
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

WITH_CONTEXT = os.getenv("WITH_CONTEXT")

new_model_local = "jcrecio/Remath-v0.5"
wandb.login(key=wandb_token)
run = wandb.init(
    project="Remath-v0.4",
    job_type="training",
    anonymous="allow",
)


max_seq_length = 2048
dtype = None
load_in_4bit = True


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=hf_token,
)

train_prompt_style = {
    "messages": [
        {
            "role": "system",
            "content": "You are an expert in Isabelle/HOL proof generation. Given a theorem statement, you will think through the proof strategy and then provide a clean, valid Isabelle/HOL proof.",
        },
        {"role": "user", "content": "Theorem: {theorem}"},
        {"role": "assistant", "thinking": "{reasoning}", "output": "proof"},
        {"role": "assistant", "content": "{proof}"},
    ]
}

inference_prompt_style = """Given a theorem in Isabelle/HOL, think through the proof strategy step by step, then output ONLY a clean, valid Isabelle/HOL proof.

Theorem: {theorem}

Think through the proof strategy:
<think>
Consider the theorem structure
Plan the proof approach
Identify necessary tactics and methods
</think>

Now provide ONLY the clean Isabelle/HOL proof:
"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    theorems = examples["theorem_statement"]
    reasoning = examples["reasoning"]
    proofs = examples["proof"]
    texts = []

    for theorem, think, proof in zip(theorems, reasoning, proofs):
        # Create prompt structure
        prompt = train_prompt_style.copy()
        prompt["messages"][1]["content"] = f"Theorem: {theorem}"
        prompt["messages"][2]["thinking"] = think
        prompt["messages"][3]["content"] = proof.strip()  # Clean proof only

        # Convert to string and add EOS token
        text = json.dumps(prompt) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}


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
