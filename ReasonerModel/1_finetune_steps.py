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

new_model_local = (
    "jcrecio/Remath-v0.1-c" if WITH_CONTEXT == "True" else "jcrecio/Remath-v0.1"
)
wandb.login(key=wandb_token)
run = wandb.init(
    project="Remath-v0.1-c" if WITH_CONTEXT == "True" else "Remath-v0.1",
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

train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are now an specialized agent to infer proofs for problems, theorem statements and lemmas written in Isabelle/HOL.
Infer a proof for the following Isabelle/HOL theorem statement.

### Theorem statement:
{}

### Proof:
<think>
{}
</think>
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func_with_context(examples):
    contexts = examples["context"]
    theorem_statements = examples["theorem_statement"]
    thinks = examples["reasoning"]
    proofs = examples["proof"]
    texts = []
    for context, theorem_statement, think, proof in zip(
        contexts, theorem_statements, thinks, proofs
    ):
        text = (
            train_prompt_style_with_context.format(
                context, theorem_statement, think, proof
            )
            + EOS_TOKEN
        )
        texts.append(text)
    return {
        "text": texts,
    }


def formatting_prompts_func(examples):
    theorem_statements = examples["theorem_statement"]
    thinks = examples["reasoning"]
    proofs = examples["proof"]
    texts = []
    for theorem_statement, think, proof in zip(theorem_statements, thinks, proofs):
        text = train_prompt_style.format(theorem_statement, think, proof) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


DATASET_FILE = (
    "afp_extractions_context_reasoning.jsonl"
    if WITH_CONTEXT == "True"
    else "afp_extractions_reasoning.jsonl"
)

dataset = load_dataset(
    "jcrecio/AFP_Theories",
    data_files=DATASET_FILE,
    split="train",
    trust_remote_code=True,
)
formatter = (
    formatting_prompts_func_with_context
    if WITH_CONTEXT == "True"
    else formatting_prompts_func
)
dataset = dataset.map(
    formatter,
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
