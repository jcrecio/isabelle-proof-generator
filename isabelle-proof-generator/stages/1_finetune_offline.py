from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch
from datasets import load_from_disk
from trl import SFTTrainer

from dotenv import load_dotenv
load_dotenv()

base_model = "mistralai/Mistral-7B-v0.1" #bn22/Mistral-7B-Instruct-v0.1-sharded
dataset_name, new_model = "jcrecio/subset_afp_mistral", "jcrecio/suboptimal_afp_7B"
model_dir = 'Mistral-7B-v0.1'

dataset = load_from_disk("afp_mistral")

bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_dir,
    device_map={"": 0}
)

# Config
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

# Prepare model
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 8,
    gradient_accumulation_steps= 2,
    optim = "paged_adamw_8bit",
    save_steps= 5000,
    logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "constant"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="proof",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

trainer.train()
trainer.model.save_pretrained(new_model)

model.config.use_cache = True
model.eval()