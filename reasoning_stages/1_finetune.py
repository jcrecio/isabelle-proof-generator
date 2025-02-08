from huggingface_hub import login
from unsloth import FastLanguageModel
import wandb
import os

wandb_token = os.getenv("WANDB_TOKEN")

hf_token = os.getenv("HF_TOKEN")
login(hf_token)


wandb.login(key=wandb_token)
run = wandb.init(
    project="isamath-v0.23-deepkseekR1-test",
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


PROMPT_TEMPLATE_QUESTION_ANSWER_WITH_CONTEXT = '<s>. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding problem, theorem statement or lemma. When you answer, please do it reasoning step by step.[INST]Given the problem context "{context}". Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement}[/INST]{proof}</s>'


prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are now an specialized agent to infer proofs for problems, theorem statements and lemmas written in Isabelle/HOL.
Infer a proof for the following Isabelle/HOL theorem statement.

### Theorem statement:
{}

### Proof:
<think>{}"""


question = "Prove that the square root of 2 is irrational."


FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print(response)
