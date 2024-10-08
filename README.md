# Isabelle-proof-generator

## Create the dataset

### JSON Dataset
In order to generate a dataset to finetune a model to infer Isabelle/HOL proofs we need to start with the PISA Dataset from https://github.com/albertqjiang/Portal-to-ISAbelle

This dataset contains the AFP problems separated in folders.

Set the following environment variables:
```
PROBLEMS_FOLDER=<root folder for the PISA extractions AFP problems to create the dataset>
AFP_FOLDER=<root folder of AFP problems>/thys
OUTPUT_FILE_DATASET=<output dataset filepath .json for the json dataset creation>
WITH_CONTEXT=False/True flag to indicate if the finetune will be contextualized

USE_MODELS_OFFLINE=<true/false>
USE_WANDB=<true/false>

MODEL_TO_USE=<choose the model>
<choose the model> can be one of the following list: https://docs.mistral.ai/getting-started/models/weights/
example: MODEL_TO_USE=Mathstral-7B-v0.1
```

1. Run the script `json_dataset_creator.py` to generate the dataset as a json file containing pairs (theorem statement, proof)

2. Run the script `proofs_to_mistral.py <dataset json file> <output jsonl file>` to get the file in the format Mistral expects for finetuning. (Includes the finetune specialized prompt)

The simple specialized prompt for finetune is:
> *"You are now an specialized agent to infer proofs for theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding theorem statement or lemma.[INST]Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement}[/INST]{proof}"*


The contextualized prompt for finetune (based on the abstract of the problems) is:
```
You are now an specialized agent to infer proofs for theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding theorem statement or lemma. When you answer, please do it reasoning step by step.[INST]Given the problem context {context}. Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement}[/INST]{proof}
```

Example:
```
You are now an specialized agent to infer proofs for theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding theorem statement or lemma. When you answer, please do it reasoning step by step.[INST]Given the problem context "We define the SLin I/O-automaton and prove its composition property. The SLin\nI/O-automaton is at the heart of the Speculative Linearizability framework.\nThis framework simplifies devising robust distributed algorithms by enabling\ntheir decomposition into independent modules that can be analyzed and proved\ncorrect in isolation. It is particularly useful when working in a distributed\nenvironment, where the need to tolerate faults and asynchrony has made current\nmonolithic protocols so intricate that it is no longer tractable to check their\ncorrectness. Our theory contains a formalization of simulation proof techniques\nin the I/O-automata of Lynch and Tuttle and a typical example of a refinement\nproof".[INST]Infer a proof for the following Isabelle/HOL theorem statement/s: \nlemma single_use:\n  fixes r rs\n  shows  "\\<bottom> \\<star> ([r]@rs) = Some (snd r)"\nproof (induct rs)\ncase Nil\nthus ?case\nby simp\nnext\ncase (Cons r rs)\nthus ?case\nby auto\nqed\nlemma bot: "\\<exists> rs . s = \\<bottom> \\<star> rs"\nproof (cases s)\ncase None\nhence "s = \\<bottom> \\<star> []"\nby auto\nthus ?thesis\nby blast\nnext\ncase (Some v)\nobtain r where "\\<bottom> \\<star> [r] = Some v"\nby force\nthus ?thesis\nusing Some\nby metis\nqed[/INST]'\nusing assms single_use \nproof - \n{ \nassume 1:"s1 \\<noteq> None" and 2:"s1 \\<noteq> s2" \nobtain r rs where 3:"s1 = \\<bottom> \\<star> ([r]@rs)" \nusing bot \nusing 1 \nby (metis append_butlast_last_id pre_RDR.exec.simps(1)) \nobtain rs\' where 4:"s2 = s1 \\<star> rs\'" \nusing assms \nby (auto simp add:less_eq_def) \nhave "s2 = \\<bottom> \\<star> ([r]@(rs@rs\'))" \nusing 3 4 \nby (metis exec_append) \nhence "s1 = s2" \nusing 3 \nby (metis single_use) \nwith 2 \nhave False \nby auto \n} \nthus ?thesis \nby blast \nqed \ninterpretation RDR \\<delta> \\<gamma> \\<bottom> \nproof (unfold_locales) \nfix s r \nassume "contains s r" \nshow "s \\<bullet> r = s" \nproof - \nobtain rs where "s = \\<bottom> \\<star> rs" and "rs \\<noteq> []" \nusing \\<open>contains s r\\<close> \nby (auto simp add:contains_def, force) \nthus ?thesis \nby (metis \\<delta>.simps(2) rev_exhaust single_use) \nqed \nnext \nfix s and r r\' :: "proc \\<times> val" \nassume 1:"fst r \\<noteq> fst r\'" \nthus "\\<gamma> s r = \\<gamma> ((s \\<bullet> r) \\<bullet> r\') r" \nby (metis \\<delta>.simps \\<gamma>.simps not_Some_eq) \nnext \nfix s1 s2 \nassume "s1 \\<preceq> s2 \\<and> s2 \\<preceq> s1" \nthus "s1 = s2" \nby (metis prec_eq_None_or_equal) \nnext \nfix s1 s2 \nshow "\\<exists> s . is_glb s s1 s2" \nby (simp add:is_glb_def is_lb_def)\n    (metis bot pre_RDR.less_eq_def prec_eq_None_or_equal) \nnext \nfix s \nshow "\\<bottom> \\<preceq> s" \nby (metis bot pre_RDR.less_eq_def) \nnext \nfix s1 s2 s3 rs \nassume "s1 \\<preceq> s2" and "s2 \\<preceq> s3" and "s3 = s1 \\<star> rs" \nthus "\\<exists> rs\' rs\'\' . s2 = s1 \\<star> rs\' \\<and> s3 = s2 \\<star> rs\'\' \n    \\<and> set rs\' \\<subseteq> set rs \\<and> set rs\'\' \\<subseteq> set rs" \nby (metis Consensus.prec_eq_None_or_equal \n      in_set_insert insert_Nil list.distinct(1)\n        pre_RDR.exec.simps(1) subsetI) \nqed \nend \nend'
```

> *"You are now an specialized agent to infer proofs for theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding theorem statement or lemma.[INST]Given the problem context: **"We define the SLin I/O-automaton and prove its composition property. The SLin\nI/O-automaton is at the heart of the Speculative Linearizability framework.\nThis framework simplifies devising robust distributed algorithms by enabling\ntheir decomposition into independent modules that can be analyzed and proved\ncorrect in isolation. It is particularly useful when working in a distributed\nenvironment, where the need to tolerate faults and asynchrony has made current\nmonolithic protocols so intricate that it is no longer tractable to check their\ncorrectness. Our theory contains a formalization of simulation proof techniques\nin the I/O-automata of Lynch and Tuttle and a typical example of a refinement\nproof"**.[INST]Infer a proof for the following Isabelle/HOL theorem statement/s: **\nlemma single_use:\n  fixes r rs\n  shows  "\\<bottom> \\<star> ([r]@rs) = Some (snd r)"\nproof (induct rs)\ncase Nil\nthus ?case\nby simp\nnext\ncase (Cons r rs)\nthus ?case\nby auto\nqed\nlemma bot: "\\<exists> rs . s = \\<bottom> \\<star> rs"\nproof (cases s)\ncase None\nhence "s = \\<bottom> \\<star> []"\nby auto\nthus ?thesis\nby blast\nnext\ncase (Some v)\nobtain r where "\\<bottom> \\<star> [r] = Some v"\nby force\nthus ?thesis\nusing Some\nby metis\nqed**[/INST]'\nusing assms single_use \nproof - \n{ \nassume 1:"s1 \\<noteq> None" and 2:"s1 \\<noteq> s2" \nobtain r rs where 3:"s1 = \\<bottom> \\<star> ([r]@rs)" \nusing bot \nusing 1 \nby (metis append_butlast_last_id pre_RDR.exec.simps(1)) \nobtain rs\' where 4:"s2 = s1 \\<star> rs\'" \nusing assms \nby (auto simp add:less_eq_def) \nhave "s2 = \\<bottom> \\<star> ([r]@(rs@rs\'))" \nusing 3 4 \nby (metis exec_append) \nhence "s1 = s2" \nusing 3 \nby (metis single_use) \nwith 2 \nhave False \nby auto \n} \nthus ?thesis \nby blast \nqed \ninterpretation RDR \\<delta> \\<gamma> \\<bottom> \nproof (unfold_locales) \nfix s r \nassume "contains s r" \nshow "s \\<bullet> r = s" \nproof - \nobtain rs where "s = \\<bottom> \\<star> rs" and "rs \\<noteq> []" \nusing \\<open>contains s r\\<close> \nby (auto simp add:contains_def, force) \nthus ?thesis \nby (metis \\<delta>.simps(2) rev_exhaust single_use) \nqed \nnext \nfix s and r r\' :: "proc \\<times> val" \nassume 1:"fst r \\<noteq> fst r\'" \nthus "\\<gamma> s r = \\<gamma> ((s \\<bullet> r) \\<bullet> r\') r" \nby (metis \\<delta>.simps \\<gamma>.simps not_Some_eq) \nnext \nfix s1 s2 \nassume "s1 \\<preceq> s2 \\<and> s2 \\<preceq> s1" \nthus "s1 = s2" \nby (metis prec_eq_None_or_equal) \nnext \nfix s1 s2 \nshow "\\<exists> s . is_glb s s1 s2" \nby (simp add:is_glb_def is_lb_def)\n    (metis bot pre_RDR.less_eq_def prec_eq_None_or_equal) \nnext \nfix s \nshow "\\<bottom> \\<preceq> s" \nby (metis bot pre_RDR.less_eq_def) \nnext \nfix s1 s2 s3 rs \nassume "s1 \\<preceq> s2" and "s2 \\<preceq> s3" and "s3 = s1 \\<star> rs" \nthus "\\<exists> rs\' rs\'\' . s2 = s1 \\<star> rs\' \\<and> s3 = s2 \\<star> rs\'\' \n    \\<and> set rs\' \\<subseteq> set rs \\<and> set rs\'\' \\<subseteq> set rs" \nby (metis Consensus.prec_eq_None_or_equal \n      in_set_insert insert_Nil list.distinct(1)\n        pre_RDR.exec.simps(1) subsetI) \nqed \nend \nend'"*

The generated datasets are published in the following links in Huggingface: 
- Normal dataset: https://huggingface.co/datasets/jcrecio/AFP_Proofs
- Contextualized dataset: https://huggingface.co/datasets/jcrecio/AFP_Proofs_Contextualized

## Finetune

The finetune will be using the datasets we generated previously or you can create a different dataset of your own.

Finetuning can be setup differently depending on the machine to be run.

### Online

If the machine has full access to the network, it will have direct access to datasets and models.
However we need to login with an access token into Hugging Face, and go into the corresponding model in HuggingFace and activate de access. 
1. Navigate to the hugging face model page and accept the conditions.
2. Go to your personal settings to create an access token.
2. Install hugging face cli
3. Login with your right token using: ` huggingface-cli login `

This is the most straightforward way to run finetuning for the base model of Mistral.

Create an .env file with the following variables
```
HF_TOKEN=<token to connect to HuggingFace>
WANDB_TOKEN=<token to register and track in WandDb>
```
Install the following libraries (regardless of the system chosen to manage dependencies: pip, venv, conda, poetry, etc)
```
# In this case we use pip
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q trl xformers wandb datasets einops sentencepiece
pip install python-dotenv
```

Run the command for finetune
```
    python isabelle-proof-generator/finetune.py
```


### Offline

Our use case for an offline finetune operation is based on the Spanish Supercomputing Network (RES), which has partial access to the network to clone git repositories.

1. Download the dataset https://huggingface.co/datasets/jcrecio/afp_mistral
2. Install Git LFS to be able to clone the Mistral Model repository that contains large binary https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.mdfiles
3. Run `git lfs install` in the folder
4. Run `git clone https://huggingface.co/mistralai/<model>` 

## Run the model

Clear the cache from pytorch finetune operations.
```
torch.cuda.empty_cache()
```

In order to avoid memory problems:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 156.81 MiB is free. Including non-PyTorch memory, this process has 23.48 GiB memory in use. Of the allocated memory 23.11 GiB is allocated by PyTorch, and 146.00 KiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

# Run
>>> torch.cuda.empty_cache()
>>> PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
```

Load the finetuned model:
```
model_dir = 'path to your finetuned model'

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_dir)
```

Load the tokenizer:
```
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mathstral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/<original model>")
# or
tokenizer = AutoTokenizer.from_pretrained(mistralai/<original model>, use_fast=True)

#### ???

tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token


#### ???


```

Load the cpu/gpu device:
```
device = "cuda" if torch.cuda.is_available() else "cpu"
```
Moves model to the gpu
```
model = model.to(device)
```

```
def generate_text(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token=tokenizer.eos_token,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```