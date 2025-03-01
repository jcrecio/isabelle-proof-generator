# Isabelle-proof-generator
:warning: It is outdated, needs to be updated!

## 0. Create the dataset

### JSON Dataset
In order to generate a dataset to finetune a model to infer Isabelle/HOL proofs we need to start with the PISA Dataset from https://github.com/albertqjiang/Portal-to-ISAbelle

This dataset contains the AFP problems separated in folders.

Set the following environment variables:
```
PROBLEMS_FOLDER=<root folder for the PISA extractions AFP problems to create the dataset>
AFP_FOLDER=<root folder of AFP problems>/thys
OUTPUT_FILE_DATASET=<output dataset filepath .json for the json dataset creation>
WITH_CONTEXT=False/True flag to indicate if the finetune will be contextualized, that is, include a context or not for each problem

USE_WANDB=<true/false>
TRAINING_FILE=jsonl training file in the HF dataset (example: train.jsonl)
MODEL_TO_USE=<choose the model>
<choose the model> can be one of the following list: https://docs.mistral.ai/getting-started/models/weights/
example: MODEL_TO_USE=Mathstral-7B-v0.1
NEW_MODEL=<your namespace/name of your model>
example: jcrecio/isamath-v0.1
```

1. Run the script `1_json_dataset_creator.py` to generate the dataset as a json file containing pairs (theorem statement, proof)

2. Run the script `2_proofs_to_mistral.py <dataset json file> <output jsonl file>` to get the file in the format Mistral expects for finetuning. (Includes the finetune specialized prompt)

3. Run the script `3_split_proofs_into_train_test.py --train-file <train file jsonl> --test-file <test file jsonl> <input file jsonl>` to generate the training and testing files.

The simple specialized prompt for finetune is:
> *"\<s>You are now an specialized agent to infer proofs for theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding theorem statement or lemma.[INST]Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement}[/INST]{proof}\</s>"*


The contextualized prompt for finetune (based on the abstract of the problems) is:
```
<s>You are now an specialized agent to infer proofs for theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding theorem statement or lemma. When you answer, please do it reasoning step by step.[INST]Given the problem context {context}. Infer a proof for the following Isabelle/HOL theorem statement/s: {theorem_statement}[/INST]{proof}</s>
```

Example:
```
<s>You are now an specialized agent to infer proofs for theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding theorem statement or lemma. When you answer, please do it reasoning step by step.[INST]Given the problem context "We define the SLin I/O-automaton and prove its composition property. The SLin\nI/O-automaton is at the heart of the Speculative Linearizability framework.\nThis framework simplifies devising robust distributed algorithms by enabling\ntheir decomposition into independent modules that can be analyzed and proved\ncorrect in isolation. It is particularly useful when working in a distributed\nenvironment, where the need to tolerate faults and asynchrony has made current\nmonolithic protocols so intricate that it is no longer tractable to check their\ncorrectness. Our theory contains a formalization of simulation proof techniques\nin the I/O-automata of Lynch and Tuttle and a typical example of a refinement\nproof".[INST]Infer a proof for the following Isabelle/HOL theorem statement/s: \nlemma single_use:\n  fixes r rs\n  shows  "\\<bottom> \\<star> ([r]@rs) = Some (snd r)"\nproof (induct rs)\ncase Nil\nthus ?case\nby simp\nnext\ncase (Cons r rs)\nthus ?case\nby auto\nqed\nlemma bot: "\\<exists> rs . s = \\<bottom> \\<star> rs"\nproof (cases s)\ncase None\nhence "s = \\<bottom> \\<star> []"\nby auto\nthus ?thesis\nby blast\nnext\ncase (Some v)\nobtain r where "\\<bottom> \\<star> [r] = Some v"\nby force\nthus ?thesis\nusing Some\nby metis\nqed[/INST]'\nusing assms single_use \nproof - \n{ \nassume 1:"s1 \\<noteq> None" and 2:"s1 \\<noteq> s2" \nobtain r rs where 3:"s1 = \\<bottom> \\<star> ([r]@rs)" \nusing bot \nusing 1 \nby (metis append_butlast_last_id pre_RDR.exec.simps(1)) \nobtain rs\' where 4:"s2 = s1 \\<star> rs\'" \nusing assms \nby (auto simp add:less_eq_def) \nhave "s2 = \\<bottom> \\<star> ([r]@(rs@rs\'))" \nusing 3 4 \nby (metis exec_append) \nhence "s1 = s2" \nusing 3 \nby (metis single_use) \nwith 2 \nhave False \nby auto \n} \nthus ?thesis \nby blast \nqed \ninterpretation RDR \\<delta> \\<gamma> \\<bottom> \nproof (unfold_locales) \nfix s r \nassume "contains s r" \nshow "s \\<bullet> r = s" \nproof - \nobtain rs where "s = \\<bottom> \\<star> rs" and "rs \\<noteq> []" \nusing \\<open>contains s r\\<close> \nby (auto simp add:contains_def, force) \nthus ?thesis \nby (metis \\<delta>.simps(2) rev_exhaust single_use) \nqed \nnext \nfix s and r r\' :: "proc \\<times> val" \nassume 1:"fst r \\<noteq> fst r\'" \nthus "\\<gamma> s r = \\<gamma> ((s \\<bullet> r) \\<bullet> r\') r" \nby (metis \\<delta>.simps \\<gamma>.simps not_Some_eq) \nnext \nfix s1 s2 \nassume "s1 \\<preceq> s2 \\<and> s2 \\<preceq> s1" \nthus "s1 = s2" \nby (metis prec_eq_None_or_equal) \nnext \nfix s1 s2 \nshow "\\<exists> s . is_glb s s1 s2" \nby (simp add:is_glb_def is_lb_def)\n    (metis bot pre_RDR.less_eq_def prec_eq_None_or_equal) \nnext \nfix s \nshow "\\<bottom> \\<preceq> s" \nby (metis bot pre_RDR.less_eq_def) \nnext \nfix s1 s2 s3 rs \nassume "s1 \\<preceq> s2" and "s2 \\<preceq> s3" and "s3 = s1 \\<star> rs" \nthus "\\<exists> rs\' rs\'\' . s2 = s1 \\<star> rs\' \\<and> s3 = s2 \\<star> rs\'\' \n    \\<and> set rs\' \\<subseteq> set rs \\<and> set rs\'\' \\<subseteq> set rs" \nby (metis Consensus.prec_eq_None_or_equal \n      in_set_insert insert_Nil list.distinct(1)\n        pre_RDR.exec.simps(1) subsetI) \nqed \nend \nend'</s>
```

> *"\<s>You are now an specialized agent to infer proofs for theorem statements or lemmas written in Isabelle/HOL. You are going to receive instructions of what you need to infer, and you will also receive some context and the corresponding theorem statement or lemma.[INST]Given the problem context: **"We define the SLin I/O-automaton and prove its composition property. The SLin\nI/O-automaton is at the heart of the Speculative Linearizability framework.\nThis framework simplifies devising robust distributed algorithms by enabling\ntheir decomposition into independent modules that can be analyzed and proved\ncorrect in isolation. It is particularly useful when working in a distributed\nenvironment, where the need to tolerate faults and asynchrony has made current\nmonolithic protocols so intricate that it is no longer tractable to check their\ncorrectness. Our theory contains a formalization of simulation proof techniques\nin the I/O-automata of Lynch and Tuttle and a typical example of a refinement\nproof"**.[INST]Infer a proof for the following Isabelle/HOL theorem statement/s: **\nlemma single_use:\n  fixes r rs\n  shows  "\\<bottom> \\<star> ([r]@rs) = Some (snd r)"\nproof (induct rs)\ncase Nil\nthus ?case\nby simp\nnext\ncase (Cons r rs)\nthus ?case\nby auto\nqed\nlemma bot: "\\<exists> rs . s = \\<bottom> \\<star> rs"\nproof (cases s)\ncase None\nhence "s = \\<bottom> \\<star> []"\nby auto\nthus ?thesis\nby blast\nnext\ncase (Some v)\nobtain r where "\\<bottom> \\<star> [r] = Some v"\nby force\nthus ?thesis\nusing Some\nby metis\nqed**[/INST]'\nusing assms single_use \nproof - \n{ \nassume 1:"s1 \\<noteq> None" and 2:"s1 \\<noteq> s2" \nobtain r rs where 3:"s1 = \\<bottom> \\<star> ([r]@rs)" \nusing bot \nusing 1 \nby (metis append_butlast_last_id pre_RDR.exec.simps(1)) \nobtain rs\' where 4:"s2 = s1 \\<star> rs\'" \nusing assms \nby (auto simp add:less_eq_def) \nhave "s2 = \\<bottom> \\<star> ([r]@(rs@rs\'))" \nusing 3 4 \nby (metis exec_append) \nhence "s1 = s2" \nusing 3 \nby (metis single_use) \nwith 2 \nhave False \nby auto \n} \nthus ?thesis \nby blast \nqed \ninterpretation RDR \\<delta> \\<gamma> \\<bottom> \nproof (unfold_locales) \nfix s r \nassume "contains s r" \nshow "s \\<bullet> r = s" \nproof - \nobtain rs where "s = \\<bottom> \\<star> rs" and "rs \\<noteq> []" \nusing \\<open>contains s r\\<close> \nby (auto simp add:contains_def, force) \nthus ?thesis \nby (metis \\<delta>.simps(2) rev_exhaust single_use) \nqed \nnext \nfix s and r r\' :: "proc \\<times> val" \nassume 1:"fst r \\<noteq> fst r\'" \nthus "\\<gamma> s r = \\<gamma> ((s \\<bullet> r) \\<bullet> r\') r" \nby (metis \\<delta>.simps \\<gamma>.simps not_Some_eq) \nnext \nfix s1 s2 \nassume "s1 \\<preceq> s2 \\<and> s2 \\<preceq> s1" \nthus "s1 = s2" \nby (metis prec_eq_None_or_equal) \nnext \nfix s1 s2 \nshow "\\<exists> s . is_glb s s1 s2" \nby (simp add:is_glb_def is_lb_def)\n    (metis bot pre_RDR.less_eq_def prec_eq_None_or_equal) \nnext \nfix s \nshow "\\<bottom> \\<preceq> s" \nby (metis bot pre_RDR.less_eq_def) \nnext \nfix s1 s2 s3 rs \nassume "s1 \\<preceq> s2" and "s2 \\<preceq> s3" and "s3 = s1 \\<star> rs" \nthus "\\<exists> rs\' rs\'\' . s2 = s1 \\<star> rs\' \\<and> s3 = s2 \\<star> rs\'\' \n    \\<and> set rs\' \\<subseteq> set rs \\<and> set rs\'\' \\<subseteq> set rs" \nby (metis Consensus.prec_eq_None_or_equal \n      in_set_insert insert_Nil list.distinct(1)\n        pre_RDR.exec.simps(1) subsetI) \nqed \nend \nend'"\</s>*

The generated datasets are published in the following links in Huggingface: 
https://huggingface.co/datasets/jcrecio/AFP_Cot_Contextualized_Proofs
https://huggingface.co/datasets/jcrecio/afp_training_test
https://huggingface.co/datasets/jcrecio/AFP_Theories

## 1. Finetune

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

:warning: for UMA AI machine, create a virtual env before installing dependencies:
python3 -m venv <env>
source <env>/bin/activate


pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q trl xformers wandb datasets einops sentencepiece
pip install python-dotenv
```

Run the command for finetune
```
python stages/1_finetune.py
```


### Offline (deprecated, not in use)

Our use case for an offline finetune operation is based on the Spanish Supercomputing Network (RES), which has partial access to the network to clone git repositories.

1. Download the dataset https://huggingface.co/datasets/jcrecio/afp_mistral
2. Install Git LFS to be able to clone the Mistral Model repository that contains large binary https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.mdfiles
3. Run `git lfs install` in the folder
4. Run `git clone https://huggingface.co/mistralai/<model>` 



### Reasoning Finetune
1. Install rest of previous dependencies but also add unsloth
```
    pip install unsloth
    pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

And to run it:
`python reasoning_stages/1_finetune.py`

### RAG inference
```
    pip install -q torch transformers accelerate bitsandbytes langchain sentence-transformers faiss-cpu openpyxl pacmap datasets langchain-community ragatouille
```

## 2. Merge and push the new model to Huggingface

Previously, finetuning the base model generated the delta weights on top of the base model.
After that we will merge both the base model plus the refined knowledge (weights) and push it to HF to make it available.

Note: **Make sure you install first LFS with the command `git install lfs`**

Set the environmental variables MODEL_TO_USE, NEW_MODEL and TOKENIZER:
```
MODEL_TO_USE=mistralai/Mathstral-7B-v0.1
NEW_MODEL=jcrecio/isamath-v0.1 # isamath is the name we put to the finetuned model
TOKENIZER=jcrecio/isamath-tokenizer-v0.1
```

Run: 
```
python stages/2_push_model.py
```
## 3. Run the model

In order to run the model interactively you have to execute the following command:
```
python stages/3_run_model.py <model_name> <device mode>
```
- \<model name> is the model to run, in our case `jcrecio-isamath-v0.1`
- \<device mode> can be gpu, cpu, half or low.
    - cpu: Infer using CPU
    - cuda: Infer using GPU
    - half: Infer using GPU with half precision
    - low: Infer using GPU with low CPU memory usage

There is a console application that will ask for an optional context, and for the problem to infer a proof from.

## 4. Bulk-run the model
In order to run the modal in bulk with a lot of samples (from the dataset jcrecio/AFP_Theories).

```
python stages/4_bulk_inference.py <model_name> <device mode> <number of problems to solve (from dataset)>
```

:warning: Trick for Picasso machine: 
Activate a conda environment and install the dependencies with pip.
`conda create -n jgpu python=3.9`

`conda activate jgpu`
```
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q trl xformers wandb datasets einops sentencepiece
pip install python-dotenv
```
## 5. Verifying proofs
Make sure you have installed Isabelle, you can download it from the AFP page and add it to the PATH
Also install all the dependencies (maybe also Nominal2)
```
isabelle components -a
isabelle build -a
isabelle build -b Nominal2
```