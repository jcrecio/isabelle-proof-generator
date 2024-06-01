# Isabelle-proof-generator

## Create the dataset

### JSON Dataset
In order to generate a dataset to finetune a model to infer Isabelle/HOL proofs we need to start with the PISA Dataset from https://github.com/albertqjiang/Portal-to-ISAbelle

This dataset contains the AFP problems separated in folders.

Set the following environment variables:
```
PROBLEMS_FOLDER=<root folder for the problems>
OUTPUT_FILE_DATASET=<output dataset file .json>
USE_MODELS_OFFLINE=<true/false>
USE_WANDB=<true/false>
```

Run the script `isabelle_proofs_dataset_creator` to generate the dataset as a json file.

Run the script `proofs_to_jsonl.py <dataset json file>` to get the file in jsonl format.

Run the script `proofs_to_mistral.py <dataset jsonl file>` to get the file in the format Mistal expects for finetuning.

The generated dataset is published in the following link in Huggingface: https://huggingface.co/datasets/jcrecio/afp_mistral

## Finetune

The finetune will be using the dataset we generated previously on https://huggingface.co/datasets/jcrecio/afp_mistral or you can create a different dataset of your own.

Finetuning can be setup differently depending on the machine to be run.

### Online

If the machine has full access to the network, it will have direct access to datasets and models.

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
4. Run `git clone https://huggingface.co/mistralai/Mistral-7B-v0.1`

