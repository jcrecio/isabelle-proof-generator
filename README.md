# Isabelle-proof-generator

## Create the dataset

### JSON Dataset
In order to generate a dataset to finetune a model to infer Isabelle/HOL proofs we need to start with the PISA Dataset from https://github.com/albertqjiang/Portal-to-ISAbelle

This dataset contains the AFP problems separated in folders.

Set the following environment variables:
```
PROBLEMS_FOLDER=<root folder for the problems>
OUTPUT_FILE_DATASET=<output dataset file .json>
```

Run the script `isabelle_proofs_dataset_creator` to generate the dataset as a json file.

Run the script `proofs_to_jsonl.py <dataset json file>` to get the file in jsonl format.

Run the script `proofs_to_mistral.py <dataset jsonl file>` to get the file in the format Mistal expects for finetuning.


## Finetune



1. Install virtualenv in the machine
2. Go to the root of the project and run:
```
    python3 -m venv venv_isabelle
```
3. Activate the environment
```
    source venv_isabelle/bin/activate
```
4. Create an .env file with the following variables
```
HF_TOKEN=<token to connect to HuggingFace>
WANDB_TOKEN=<token to register and track in WandDb>
```
5. Install the following libraries only for the first time in the environment
```
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q trl xformers wandb datasets einops sentencepiece
```
6. Run the command for finetune
```
    python isabelle-proof-generator/finetune.py
```