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

Run the `script proofs_to_jsonl.py <dataset json file>` to get the file in jsonl format.

