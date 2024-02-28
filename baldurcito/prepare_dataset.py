import re
from datasets import load_dataset
import pandas as pd
import jsonlines

def transform(entry, pattern):
    match = re.match(pattern, entry)

    if match:
        title = match.group(1)
        question = match.group(2)
        reply = match.group(3)

        output_string = f"Conversation between a Human and a proofer assistant [INS] {title}: {question} [/INS] {reply}"

        return output_string
    else:
        return "Input format not recognized."

def transform_question(question):
    pattern = (
        r"TITLE: (.+) QUESTION \[(\d+) upvotes\]: (.+) REPLY \[(\d+) votes\]: (.+)"
    )
    return transform(question, pattern)

def transform_paper(paper):
    pattern = r"\\begin{document} \\title\{(.+?)\} (.+?)\\bibliographystyle(.+)"
    return transform(paper, pattern)

def map_sample(sample):
    try:
        text = sample.get("text")
        meta = sample.get("meta")

        if "question_id" in meta:
            new_text = transform_question(text)
            new_sample = {"text": new_text, "meta": meta}
            return new_sample
        elif "config" in meta:
            if text.startswith("\begin"):
                new_text = transform_paper(text)
                new_sample = {"text": new_text, "meta": meta}
                return new_sample

        return sample

    except Exception as e:
        print("transform_sample error " + e)
        return sample

def save_dataset_as_jsonl(dataset, filename):
    with jsonlines.open(filename + ".jsonl", mode="w") as writer:
        for _, row in pd.DataFrame(dataset).iterrows():
            transformed_row = map_sample(row.to_dict())
            writer.write(transformed_row)

def main():

    # Training
    dataset = load_dataset(
        "hoskinson-center/proof-pile",
        # streaming=True,
        split="train",
        trust_remote_code=True,
    )
    save_dataset_as_jsonl(dataset, 'dataset-train')



    # Testing
    dataset = load_dataset(
        "hoskinson-center/proof-pile",
        # streaming=True,
        split="test",
        trust_remote_code=True
    )
    save_dataset_as_jsonl(dataset, 'dataset-test')



    # Validation
    dataset = load_dataset(
        "hoskinson-center/proof-pile",
        # streaming=True,
        split="validation",
        trust_remote_code=True
    )
    save_dataset_as_jsonl(dataset, 'dataset-validation')

if __name__ == "__main__":
    main()
