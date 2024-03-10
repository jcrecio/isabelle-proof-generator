import json
import jsonlines
from isabelle_text2mistral import isabelle2mistral

def convert_dataset_to_mistral(input_filepath: str, output_filepath):
    result = []
    with open(input_filepath, 'r',  encoding='utf-8') as input:
        with jsonlines.open(output_filepath + ".jsonl", mode="w") as output:
            for line in input:
                input_line = json.loads(line)
                text = input_line.get('text')
                [question, answer] = isabelle2mistral(text)
                output_line = { "question": question, "answer": answer}
                output.write(output_line)

    return result
convert_dataset_to_mistral('.\\afp-dataset-train.jsonl','.\\afp-dataset-train-mistral')
convert_dataset_to_mistral('.\\afp-dataset-test.jsonl','.\\afp-dataset-test-mistral')
convert_dataset_to_mistral('.\\afp-dataset-validation.jsonl','.\\afp-dataset-validation-mistral')
