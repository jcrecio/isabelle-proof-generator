import sys
import jsonlines
import random

def sample_jsonl(input_file, output_file, percentage_str):
    percentage = int(percentage_str)
    with jsonlines.open(input_file, 'r') as reader:
        lines = list(reader)
        num_lines = len(lines)
        num_samples = int(num_lines * percentage / 100)
        sampled_lines = random.sample(lines, num_samples)

    with jsonlines.open(output_file, 'w') as writer:
        for line in sampled_lines:
            writer.write(line)

sample_jsonl(sys.argv[1], sys.argv[2], sys.argv[3])