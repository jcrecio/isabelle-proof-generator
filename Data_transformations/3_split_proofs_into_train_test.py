import json
import random
import argparse
from pathlib import Path


def split_jsonl(
    input_file: str, train_file: str, test_file: str, train_ratio: float = 0.7
):
    """
    Split a JSONL file into two files with random sampling.

    Args:
        input_file (str): Path to input JSONL file
        train_file (str): Path to output training file (70% of data)
        test_file (str): Path to output test file (30% of data)
        train_ratio (float): Ratio of data to put in training file (default: 0.7)
    """
    # Read all lines from input file
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Randomly shuffle the lines
    random.shuffle(lines)

    # Calculate split point
    split_idx = int(len(lines) * train_ratio)

    # Split into train and test sets
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    # Write training file
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    # Write test file
    with open(test_file, "w", encoding="utf-8") as f:
        f.writelines(test_lines)

    # Print statistics
    print(f"Total lines: {len(lines)}")
    print(
        f"Training lines: {len(train_lines)} ({len(train_lines)/len(lines)*100:.1f}%)"
    )
    print(f"Test lines: {len(test_lines)} ({len(test_lines)/len(lines)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Split JSONL file into train and test sets"
    )
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument(
        "--train-file",
        default="train.jsonl",
        help="Output training file (default: train.jsonl)",
    )
    parser.add_argument(
        "--test-file",
        default="test.jsonl",
        help="Output test file (default: test.jsonl)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio of data to use for training (default: 0.7)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Validate ratio is between 0 and 1
    if not 0 < args.train_ratio < 1:
        raise ValueError("Train ratio must be between 0 and 1")

    split_jsonl(args.input_file, args.train_file, args.test_file, args.train_ratio)


if __name__ == "__main__":
    main()
