import os
import json


def explore_thy_files(directory):
    """
    Recursively explore a directory for .thy files and create a JSONL file.

    Args:
        directory (str): The root directory to start searching from

    Returns:
        str: Path to the generated JSONL file
    """
    thy_contents = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".thy"):
                full_path = os.path.join(root, file)

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    key = os.path.splitext(file)[0]
                    thy_contents[key] = content

                except Exception as e:
                    print(f"Error reading file {full_path}: {e}")

    output_file = os.path.join(directory, "thy_files_output.jsonl")

    with open(output_file, "w", encoding="utf-8") as jsonl_file:
        for key, value in thy_contents.items():
            json.dump({key: value}, jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")

    print(f"JSONL file created at: {output_file}")
    print(f"Total .thy files processed: {len(thy_contents)}")

    return output_file


def main():
    directory = input("Enter the directory path to explore .thy files: ").strip()

    if not os.path.isdir(directory):
        print("Invalid directory path.")
        return

    explore_thy_files(directory)


if __name__ == "__main__":
    main()
