import re
from pathlib import Path
from typing import List, Optional

thy_folder = "C:\repos\isabelle-proof-generator\afp-current\afp-2024-03-04\thys\Abortable_Linearizable_Modules"


def read_root_file(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def extract_theories(content: str) -> List[str]:
    theories_pattern = r"theories\s*((?:[^\s]+\s*)*)"
    theories_match = re.search(theories_pattern, content)

    if not theories_match:
        return []

    theories_text = theories_match.group(1)
    theories = [theory.strip() for theory in theories_text.split()]

    return [f"{theory}.thy" for theory in theories if theory]


def process_root_file(file_path: str) -> List[str]:
    content = read_root_file(file_path)
    if content is None:
        return []

    return extract_theories(content)


if __name__ == "__main__":
    file_path = f"{thy_folder}/ROOT"

    theories = process_root_file(file_path)

    if theories:
        print("Extracted theories:")
        for theory in theories:
            print(f"  {theory}")
    else:
        print("No theories found or error processing file")
