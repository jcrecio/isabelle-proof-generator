import json
import os
import re
import sys
from typing import Any, List, Optional, TextIO
from datetime import datetime

VERBOSE = True


def write_json_file(json_file_path: str, content: str):
    with open(json_file_path, "w") as file:
        json.dump(content, file)


def read_file(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def read_json_file(json_file_path: str):
    with open(json_file_path, "r") as file:
        return json.load(file)


def log(
    *values: Any,
    sep: str = " ",
    end: str = "\n",
    file: Optional[TextIO] = None,
    flush: bool = False,
    with_time: bool = True,
) -> None:

    if not VERBOSE:
        return
    message = sep.join(str(value) for value in values)

    timestamp = f"[{datetime.now()}]" if with_time else ""
    formatted_message = f"{timestamp} {message}"

    output_file = file if file is not None else sys.stdout

    print(formatted_message, end=end, file=output_file, flush=flush)


def get_subfolders(folder: str) -> List[str]:
    return [f.path for f in os.scandir(folder) if f.is_dir()]


def get_files_in_folder(folder: str) -> List[str]:
    return [f.path for f in os.scandir(folder) if f.is_file()]


def extract_theory_name(extractions_file_name, prefix=None):
    try:
        parts = extractions_file_name.split("_ground_truth")
        if len(parts) != 2:
            return None

        pre_ground = parts[0]

        if prefix:
            if prefix not in pre_ground:
                return None
            result = pre_ground.split(prefix)[-1]
            return result.lstrip("_")
        else:
            return pre_ground.split("_")[-1]

    except Exception as _:
        return None


def extract_theories_files(content: str) -> List[str]:
    theories_pattern = r"theories\s*(.*?)(?=\s+document_files|\s*$)"
    theories_match = re.search(theories_pattern, content, re.DOTALL)

    if not theories_match:
        return []

    theories_text = theories_match.group(1)
    theories = [theory.strip() for theory in theories_text.split() if theory.strip()]

    theory_files = [f"{theory}.thy" for theory in theories]
    return theory_files


def find_lemma_index_in_translations(lemma, translations):
    for index, pair in enumerate(translations):
        if pair[1] == lemma:
            return index
    return -1


def get_lemmas_proofs_for_file(extraction_file_path: str):
    file_content = read_json_file(extraction_file_path)
    translations = file_content.get("translations")
    if translations is None or len(translations) == 0:
        return []

    problem_names = file_content.get("problem_names")
    problem_names_len = len(problem_names)

    lemmas_with_proofs = []

    index = 0
    lemma = None

    while index < problem_names_len:
        lemma = problem_names[index]
        if lemma[0:5] == "lemmas":
            index += 1
            continue

        current_proof = ""

        next_lemma_translations_index = None
        if index + 1 < problem_names_len:
            next_lemma = problem_names[index + 1]
            next_lemma_translations_index = find_lemma_index_in_translations(
                next_lemma, translations
            )
        if next_lemma_translations_index is None:
            next_lemma_translations_index = len(translations)

        lemma_index = find_lemma_index_in_translations(lemma, translations)
        for i in range(lemma_index + 1, next_lemma_translations_index):
            current_proof = f"{current_proof} {translations[i][1]}".strip()

        lemmas_with_proofs.append((lemma, current_proof))
        index += 1

    return lemmas_with_proofs


def generate_all_predocuments(afp_extractions_folder, afp_extractions_original):
    predocuments = []

    sessions = get_subfolders(afp_extractions_folder)

    for session in sessions:
        session_name = session.split("/")[-1]
        theory_files = get_files_in_folder(session)
        for theory_file in theory_files:
            theory_name = extract_theory_name(
                theory_file.split("/")[-1], session.split("/")[-1]
            )
            original_theory_file = (
                f"{afp_extractions_original}/thys/{session_name}/{theory_name}.thy"
            )

            if not os.path.exists(original_theory_file):
                log(f"Original theory not found: {original_theory_file}")
            else:
                log(f"Processing theory: {theory_file}")
                log("\n")

                lemmas_and_proofs = get_lemmas_proofs_for_file(theory_file)
                for lemma_index, (lemma, ground_proof) in enumerate(lemmas_and_proofs):
                    log(f"Generating source for lemma: {lemma}")
                    original_theory_file = f"{afp_extractions_original}/thys/{session_name}/{theory_name}.thy"

                    source = {
                        "theory_file": original_theory_file,
                        "theory_name": theory_name,
                        "lemma": lemma,
                    }

                    content = ground_proof

                    predocuments.append({"source": source, "content": content})
    return predocuments


# afp_extractions_folder = (
#     os.getenv("PROBLEMS_FOLDER")
#     or "/home/jcrecio/repos/isabelle_server/isabelle-proof-generator/afp_extractions/afp_extractions"
# )
# afp_extractions_original = (
#     os.getenv("AFP_FOLDER")
#     or "/home/jcrecio/repos/isabelle_server/isabelle-proof-generator/afp-current-extractions"
# )
afp_extractions_folder = "/home/jcrecio/repos/isabelle_server/isabelle-proof-generator/afp_extractions/afp_extractions"
afp_extractions_original = "/home/jcrecio/repos/isabelle_server/isabelle-proof-generator/afp-current-extractions"

all_predocuments = generate_all_predocuments(
    afp_extractions_folder, afp_extractions_original
)

with open("rag_predocuments.jsonl", "w") as f:
    for predocument in all_predocuments:
        f.write(json.dumps(predocument) + "\n")
