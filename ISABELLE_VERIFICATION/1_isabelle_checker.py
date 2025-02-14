import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from timeit import Timer
from typing import Any, List, Optional, TextIO
from pathlib import Path
from datetime import datetime

VERBOSE = True
ISABELLE_PATH = "/home/jcrecio/repos/isabelle_server/Isabelle2024/bin/isabelle"
ISABELLE_COMMAND = f"{ISABELLE_PATH} build -D"
# ISABELLE_COMMAND = "isabelle build -D"


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


def convert_to_command(command: str):
    return command.split()


def convert_to_shell_command(command: str):
    return [command]


def run_command_with_output(command, execution_dir=None, timeout=None):
    full_output = []
    full_error = []

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=execution_dir,
        env=os.environ.copy(),
        shell=True,
        preexec_fn=None if timeout is None else os.setsid,
    )

    def kill_process():
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            time.sleep(1)
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = None
    if timeout:
        timer = Timer(timeout, kill_process)
        timer.start()

    try:
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()

            if not output and not error and process.poll() is not None:
                break

            if output:
                log(output.rstrip())
                full_output.append(output)
                sys.stdout.flush()

            if error:
                log(error.rstrip(), file=sys.stderr)
                full_error.append(error)
                sys.stderr.flush()

        return_code = process.poll()

        if timer and timer.is_alive():
            timer.cancel()

        if return_code is None:
            raise TimeoutError(f"Command execution timed out after {timeout} seconds")

        return return_code, "".join(full_output), "".join(full_error)

    finally:
        if timer:
            timer.cancel()
        if process.poll() is None:
            kill_process()


def read_json_file(json_file_path: str):
    with open(json_file_path, "r") as file:
        return json.load(file)


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


def extract_theories_files(content: str) -> List[str]:
    theories_pattern = r"theories\s*(.*?)(?=\s+document_files|\s*$)"
    theories_match = re.search(theories_pattern, content, re.DOTALL)

    if not theories_match:
        return []

    theories_text = theories_match.group(1)
    theories = [theory.strip() for theory in theories_text.split() if theory.strip()]

    theory_files = [f"{theory}.thy" for theory in theories]
    return theory_files


def get_subfolders(folder: str) -> List[str]:
    return [f.path for f in os.scandir(folder) if f.is_dir()]


def get_files_in_folder(folder: str) -> List[str]:
    return [f.path for f in os.scandir(folder) if f.is_file()]


def process_root_file(file_path: str) -> List[str]:
    content = read_file(file_path)
    if content is None:
        return []

    return extract_theories_files(content)


def verify_isabelle_session(project_folder: str) -> bool:
    command_string = f"{ISABELLE_COMMAND} {project_folder}"
    command = convert_to_shell_command(command_string)

    log("\n")
    log(f"Verifying Isabelle project... {project_folder.split('/')[-1]}")
    output = run_command_with_output(command)
    if "error" in output[1]:
        return ["error", output[1]]
    return ["success", output[1]]


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


def infer_proof(lemma):
    return ""


def replace_lemma_proof(content: str, start_line: str, new_proof: str) -> str:
    lines = content.split("\n")

    try:
        start_idx = lines.index(start_line)
    except ValueError:
        raise ValueError(f"Start line '{start_line}' not found in content")

    proof_idx = None
    for i in range(start_idx + 1, len(lines)):
        if "proof" in lines[i]:
            proof_idx = i
            break

    if proof_idx is None:
        raise ValueError("No 'proof' keyword found after the specified start line")

    result_lines = lines[: start_idx + 1] + [new_proof] + lines[proof_idx:]

    return "\n".join(result_lines)


def find_string_in_file(filename, search_string, case_sensitive=True):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            if not case_sensitive:
                search_string = search_string.lower()

            for line_number, line in enumerate(file, 1):
                current_line = line.rstrip()
                compare_line = current_line if case_sensitive else current_line.lower()

                if search_string in compare_line:
                    return line_number, current_line

        return None, None

    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{filename}' was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {str(e)}")


def create_text_file(filename, content):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error creating file: {e}")
        return False


"""
    afp_extractions_folder: Folder with the PISA dataset with all the lemmas separated
    afp_extractions_original: Folder with the original theories from AFP
"""


def verify_all_sessions(afp_extractions_folder, afp_extractions_original):
    sessions = get_subfolders(afp_extractions_folder)

    for session in sessions:
        session_name = session.split("/")[-1]
        theory_files = get_files_in_folder(session)
        for theory_file in theory_files:
            theory_name = theory_file.split("/")[-1].replace(".thy", "")
            theory_original = f"{afp_extractions_original}/{theory_name}.thy"

            if not os.path.exists(theory_original):
                log(f"Original theory not found: {theory_original}")
            else:
                log(f"Processing theory: {theory_file}")
                log("\n")

                lemmas_and_proofs = get_lemmas_proofs_for_file(theory_file)
                for lemma, ground_proof in lemmas_and_proofs:
                    log(f"Processing lemma: {lemma}")
                    original_theory_file = f"{afp_extractions_original}/thys/{session_name}/{theory_file}.thy"
                    backup_theory_file = f"{afp_extractions_original}/thys/{session_name}/{theory_file}.thy"
                    shutil.copy(theory_file, backup_theory_file)

                    generated_proof = infer_proof(lemma)
                    theory_content = read_file(theory_file)
                    line_num, line_content = find_string_in_file(
                        backup_theory_file, lemma
                    )
                    new_theory_content = replace_lemma_proof(
                        theory_content, line_num, generated_proof
                    )
                    create_text_file(original_theory_file, new_theory_content)

                    result = verify_isabelle_session(session)
                if result[0] == "error":
                    log(f"Error details: {result[1]}", file=sys.stderr)

        result = verify_isabelle_session(session)
        if result[0] == "error":
            log(f"Error details: {result[1]}", file=sys.stderr)


if __name__ == "__main__":
    verify_all_sessions(
        "/home/jcrecio/repos/isabelle_server/isabelle-proof-generator/afp_extractions/afp_extractions",
        "isabelle-proof-generator/afp-current-extractions",
    )
