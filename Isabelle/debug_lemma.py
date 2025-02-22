import json
import os
import shutil
import signal
import subprocess
import sys
from threading import Timer
import time
import uuid

ISABELLE_PATH = "/home/jcrecio/repos/isabelle_server/Isabelle2024/bin"
ISABELLE_COMMAND = f"{ISABELLE_PATH} build -D"
afp_extractions_original = "/home/jcrecio/repos/isabelle_server/isabelle-proof-generator/afp-current-extractions"
session_name = "HOL-Analysis"
theory_name = "ADS_Construction"
theory_file = "/home/jcrecio/repos/isabelle_server/isabelle-proof-generator/afp_extractions/afp_extractions/ADS_Functor/_home_qj213_afp-2021-10-22_thys_ADS_Functor_ADS_Construction_ground_truth.json"


def read_file(file_path: str):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def read_json_file(json_file_path: str):
    with open(json_file_path, "r") as file:
        return json.load(file)


# original_theory_file = (
#     f"{afp_extractions_original}/thys/{session_name}/{theory_name}.thy"
# )

original_theory_file = "/home/jcrecio/repos/isabelle_server/isabelle-proof-generator/afp-current-extractions/thys/ADS_Functor/ADS_Construction.thy"
backup_original_theory_file = "/home/jcrecio/repos/isabelle_server/isabelle-proof-generator/afp-current-extractions/thys/ADS_Functor/ADS_Construction_backup.thy"
theory_content = read_file(original_theory_file)
generated_proof = "by unfold_locales(auto simp add: merge_discrete_def)"
lemma_index = 2
lemma = 'lemma merge_on_discrete [locale_witness]:\n  "merge_on UNIV hash_discrete blinding_of_discrete merge_discrete"'


def find_lemma_index_in_translations(lemma, translations):
    for index, pair in enumerate(translations):
        if pair[1] == lemma:
            return index
    return -1


def find_text_and_next_line(content, search_text):
    start_pos = content.find(search_text)
    if start_pos == -1:
        return None

    end_pos = start_pos + len(search_text)

    next_line_start = content.find("\n", end_pos) + 1
    if next_line_start == 0:
        return None

    next_line_end = content.find("\n", next_line_start)
    if next_line_end == -1:
        next_line_end = len(content)

    return next_line_start


def find_text_and_next_line2(content, search_text):
    pos = content.find(search_text)
    if pos == -1:
        return None

    return pos


def replace_lemma_proof2(
    content: str, current_lemma: str, next_lemma: str, new_proof_raw: str
) -> str:

    # If the lemma is part of the generated proof, leave it, otherwise add it to the proof
    new_proof = (
        new_proof_raw
        if current_lemma in new_proof_raw
        else f"{current_lemma} {new_proof_raw}"
    )

    # start and end lines where to remove
    start_pos = find_text_and_next_line2(content, current_lemma)
    end_pos = find_text_and_next_line2(content, next_lemma)

    result_content = content[:start_pos] + new_proof + content[end_pos:]

    return result_content


def replace_lemma_proof(
    content: str, current_lemma: str, next_lemma: str, new_proof_raw: str
) -> str:
    # If the lemma is part of the generated proof, remove it from the proof
    new_proof = (
        new_proof_raw.replace(current_lemma, "")
        if current_lemma in new_proof_raw
        else new_proof_raw
    )

    lines = [line for line in content.split("\n")]

    start_idx = -1
    try:
        start_idx = lines.index(current_lemma)
    except ValueError:
        # not finding it directly then try to find in the whole file content
        # raise ValueError(f"Start line '{start_line}' not found in content")
        start_idx = find_text_and_next_line(content, current_lemma)

    if start_idx is None:
        return None

    proof_idx = None
    for i in range(start_idx + 1, len(lines)):
        if next_lemma == lines[i]:
            proof_idx = i
            break

    if proof_idx is None:
        duplicate_lemma_new_proof(content, current_lemma, new_proof)

    result_lines = lines[: start_idx + 1] + [new_proof] + lines[proof_idx:]

    return "\n".join(result_lines)


def duplicate_lemma_new_proof(content: str, current_lemma: str, new_proof: str) -> str:
    lines = [line.strip() for line in content.split("\n")]

    start_idx = -1
    try:
        start_idx = lines.index(current_lemma)
    except ValueError:
        # not finding it directly then try to find in the whole file content
        # raise ValueError(f"Start line '{start_line}' not found in content")
        start_idx = find_text_and_next_line(content, current_lemma)
    start_idx = start_idx - 1
    lines.insert(start_idx, f"lemma {str(uuid.uuid4())} {current_lemma[5:]}")
    lines.insert(start_idx + 1, f"{new_proof}")
    return "\n".join(lines)


def create_text_file(filename, content):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error creating file: {e}")
        return False


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
                print(output.rstrip())
                full_output.append(output)
                sys.stdout.flush()

            if error:
                print(error.rstrip(), file=sys.stderr)
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


def verify_isabelle_session(project_folder: str):
    command_string = f"{ISABELLE_COMMAND} {project_folder}"
    command = convert_to_shell_command(command_string)

    print("<br>")
    print(f"<b>Verifying Isabelle project... {project_folder.split('/')[-1]}</b>")
    print("<br>")
    output = run_command_with_output(command)
    if len(output) == 0 or len(output) < 2:
        return ["inconclusive", output]
    if (
        "error" in output[1]
        or "Error" in output[1]
        or "Malformed" in output[1]
        or "malformed" in output[1]
        or "Invalid" in output[1]
        or "invalid" in output[1]
        or "denied" in output[1]
    ):
        return ["error", output[1]]
    return ["success", output[1]]


lemmas_and_proofs = get_lemmas_proofs_for_file(theory_file)

next_lemma = None
if (lemma_index + 1) < len(lemmas_and_proofs):
    next_lemma = lemmas_and_proofs[lemma_index + 1][0]
new_theory_content = replace_lemma_proof2(
    theory_content, lemma, next_lemma, generated_proof
)
if new_theory_content is None:
    exit

_ = shutil.move(original_theory_file, backup_original_theory_file)
create_text_file(original_theory_file, new_theory_content)

result = verify_isabelle_session(f"{afp_extractions_original}/thys/{session_name}")

os.remove(original_theory_file)
_ = shutil.move(backup_original_theory_file, original_theory_file)
