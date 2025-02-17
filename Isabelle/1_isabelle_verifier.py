# How to run: python 1_isabelle_verifier.py UNSLOTH(True, False) BASE_ONLY(True, False) BASE_MODEL LORA_MODEL(N/A for nothing) PROMPT(MATH/REASONING)


import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from timeit import Timer
from typing import Any, List, Optional, TextIO
from pathlib import Path
from datetime import datetime
import sys
import datasets
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset, load_dataset
from tqdm.std import tqdm as tqdm_std
from transformers import AutoTokenizer
from transformers import pipeline
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import torch
import pickle
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

VERBOSE = True
ISABELLE_PATH = "/home/jcrecio/repos/Isabelle2024/bin/isabelle"
ISABELLE_COMMAND = f"{ISABELLE_PATH} build -D"
# ISABELLE_COMMAND = "isabelle build -D"

MODEL: Any = None
TOKENIZER: Any = None


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


def verify_isabelle_session(project_folder: str):
    command_string = f"{ISABELLE_COMMAND} {project_folder}"
    command = convert_to_shell_command(command_string)

    log("<br>")
    log(f"<b>Verifying Isabelle project... {project_folder.split('/')[-1]}</b>")
    output = run_command_with_output(command)
    if "error" in output[1]:
        return [False, output[1]]
    return [True, output[1]]


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


def replace_lemma_proof(
    content: str, current_lemma: str, next_lemma: str, new_proof: str
) -> str:
    """there is a bug in this method"""
    lines = [line.strip() for line in content.split("\n")]

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


KNOWLEDGE_VECTOR_DATABASE = None
embedding_model = None


def load_rag():
    repo_id = "jcrecio/afp_rag"

    faiss_path = hf_hub_download(repo_id=repo_id, filename="index.faiss")
    pkl_path = hf_hub_download(repo_id=repo_id, filename="index.pkl")

    with open(pkl_path, "rb") as f:
        index_metadata = pickle.load(f)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local("faiss_index", embedding_model)
    KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(faiss_path, embedding_model)


def verify_all_sessions(afp_extractions_folder, afp_extractions_original):
    sessions = get_subfolders(afp_extractions_folder)

    successes = 0
    failures = 0

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
                    log(f"Processing lemma: {lemma}")

                    original_theory_file = f"{afp_extractions_original}/thys/{session_name}/{theory_name}.thy"
                    backup_original_theory_file = f"{afp_extractions_original}/thys/{session_name}/{theory_name}_backup.thy"
                    theory_content = read_file(original_theory_file)
                    generated_proof = infer_proof(lemma)[0]

                    print(f"<b>Ground proof:</b><pre><code>{ground_proof}</code></pre>")
                    print(
                        f"<b>Generated proof:</b><pre><code>{generated_proof}</code></pre>"
                    )
                    # generated_proof_without_tags = generated_proof.replace(
                    #     "['<｜begin▁of▁sentence｜>", ""
                    # ).replace("['<｜end▁of▁sentence｜>", "")

                    new_theory_content = theory_content.replace(
                        ground_proof, generated_proof
                    )
                    next_lemma = None
                    if (lemma_index + 1) < len(lemmas_and_proofs):
                        next_lemma = lemmas_and_proofs[lemma_index + 1][0]
                    new_theory_content = replace_lemma_proof(
                        theory_content, lemma, next_lemma, generated_proof
                    )
                    if new_theory_content is None:
                        continue

                    _ = shutil.move(original_theory_file, backup_original_theory_file)
                    create_text_file(original_theory_file, new_theory_content)

                    result = verify_isabelle_session(
                        f"{afp_extractions_original}/thys/{session_name}"
                    )
                if result[0] is False:
                    failures += 1
                    log(f"Error details: {result[1]}", file=sys.stderr)
                else:
                    successes += 1

        result = verify_isabelle_session(session)
        if result[0] == "error":
            log(f"Error details: {result[1]}", file=sys.stderr)


WITH_RAG = False
PROMPT_STYLE = "Math"

EMBEDDING_MODEL_NAME = "thenlper/gte-large"


def load_model():
    if len(sys.argv) != 5:
        print("Run the programm as follows: \n")
        print(
            "python 1_isabelle_verifier.py UNSLOTH(True, False) BASE_ONLY(True, False) BASE_MODEL LORA_MODEL(N/A for nothing)"
        )
        exit

    print(sys.argv[0])
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
    print(sys.argv[4])
    print(sys.argv[5])

    UNSLOTH = True if sys.argv[1] == "True" else False
    BASE_ONLY = True if sys.argv[2] == "True" else False
    base_model_name = sys.argv[3]
    model_name = sys.argv[4]
    PROMPT_STYLE = sys.argv[5]

    if UNSLOTH:
        from langchain.docstore.document import Document as LangchainDocument
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores.utils import DistanceStrategy
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            base_model_name,
            base_model_name=4096,
            dtype=None,  # Uses bfloat16 if available, else float16
            load_in_4bit=True,  # Enable 4-bit quantization
        )
        FastLanguageModel.for_inference(model)
        return tokenizer
    else:
        base_model_name = base_model_name
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            token=hf_token,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if BASE_ONLY:
            return base_model, tokenizer

        adapter_path = model_name
        model = PeftModel.from_pretrained(base_model, adapter_path)
        return model, tokenizer


math_prompt_style = """
You are now an specialized agent to infer proofs for problems, theorem statements or lemmas written in Isabelle/HOL.
[INST]Infer a proof for the following Isabelle/HOL lemma: {}. Answer only with an Isabelle/HOL proof.[/INST]
{}
"""

reasoning_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are now an specialized agent to infer proofs for problems, theorem statements and lemmas written in Isabelle/HOL.
Infer a proof for the following Isabelle/HOL theorem statement.

### Theorem statement:
{}

### Proof:
<think>{}"""


def infer_proof(theorem_statement, device="cuda"):
    prompt_to_use = (
        math_prompt_style if PROMPT_STYLE == "Math" else reasoning_prompt_style
    )

    if WITH_RAG:
        load_rag()

        prompt_to_use = (
            math_prompt_style if PROMPT_STYLE == "Math" else reasoning_prompt_style
        )

        math_prompt_style
        inputs = TOKENIZER(
            [prompt_to_use.format(theorem_statement, "")],
            return_tensors="pt",
        ).to(device)

        outputs = MODEL.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=4096,
            use_cache=True,
        )
        response = TOKENIZER.batch_decode(outputs)
        return response
    else:
        inputs = TOKENIZER(
            [prompt_to_use.format(theorem_statement, "")],
            return_tensors="pt",
        ).to(device)

        outputs = MODEL.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=4096,
            use_cache=True,
        )
        response = TOKENIZER.batch_decode(outputs)
        return response


MODEL, TOKENIZER = load_model()
verify_all_sessions(
    "/home/jcrecio/repos/isabelle-proof-generator/afp_extractions/afp_extractions",
    "/home/jcrecio/repos/isabelle-proof-generator/afp-current-extractions",
)
