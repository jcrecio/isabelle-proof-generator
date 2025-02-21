# How to run: python 1_isabelle_verifier.py <model> [RAG]


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
from unsloth import FastLanguageModel
import numpy as np
import pacmap
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

GENERATE = True
VERIFY = False

BEGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Formatting</title>
    <style>
        pre {
            white-space: pre-wrap;
            word-wrap: break-word; 
            overflow-x: auto;
            max-height: 400px;
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
"""

END_TEMPLATE = """
</body>
</html>
"""

WITH_RAG = False

EMBEDDING_MODEL_NAME = "thenlper/gte-large"


model_to_load = sys.argv[1]
RAG = True if len(sys.argv) > 3 and sys.argv[2] == "RAG" else False

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

VERBOSE = False
LOG_TIME = False
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

    if LOG_TIME:
        timestamp = f"[{datetime.now()}]" if with_time else ""
        formatted_message = f"{timestamp} {message}"
    else:
        formatted_message = message

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
        log(f"Error reading file: {e}")
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
    log("<br>")
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
    ):
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
    content: str, current_lemma: str, next_lemma: str, new_proof_raw: str
) -> str:
    # If the lemma is part of the generated proof, remove it from the proof
    new_proof = (
        new_proof_raw.replace(current_lemma, "").strip()
        if current_lemma in new_proof_raw
        else new_proof_raw
    )

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
        log(f"Error creating file: {e}")
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

    except Exception:
        return None


KNOWLEDGE_VECTOR_DATABASE = None
EMBEDDING_PROJECTOR = None
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

    EMBEDDING_PROJECTOR = pacmap.PaCMAP(
        n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1
    )

    return KNOWLEDGE_VECTOR_DATABASE, EMBEDDING_PROJECTOR


def verify_all_sessions(afp_extractions_folder, afp_extractions_original):
    sessions = get_subfolders(afp_extractions_folder)

    per_page = 5  # sessions per page
    page = 1
    accumulated_per_page = 0

    successes = 0
    failures = 0
    inconclusives = 0

    for session in sessions:
        if accumulated_per_page == per_page:
            accumulated_per_page = 0
            page += 1
        with open(f"logfile-{model_to_load}-{page}.html", "a") as log_file:
            if accumulated_per_page == 0:
                log(BEGIN_TEMPLATE, file=log_file)

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
                    log(
                        f"Original theory not found: {original_theory_file}<br>",
                        file=log_file,
                    )
                else:
                    log(f"Processing theory: {theory_file}<br>", file=log_file)
                    log("<br>", file=log_file)

                    lemmas_and_proofs = get_lemmas_proofs_for_file(theory_file)
                    for lemma_index, (lemma, ground_proof) in enumerate(
                        lemmas_and_proofs
                    ):
                        log("<hr><hr><hr><hr>", file=log_file)
                        log(f"<h2>{lemma}</h2><br>", file=log_file)

                        original_theory_file = f"{afp_extractions_original}/thys/{session_name}/{theory_name}.thy"
                        backup_original_theory_file = f"{afp_extractions_original}/thys/{session_name}/{theory_name}_backup.thy"

                        try:
                            theory_content = read_file(original_theory_file)
                            generated_proof = generate_proof(MODEL, TOKENIZER, lemma)

                            if GENERATE:
                                with open(
                                    f"generated_proofs_{model_to_load}.jsonl", "a"
                                ) as f:
                                    f.write(
                                        f"""{ "lemma": lemma, "proof": generated_proof }\n"""
                                    )

                            log(
                                f"<b>Ground proof:</b> <br><pre><code>{ground_proof}</code></pre>",
                                file=log_file,
                            )
                            log(
                                f"<b>Generated proof:</b><pre><code>{generated_proof}</code></pre><br><br>",
                                file=log_file,
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

                            _ = shutil.move(
                                original_theory_file, backup_original_theory_file
                            )
                            create_text_file(original_theory_file, new_theory_content)

                            if VERIFY:
                                result = verify_isabelle_session(
                                    f"{afp_extractions_original}/thys/{session_name}"
                                )

                            log(
                                f"<b>Old content:</b><pre><code>{theory_content}</code></pre><br><br>",
                                file=log_file,
                            )
                            log(
                                f"<b>New content:</b><pre><code>{new_theory_content}</code></pre><br><br>",
                                file=log_file,
                            )

                            if result[0] == "inconclusive":
                                inconclusives += 1
                                log(
                                    f"""
                                    <div style="border:1px solid black">
                                        <span style="color: orange; font-stye: bold">Inconclusive Isabelle/HOL proof.</span><br>
                                        Error details: <br>
                                        <div style="font-style: italic;">{result[1]}</div>
                                    </div><br>
                                    """,
                                    file=log_file,
                                )
                            elif result[0] == "error":
                                failures += 1
                                log(
                                    f"""
                                    <div style="border:1px solid black">
                                        <span style="color: red; font-stye: bold">Failing Isabelle/HOL proof.</span><br>
                                        Error details: <br>
                                        <div style="font-style: italic;">{result[1]}</div>
                                    </div><br>
                                    """,
                                    file=log_file,
                                )
                            elif result[0] == "success":
                                successes += 1
                                log(
                                    """
                                    <div style="border:1px solid black">
                                        <span style="color: green; font-stye: bold">Successful Isabelle/HOL proof.</span>
                                    </div><br>
                                    """,
                                    file=log_file,
                                )
                                # Clean isabelle theory files after verification
                            os.remove(original_theory_file)
                            _ = shutil.move(
                                backup_original_theory_file, original_theory_file
                            )

                            log(
                                f"""
                                Successes: {successes} <-|-> Failures: {failures}<br>
                                """,
                                file=log_file,
                            )
                        except Exception as e:
                            log(
                                f"""
                                <span style="color: red"> Unexpected error: {e}</span?<br>
                                """
                            )


max_seq_length = 2048


def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_to_load,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


math_prompt_style = """Given a theorem in Isabelle/HOL, output ONLY a clean and valid Isabelle/HOL proof without any natural language explanation. I attach one example for you to follow.

Example:

Theorem:
"∀x. x + 0 = x"

Proof:
"by simp"

Theorem:
{theorem}

Proof:
"""


# PROMPT TO INFERENCE JSON STYLE TO PASS THE CONTEXT?
math_prompt_style = """Given a theorem in Isabelle/HOL, output ONLY a clean and valid Isabelle/HOL proof without any natural language explanation. I attach one example for you to follow.

Example:

Theorem:
"∀x. x + 0 = x"

Proof:
"by simp"

Theorem:
{theorem}

Proof:
"""

math_prompt_style_rag = """Given a theorem in Isabelle/HOL and given some context that contains related theorems with their proofs, output ONLY a clean and valid Isabelle/HOL proof without any natural language explanation.

Example:

Theorem:
"∀x. x + 0 = x"

Proof:
"by simp"

Theorem:
{theorem}

Proof:
"""

max_seq_length = 2048


def generate_proof(model, tokenizer, theorem):
    if RAG:
        retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=theorem, k=5)
        retrieved_docs_text = [
            (doc.metadata["source"], {doc.page_content}) for doc in retrieved_docs
        ]

        context = "".join(
            [
                f"Theorem {str(i)}: {doc_source}\n" + doc_content
                for i, (doc_source, doc_content) in enumerate(retrieved_docs_text)
            ]
        )

        formatted_prompt = math_prompt_style_rag.format(
            theorem=theorem, context=context
        )
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_seq_length,
            use_cache=True,
            temperature=0.7,
            top_p=0.95,
        )

        response = tokenizer.batch_decode(outputs)[0]

        proof = response.split("Now provide ONLY the clean Isabelle/HOL proof:")[
            -1
        ].strip()
        proof = (
            proof.replace("<think>", "")
            .replace("</think>", "")
            .replace("<｜end▁of▁sentence｜>", "")
            .replace("```isar", "")
            .replace("```", "")
            .replace("isabelle", "")
            .strip()
        )

        return proof

    else:
        formatted_prompt = math_prompt_style.format(theorem=theorem)
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_seq_length,
            use_cache=True,
            temperature=0.7,
            top_p=0.95,
        )

        response = tokenizer.batch_decode(outputs)[0]

        proof = response.split("Now provide ONLY the clean Isabelle/HOL proof:")[
            -1
        ].strip()
        proof = (
            proof.replace("<think>", "")
            .replace("</think>", "")
            .replace("<｜end▁of▁sentence｜>", "")
            .replace("```isar", "")
            .replace("```", "")
            .replace("isabelle", "")
            .strip()
        )

        return proof


if RAG:
    KNOWLEDGE_VECTOR_DATABASE = load_rag()

MODEL, TOKENIZER = load_model()
verify_all_sessions(
    "/home/jcrecio/repos/isabelle-proof-generator/afp_extractions/afp_extractions",
    "/home/jcrecio/repos/isabelle-proof-generator/afp-current-extractions",
)
