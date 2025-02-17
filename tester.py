# How to run: python tester.py UNSLOTH(True, False) BASE_ONLY(True, False) BASE_MODEL LORA_MODEL(N/A for nothing) PROMPT(MATH/REASONING) MODE(FULL/PROOF)


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

MODEL: Any = None
TOKENIZER: Any = None


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


WITH_RAG = False

EMBEDDING_MODEL_NAME = "thenlper/gte-large"


def load_model():
    if len(sys.argv) != 5:
        print("Run the programm as follows: \n")
        print(
            "python 1_isabelle_verifier.py UNSLOTH(True, False) BASE_ONLY(True, False) BASE_MODEL LORA_MODEL(N/A for nothing)"
        )
        exit

    print("0", sys.argv[0])
    print("1", sys.argv[1])
    print("2", sys.argv[2])
    print("3", sys.argv[3])
    print("4", sys.argv[4])
    print("5", sys.argv[5])

    UNSLOTH = True if sys.argv[1] == "True" else False
    BASE_ONLY = True if sys.argv[2] == "True" else False
    base_model_name = sys.argv[3]
    model_name = sys.argv[4]

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
            max_seq_length=4096,
            dtype=None,  # Uses bfloat16 if available, else float16
            load_in_4bit=True,  # Enable 4-bit quantization
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
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
Infer a proof for the following Isabelle/HOL theorem statement. After thinking, respond only with the pure Isabelle/HOL proof.

### Theorem statement:
{}

### Proof:
<think>{}"""


def infer_proof(theorem_statement, device="cuda"):
    PROMPT_STYLE = sys.argv[5]
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


def infer(prompt, device="cuda"):
    if WITH_RAG:
        load_rag()

        math_prompt_style
        inputs = TOKENIZER(
            [prompt],
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
            [prompt],
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

mode = sys.argv[6]

while True:
    print(
        "******************************************************************************************"
    )
    print(
        "******************************************************************************************"
    )
    if mode == "full":
        prompt = input("Please enter your prompt:")
        if prompt == "EXIT":
            break
        response = infer(prompt)
        print(response)
    else:
        theorem_statement = input("Please enter the theorem statement:")
        if theorem_statement == "EXIT":
            break
        response = infer_proof(theorem_statement)
        print(response)
