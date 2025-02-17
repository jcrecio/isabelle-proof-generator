# How to run: python 2_inference.py <model_to_load>


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
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from unsloth import FastLanguageModel

model_to_load = sys.argv[1]

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

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


def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_to_load,
        max_seq_length=4096,
        dtype=None,  # Uses bfloat16 if available, else float16
        load_in_4bit=True,  # Enable 4-bit quantization
    )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


reasoning_prompt_style = """Given a theorem in Isabelle/HOL, think through the proof strategy step by step, then output ONLY a clean, valid Isabelle/HOL proof.

Theorem: {theorem}

Think through the proof strategy:
<think>
Consider the theorem structure
Plan the proof approach
Identify necessary tactics and methods
</think>

Now provide ONLY the clean Isabelle/HOL proof:
"""


def generate_proof(model, tokenizer, theorem):
    formatted_prompt = reasoning_prompt_style.format(theorem=theorem)

    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
        temperature=0.7,
        top_p=0.95,
    )

    response = tokenizer.batch_decode(outputs)[0]

    proof = response.split("Now provide ONLY the clean Isabelle/HOL proof:")[-1].strip()
    proof = proof.replace("<think>", "").replace("</think>", "").strip()

    return proof


MODEL, TOKENIZER = load_model()

while True:
    print(
        "******************************************************************************************"
    )
    print(
        "******************************************************************************************"
    )
    theorem_statement = input("Please enter the theorem statement:")
    if theorem_statement == "EXIT":
        break
    response = generate_proof(MODEL, TOKENIZER, theorem_statement)
    print(response)
