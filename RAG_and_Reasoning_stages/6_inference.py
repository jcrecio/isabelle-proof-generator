import sys
import datasets
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset, load_dataset
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm.std import tqdm as tqdm_std
from transformers import AutoTokenizer
from transformers import pipeline
from unsloth import FastLanguageModel

# EMBEDDING_MODEL_NAME = "jcrecio/risamath-v0.1"
EMBEDDING_MODEL_NAME = "thenlper/gte-large"

# Load the knowledge vector database

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)
KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local("faiss_index", embedding_model)

# Load the LLM reader

model_name = "jcrecio/risamath-v0.1"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=4096,  # Set max sequence length
    dtype=None,  # Uses bfloat16 if available, else float16
    load_in_4bit=True,  # Enable 4-bit quantization
)

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are now an specialized agent to infer proofs for problems, theorem statements and lemmas written in Isabelle/HOL.
Infer a proof for the following Isabelle/HOL theorem statement.

### Context:
{}

### Theorem statement:
{}

### Proof:
<think>{}"""


def infer_proof(context, theorem_statement, device):
    inputs = tokenizer(
        [prompt_style.format(context, theorem_statement, "")], return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=4096,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    return response


while True:
    context = input(
        "Please enter the context for the problem (or leave it empty if no context), or write EXIT to quit:"
    )
    if context == "EXIT":
        break
    theorem_statement = input(
        "Please enter the theorem statement you want to infer a proof for, or write EXIT to quit:"
    )
    if theorem_statement == "EXIT":
        break

    user_query = prompt_style.format(context, theorem_statement, "")
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    additional_context = "\nExtracted documents:\n"
    additional_context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
    )

    proof = infer_proof(f"{additional_context} {context}", theorem_statement, "cuda")
    print("Inferred proof:\n")
    print(proof)
