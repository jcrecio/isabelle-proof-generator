from typing import Optional, List, Tuple
from datasets import Dataset, load_dataset
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm.std import tqdm as tqdm_std
from transformers import AutoTokenizer

SPLIT_DOCUMENTS = False

dataset = load_dataset(
    "jcrecio/AFP_Theories",
    data_files="rag_predocuments.jsonl",
    split="train",
    trust_remote_code=True,
)

RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["content"], metadata={"source": doc["source"]})
    for doc in tqdm_std(dataset)
]

# work on better separators for isabelle theorems lemmas and proofs
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

# jcrecio: choose a different embedding model? which criteria to use? Mathstral? generico?
EMBEDDING_MODEL_NAME = "thenlper/gte-small"


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


docs_processed = (
    split_documents(
        300,
        RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )
    if SPLIT_DOCUMENTS
    else RAW_KNOWLEDGE_BASE
)


tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

print("*********************************** Loading the embedding model...")

#  nearest neighbor search algorithm: FAISS + cosine similarity
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=False,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

print("*********************************** Creating the knowledge vector database...")


KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
)

KNOWLEDGE_VECTOR_DATABASE.save_local("isabelle-hol-vectors")
