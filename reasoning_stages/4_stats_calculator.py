from datasets import load_dataset
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from collections import Counter
import json


def get_stats(lenghts):
    if not lenghts:
        return "No documents"

    max_value = max(lenghts)
    min_value = min(lenghts)
    avg_value = sum(lenghts) / len(lenghts)

    counter = Counter(lenghts)
    most_common = counter.most_common(1)[0]  # (value, count)

    return {
        "Max": max_value,
        "Min": min_value,
        "Average": avg_value,
        "Most Repetitive": most_common[0],
        "Occurrences": most_common[1],
        "Sum": sum(lenghts),
        "Length": len(lenghts),
        "Unique Count": len(set(lenghts)),
    }


dataset = load_dataset(
    "jcrecio/AFP_Theories",
    data_files="rag_database.jsonl",
    split="train",
    trust_remote_code=True,
)

RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["content"], metadata={"source": doc["source"]})
    for doc in tqdm(dataset)
]


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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # The maximum number of characters in a chunk: we selected this value arbitrarily
    chunk_overlap=100,  # The number of characters to overlap between chunks
    add_start_index=True,  # If `True`, includes chunk's start index in metadata
    strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
    separators=MARKDOWN_SEPARATORS,
)

docs_processed = []
for doc in RAW_KNOWLEDGE_BASE:
    docs_processed += text_splitter.split_documents([doc])


print(
    f"Model's maximum sequence length: {SentenceTransformer('jcrecio/risamath-v0.1-merged').max_seq_length}"
)


tokenizer = AutoTokenizer.from_pretrained("jcrecio/risamath-v0.1-merged")
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]


stats = get_stats(lengths)
print(json.dumps(stats, indent=4))
