import re
import bm25s
import json
from pathlib import Path
from models import MinimalSource


def load_metadata():
    output_dir = Path("data/processed/bm25_index")
    metadata_file_path = output_dir / "metadata.json"

    metadata = []
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    pydantic_sources = [MinimalSource(**item) for item in metadata]
    return pydantic_sources


def load_index():
    return bm25s.BM25.load('data/processed/')


def main():
    metadata = load_metadata()
    retriever = load_index()

    query = "supported OpenAI compatible models architectures"
    k = 5

    query_tokens = bm25s.tokenize(query)

    documents, scores = retriever.retrieve(query_tokens=query_tokens, k=k)

    for i, document in enumerate(documents[0]):
        print(f"Result #{i + 1}: [Score: {scores[0][i]:.2f}]")
        print(metadata[document])


if __name__ == "__main__":
    main()
