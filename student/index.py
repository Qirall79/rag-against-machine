import re
import bm25s
import json
import argparse
import numpy as np
from pathlib import Path
from student.models import MinimalSource
from sentence_transformers import SentenceTransformer


def load_args():
    # parse command line args
    parser = argparse.ArgumentParser(
        description='RAG Pipeline Retrieval Engine')

    parser.add_argument(
        '--max_chunk_size',
        type=str,
        required=True,
        help='Maximum chunk size'
    )

    return parser.parse_args()


def list_knowledge_files():
    root_dir = Path(__file__).parent.parent.resolve()
    repo_path = root_dir / 'data' / 'raw' / 'vllm-0.10.1'
    python_files = repo_path.rglob(pattern='*.py')
    md_files = repo_path.rglob(pattern='*.md')

    return [
        list(python_files),
        list(md_files)
    ]


def read_file_content(path: str) -> str:
    content = ''
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def expand_code_identifiers(text: str) -> str:

    # to expand CamelCase and snake_case texts in code blocks

    text = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', text)

    text = re.sub(r'(?<=[A-Z])([A-Z][a-z])', r' \1', text)

    text = re.sub(r'[_]+', ' ', text)

    return re.sub(r'\s+', ' ', text).strip()


def chunk_file(file_path: str, max_chunk_size=2000, chunk_overlap=500) -> list[str]:
    content = read_file_content(file_path)

    if chunk_overlap >= max_chunk_size // 2:
        chunk_overlap = max_chunk_size // 4

    chunks = []

    split_positions = []

    if file_path.endswith('.md'):
        split_positions = [match.end()
                           for match in re.finditer(r'\n\n', content)]
    else:
        split_positions = [match.start() for match in re.finditer(
            r'^(def|class)\s', content, re.MULTILINE)]

    # adding file boundaries
    split_positions = [0] + split_positions + [len(content)]

    chunk = ''

    start_index = 0
    last_valid_split = 0
    is_new_chunk = True

    i = 0
    length = len(split_positions)

    while i < length:
        pos = split_positions[i]

        if pos - start_index > max_chunk_size:

            # means it's a single chunk with more than max_chunk_size
            if is_new_chunk:
                chunk = content[start_index:start_index + max_chunk_size]
                chunks.append({
                    'content': chunk,
                    'first_character_index': start_index,
                    'last_character_index': start_index + len(chunk),
                    'file_path': file_path
                })

                start_index = start_index + len(chunk)

                if split_positions[i] <= start_index:
                    i += 1
            else:
                chunk = content[start_index:last_valid_split]
                chunks.append({
                    'content': chunk,
                    'first_character_index': start_index,
                    'last_character_index': last_valid_split,
                    'file_path': file_path
                })

                start_index = max(0, last_valid_split - chunk_overlap)
                is_new_chunk = True

        else:
            last_valid_split = pos
            is_new_chunk = False
            i += 1

    # cleanup
    if start_index < len(content):
        final_chunk = content[start_index:len(content)]
        chunks.append({
            'content': final_chunk,
            'first_character_index': start_index,
            'last_character_index': len(content),
            'file_path': file_path
        })

    return chunks


def build_knowledge_base(max_chunk_size: int):

    root_dir = Path(__file__).parent.parent.resolve()

    python_files, markdown_files = list_knowledge_files()

    python_chunks_content = []
    python_chunks_metadata = []

    markdown_chunks_content = []
    markdown_chunks_metadata = []

    for file in python_files:
        full_path = Path(file)

        try:
            parts = full_path.parts
            data_index = parts.index("data")
            clean_unix_path = "/".join(parts[data_index:])
        except ValueError:
            clean_unix_path = full_path.as_posix()

        chunks = chunk_file(str(full_path), max_chunk_size)
        for chunk in chunks:
            python_chunks_content.append(chunk['content'])

            # for unix-style paths
            chunk_data = chunk.copy()
            chunk_data['file_path'] = clean_unix_path
            python_chunks_metadata.append(MinimalSource(**chunk_data))

    for file in markdown_files:
        full_path = Path(file)

        try:
            parts = full_path.parts
            data_index = parts.index("data")
            clean_unix_path = "/".join(parts[data_index:])
        except ValueError:
            clean_unix_path = full_path.as_posix()

        chunks = chunk_file(str(full_path), max_chunk_size)
        for chunk in chunks:
            markdown_chunks_content.append(chunk['content'])

            chunk_data = chunk.copy()
            chunk_data['file_path'] = clean_unix_path
            markdown_chunks_metadata.append(MinimalSource(**chunk_data))

    return python_chunks_content, python_chunks_metadata, markdown_chunks_content, markdown_chunks_metadata


def index(python_corpus, markdown_corpus):

    corpus = python_corpus + markdown_corpus
    project_root = Path(__file__).parent.parent.resolve()

    # semantic indexing
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(corpus, normalize_embeddings=True)
    output_dir = project_root / "data" / "processed" / "embeddings.npy"
    np.save(file=output_dir, arr=embeddings)

    # BM25s indexing
    output_dir = project_root / "data" / "processed" / "bm25_index"
    corpus_tokens = bm25s.tokenize(corpus, stopwords='en')
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)
    retriever.save(output_dir)

    return retriever


def save_metadata(python_metadata, markdown_metadata):
    project_root = Path(__file__).parent.parent.resolve()
    output_dir = project_root / "data" / "processed" / "bm25_index"
    metadata_file_path = output_dir / "metadata.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = python_metadata + markdown_metadata

    serializable_metadata = [item.model_dump() for item in metadata]

    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_metadata, f, indent=4)
