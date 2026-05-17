import re
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from models import MinimalSource


def list_knowledge_files():
    repo_path = Path('data/vllm-0.10.1')
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


def chunk_file(file_path: str, max_chunk_size=2000) -> list[str]:
    content = read_file_content(file_path)

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
            if is_new_chunk:
                chunk = content[start_index:start_index + max_chunk_size]
                chunks.append({
                    'content': chunk,
                    'first_char_index': start_index,
                    'last_char_index': start_index + len(chunk),
                    'file_path': file_path
                })

                start_index = start_index + len(chunk)
            else:
                chunk = content[start_index:last_valid_split]
                chunks.append({
                    'content': chunk,
                    'first_char_index': start_index,
                    'last_char_index': last_valid_split,
                    'file_path': file_path
                })

                start_index = last_valid_split
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
            'first_char_index': start_index,
            'last_char_index': len(content),
            'file_path': file_path
        })

    return chunks


def build_knowledge_base():
    python_files, markdown_files = list_knowledge_files()

    python_chunks_content = []
    python_chunks_metadata = []

    markdown_chunks_content = []
    markdown_chunks_metadata = []

    for file in python_files:
        chunks = chunk_file(str(file))
        for chunk in chunks:
            python_chunks_content.append(chunk['content'])
            python_chunks_metadata.append(MinimalSource(
                file_path=chunk['file_path'], first_character_index=chunk['first_char_index'], last_character_index=chunk['last_char_index']))

    for file in markdown_files:
        chunks = chunk_file(str(file))
        for chunk in chunks:
            markdown_chunks_content.append(chunk['content'])
            markdown_chunks_metadata.append(MinimalSource(
                file_path=chunk['file_path'], first_character_index=chunk['first_char_index'], last_character_index=chunk['last_char_index']))

    return python_chunks_content, python_chunks_metadata, markdown_chunks_content, markdown_chunks_metadata


def main():
    python_chunks_content, python_chunks_metadata, markdown_chunks_content, markdown_chunks_metadata = build_knowledge_base()
    
    print(python_chunks_content)


if __name__ == "__main__":
    main()
