import re
import bm25s
import json
import argparse
from pathlib import Path
from models import MinimalSource, UnansweredQuestion, AnsweredQuestion, MinimalSearchResults


def load_metadata() -> list[MinimalSource]:
    output_dir = Path("data/processed/bm25_index")
    metadata_file_path = output_dir / "metadata.json"

    metadata = []
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    pydantic_sources = [MinimalSource(**item) for item in metadata]
    return pydantic_sources


def load_index():
    return bm25s.BM25.load('data/processed/bm25_index')


def load_json_file(path: str) -> list[UnansweredQuestion]:
    questions = []

    with open(path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    return [UnansweredQuestion(question_id=question.get('query_id'), question=question.get('text')) for question in questions]


def write_answers(path: str, results: list[MinimalSearchResults]):

    output = {}

    for result in results:
        output[result.question_id] = [minimal_source.model_dump()
                                      for minimal_source in result.retrieved_sources]

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)


def get_answer(query: UnansweredQuestion, k, retriever, metadata) -> MinimalSearchResults:
    query_tokens = bm25s.tokenize(query.question)

    documents, scores = retriever.retrieve(query_tokens=query_tokens, k=k)

    sources = []
    for document in documents[0]:
        source_model = metadata[document]
        sources.append(source_model)

    return MinimalSearchResults(question_id=query.question_id, question=query.question, retrieved_sources=sources)


def load_args():
    # parse command line args
    parser = argparse.ArgumentParser(
        description='RAG Pipeline Retrieval Engine')

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input queries JSON file'
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where the predictions JSON should be saved"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top matching sources to retrieve per query (default: 5)"
    )

    return parser.parse_args()


def main():
    metadata = load_metadata()
    retriever = load_index()

    args = load_args()

    queries = load_json_file(args.input)

    results = []

    for query in queries:
        results.append(get_answer(query, args.k, retriever, metadata))

    write_answers(args.output, results)


if __name__ == "__main__":
    main()
