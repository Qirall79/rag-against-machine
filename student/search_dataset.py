import bm25s
import json
from pathlib import Path
from student.models import MinimalSource, UnansweredQuestion, AnsweredQuestion, MinimalSearchResults


def load_metadata() -> list[MinimalSource]:
    project_root = Path(__file__).parent.parent.resolve()
    output_dir = project_root / "data" / "processed" / "bm25_index"
    metadata_file_path = output_dir / "metadata.json"

    metadata = []
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    pydantic_sources = [MinimalSource(**item) for item in metadata]
    return pydantic_sources


def load_index():
    project_root = Path(__file__).parent.parent.resolve()
    output_dir = project_root / "data" / "processed" / "bm25_index"
    return bm25s.BM25.load(output_dir)


def load_json_file(dataset_path: str) -> list[UnansweredQuestion]:
    current_dir = Path(__file__).parent.resolve()
    absolute_dataset_path = current_dir / dataset_path

    print(absolute_dataset_path)

    questions = []

    with open(str(absolute_dataset_path), 'r', encoding='utf-8') as f:
        questions = json.load(f).get('rag_questions')

    return [UnansweredQuestion(**question) for question in questions]


def write_answers(save_file_path: str, results: list[MinimalSearchResults], k: int):
    current_dir = Path(__file__).parent.resolve()
    absolute_save_path = current_dir / save_file_path

    absolute_save_path.parent.mkdir(parents=True, exist_ok=True)

    search_results = []

    for result in results:
        search_results.append({
            "question_id": result.question_id,
            "question_str": result.question,
            "retrieved_sources": [minimal_source.model_dump()
                                  for minimal_source in result.retrieved_sources]
        })

    with open(absolute_save_path, 'w', encoding='utf-8') as f:
        json.dump({"search_results": search_results, "k": k}, f, indent=4)


def get_answer(query: UnansweredQuestion, k, retriever, metadata) -> MinimalSearchResults:
    query_tokens = bm25s.tokenize(query.question)

    documents, _ = retriever.retrieve(query_tokens=query_tokens, k=k)

    sources = []
    for document in documents[0]:
        source_model = metadata[document]
        sources.append(source_model)

    return MinimalSearchResults(question_id=query.question_id, question=query.question, retrieved_sources=sources)
