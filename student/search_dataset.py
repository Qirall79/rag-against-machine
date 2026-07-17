import bm25s
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
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
    index_dir = project_root / "data" / "processed" / "bm25_index"
    return bm25s.BM25.load(index_dir)


def load_embeddings():
    project_root = Path(__file__).parent.parent.resolve()
    embeddings_file = project_root / "data" / "processed" / "embeddings.npy"
    return np.load(file=embeddings_file)


def load_json_file(dataset_path: str) -> list[UnansweredQuestion]:
    root_dir = Path(__file__).parent.parent.resolve()
    absolute_dataset_path = root_dir / dataset_path

    print(absolute_dataset_path)

    questions = []

    with open(str(absolute_dataset_path), 'r', encoding='utf-8') as f:
        questions = json.load(f).get('rag_questions')

    return [UnansweredQuestion(**question) for question in questions]


def write_answers(save_file_path: str, results: list[MinimalSearchResults], k: int):
    root_dir = Path(__file__).parent.parent.resolve()
    absolute_save_path = root_dir / save_file_path

    absolute_save_path.parent.mkdir(parents=True, exist_ok=True)

    search_results = []

    for result in results:
        search_results.append({
            "question_id": result.question_id,
            "question": result.question,
            "retrieved_sources": [minimal_source.model_dump()
                                  for minimal_source in result.retrieved_sources]
        })

    with open(absolute_save_path, 'w', encoding='utf-8') as f:
        json.dump({"search_results": search_results, "k": k}, f, indent=4)


def get_lexical_result(query: UnansweredQuestion, k, retriever):
    query_tokens = bm25s.tokenize(query.question)

    documents, _ = retriever.retrieve(query_tokens=query_tokens, k=k)

    results = [int(doc_id) for doc_id in documents[0]]

    return results


def get_semantic_result(query: UnansweredQuestion, corpus_embeddings, k, model):
    query_embedding = model.encode(query.question, normalize_embeddings=True)

    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=k)

    results = [int(hit['corpus_id']) for hit in hits[0]]

    return results


def calculate_rrf(lexical_results, semantic_results, w_lex=1.0, w_sem=1.0, k=60):
    scores = {}
    for rank, doc_id in enumerate(lexical_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + w_lex * (1 / (k + rank))
    for rank, doc_id in enumerate(semantic_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + w_sem * (1 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def get_answer(query: UnansweredQuestion, k, retriever, corpus_embeddings, metadata, model) -> MinimalSearchResults:

    lexical_results = get_lexical_result(query=query, k=k, retriever=retriever)
    semantic_results = get_semantic_result(
        query=query, k=k, corpus_embeddings=corpus_embeddings, model=model)
    
    # semantic_results = []
    # lexical_results = []
    
    
    final_ranking = calculate_rrf(
        lexical_results=lexical_results, semantic_results=semantic_results)[:k]

    sources = [metadata[doc[0]] for doc in final_ranking]

    return MinimalSearchResults(question=query.question, question_id=query.question_id, retrieved_sources=sources)
