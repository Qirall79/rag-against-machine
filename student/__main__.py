import fire
from pathlib import Path
from student.index import build_knowledge_base, index, save_metadata
from student.search_dataset import load_metadata, load_index, load_json_file, get_answer, write_answers
from student.answer_dataset import load_search_results, get_answers, read_chunk, generate_response, write_search_results_answers
from student.models import UnansweredQuestion, StudentSearchResultsAndAnswer


class StudentRAGCli:
    """
    RAG Pipeline Retrieval Engine CLI managed by Python Fire
    """

    def index(self, max_chunk_size: int = 2000):
        """
        Index the repository codebase and documentation
        """
        print(f"Starting ingestion with max_chunk_size: {max_chunk_size}...")

        python_corpus, python_metadata, markdown_corpus, markdown_metadata = build_knowledge_base(
            max_chunk_size)

        index(python_corpus, markdown_corpus)

        save_metadata(python_metadata, markdown_metadata)

        print(
            "Ingestion complete! Indices saved under data/processed/")

    def search_dataset(self, dataset_path: str, k: int = 10, save_directory: str = "data/output/search_results"):
        """
        Process multiple questions from JSON datasets and save coordinate results
        """
        print(f"Loading queries from: {dataset_path}")

        metadata = load_metadata()
        retriever = load_index()
        queries = load_json_file(dataset_path)

        results = []
        for query in queries:
            search_result = get_answer(query, k, retriever, metadata)
            results.append(search_result)

        output_name = Path(dataset_path).name
        output_file_path = Path(save_directory) / output_name

        write_answers(str(output_file_path), results, k)
        print(f"Saved student_search_results to {output_file_path}")

    def answer_dataset(self, student_search_results_path: str, save_directory: str = "data/output/search_results_and_answer", k: int = 5):
        answers = StudentSearchResultsAndAnswer(search_results=get_answers(load_search_results(student_search_results_path)), k=k) 
        
        output_name = Path(student_search_results_path).name
        output_file_path = Path(save_directory) / output_name

        write_search_results_answers(str(output_file_path), answers, k)
        print(f"Saved student_search_resultsAndAnswer to {output_file_path}")

    def answer(self, question: str, k: int = 10):
        metadata = load_metadata()
        retriever = load_index()
        
        query = UnansweredQuestion(question=question)
        search_result = get_answer(query, k, retriever, metadata)
        relevant_chunks = []
        for source in search_result.retrieved_sources:
            relevant_chunks.append(read_chunk(source.file_path, source.first_character_index, source.last_character_index))

        context = "\n---\n".join(relevant_chunks)
        answer = generate_response(search_result.question, context)
        
        print(answer)

def main():
    fire.Fire(StudentRAGCli)


if __name__ == "__main__":
    main()
