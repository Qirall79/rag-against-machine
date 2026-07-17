import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
from student.models import StudentSearchResults, MinimalAnswer, MinimalSearchResults, MinimalSource, StudentSearchResultsAndAnswer
from student.index import read_file_content
from pathlib import Path

load_dotenv()

model_name = 'Qwen/Qwen3-0.6B'

model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True)


def load_search_results(path: str) -> StudentSearchResults:
    root_dir = Path(__file__).parent.parent.resolve()
    absolute_file_path = root_dir / path

    search_results = {}

    with open(absolute_file_path, 'r', encoding='utf-8') as f:
        search_results = json.load(f)

    minimal_search_results = [MinimalSearchResults(question=search_result.get('question'), question_id=search_result.get('question_id'), retrieved_sources=[
                                                   MinimalSource(**source) for source in search_result.get('retrieved_sources')]) for search_result in search_results.get('search_results')]

    return StudentSearchResults(k=search_results.get('k'),
                                search_results=minimal_search_results
                                )


def read_chunk(file_path: str, first_character_index: int, last_character_index: int):
    root_dir = Path(__file__).parent.parent.resolve()
    absolute_file_path = root_dir / file_path

    full_content = read_file_content(str(absolute_file_path))
    return full_content[first_character_index:last_character_index]


def generate_response(prompt: str, context: str):

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a faithful code assistant. Answer the question using ONLY the provided context. "
                "If the context doesn't contain enough information, say 'I don't know'. Do not hallucinate.\n\n"
                f"Context:\n{context}"
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    response = generator(formatted_prompt, max_new_tokens=1024,
                         max_length=None, return_full_text=False)
    raw_answer = response[0]['generated_text'].strip()

    # remove the thought process
    return re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()


def get_answers(student_search_results: StudentSearchResults) -> list[MinimalAnswer]:

    answers = []

    for search_result in student_search_results.search_results:
        relevant_chunks = []
        for source in search_result.retrieved_sources:
            relevant_chunks.append(read_chunk(
                source.file_path, source.first_character_index, source.last_character_index))

        context = "\n---\n".join(relevant_chunks)
        answer = MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources=search_result.retrieved_sources,
            answer=generate_response(search_result.question, context)
        )
        answers.append(answer)

    return answers


def write_search_results_answers(save_file_path: str, results: StudentSearchResultsAndAnswer, k: int):
    root_dir = Path(__file__).parent.parent.resolve()
    absolute_save_path = root_dir / save_file_path

    absolute_save_path.parent.mkdir(parents=True, exist_ok=True)

    search_results = []

    for result in results.search_results:
        search_results.append({
            "question_id": result.question_id,
            "question": result.question,
            "answer": result.answer,
            "retrieved_sources": [minimal_source.model_dump()
                                  for minimal_source in result.retrieved_sources]
        })

    with open(absolute_save_path, 'w', encoding='utf-8') as f:
        json.dump({"search_results": search_results, "k": k}, f, indent=4)
