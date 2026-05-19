import re
import bm25s
import json
import argparse
from pathlib import Path
from models import MinimalSource, UnansweredQuestion, AnsweredQuestion, MinimalSearchResults


def load_predictions(path: str) -> dict:
    predictions: dict = {}

    with open(path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    return predictions


def load_answers(path: str) -> list[AnsweredQuestion]:
    answers = []

    with open(path, 'r', encoding='utf-8') as f:
        answers = json.load(f).get('rag_questions')

    return [AnsweredQuestion(question_id=answer.get('question_id'), question=answer.get('question'), answer=answer.get('answer'), sources=answer.get('sources')) for answer in answers]


def create_queries_file(questions: list[AnsweredQuestion]):
    queries = [{'query_id': question.question_id, 'text': question.question}
               for question in questions]

    with open('data/queries.json', 'w', encoding='utf-8') as f:
        json.dump(queries, f)

def check_overlap(source_a: MinimalSource, source_b: MinimalSource) -> bool:
    
    # to avoid different path styles (across different OS)
    path_a = Path(source_a.file_path).as_posix()
    path_b = Path(source_b.file_path).as_posix()
    
    if path_a != path_b:
        return False

    overlap_start = max(source_a.first_character_index, source_b.first_character_index)
    overlap_end = min(source_a.last_character_index, source_b.last_character_index)

    return overlap_start < overlap_end

def evaluate_dataset(name: str, answers: list[AnsweredQuestion], predictions: dict):
    total_questions = 0
    successful_answers = 0
    
    for answer in answers:
        total_questions += 1
        
        predicted_list = predictions.get(answer.question_id, [])
        predicted_sources = [MinimalSource(**predicted_source) for predicted_source in predicted_list]
        
        is_hit = False
        for predicted_source in predicted_sources:
            for true_source in answer.sources:
                if check_overlap(predicted_source, true_source):
                    is_hit = True
                    break
            if is_hit:
                successful_answers += 1
                break
    
    recall_score = (successful_answers / total_questions) * 100 if total_questions > 0 else 0.0
    print(f"--- {name} Results ---")
    print(f"Total Evaluated Questions: {total_questions}")
    print(f"Successful Hits: {successful_answers}")
    print(f"Final Recall Score: {recall_score:.2f}%\n")
        
        
    

def main():
    docs_questions = load_answers('public/AnsweredQuestions/dataset_docs_public.json') 
    code_questions = load_answers('public/AnsweredQuestions/dataset_code_public.json')

    predictions = load_predictions('data/predictions.json')
    
    evaluate_dataset("Documentation Dataset", docs_questions, predictions)
    
    print("\n=========================================\n")
    
    evaluate_dataset("Codebase Dataset", code_questions, predictions)
    


if __name__ == "__main__":
    main()
