import logging

import sys
from pathlib import Path

from datasets import load_dataset, Dataset, DatasetDict
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT


logger = logging.getLogger(__name__)

def extract_data(dataset: Dataset) -> tuple[list, list, list, int]:
    problems = []
    solutions = []
    answers = []

    for data in dataset:
        problem = data.get("problem", None)
        solution = data.get("solution", None)
        answer = data.get("answer", None)

        if problem and solution and answer:
            problems.append(problem)
            solutions.append(solution)
            answers.append(answer)
    return problems, solutions, answers, len(problem)


def gen_prompt(data:list, max_token: int):
    for i in range(len(data)):
        data[i] = QUESTION_PROMPT.format(
            max_token=max_token,
            problem_text=data[i]
        ) 


def main():
    dataset_loader = LoadDataset(
        dataset_name='HuggingFaceH4/MATH-500',
        split='test',
        local_path='./datasets/data/MATH-500'
    )

    train_problems, train_solutions, train_answers, train_len = extract_data(dataset_loader.get_dataset())

    # print(f"Train set: {train_len} examples")
    print(f"Train set: {train_problems[0]}")
    gen_prompt(train_problems, max_token=512)
    # print(f"Train set: {train_problems[0]}")
    # print(train_solutions[0])
    # print(train_answers[0])
    
    print(train_problems[0])
    
    
    
if __name__ == "__main__":
    main()