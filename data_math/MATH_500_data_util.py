import logging
from datasets import Dataset
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT
from configs import GRPOConfig


logger = logging.getLogger(__name__)


class Math_500:
    def __init__(self, config: GRPOConfig):
        dataset_loader = LoadDataset(
            dataset_name='HuggingFaceH4/MATH-500',
            split='test',
            local_path='./datasets/data/MATH-500'
        )

        self.train_problems, self.train_solutions, self.train_answers, self.train_len = self.extract_data(
            dataset_loader.get_dataset())
        self.gen_prompt(self.train_problems, max_token=GRPOConfig.thinking_max_tokens)
        

    def extract_data(self, dataset: Dataset) -> tuple[list, list, list, int]:
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

    def gen_prompt(self,data: list, max_token: int = 512):
        for i in range(len(data)):
            data[i] = QUESTION_PROMPT.format(
                max_token=max_token,
                problem_text=data[i]
            )
            
    def get_data(self):
        return self.train_problems, self.train_solutions, self.train_answers, self.train_len


def main():
   math_500 = Math_500(config=GRPOConfig)
   train_problems, train_solutions, train_answers, train_len = math_500.get_data()
   print(train_problems[0])


if __name__ == "__main__":
    main()
