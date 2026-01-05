import logging
from datasets import Dataset
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT, ANSWER_PROMPT
from configs import GRPOConfig
from sklearn.model_selection import train_test_split 
from .math_dataset import Math_DataSet


logger = logging.getLogger(__name__)


class Math_500:
    def __init__(self, config: GRPOConfig):
        dataset_loader = LoadDataset(
            dataset_name='HuggingFaceH4/MATH-500',
            split='test',
            local_path='./datasets/data/MATH-500'
        )


        data_size = min(config.max_samples, dataset_loader)
        dataset_loader = dataset_loader[:data_size]
        self.problems, self.solutions, self.answers, self.data_len = self.extract_data(
            dataset_loader.get_dataset())
        self.gen_prompt(self.problems, max_token=GRPOConfig.thinking_max_tokens)
        self.gen_answer(self.answers)
        
        (self.train_problems, self.test_problems,
         self.train_solutions, self.test_solutions,
         self.train_answers, self.test_answers) = train_test_split(
            self.problems, self.solutions, self.answers,
            test_size=0.2, 
            random_state=42,  
            shuffle=True  
        )

        self.train_data =  Math_DataSet(self.train_problems, self.train_solutions,  self.train_answers)
        self.test_data = Math_DataSet(self.test_problems, self.test_solutions, self.test_answers)

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
        return problems, solutions, answers, len(problems)

    def gen_prompt(self, data: list, max_token: int = 512):
        for i in range(len(data)):
            data[i] = QUESTION_PROMPT.format(
                max_token=max_token,
                problem_text=data[i]
            )
            
    def gen_answer(self, data: list):
        for i in range(len(data)):
            data[i] = ANSWER_PROMPT.format(
                answer = data[i]
            )
            
    def get_data(self):
        return self.train_problems, self.train_solutions, self.train_answers


    def get_train_data(self):
        return self.train_data
    
    def get_test_data(self):
        return self.test_data
    
    def get_dataset(self):
        return self.train_data, self.test_data
    
def main():
   math_500 = Math_500(config=GRPOConfig)
   train_problems, train_solutions, train_answers = math_500.get_data()
   print("problems:" + train_problems[0])
   print("train_answer:" + train_answers[0])
         


if __name__ == "__main__":
    main()
