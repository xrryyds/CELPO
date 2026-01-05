import logging
from datasets import Dataset
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT, ANSWER_PROMPT
from configs import GRPOConfig
from sklearn.model_selection import train_test_split 
from .math_dataset import Math_DataSet


logger = logging.getLogger(__name__)


class GSM8K:
    def __init__(self, config: GRPOConfig):
        dataset_loader = LoadDataset(
            dataset_name='gsm8k',
            split='train',
            local_path='./datasets/data/gsm8k',
            config='main'
        )

        if config.max_samples is not None:
            total_size = len(dataset_loader)
            actual_size = min(config.max_samples, total_size)
            dataset_loader.set_dataset_size(actual_size)
        

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
            question = data.get("question", "").strip()
            answer_text = data.get("answer", "").strip()

            if not question or not answer_text:
                continue
            if "####" in answer_text:
                parts = answer_text.split("####")
                solution_text = parts[0].strip()  
                final_answer = parts[1].strip()   

            if solution_text and final_answer:
                problems.append(question)
                solutions.append(solution_text)
                answers.append(final_answer)

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
   gms8k = GSM8K(config=GRPOConfig)
   train_problems, train_solutions, train_answers = gms8k.get_data()
   print("problems:" + train_problems[0])
   print("train_answer:" + train_answers[0])
         


if __name__ == "__main__":
    main()
