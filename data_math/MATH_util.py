import logging
from datasets import Dataset
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT, ANSWER_PROMPT
from configs import GRPOConfig
from sklearn.model_selection import train_test_split 
from .math_dataset import Math_DataSet
from .math_data_util import Math_data
from utils import extract_boxed_content


logger = logging.getLogger(__name__)


class Math():
    def __init__(self):
        dataset_loader = LoadDataset(
            dataset_name='HuggingFaceH4/MATH',
            split='train',
            local_path='./datasets/data/MATH/train'
        )
        
        print(len(dataset_loader.get_dataset()))

        self.problems, self.solutions, self.answers, self.level, self.type, self.data_len = self.extract_data(
            dataset_loader.get_dataset())


        self.data =  Math_DataSet(self.problems, self.solutions,  self.answers)

    def extract_data(self, dataset: Dataset):
        problems = []
        solutions = []
        answers = []
        levels= []
        types = []
        print(f"datasize:{len(dataset)}")
        for data in dataset:
            solution = data.get("solution", None)
            answer = extract_boxed_content(solution)
            if not answer:
                continue

            problem = data.get("problem", None)
            level = data.get("level", None)
            type = data.get("type", None)

            if problem and solution and answer and level and type:
                problems.append(problem)
                solutions.append(solution)
                answers.append(answer)
                types.append(type)
                levels.append(level)

        return problems, solutions, answers, level, type, len(problems)

    def gen_prompt(self, data: list, max_token: int = 512):
        for i in range(len(data)):
            data[i] = QUESTION_PROMPT.format(
                max_token=max_token,
                problem_text=data[i]
            )
            
    
def main():
    math = Math()

    spilt = "=============================="


    print("problems:" + math.problems[0])
    print(spilt)
    print("train_answer:" + math.solutions[0])
    print(spilt)
    print("answet:" + math.answers[0])
    print(spilt)
    print(f"size:{len(math.problems)}")
         


if __name__ == "__main__":
    main()
