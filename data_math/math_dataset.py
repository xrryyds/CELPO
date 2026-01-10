from datasets import Dataset 
from prompt import GEN_ENHANCE_PROMPT

class Math_DataSet():
    def __init__(self, problems, solutions ,answers):
        self.problems = problems
        self.solutions = solutions
        self.answers = answers
        self.enhance_data = [""] * len(problems)
        
    def __len__(self): return len(self.problems)
    
    def __getitem__(self, idx):
        return {
            'prompt': self.problems[idx],
            'reference_answer': self.answers[idx],
            'reference_solution': self.solutions[idx]
        }

    def to_hf_dataset(self):
        return Dataset.from_dict({
            'prompt': self.problems,
            'reference_solution': self.answers
        })
    
    def gen_enhance_prompt(self):
        for i in range(len(self.problems)):
            self.problems[i] = GEN_ENHANCE_PROMPT.format(
                hints = self.enhance_data[i],
                question = self.problems[i]
            )

        
        
class Math_DataSet_Judge():
    def __init__(self, problems, solutions ,answers, conclusion, reason, flag: bool=True):
        self.problems = problems
        self.solutions = solutions
        self.answers = answers
        self.conclusion = conclusion
        self.reason = reason
        self.flag = flag
        
    def __len__(self): return len(self.problems)
    
    ########
    def __getitem__(self, idx):
        return {
            'prompt': self.problems[idx],
            'reference_answer': self.answers[idx]
        }
        
    def gen_incor_reason(self):
        if self.flag:
            return
        else:
            ## call gtp
            return