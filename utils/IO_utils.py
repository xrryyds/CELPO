import json

class FileIOUtils:
    def __init__(self, exam_file_path: str, mistake_file_path: str ,hints_file_path: str):
        self.exam_file_path = exam_file_path
        self.mistake_file_path = mistake_file_path
        self.hints_file_path = hints_file_path
        self.data = []
        self.mistakes = []
    
    def load_exam(self) -> bool:
        try:
            with open(self.exam_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            return True
        except Exception as e:
            print(f"load fail: {e}")
            return False
        

    def load_mistakes(self) -> bool:
        try:
            with open(self.mistake_file_path, 'r', encoding='utf-8') as f:
                self.mistakes = json.load(f)
            return True
        except Exception as e:
            print(f"load fail: {e}")
            return False
        
    def parse_data(self, data: list):
        size = len(data)
        question = [] * size
        answer = [] * size
        ref_answer = [] * size
        ref_solution = [] * size
        for idx, item in enumerate(self.data):
            question.append(item.get("question", ""))
            answer.append(item.get("answer", ""))
            ref_answer.append(item.get("ref_answer", ""))
            ref_solution.append(item.get("ref_solution", ""))
        return question, answer, ref_answer, ref_solution

    def save_hints(self, question: list, hints: list, ref_solution: list, ref_answer: list) -> bool:
        try:
            size = len(question)
            self.correct_data = [] * size
            data = []
            for idx in range(size):
                item = {
                    "question": question[idx],
                    "hint": hints[idx],
                    "ref_solution": ref_solution[idx],
                    "ref_answer": ref_answer[idx]
                }
                data.append(item)

            with open(self.hints_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"save fail: {e}")
            return False
        

    def save_mistakes(self, question: list, answers: list, ref_solution: list, ref_answer: list) -> bool:
        try:
            size = len(question)
            data = []
            for idx in range(size):
                item = {
                    "question": question[idx],
                    "answer": answers[idx],
                    "ref_solution": ref_solution[idx],
                    "ref_answer": ref_answer[idx]
                }
                data.append(item)

            with open(self.mistake_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"save fail: {e}")
            return False
        

