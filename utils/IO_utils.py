import json
from prompt import GEN_ENHANCE_PROMPT

class FileIOUtils:
    def __init__(self, exam_file_path: str, mistake_file_path: str ,hints_file_path: str, student_correct_output_path: str):
        self.exam_file_path = exam_file_path
        self.mistake_file_path = mistake_file_path
        self.hints_file_path = hints_file_path
        self.student_correct_output_path = student_correct_output_path

        self.data = []
        self.mistakes = []
        self.question_with_hints = []
    
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
        
    def load_question_with_hints(self) -> bool:
        try:
            with open(self.hints_file_path, 'r', encoding='utf-8') as f:
                self.question_with_hints = json.load(f)
            return True
        except Exception as e:
            print(f"load fail: {e}")
            return False
        
    def parse_data(self, data: list):
        question = []
        answer = []
        ref_answer = []
        ref_solution = []
        for idx, item in enumerate(data):
            question.append(item.get("question", ""))
            answer.append(item.get("answer", ""))
            ref_answer.append(item.get("ref_answer", ""))
            ref_solution.append(item.get("ref_solution", ""))
        return question, answer, ref_answer, ref_solution
    
    def parse_hints_exam(self, data: list):
        question = []
        question_with_hint = []
        hints = []
        ref_answer = []
        ref_solution = []
        for idx, item in enumerate(data):
            question.append(item.get("question", ""))
            hints.append(item.get("hint", ""))
            ref_answer.append(item.get("ref_answer", ""))
            ref_solution.append(item.get("ref_solution", ""))

        for idx in range(len(question)):
            question_with_hint.append(GEN_ENHANCE_PROMPT.format(question=question[idx], hints=hints[idx]))
        return question_with_hint, ref_solution, ref_answer

    def save_hints(self, question: list, hints: list, ref_solution: list, ref_answer: list, student_answer: list) -> bool:
        try:
            size = len(question)
            data = []
            for idx in range(size):
                item = {
                    "question": question[idx],
                    "hint": hints[idx],
                    "ref_solution": ref_solution[idx],
                    "ref_answer": ref_answer[idx],
                    "student_answer": student_answer[idx]
                }
                data.append(item)

            with open(self.hints_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"save fail: {e}")
            return False
        

    def save_mistakes(self, question: list, answers: list, ref_solution: list, ref_answer: list) -> bool:
        self.save_Q_and_A(question, answers, ref_solution, ref_answer, self.mistake_file_path)


    def save_student_correct(self, question: list, answers: list, ref_solution: list, ref_answer: list) -> bool:
        self.save_Q_and_A(question, answers, ref_solution, ref_answer, self.student_correct_output_path)    

    def save_Q_and_A(self, question: list, answers: list, ref_solution: list, ref_answer: list, path:str) -> bool:
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

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"save fail: {e}")
            return False

