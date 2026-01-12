from utils import FileIOUtils, extract_hints
from openai import OpenAI
from prompt.prompts import TEACHER_CORRECT_PROMPT, OREAL_CORRECT_PROMPT
import os

base_url = "https://wanqing-api.corp.kuaishou.com/api/agent/v1/apps"
api_key = "k1y21hll8l0eurf7t3dg4enb56g0hhjjszf4"

class TeacherCorrect:
    def __init__(self, 
                 exam_file_path: str = "/Users/xiongrengrong/项目/CELPO/datasets/exam/exam.json", 
                 hints_file_path: str = "/Users/xiongrengrong/项目/CELPO/datasets/exam/hints.json",
                 mistake_collection_book: str = "/Users/xiongrengrong/项目/CELPO/datasets/exam/mistake_collection_book.json"):
        self.file = FileIOUtils(exam_file_path,
                                mistake_collection_book,
                                hints_file_path)
        self.file.load_exam()
        self.question, self.answer, self.ref_answer, self.ref_solution = self.file.parse_data(self.file.data)
        self.size = len(self.question)
        self.acc = 0




    def teacher_hints(self) -> bool:
        print("Starting teacher hinting...")
        print("load mistakes...")
        self.file.load_mistakes()
        m_question, m_answer, m_ref_answer, m_ref_solution = self.file.parse_data(self.file.mistakes)
        print("mistakes size:", len(self.question))


        h_question = []
        h_hints = []
        h_ref_solution = []
        h_ref_answer = []
        print("generating hints...")
        client = OpenAI(
            base_url = base_url,
            api_key = api_key,
        )
        print("----- standard request -----")
        for idx in range(len(m_question)):
            prompt = TEACHER_CORRECT_PROMPT.format(
                problem=m_question[idx],
                student_answer=m_answer[idx],
                ref_solution=m_ref_solution[idx]
            )
            completion = client.chat.completions.create(
                model="app-7c54im-1766977238437488331",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who good at math"},
                    {"role": "user", "content": prompt},
                ],
            )
            resounse = completion.choices[0].message.content
            hints = extract_hints(resounse)
            h_question.append(m_question[idx])
            h_hints.append(hints)
            h_ref_solution.append(m_ref_solution[idx])
            h_ref_answer.append(m_ref_answer[idx])
        print("saving hints...")
        self.file.save_hints(h_question, h_hints, h_ref_solution, h_ref_answer)
        return True
       


    def teacher_correct(self) -> bool:
        print("Starting teacher correction...")
        client = OpenAI(
            base_url = base_url,
            api_key = api_key,
        )
        acc_cnt = 0
        err_cnt = 0
        err_questions = []
        err_answers = []
        err_ref_solutions = []
        err_ref_answers = []
        
        print("----- standard request -----")
        for idx in range(len(self.question)):
            prompt = OREAL_CORRECT_PROMPT.format(
                question=self.question[idx],
                gold_answer=self.ref_answer[idx],
                answer=self.answer[idx]
            )
            completion = client.chat.completions.create(
                model="app-7c54im-1766977238437488331",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who good at math"},
                    {"role": "user", "content": prompt},
                ],
            )
            response = completion.choices[0].message.content
            if response.strip().lower() == "a":
                acc_cnt += 1
                if len(gold_answer) >= len(answer) * 4:
                    err_questions.append(self.question[idx])
                    err_answers.append(self.answer[idx])
                    err_ref_solutions.append(self.ref_solution[idx])
                    err_ref_answers.append(self.ref_answer[idx])
            else:
                err_cnt += 1
                err_questions.append(self.question[idx])
                err_answers.append(self.answer[idx])
                err_ref_solutions.append(self.ref_solution[idx])
                err_ref_answers.append(self.ref_answer[idx])
            if idx % 5:
                left = self.size - idx
                print(f"finished: {idx}, left: {left}")
        print(f"Accuracy: {acc_cnt}/{self.size}")
        print(f"Error count: {err_cnt}")
        self.err_conunt = err_cnt
        self.file.save_mistakes(err_questions, err_answers, err_ref_solutions, err_ref_answers)
        return True
            
    def judge_and_gen_hints(self):
        print("Starting judge and generate hints...")
        self.teacher_correct()
        # self.teacher_hints()





    
if __name__ == "__main__":
    corrector = TeacherCorrect()
    # corrector.judge_and_gen_hints()
    


    