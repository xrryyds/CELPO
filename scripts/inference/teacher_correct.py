from utils import FileIOUtils, extract_hints ,extract_boxed_content
from openai import OpenAI
from prompt.prompts import TEACHER_CORRECT_PROMPT, OREAL_CORRECT_PROMPT
import os
import time

base_url = "https://wanqing-api.corp.kuaishou.com/api/agent/v1/apps"
api_key = "k1y21hll8l0eurf7t3dg4enb56g0hhjjszf4"

class TeacherCorrecter:
    def __init__(self, 
                 exam_file_path: str, 
                 hints_file_path: str,
                 mistake_collection_book: str,
                 student_correct_output_path: str):
        self.file = FileIOUtils(exam_file_path,
                                mistake_collection_book,
                                hints_file_path,
                                student_correct_output_path)
        self.file.load_exam()
        self.question, self.answer, self.ref_answer, self.ref_solution = self.file.parse_data(self.file.data)
        self.size = len(self.question)

        self.student_correct_output_path = student_correct_output_path
        self.acc = 0
        self.err_conunt = 0
        self.toolong_count = 0
        self.acc_count = 0




    def teacher_hints(self) -> bool:
        print("Starting teacher hinting...")
        print("load mistakes...")
        self.file.load_mistakes()
        m_question, m_answer, m_ref_answer, m_ref_solution = self.file.parse_data(self.file.mistakes)
        print("mistakes size:", len(m_question))


        h_question = []
        h_hints = []
        h_ref_solution = []
        h_ref_answer = []
        print(f"generating hints({len(m_question)})...")
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
        self.file.save_hints(h_question, h_hints, h_ref_solution, h_ref_answer, m_answer)
        return True
       


    def teacher_correct(self) -> bool:
        print("Starting teacher correction...")
        client = OpenAI(
            base_url = base_url,
            api_key = api_key,
        )

        self.acc_count = 0
        self.err_conunt = 0
        self.toolong_count = 0

        err_questions = []
        err_answers = []
        err_ref_solutions = []
        err_ref_answers = []
        
        print("----- standard request -----")
        for idx in range(len(self.question)):
            # answer too long
            # if len(self.ref_solution[idx]) * 4 <= len(self.answer[idx]):
            #     self.toolong_count += 1
            #     err_questions.append(self.question[idx])
            #     err_answers.append(self.answer[idx])
            #     err_ref_solutions.append(self.ref_solution[idx])
            #     err_ref_answers.append(self.ref_answer[idx])
            #     continue
            
            # answer is correct
            final_answer = extract_boxed_content(self.answer[idx])
            if final_answer == self.ref_answer[idx]:
                self.acc_count += 1
                if idx % 5 == 0:
                    left = self.size - idx
                    print(f"finished: {idx}, left: {left}, acc:{self.acc_count}, err:{self.err_conunt}, toolong:{self.toolong_count}")
                if idx % 20 == 0:
                    self.file.save_mistakes(err_questions, err_answers, err_ref_solutions, err_ref_answers)
                    print(f"sleep in idx：{idx}")
                    time.sleep(10)                
                continue
            
            # need teacher correct
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
                self.acc_count += 1
            else:
                self.err_conunt += 1
                err_questions.append(self.question[idx])
                err_answers.append(self.answer[idx])
                err_ref_solutions.append(self.ref_solution[idx])
                err_ref_answers.append(self.ref_answer[idx])
            if idx % 5 == 0:
                left = self.size - idx
                print(f"finished: {idx}, left: {left}, acc:{self.acc_count}, err:{self.err_conunt}, toolong:{self.toolong_count}")
            if idx % 20 == 0:
                self.file.save_mistakes(err_questions, err_answers, err_ref_solutions, err_ref_answers)
                print(f"sleep in idx：{idx}")
                time.sleep(10)
        print(f"Accuracy: {self.acc_count}/{self.size}")
        print(f"Error count: {self.err_conunt}")
        self.err_conunt = self.err_conunt
        self.file.save_mistakes(err_questions, err_answers, err_ref_solutions, err_ref_answers)
        return True
            
    def judge_and_gen_hints(self):
        print("Starting judge and generate hints...")
        self.teacher_correct()
        # self.teacher_hints()
        
    def get_question_with_hints(self):
        print("load question with hints...")
        self.file.load_question_with_hints()
        h_question, h_ref_solution, h_ref_answer = self.file.parse_hints_exam(self.file.question_with_hints)
        return h_question, h_ref_solution, h_ref_answer

    
# if __name__ == "__main__":
    # corrector = TeacherCorrect()
    # corrector.judge_and_gen_hints()
    # corrector.student_correct()
    


    