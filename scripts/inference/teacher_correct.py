from utils import FileIOUtils, extract_hints
from openai import OpenAI
from prompt.prompts import TEACHER_CORRECT_PROMPT
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
        self.file.load()
        self.question, self.answer, self.ref_answer, self.ref_solution = self.file.parse_data()
        self.size = len(self.question)




    def teacher_hints(self) -> bool:
        print("Starting teacher hinting...")
        client = OpenAI(
            base_url = base_url,
            api_key = api_key,
        )
        print("----- standard request -----")
        for idx in range(1):
            prompt = TEACHER_CORRECT_PROMPT.format(
                question=self.question[idx],
                student_answer=self.answer[idx],
                ref_solution=self.ref_solution[idx]
            )
            print("Prompt:\n", prompt)
            completion = client.chat.completions.create(
                model="app-7c54im-1766977238437488331",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who good at math"},
                    {"role": "user", "content": prompt},
                ],
            )
        resounse = completion.choices[0].message.content
        print("Response:\n", resounse)
        hints = extract_hints(resounse)
        print("Hints:\n", hints)


    def teacher_correct(self) -> bool:
        print("Starting teacher correction...")
        client = OpenAI(
            base_url = base_url,
            api_key = api_key,
        )
        print("----- standard request -----")
        for idx in range(1):
            prompt = TEACHER_CORRECT_PROMPT.format(
                question=self.question[idx],
                student_answer=self.answer[idx],
                ref_solution=self.ref_solution[idx]
            )
            print("Prompt:\n", prompt)
            completion = client.chat.completions.create(
                model="app-7c54im-1766977238437488331",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who good at math"},
                    {"role": "user", "content": prompt},
                ],
            )
        resounse = completion.choices[0].message.content
        print("Response:\n", resounse)
        hints = extract_hints(resounse)
        print("Hints:\n", hints)    


    
if __name__ == "__main__":
    corrector = TeacherCorrect()
    # print(corrector.question[0])
    # print("-----")
    # print(corrector.answer[0])
    # print("-----")
    # print(corrector.ref_answer[0])
    # print("-----")
    # print(corrector.ref_solution[0])
    # print("-----")
    print("Loaded data size:", corrector.size)
    print("running teacher correct")
    corrector.teacher_correct()
    


    