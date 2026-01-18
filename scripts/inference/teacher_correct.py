from utils import FileIOUtils, extract_hints ,extract_boxed_content, normalize_answer
from openai import OpenAI
from prompt.prompts import TEACHER_CORRECT_PROMPT, OREAL_CORRECT_PROMPT
import time

base_url = "https://wanqing-api.corp.kuaishou.com/api/agent/v1/apps"
api_key = "k1y21hll8l0eurf7t3dg4enb56g0hhjjszf4"

class TeacherCorrecter:
    def __init__(self):
        self.file = FileIOUtils()
        self.acc = 0
        self.err_count = 0
        self.toolong_count = 0
        self.acc_count = 0

    def teacher_hints(self) -> bool:
        print("Starting teacher hinting...")
        print("load mistakes...")
        self.file.load_mistakes()
        m_question_idx, m_question, m_answer, m_ref_answer, m_ref_solution = self.file.parse_data(self.file.mistakes)
        print("mistakes size:", len(m_question))


        h_question = []
        h_hints = []
        h_ref_solution = []
        h_ref_answer = []
        h_question_idx = []

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
            response = None
            while True:
                try:
                    completion = client.chat.completions.create(
                        model="app-7c54im-1766977238437488331",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant who good at math"},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    response = completion.choices[0].message.content
                    break 
                
                except openai.RateLimitError:
                    print(f"Rate limit reached at idx {idx}. Sleeping for 20 seconds...")
                    time.sleep(20)
                except Exception as e:
                    print(f"An unexpected error occurred at idx {idx}: {e}")
                    raise e
                
            hints = extract_hints(response)
            h_question_idx.append(m_question_idx[idx])
            h_question.append(m_question[idx])
            h_hints.append(hints)
            h_ref_solution.append(m_ref_solution[idx])
            h_ref_answer.append(m_ref_answer[idx])
            h_question_idx.append(idx)
            
        print("saving hints...")
        self.file.save_hints(h_question, h_hints, h_ref_solution, h_ref_answer, h_question_idx, m_answer)
        return True
       
    def teacher_mark_paper_with_save(self) -> bool:
        err_question_idx, err_questions, err_answers, err_ref_solutions, err_ref_answers = self.teacher_mark_paper()
        self.file.save_mistakes(err_question_idx, err_questions, err_answers, err_ref_solutions, err_ref_answers)
        return True
            
    def judge_and_gen_hints(self):
        print("Starting judge and generate hints...")
        self.teacher_mark_paper_with_save()
        self.teacher_hints()
        
    def get_question_with_hints(self):
        print("load question with hints...")
        self.file.load_question_with_hints()
        h_question, h_ref_solution, h_ref_answer = self.file.parse_hints_exam(self.file.question_with_hints)
        return h_question, h_ref_solution, h_ref_answer
    

    def teacher_mark_paper(self):
        print("Starting teacher marking...")
        self.file.load_exam()
        question_idx, question, answer, ref_answer, ref_solution = self.file.parse_data(self.file.data)
        size = len(question)
        client = OpenAI(
            base_url = base_url,
            api_key = api_key,
        )

        self.acc_count = 0
        self.err_count = 0
        self.toolong_count = 0

        err_question_idx = []
        err_questions = []
        err_answers = []
        err_ref_solutions = []
        err_ref_answers = []
        
        print("----- standard request -----")
        for idx in range(size):
            final_answer = extract_boxed_content(answer[idx])
            final_answer = normalize_answer(final_answer)
            ref_final_answer = normalize_answer(ref_answer[idx])
            # 如果答案直接匹配，跳过 API 请求
            if final_answer == ref_final_answer:
                self.acc_count += 1
                if idx % 5 == 0:
                    left = size - idx
                    print(f"finished: {idx}, left: {left}, acc:{self.acc_count}, err:{self.err_count}, toolong:{self.toolong_count}")
                # 原有的休眠逻辑保留
                if idx % 20 == 0:
                    print(f"sleep in idx：{idx}")
                    time.sleep(10)                
                continue
            
            prompt = OREAL_CORRECT_PROMPT.format(
                question=question[idx],
                gold_answer=ref_answer[idx],
                answer=answer[idx]
            )

            response = None
            while True:
                try:
                    completion = client.chat.completions.create(
                        model="app-7c54im-1766977238437488331",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant who good at math"},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    response = completion.choices[0].message.content
                    break 
                
                except openai.RateLimitError:
                    print(f"Rate limit reached at idx {idx}. Sleeping for 20 seconds...")
                    time.sleep(20)
                except Exception as e:
                    print(f"An unexpected error occurred at idx {idx}: {e}")
                    raise e

            if response.strip().lower() == "a":
                self.acc_count += 1
            else:
                self.err_count += 1
                err_question_idx.append(question_idx[idx])
                err_questions.append(question[idx])
                err_answers.append(answer[idx])
                err_ref_solutions.append(ref_solution[idx])
                err_ref_answers.append(ref_answer[idx])
            
            if idx % 5 == 0:
                left = size - idx
                print(f"finished: {idx}, left: {left}, acc:{self.acc_count}, err:{self.err_count}, toolong:{self.toolong_count}")
            
            if idx % 20 == 0:
                print(f"sleep in idx：{idx}")
                time.sleep(10)
                
        print(f"Accuracy: {self.acc_count}/{size}")
        print(f"Error count: {self.err_count}")
        return err_question_idx, err_questions, err_answers, err_ref_solutions, err_ref_answers

    
# if __name__ == "__main__":
    # corrector = TeacherCorrecter()
    # corrector.judge_and_gen_hints()
    # corrector.student_correct()
    


    