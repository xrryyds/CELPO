import os
from scripts import TakeExam, TeacherCorrecter
from utils import FileIOUtils
from configs import GRPOConfig
from data_math import Math_500, GSM8K
from utils import extract_KNOWN

exam_paper = FileIOUtils()

def student_correct():
    exam_paper.load_question_with_hints()
    question_idx, question, question_with_hint, ref_solution, ref_answer, student_answer = exam_paper.parse_hints_exam(exam_paper.question_with_hints)
    # student_exam = TakeExam()
    # student_exam.exam(question_with_hint, ref_solution, ref_answer ,question_idx)

    teacher = TeacherCorrecter()
    err_question_idx, err_questions, err_answers, err_ref_solutions, err_ref_answers = teacher.teacher_mark_paper()
    exam_paper.save_mistakes(err_question_idx, err_questions, err_answers, err_ref_solutions, err_ref_answers)

    err_idx_set = set(err_question_idx)
    correct_group = []
    incorrect_group = []
    total_data = zip(question_idx, question, question_with_hint, ref_solution, ref_answer, student_answer)

    for q_id, q, q_hint, r_sol, r_ans, s_ans in total_data:
        item = {
            "question_idx": q_id,
            "question": q,
            "question_with_hints": q_hint, 
            "ref_solution": r_sol,
            "ref_answer": r_ans,
            "student_answer": s_ans
        }
        
        if q_id in err_idx_set:
            incorrect_group.append(item)
        else:
            correct_group.append(item)

    data_for_teacher_grpo = []
    for item in correct_group:
        data_for_teacher_grpo .append({
            "question_idx": item["question_idx"],
            "question": item["question"],
            "hints": extract_KNOWN(item["question_with_hints"]),
            "student_answer": item["student_answer"],
            "success": True
        })
        
    for item in incorrect_group:
        data_for_teacher_grpo.append({
            "question_idx": item["question_idx"],
            "question": item["question"],
            "hints": extract_KNOWN(item["question_with_hints"]),
            "student_answer": item["student_answer"],
            "success": False
        })
    
    data_for_student_celpo = []
    for item in correct_group:
        data_for_student_celpo.append({
            "question_idx": item["question_idx"],
            "question": item["question"],
            "hints": extract_KNOWN(item["question_with_hints"]),
            "student_answer": item["student_answer"],
            "ref_solution": item["ref_solution"],
            "ref_answer": item["ref_answer"]
        })
    
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    celpo_dataset_path = os.path.join(project_root, "CELPO", "datasets", "exam", "adv_hints.json")
    grpo_dataset_path = os.path.join(project_root, "CELPO", "datasets", "exam", "grpo_data.json")

    exam_paper.save_results_to_json(data_for_teacher_grpo, grpo_dataset_path)
    exam_paper.save_results_to_json(data_for_student_celpo, celpo_dataset_path)



def teacher_correct():
    teacher = TeacherCorrecter()
    teacher.teacher_mark_paper_with_save()
    teacher.teacher_hints()
    del teacher

def single_qusestion(qusetion):
    student_exam = TakeExam("/root/project/data/xrr/Qwen/Qwen2.5-Math-7B-Instruct")
    return student_exam.answer_single_question(qusetion)



def student_first_take_exam_Math500():
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    exam_file_path = os.path.join(project_root, "CELPO", "configs", "celpo_train.yaml")
    config = GRPOConfig.load_yaml(exam_file_path)
    math_500 = Math_500(config)
    test_dataset = math_500.get_test_data()
    train_dataset= math_500.get_train_data()
    question = test_dataset.problems + train_dataset.problems
    solution = test_dataset.solutions + train_dataset.solutions
    answer = test_dataset.answers + train_dataset.answers
    print(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    take_exam = TakeExam()
    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


def student_first_take_exam_Gsm8k():
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    exam_file_path = os.path.join(project_root, "CELPO", "configs", "celpo_train.yaml")
    config = GRPOConfig.load_yaml(exam_file_path)
    gsm8k = GSM8K()
    question = gsm8k.problems
    solution = gsm8k.solutions
    answer = gsm8k.answers
    print(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    take_exam = TakeExam("/root/project/data/xrr/Qwen/Qwen2.5-Math-7B-Instruct")
    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)

if __name__ == "__main__":
    # #1. student first take exam
    # student_first_take_exam()
    # #2. teacher judges and gives hints
    teacher = TeacherCorrecter()
    # teacher.teacher_mark_paper_with_save()
    # teacher.teacher_hints()
    # student_correct()
    #3. teacher correct
    print(single_qusestion("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")
)



