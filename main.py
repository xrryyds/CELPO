import os
from scripts import TakeExam, TeacherCorrecter
from utils import FileIOUtils, remove_null_hints
from configs import GRPOConfig
from data_math import Math_500, GSM8K
from utils import extract_KNOWN, filter_json_by_question_idx

exam_paper = FileIOUtils()

def student_correct():
    exam_paper.load_question_with_hints()
    question_idx, question, question_with_hint, ref_solution, ref_answer, student_answer, hints = exam_paper.parse_hints_exam(exam_paper.question_with_hints)
    student_exam = TakeExam()
    student_exam.exam(question_with_hint, ref_solution, ref_answer ,question_idx)

    teacher = TeacherCorrecter()
    err_question_idx, err_questions, err_answers, err_ref_solutions, err_ref_answers = teacher.teacher_mark_paper()

    err_idx_set = set(err_question_idx)
    correct_group = []
    incorrect_group = []
    total_data = zip(question_idx, question, question_with_hint, ref_solution, ref_answer, student_answer, hints)

    for q_id, q, q_hint, r_sol, r_ans, s_ans, s_hint in total_data:
        item = {
            "question_idx": q_id,
            "question": q,
            "question_with_hints": q_hint, 
            "ref_solution": r_sol,
            "ref_answer": r_ans,
            "student_answer": s_ans,
            "hints": s_hint
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
            "hints": item["hints"],
            "student_answer": item["student_answer"],
            "success": True
        })
        
    for item in incorrect_group:
        data_for_teacher_grpo.append({
            "question_idx": item["question_idx"],
            "question": item["question"],
            "hints": item["hints"],
            "student_answer": item["student_answer"],
            "success": False
        })
    
    data_for_student_adv_hints = []
    for item in correct_group:
        data_for_student_adv_hints.append({
            "question_idx": item["question_idx"],
            "question": item["question"],
            "hints": item["hints"],
            "student_answer": item["student_answer"],
            "ref_solution": item["ref_solution"],
            "ref_answer": item["ref_answer"]
        })


    data_for_student_disadv_hints = [] 
    for item in incorrect_group:
        data_for_student_disadv_hints.append({
            "question_idx": item["question_idx"],
            "question": item["question"],
            "hints": item["hints"],
            "student_answer": item["student_answer"],
            "ref_solution": item["ref_solution"],
            "ref_answer": item["ref_answer"]
        })   
    
    adv_hints_dataset_path = exam_paper.adv_hints_dataset_path
    disadv_hints_dataset_path = exam_paper.disadv_hints_dataset_path
    grpo_dataset_path = exam_paper.grpo_dataset_path

    exam_paper.save_results_to_json(data_for_teacher_grpo, grpo_dataset_path)
    exam_paper.save_results_to_json(data_for_student_adv_hints,  adv_hints_dataset_path)
    exam_paper.save_results_to_json(data_for_student_disadv_hints, disadv_hints_dataset_path)



def teacher_correct():
    teacher = TeacherCorrecter()
    teacher.teacher_mark_paper_with_save()
    teacher.teacher_hints()
    remove_null_hints(exam_paper.hints_file_path)
    filter_json_by_question_idx(exam_paper.exam_file_path, exam_paper.hints_file_path, exam_paper.corr_path)
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



def student_take_exam_Gsm8k_test():
    gsm8k = GSM8K(False)
    question = gsm8k.problems
    solution = gsm8k.solutions
    answer = gsm8k.answers
    print(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    take_exam = TakeExam("/root/project/data/xrr/Qwen/Qwen2.5-Math-7B-Instruct")
    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    print(take_exam.exam_test(question, solution, answer, question_idx))


if __name__ == "__main__":
    # #1. student first take exam
    # student_first_take_exam()
    # #2. teacher judges and gives hints
    # teacher = TeacherCorrecter()
    # teacher.teacher_mark_paper_with_save()
    # student_first_take_exam_Gsm8k()
    # teacher.teacher_hints()
    # student_correct()
    filter_json_by_question_idx(exam_paper.exam_file_path, exam_paper.hints_file_path, exam_paper.corr_path)
    #3. teacher correct
    # student_first_take_exam_Gsm8k()
    # student_take_exam_Gsm8k_test()
    # remove_null_hints(exam_paper.hints_file_path)


