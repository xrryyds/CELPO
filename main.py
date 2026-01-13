import os
from scripts import TakeExam, TeacherCorrect

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path)) 
                               
exam_file_path = os.path.join(project_root, "datasets", "exam", "exam.json")
hints_file_path = os.path.join(project_root, "datasets", "exam", "hints.json")
mistakes_file_path = os.path.join(project_root, "datasets", "exam", "mistake_collection_book.json")
student_correct_output_path = os.path.join(project_root, "datasets", "exam", "correct.json")
correct_file_path = os.path.join(project_root, "datasets", "exam", "correct.json")

corrector = TeacherCorrect(
    exam_file_path=exam_file_path,
    hints_file_path=hints_file_path,
    mistake_collection_book=mistakes_file_path,
    student_correct_output_path=student_correct_output_path
)

print("Starting student correction...")
print("load question with hints...")
h_question, h_ref_solution, h_ref_answer = corrector.get_question_with_hints()

take_exam = TakeExam(correct_file_path)
take_exam.exam(h_question, h_ref_solution, h_ref_answer)

