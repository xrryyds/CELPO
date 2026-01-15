import os
from scripts import TakeExam, TeacherCorrecter

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path)) 
                               
exam_file_path = os.path.join(project_root, "CELPO", "datasets", "exam", "exam.json")
hints_file_path = os.path.join(project_root,  "CELPO", "datasets", "exam", "hints.json")
mistakes_file_path = os.path.join(project_root,  "CELPO", "datasets", "exam", "mistake_collection_book.json")
student_correct_output_path = os.path.join(project_root,  "CELPO", "datasets", "exam", "correct.json")

corrector = TeacherCorrecter(
    exam_file_path=exam_file_path,
    hints_file_path=hints_file_path,
    mistake_collection_book=mistakes_file_path,
    student_correct_output_path=student_correct_output_path
)

print("Starting student correction...")
corrector.teacher_correct()
# h_question, h_ref_solution, h_ref_answer = corrector.get_question_with_hints()
# takeExam = TakeExam(student_correct_output_path)
# takeExam.exam(h_question,h_ref_solution,h_ref_answer)

