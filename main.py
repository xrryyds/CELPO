import os
from scripts import TakeExam, TeacherCorrecter
from utils import FileIOUtils

exam_paper = FileIOUtils()
exam_paper.load_question_with_hints()
question, question_with_hint, ref_solution, ref_answer = exam_paper.parse_hints_exam(exam_paper.question_with_hints)

student_exam = TakeExam()
student_exam.exam(question_with_hint, ref_solution, ref_answer)

