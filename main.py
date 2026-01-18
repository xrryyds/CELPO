import os
from scripts import TakeExam, TeacherCorrecter
from utils import FileIOUtils

exam_paper = FileIOUtils()

def take_exam():
    exam_paper.load_question_with_hints()
    question, question_with_hint, ref_solution, ref_answer = exam_paper.parse_hints_exam(exam_paper.question_with_hints)
    student_exam = TakeExam()
    student_exam.exam(question_with_hint, ref_solution, ref_answer)


def teacher_correct():
    teacher = TeacherCorrecter()
    # teacher.judge_and_gen_hints()
    teacher.teacher_mark_paper()


if __name__ == "__main__":
    take_exam()
    # teacher_correct()