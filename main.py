import os
from scripts import TakeExam, TeacherCorrecter
from utils import FileIOUtils
from configs import GRPOConfig
from data_math import Math_500
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
    student_exam = TakeExam("/root/project/data/xrr/Qwen/Qwen2.5-7B-Instruct")
    return student_exam.answer_single_question(qusetion)



def student_first_take_exam():
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

if __name__ == "__main__":
    # #1. student first take exam
    # student_first_take_exam()

    # #2. teacher judges and gives hints
    teacher = TeacherCorrecter()
    # teacher.teacher_mark_paper_with_save()
    # teacher.teacher_hints()
    # student_correct()
    #3. teacher correct
    print(single_qusestion("""**Role:** Heuristic Logic Mentor & Knowledge Bridge

    **Task:**
    Your task is to analyze the **Student's Solution** in comparison to the **Reference Answer**. Instead of acting as a simple grader, you must identify the specific **missing logical link** or **knowledge gap** that prevents the student from reaching the correct conclusion.

    **Goal:**
    Provide the student with the necessary "Knowledge Hints" so that, based on their existing correct reasoning, they can bridge the gap and solve the problem themselves.

    **Critical Constraints for Hints:**
    1.  **Universal Knowledge Only:** Hints must be provided as **general problem-solving methods, formulas, theorems, or definitions** (e.g., "Recall the formula for the area of a circle: $S = \\pi r^2$" or "Use the derivatives of trigonometric functions: $(sin x)' = cos x$").
    2.  **No Specific Calculations:** Do not calculate the result for the specific numbers in the problem. Provide the *tool*, not the *answer*.
    3.  **Targeted:** The hint must directly address the specific step where the student's logic broke down or stopped.
    4.  **Unidentifiable Gap:** If it is impossible to determine if the student lacks specific universal knowledge (e.g., the error is purely a calculation/arithmetic mistake, a typo, the answer is empty/irrelevant, or the reasoning is too unclear to pinpoint a missing theorem), return `null`.

    # Problem:
    For some value of $x,$ $0 < x < 180,$
    \\[\\tan 53^\\circ \\tan 81^\\circ \\tan x^\\circ = \\tan 53^\\circ + \\tan 81^\\circ + \\tan x^\\circ.\\]
    Find $x.$

    # Student's Answer:
    I've got this trigonometry problem here. It says that for some value of $ x $, where $ 0 < x < 180 $, the equation $\\tan 53^\\circ \\tan 81^\\circ \\tan x^\\circ = \\tan 53^\\circ + \\tan 81^\\circ + \\tan x^\\circ$ holds true. I need to find the value of $ x $.

    First, I should recall some trigonometric identities that might help me simplify or manipulate this equation. I know that the tangent of an angle in a right triangle is the ratio of the opposite side to the adjacent side, but I'm not sure if that's directly helpful here.

    Maybe I can consider the tangent addition formula. I recall that:

    $$
    \\tan(a + b) = \\frac{\\tan a + \\tan b}{1 - \\tan a \\tan b}
    $$

    But in this problem, I have a product of tangents on one side and a sum on the other. It's not immediately clear how to apply the addition formula here.

    Let me think differently. Perhaps I can consider specific values for the angles given, like 53° and 81°, and see if there's a relationship between them.

    I notice that 53° and 81° are being used, and their tangents are involved. Maybe there's a way to relate these angles somehow. Let's see, 53° and 81° don't add up to 90°, so they aren't complementary angles. Complementary angles would be 53° and 37°, since 53° + 37° = 90°, but 81° is different.

    Wait, maybe I can look at the tangent of complementary angles. I know that $\\tan(90^\\circ - a) = \\cot a$. So, $\\tan 81^\\circ = \\cot 9^\\circ$, because 81° is 9° less than 90°. Similarly, $\\tan 53^\\circ = \\cot 37^\\circ$, since 53° is 37° less than 90°.

    But I'm not sure if that helps directly. Let's see if I can express everything in terms of sine and cosine, since tangent is sine over cosine.

    So, $\\tan a = \\frac{\\sin a}{\\cos a}$. Therefore, the equation becomes:

    $$
    \\frac{\\sin 53^\\circ}{\\cos 53^\\circ} \\cdot \\frac{\\sin 81^\\circ}{\\cos 81^\\circ} \\cdot \\frac{\\sin x^\\circ}{\\cos x^\\circ} = \\frac{\\sin 53^\\circ}{\\cos 53^\\circ} + \\frac{\\sin 81^\\circ}{\\cos 81^\\circ} + \\frac{\\sin x^\\circ}{\\cos x^\\circ}
    $$

    This looks complicated. Maybe there's a better approach.

    Let me consider that the equation is $\\tan 53^\\circ \\tan 81^\\circ \\tan x^\\circ = \\tan 53^\\circ + \\tan 81^\\circ + \\tan x^\\circ$. It resembles the tangent addition formula for three angles, where $\\tan(a + b + c) = \\frac{\\tan a + \\tan b + \\tan c - \\tan a \\tan b \\tan c}{1 - (\\tan a \\tan b + \\tan a \\tan c + \\tan b \\tan c)}$, but I'm not sure if that's applicable here.

    Alternatively, maybe I can consider that $\\tan 53^\\circ \\tan 81^\\circ \\tan x^\\circ = \\tan 53^\\circ + \\tan 81^\\circ + \\tan x^\\circ$ suggests some sort of equality that can be achieved when the angles are related in a particular way.

    Let me try to isolate $\\tan x^\\circ$. Let's denote $\\tan 53^\\circ = a$, $\\tan 81^\\circ = b$, and $\\tan x^\\circ = c$. Then the equation becomes:

    $$
    a b c = a + b + c
    $$

    I can rearrange this to:

    $$
    a b c - a - b - c = 0
    $$

    This seems a bit messy. Maybe I can solve for $ c $ in terms of $ a $ and $ b $:

    $$
    c (a b - 1) = a + b

    c = \\frac{a + b}{a b - 1}
    $$

    So, $\\tan x^\\circ = \\frac{\\tan 53^\\circ + \\tan 81^\\circ}{\\tan 53^\\circ \\tan 81^\\circ - 1}$

    Now, I recall that $\\tan(a + b) = \\frac{\\tan a + \\tan b}{1 - \\tan a \\tan b}$, but in this case, it's $\\frac{\\tan a + \\tan b}{\\tan a \\tan b - 1}$, which is similar but with a negative sign in the denominator.

    Wait a minute, $\\tan(a + b) = \\frac{\\tan a + \\tan b}{1 - \\tan a \\tan b}$, so $\\frac{\\tan a + \\tan b}{\\tan a \\tan b - 1} = -\\tan(a + b)$.

    Therefore, $\\tan x^\\circ = -\\tan(53^\\circ + 81^\\circ) = -\\tan(134^\\circ)$

    So, $ x^\\circ = 180^\\circ - 134^\\circ = 46^\\circ $, because $\\tan(180^\\circ - a) = -\\tan a$.

    Therefore, $ x = 46 $.

    But I should verify this solution to make sure it's correct.

    Let me plug $ x = 46^\\circ $ back into the original equation:

    $\\tan 53^\\circ \\tan 81^\\circ \\tan 46^\\circ = \\tan 53^\\circ + \\tan 81^\\circ + \\tan 46^\\circ$

    I know that $\\tan 46^\\circ = \\cot 44^\\circ$, but I'm not sure if that helps directly. Maybe I can calculate the numerical values:

    First, $\\tan 53^\\circ$ is approximately 1.3270,

    $\\tan 81^\\circ$ is approximately 5.6713,

    $\\tan 46^\\circ$ is approximately 1.0355.

    So, the left side is $1.3270 \\times 5.6713 \\times 1.0355$ ≈ $7.502$

    The right side is $1.3270 + 5.6713 + 1.0355$ ≈ $8.0338$

    These are not equal, which suggests that my solution might be wrong.

    ...

    # Reference Answer:
    Isolating $\\tan x^\\circ,$ we find
    \\begin{align*}
    \\tan x &= \\frac{\\tan 53^\\circ + \\tan 81^\\circ}{\\tan 53^\\circ \\tan 81^\\circ - 1} \\\\
    &= -\\frac{\\tan 53^\\circ + \\tan 81^\\circ}{1 - \\tan 53^\\circ \\tan 81^\\circ}.
    \\end{align*}
    From the angle addition formula, this is equal to
    \\[-\\tan (53^\\circ + 81^\\circ) = -\\tan 134^\\circ = \\tan 46^\\circ.\\]
    Therefore, $x = \\boxed{46}.$

    **Output Format:**

    **Condition A:** If Constraint #4 applies (unable to identify a specific general knowledge gap), output strictly:
    `null`

    **Condition B:** If a knowledge gap is identified, respond in the following XML format:
    <Logic Gap Analysis>
    [Briefly explain what specific concept or step the student missed.]
    </Logic Gap Analysis>

    <hints>
    [Provide the general formula, theorem, or principle. Use LaTeX for math expressions.]
    1. ...
    2. ...
    </hints>

    **Example:**
    *   **Student's Error:** Calculated probability as $P(A)+P(B)$ but events were not mutually exclusive.
    *   **Your Output:**
    <Logic Gap Analysis>
    The student directly sums the two probabilities but ignores the possibility that these two events may occur simultaneously (i.e., they are not mutually exclusive).
    </Logic Gap Analysis>

    <hints>
    1. **Inclusion-Exclusion Principle**: For any two events $A$ and $B$, the formula for the probability of their union is $P(A \\cup B) = P(A) + P(B) - P(A \\cap B)$.
    </hints>
    """)
)



