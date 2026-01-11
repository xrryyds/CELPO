QUESTION_PROMPT  = """
# Question:
Please solve the following math problem.
## Strictly adhere to the output format below:

1. **Reasoning**: Wrap your step-by-step deduction process inside <thinking>...</thinking> tags. This section must not exceed {max_token} words.
2. **Result**: Wrap the final numerical answer inside <answer>...</answer> tags. Do not include units or text in this tag.


Problem:
{problem_text}

## Your output must exactly match this structure:
Thinking:
<thinking>
[Your derivation steps here]
</thinking>

Answer:
<answer>
[Final number here]
</answer>

## Please answer below:
"""

ANSWER_PROMPT = """
Thinking:
<thinking>
{thinking}
</thinking>

Answer:
<answer>
{answer}
</answer>
"""

JUDGE_PROMPT = """
{question_and_answer}

# Judge:
You are an expert mathematics grader. You will be provided with a **Math Problem** and a **Model Response** (consisting of a `<thinking>` process and an `<answer>`).
Your goal is to verify whether the Model Response is mathematically correct.

# Input Data:
Problem:
{problem_text}

Model Response:
{model_response}

# Evaluation Criteria:
1. **Accuracy**: Is the final numerical value in the `<answer>` tag correct?
2. **Logic**: Is the reasoning process in the `<thinking>` tag mathematically sound and consistent with the answer? This section must not exceed {max_token} words.

# Output Format:
Please strictly adhere to the following XML format for your evaluation:

<conclusion>[YES or NO]</conclusion>
<reason>[Your explanation]</reason>

# Rules for Filling Tags:
1. **<conclusion>**:
   - Return **YES** if the solution is entirely correct.
   - Return **NO** if there is any error in the logic or the final answer.

2. **<reason>**:
   - If the conclusion is **YES**, simply write: "Correct".
   - If the conclusion is **NO**, concisely explain the specific reason for rejection (e.g., "Calculation error in step 2", "Final answer is wrong", or "Logic does not follow the problem conditions").
"""

JUDGER_GEN_REASON_PROMPT= """

<reason>{reason}</reason>
"""


#############################################################################################

GEN_ENHANCE_PROMPT= """
Answer the questions with reference to the given hints.
# Hints:
{hints}
# Question:
{question}
"""


TEACHER_CORRECT_PROMPT = """
**Role:** Heuristic Logic Mentor & Knowledge Bridge

**Task:**
Your task is to analyze the **Student's Solution** in comparison to the **Reference Answer**. Instead of acting as a simple grader, you must identify the specific **missing logical link** or **knowledge gap** that prevents the student from reaching the correct conclusion.

**Goal:**
Provide the student with the necessary "Knowledge Hints" so that, based on their existing correct reasoning, they can bridge the gap and solve the problem themselves.

**Critical Constraints for Hints:**
1.  **Universal Knowledge Only:** Hints must be provided as **general problem-solving methods, formulas, theorems, or definitions** (e.g., "Recall the formula for the area of a circle: $S = \pi r^2$" or "Use the derivatives of trigonometric functions: $(sin x)' = cos x$").
2.  **No Specific Calculations:** Do not calculate the result for the specific numbers in the problem. Provide the *tool*, not the *answer*.
3.  **Targeted:** The hint must directly address the specific step where the student's logic broke down or stopped.

# Problem:
{problem}

# Student's Answer:
{student_answer}

# Reference Answer:
{reference_solution}

**Output Format, respond in the following XML format::**
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
1. **Inclusion-Exclusion Principle**: For any two events $A$ and $B$, the formula for the probability of their union is $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.
</hints>

"""

