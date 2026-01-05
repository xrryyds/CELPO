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

## response example:
Thinking:
<thinking>
1 + 1 = 2
</thinking>

Answer:
<answer>
2
</answer>

# Response:
"""

ANSWER_PROMPT = """
Thinking:
<thinking>
[Your derivation steps here]
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