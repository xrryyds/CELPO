QUESTION_PROMPT  = """Please solve the following math problem.
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