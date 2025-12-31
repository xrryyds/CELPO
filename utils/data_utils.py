import re

def extract_answer(text: str) -> Optional[str]:
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    if matches: return matches[-1].strip()
    patterns = [r'[Tt]he answer is:?\s*([^\n\.]+)', r'[Ff]inal answer:?\s*([^\n\.]+)', r'=\s*([^\n]+)$']
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches: return matches[-1].strip()
    return None

def normalize_answer(answer: str) -> str:
    if answer is None: return ""
    answer = answer.replace(" ", "").lower()
    answer = re.sub(r'\\[a-zA-Z]+', '', answer) 
    answer = re.sub(r'[^0-9a-zA-Z\+\-\*/=\.\,]', '', answer)
    return answer

class MathRewardFunction:
    def __init__(self):
        self.correct_reward = 1.0
        self.wrong_reward = -1.0
        self.format_error_penalty = -0.5
        
    def __call__(self, generated_text: str, reference_solution: str) -> float:
        pred_answer = extract_answer(generated_text)
        ref_answer = extract_answer(reference_solution)
        if pred_answer is None: return self.format_error_penalty
        if normalize_answer(pred_answer) == normalize_answer(ref_answer) and ref_answer:
            return self.correct_reward
        return self.wrong_reward
