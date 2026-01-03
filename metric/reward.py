from utils import extract_answer, normalize_answer

class GRPOMathReward:
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