from utils import extract_answer, normalize_answer, extract_thinking

class GRPOMathReward:
    def __init__(self):
        self.correct_reward = 2.0
        self.wrong_reward = -1.0
        self.format_error_penalty = -2.0
        self.no_thinking = -1.0
        
    def __call__(self, generated_text: str, reference_solution: str) -> float:
        pred_answer = extract_answer(generated_text)
        ref_answer = extract_answer(reference_solution)
        thinking = extract_thinking(generated_text)
        if pred_answer is None: return self.format_error_penalty
        reward = 0.0
        if normalize_answer(pred_answer) == normalize_answer(ref_answer) and ref_answer:
            reward += self.correct_reward
        else: reward += self.wrong_reward
        if not thinking: reward += self.no_thinking
        return reward
    
    
    