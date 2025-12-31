import re
from typing import Optional
import prompt

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



