import re
from typing import Optional
import prompt

def extract_answer(text: str) -> Optional[str]:
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None

def extract_thinking(text: str) -> Optional[str]:
    pattern = r'<thinking>\s*(.*?)\s*</thinking>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None

def extract_conclusion(text: str) -> Optional[str]:
    pattern = r'<conclusion>\s*(.*?)\s*</conclusion>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None

def extract_reason(text: str) -> Optional[str]:
    pattern = r'<reason>\s*(.*?)\s*</reason>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None

def normalize_answer(answer: str) -> str:
    if answer is None: return ""
    answer = answer.replace(" ", "").lower()
    answer = re.sub(r'\\[a-zA-Z]+', '', answer) 
    answer = re.sub(r'[^0-9a-zA-Z\+\-\*/=\.\,]', '', answer)
    return answer

def collate_fn(batch):
    return {
            'prompts': [item['prompt'] for item in batch],
            'reference_answers': [item['reference_answer'] for item in batch]
            }


