import re

def extract_boxed_answer(text: str) -> Optional[str]:
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None

def extract_final_answer(text: str) -> Optional[str]:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    patterns = [
        r'[Tt]he answer is:?\s*([^\n\.]+)',
        r'[Ff]inal answer:?\s*([^\n\.]+)',
        r'=\s*([^\n]+)$',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    return None

def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    answer = answer.replace(" ", "")
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)
    answer = re.sub(r'[^0-9a-zA-Z\+\-\*/=\.\,]', '', answer)
    return answer.lower()