import re
from typing import Optional
import prompt

def extract_answer(text: str) -> Optional[str]:
    # 优先匹配 <answer>...</answer> 标签中的内容（支持换行、空格等）
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        # 返回最后一个匹配项（类似原逻辑），并去除首尾空白
        return matches[-1].strip()
    return None

def normalize_answer(answer: str) -> str:
    if answer is None: return ""
    answer = answer.replace(" ", "").lower()
    answer = re.sub(r'\\[a-zA-Z]+', '', answer) 
    answer = re.sub(r'[^0-9a-zA-Z\+\-\*/=\.\,]', '', answer)
    return answer



