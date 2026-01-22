import re
from typing import Optional
import prompt
import os
import json


def extract_KNOWN(text: str) -> Optional[str]:
    pattern = r'<KNOWN>\s*(.*?)\s*</KNOWN>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None

def extract_hints(text: str) -> Optional[str]:
    pattern = r'<hints>\s*(.*?)\s*</hints>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


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

def extract_boxed_content(text: str) -> Optional[str]:
    pattern = r'\\boxed\{(.*?)\}'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[-1].strip()
    return ""

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


def remove_null_hints(file_path):
    if not os.path.exists(file_path):
        print(f"erro, can not find: {file_path}")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_count = len(data)

        filtered_data = [item for item in data if item.get('hints') is not None]
        
        new_count = len(filtered_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)

        print(f"finished: {file_path}")
        print(f"from: {original_count}")
        print(f"to: {new_count}")
        print(f"deleted: {original_count - new_count}")

    except Exception as e:
        print(f"error: {e}")