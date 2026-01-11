import os
import json
import logging
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
from functools import lru_cache
import time
from peft import PeftModel, LoraConfig, TaskType
from utils.data_utils import extract_answer, normalize_answer
from configs import GRPOConfig, GRPOConfigInference
from data_math import Math_500, GSM8K
from prompt import QUESTION_PROMPT

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def exam(question, solution, answer):
    MAX_SEQ_LENGTH = 2048  
    MAX_NEW_TOKENS = 512
    BATCH_SIZE = 4
    
    LOCAL_MODEL_PATH = "/home/xrrfolder/models/internlm/OREAL-7B"
    OUTPUT_JSON_PATH = "/home/xrrfolder/CELPO/datasets/exam/exam.json"
    
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    print(f"Loading model from {LOCAL_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        device_map="auto",
        torch_dtype="auto", 
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    tokenizer.padding_side = "left" 

    results = []

    total_batches = (len(question) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(question), BATCH_SIZE), total=total_batches, desc="Inferencing"):
        batch_questions = question[i:i+BATCH_SIZE]
        batch_ref_answers = answer[i:i+BATCH_SIZE]
        batch_ref_solution = solution[i:i+BATCH_SIZE]

        try:
            batch_prompts = []
            for q in batch_questions:
                q_text = str(q)
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q_text}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_prompts.append(prompt)

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.7, 
                    top_p=0.9,
                )

            input_ids_len = inputs["input_ids"].shape[1]
            decoded_outputs = tokenizer.batch_decode(
                outputs[:, input_ids_len:], 
                skip_special_tokens=True
            )

            for idx, generated_text in enumerate(decoded_outputs):
                results.append({
                    "question": batch_questions[idx],
                    "answer": generated_text.strip(),
                    "ref_answer": batch_ref_answers[idx].strip(),
                    "ref_solution": batch_ref_solution[idx].strip()
                })
                
            if (i // BATCH_SIZE) % 10 == 0:
                 with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"\n[Error] Batch {i//BATCH_SIZE} failed: {e}")
            print(f"Skipping questions: {batch_questions}")
            torch.cuda.empty_cache()
            continue

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done! Results saved to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
	config = GRPOConfig.load_yaml("/home/xrrfolder/CELPO/configs/celpo_train.yaml")
	math_500 = Math_500(config)
	test_dataset = math_500.get_test_data()
	train_dataset= math_500.get_train_data()
	question = test_dataset.problems + train_dataset.problems
	solution = test_dataset.solutions + train_dataset.solutions
	answer = test_dataset.answers + train_dataset.answers
	print(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
	exam(question, solution, answer)
	

