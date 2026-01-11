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
    # --- 配置区域 ---
    # 建议调大长度，Math题目的推理过程很长
    MAX_SEQ_LENGTH = 2048  
    MAX_NEW_TOKENS = 512
    BATCH_SIZE = 4
    
    # 路径配置
    LOCAL_MODEL_PATH = "/home/xrrfolder/models/internlm/OREAL-7B"
    OUTPUT_JSON_PATH = "/home/xrrfolder/CELPO/datasets/exam/exam.json"
    
    # 环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 尝试减少碎片化导致的OOM
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    print(f"Loading model from {LOCAL_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        device_map="auto",
        torch_dtype="auto", # 自动选择 float16 或 bfloat16
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 确保 pad_token 存在，这对于批量推理至关重要
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # 左填充对于生成任务通常更安全（虽然现在的模型很多支持右填充，但左填充兼容性更好）
    tokenizer.padding_side = "left" 

    results = []

    # 使用 tqdm 显示进度条
    total_batches = (len(question) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(question), BATCH_SIZE), total=total_batches, desc="Inferencing"):
        batch_questions = question[i:i+BATCH_SIZE]
        batch_ref_answers = answer[i:i+BATCH_SIZE]
        batch_ref_solution = solution[i:i+BATCH_SIZE]

        try:
            # --- 核心修复：正确处理批量 Chat Template ---
            batch_prompts = []
            for q in batch_questions:
                # 确保输入是字符串
                q_text = str(q)
                # 生成单个对话的 prompt 字符串，而不是传给 tokenizer 列表
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q_text}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_prompts.append(prompt)

            # 批量 Tokenize
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
                    temperature=0.7, # 稍微调高一点温度，0.1可能导致由于重复生成的死循环
                    top_p=0.9,
                )

            # 解码
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
                
            # 每10批保存一次，防止程序崩溃数据全丢
            if (i // BATCH_SIZE) % 10 == 0:
                 with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"\n[Error] Batch {i//BATCH_SIZE} failed: {e}")
            print(f"Skipping questions: {batch_questions}")
            # 释放显存
            torch.cuda.empty_cache()
            continue

    # 最终保存
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
	

