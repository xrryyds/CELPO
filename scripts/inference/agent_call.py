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

def exam(question, solution, answer):
	os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
	os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

	LOCAL_MODEL_PATH = "/home/xrrfolder/models/internlm/OREAL-7B"

	OUTPUT_JSON_PATH = "/home/xrrfolder/CELPO/datasets/exam/exam.json"
	BATCH_SIZE = 4  
	MAX_NEW_TOKENS = 500  

	tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
	model = AutoModelForCausalLM.from_pretrained(
		LOCAL_MODEL_PATH,
		device_map="auto",  
		torch_dtype="auto", 
		low_cpu_mem_usage=True
	)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		model.config.pad_token_id = model.config.eos_token_id

	results = []

	for i in range(0, len(question), BATCH_SIZE):
		batch_questions = question[i:i+BATCH_SIZE]
		batch_ref_answers = answer[i:i+BATCH_SIZE]
		batch_ref_solution = solution[i:i+BATCH_SIZE]

		print(f"正在处理第 {i//BATCH_SIZE + 1} 批问题（共 {len(batch_questions)} 个）...")
		
		batch_messages = [
			{"role": "user", "content": q} for q in batch_questions
		]
		inputs = tokenizer.apply_chat_template(
			batch_messages,
			add_generation_prompt=True, 
			tokenize=True,
			return_dict=True,
			return_tensors="pt",
			padding=True,  
			truncation=True, 
			max_length=512  
		).to(model.device)
		
		with torch.no_grad(): 
			outputs = model.generate(
				**inputs,
				max_new_tokens=MAX_NEW_TOKENS,
				pad_token_id=tokenizer.pad_token_id,
				do_sample=True,  
				temperature=0.1, 
				top_p=0.9,  
				num_return_sequences=1  
			)
		
		input_length = inputs["input_ids"].shape[1] 
		for idx, output in enumerate(outputs):
			answer = tokenizer.decode(
				output[input_length:],
				skip_special_tokens=True, 
				clean_up_tokenization_spaces=True  
			)
			results.append({
				"question": batch_questions[idx],
				"answer": answer.strip(),
				"ref_answer": batch_ref_answers[idx].strip(),
				"ref_solution": batch_ref_solution[idx].strip()
			})
	
	# 新增：保存到JSON文件（原代码缺少保存步骤，补充完整）
	with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
		json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
	config = GRPOConfig.load_yaml("/home/xrrfolder/CELPO/configs/celpo_train.yaml")
	math_500 = Math_500(config)
	test_dataset = math_500.get_test_data()
	train_dataset= math_500.get_train_data()
	question = test_dataset.problems + train_dataset.problems
	solution = test_dataset.solutions + train_dataset.solutions
	answer = test_dataset.answers + train_dataset.answers
	exam(question, solution, answer)
	

