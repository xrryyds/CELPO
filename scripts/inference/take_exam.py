from debugpy._vendored import project_root
from builtins import print
from builtins import print
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm



class TakeExam:
    def __init__(self, exam_result_json_path: str = "/root/celpo/CELPO/datasets/exam/exam.json"):
        self.BATCH_SIZE = 8  
        self.MAX_NEW_TOKENS = 4096
        self.MAX_SEQ_LENGTH = 6000
    
        self.LOCAL_MODEL_PATH = "/root/project/data/xrr/OREAL-7B"
        self.OUTPUT_JSON_PATH = exam_result_json_path
        
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

        print(f"Loading model from {self.LOCAL_MODEL_PATH}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_MODEL_PATH, trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.LOCAL_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.tokenizer.padding_side = "left" 





    def exam(self, question, solution, answer):
        results = []
        total_batches = (len(question) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        
        for i in tqdm(range(0, len(question), self.BATCH_SIZE), total=total_batches, desc="Inferencing"):
            batch_questions = question[i:i+self.BATCH_SIZE]
            batch_ref_answers = answer[i:i+self.BATCH_SIZE]
            batch_ref_solution = solution[i:i+self.BATCH_SIZE]

            try:
                batch_prompts = []
                for q in batch_questions:
                    q_text = str(q)
                    prompt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": q_text}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    batch_prompts.append(prompt)

                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_SEQ_LENGTH
                ).to(self.model.device)

                # 【优化保留】Inference Mode
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.MAX_NEW_TOKENS,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=0.7, 
                        top_p=0.9,
                        use_cache=True 
                    )

                input_ids_len = inputs["input_ids"].shape[1]
                decoded_outputs = self.tokenizer.batch_decode(
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
                    
                if (i // self.BATCH_SIZE) % 10 == 0:
                    with open(self.OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"\n[Error] Batch {i//self.BATCH_SIZE} failed: {e}")
                if "out of memory" in str(e):
                    print("显存不足提示: 如果 BS=8 依然 OOM，请改回 4。")
                torch.cuda.empty_cache()
                continue

        with open(self.OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Done! Results saved to {self.OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    from configs import GRPOConfig
    from data_math import Math_500
    
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    project_root = os.path.dirname(os.path.dirname(project_root))
    exam_file_path = os.path.join(project_root, "CELPO", "configs", "celpo_train.yaml")
    print(exam_file_path)
    config = GRPOConfig.load_yaml(exam_file_path)
    math_500 = Math_500(config)
    test_dataset = math_500.get_test_data()
    train_dataset= math_500.get_train_data()
    question = test_dataset.problems + train_dataset.problems
    solution = test_dataset.solutions + train_dataset.solutions
    answer = test_dataset.answers + train_dataset.answers
    print(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    take_exam = TakeExam()
    take_exam.exam(question, solution, answer)
