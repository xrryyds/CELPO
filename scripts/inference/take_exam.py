import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def exam(question, solution, answer):
    # 【优化保留】BATCH_SIZE=8
    # 你的显存充裕，这能大幅提升吞吐量
    BATCH_SIZE = 8  
    MAX_NEW_TOKENS = 14000
    MAX_SEQ_LENGTH = 16384
    
    LOCAL_MODEL_PATH = "/root/autodl-tmp/xrrfolder/models/internlm/OREAL-7B"
    OUTPUT_JSON_PATH = "/root/celpo/CELPO/datasets/exam/exam.json"
    
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    print(f"Loading model from {LOCAL_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    
    # 【关键修改】
    # 1. 移除了 attn_implementation="flash_attention_2"，解决报错。
    # 2. 保留 torch_dtype=torch.bfloat16。
    #    PyTorch 2.4 会自动检测到 bf16 并在底层使用 SDPA (Scaled Dot Product Attention) 加速。
    #    这就是“原生 Flash Attention”，速度非常快，且不需要安装额外库。
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16, 
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

            # 【优化保留】Inference Mode
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.7, 
                    top_p=0.9,
                    use_cache=True 
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
            if "out of memory" in str(e):
                print("显存不足提示: 如果 BS=8 依然 OOM，请改回 4。")
            torch.cuda.empty_cache()
            continue

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done! Results saved to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    from configs import GRPOConfig
    from data_math import Math_500
    
    config = GRPOConfig.load_yaml("/root/celpo/CELPO/configs/celpo_train.yaml")
    math_500 = Math_500(config)
    test_dataset = math_500.get_test_data()
    train_dataset= math_500.get_train_data()
    question = test_dataset.problems + train_dataset.problems
    solution = test_dataset.solutions + train_dataset.solutions
    answer = test_dataset.answers + train_dataset.answers
    print(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    exam(question, solution, answer)
