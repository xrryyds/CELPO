import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import transformers
import logging
from datetime import datetime, timedelta
from dataclasses import asdict
from collections import OrderedDict

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainerCallback,
    set_seed,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from utils import FileIOUtils 

# =============================================================================
# Consistency Reward Function (带 Debug 打印 & Mask 修复)
# =============================================================================
class ConsistencyRewardFunc:
    def __init__(self, model, tokenizer, alpha=2.0, k=0.5, log_file=None, inference_batch_size=2):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.k = k
        self.log_file = log_file
        self.inference_batch_size = inference_batch_size
        self.__name__ = "consistency_reward" 
        
        # [DEBUG设置] 只打印前 3 次调用，用于快速验证
        self.debug_log_count = 3
        
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def __call__(self, prompts, completions, question_with_hints, **kwargs):
        rewards = []
        trajectories = []
        
        # 拼接输入
        inputs_str_1 = [f"{q_h}{c}" for q_h, c in zip(question_with_hints, completions)]
        inputs_str_2 = [f"{p}{c}" for p, c in zip(prompts, completions)]
        
        # =========================================================================
        # [DEBUG] 打印输入样本进行检查 (快速测试核心)
        # =========================================================================
        if self.debug_log_count > 0:
            print(f"\n{'='*20} DEBUG: Reward Input Check (Remaining: {self.debug_log_count}) {'='*20}")
            print(f"[Context 1] (Hint Path):\n{inputs_str_1[0]}")
            print(f"\n[Context 2] (Direct Path) :\n{inputs_str_2[0]}")
            print(f"\n[Completion]:\n{completions[0]}")
            print("="*80 + "\n")
            self.debug_log_count -= 1
        # =========================================================================

        device = self.model.device
        total_len = len(prompts)

        # Mini-Batch 推理循环
        for i in range(0, total_len, self.inference_batch_size):
            batch_end = min(i + self.inference_batch_size, total_len)
            
            sub_inputs_1 = inputs_str_1[i:batch_end]
            sub_inputs_2 = inputs_str_2[i:batch_end]
            
            sub_prompts = prompts[i:batch_end]
            sub_hints = question_with_hints[i:batch_end]
            sub_completions = completions[i:batch_end]

            with torch.no_grad():
                # [修复] 显式传入 attention_mask
                inputs_1_tokens = self.tokenizer(sub_inputs_1, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                inputs_2_tokens = self.tokenizer(sub_inputs_2, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                
                outputs_1 = self.model(input_ids=inputs_1_tokens.input_ids, attention_mask=inputs_1_tokens.attention_mask)
                outputs_2 = self.model(input_ids=inputs_2_tokens.input_ids, attention_mask=inputs_2_tokens.attention_mask)
                
                logits_batch_1 = outputs_1.logits
                logits_batch_2 = outputs_2.logits

            for j in range(len(sub_prompts)):
                ctx_str_1 = sub_hints[j]
                ctx_str_2 = sub_prompts[j]
                
                len_ctx_1 = len(self.tokenizer(ctx_str_1, add_special_tokens=False)["input_ids"])
                len_ctx_2 = len(self.tokenizer(ctx_str_2, add_special_tokens=False)["input_ids"])

                curr_logits_1 = logits_batch_1[j]
                curr_logits_2 = logits_batch_2[j]

                slice_1 = curr_logits_1[len_ctx_1 - 1 : -1]
                slice_2 = curr_logits_2[len_ctx_2 - 1 : -1]

                completion_len_1 = inputs_1_tokens.input_ids[j].ne(self.tokenizer.pad_token_id).sum().item() - len_ctx_1
                completion_len_2 = inputs_2_tokens.input_ids[j].ne(self.tokenizer.pad_token_id).sum().item() - len_ctx_2
                
                valid_len = min(completion_len_1, completion_len_2, slice_1.shape[0], slice_2.shape[0])

                if valid_len <= 0:
                    reward = 0.0
                    kl_val = 0.0
                else:
                    slice_1_valid = slice_1[:valid_len]
                    slice_2_valid = slice_2[:valid_len]

                    p1_probs = F.softmax(slice_1_valid, dim=-1)
                    p2_log_probs = F.log_softmax(slice_2_valid, dim=-1)

                    kl_val = F.kl_div(p2_log_probs, p1_probs, reduction='batchmean').item()
                    reward = self.alpha * torch.exp(torch.tensor(-self.k * kl_val)).item()

                rewards.append(reward)

                if self.log_file:
                    trajectories.append({
                        "step_timestamp": datetime.now().isoformat(),
                        "prompt": sub_prompts[j],
                        "hint_context": sub_hints[j],
                        "completion": sub_completions[j],
                        "kl_divergence": kl_val,
                        "reward": reward
                    })
            
            del logits_batch_1, logits_batch_2, outputs_1, outputs_2
            torch.cuda.empty_cache()

        if self.log_file and len(trajectories) > 0:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for t in trajectories:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            
        return rewards


# 配置 logger
logger = logging.getLogger(__name__)

# =============================================================================
# 辅助函数
# =============================================================================
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_format = "[%(asctime)s][%(levelname)s][Rank %(process)d] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(os.path.join(output_dir, "train.log"), encoding='utf-8')
        ]
    )
    return os.path.join(output_dir, "metrics.jsonl")

def log_environment(args):
    import trl
    import peft
    env_info = OrderedDict()
    env_info["Python"] = sys.version.split()[0]
    env_info["PyTorch"] = torch.__version__
    env_info["Transformers"] = transformers.__version__
    env_info["TRL"] = trl.__version__
    env_info["PEFT"] = peft.__version__
    env_info["CUDA"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
    env_info["GPUs"] = torch.cuda.device_count()
    
    logger.info("*" * 40)
    logger.info("Runtime Environment:")
    for k, v in env_info.items():
        logger.info(f"{k}: {v}")
    logger.info("*" * 40)
    
    with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding='utf-8') as f:
        json.dump(args.to_dict(), f, indent=4)

class CELPOLoggingCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs is not None:
            log_entry = {
                "step": state.global_step,
                "timestamp": datetime.now().isoformat(),
                "epoch": state.epoch,
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
            }
            log_entry.update(logs)
            with open(self.log_file_path, "a", encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")

def format_dataset(questions, questions_with_hints, ref_solutions, ref_answers):
    data = []
    for q, q_hint, sol, ans in zip(questions, questions_with_hints, ref_solutions, ref_answers):
        entry = {
            "prompt": q,
            "question_with_hints": q_hint, 
            "ref_solutions": sol, 
            "ref_answer": ans
        }
        data.append(entry)
    return Dataset.from_list(data)

# =============================================================================
# CELPO Trainer 主类
# =============================================================================
class CELPOTrainer:
    def __init__(self, model_name: str):
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path)) 
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(project_root, "outputs", "celpo_output", timestamp)
        
        self.metrics_log_path = setup_logging(self.output_dir)
        self.trajectory_log_path = os.path.join(self.output_dir, "trajectories.jsonl")
        
        self.model_name = model_name
        self.file_io = FileIOUtils()
        
        logger.info("Loading dataset...")
        self.file_io.load_question_with_hints()
        question, question_with_hint, ref_solution, ref_answer = self.file_io.parse_hints_exam(self.file_io.question_with_hints)
        self.dataset = format_dataset(question, question_with_hint, ref_solution, ref_answer)
        logger.info(f"Dataset loaded. Size: {len(self.dataset)}")

    def train(self):
        set_seed(42)
        
        # [修复] Qwen需要 trust_remote_code=True, 且 padding_side="left"
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            padding_side="left", 
            trust_remote_code=True
        )
        
        # [修复逻辑] 针对 Qwen2 的特定处理
        # 1. 如果 PAD 存在，就用现有的（Qwen2通常是 <|endoftext|>），不要覆盖
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Pad token was None, set to EOS.")
        else:
            logger.info(f"Using existing pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
            
        # 2. [关键修复] 如果 BOS 缺失（Qwen2 默认 None），手动设为 EOS 避免报错
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            tokenizer.bos_token_id = tokenizer.eos_token_id
            logger.info(f"Warning: bos_token was None. Set bos_token to eos_token: {tokenizer.bos_token}")

        logger.info("Loading model...")
        attn_impl = "sdpa" if torch.cuda.is_available() else "eager"
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16, 
            device_map="auto",
            attn_implementation=attn_impl,
            trust_remote_code=True
        )

        # [修复] 强制对齐 Model 和 Tokenizer 的配置
        # 这一步确保 bos_token_id 不再是 None，解决 generation 警告
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        
        # 显式更新 Generation Config
        if model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id
            model.generation_config.bos_token_id = tokenizer.bos_token_id

        # 预处理模型
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        
        # [稳定性] 确保 LoRA 权重为 float32
        logger.info("Casting trainable parameters to float32 for mixed precision stability...")
        for name, param in model.named_parameters():
            if "lora" in name or param.requires_grad:
                param.data = param.data.to(torch.float32)

        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}")

        # 实例化 Reward Function
        consistency_reward = ConsistencyRewardFunc(
            model=model, 
            tokenizer=tokenizer, 
            alpha=2.0,
            k=0.5,
            log_file=self.trajectory_log_path,
            inference_batch_size=2 
        )

        training_args = GRPOConfig(
            output_dir=self.output_dir,
            run_name=f"celpo_{datetime.now().strftime('%Y%m%d')}",
            learning_rate=1e-5,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_steps=50,
            save_strategy="steps",
            save_total_limit=3,
            
            bf16=False,
            fp16=True, 
            
            per_device_train_batch_size=1, 
            gradient_accumulation_steps=8, 
            num_generations=4,              
            max_prompt_length=1536,         
            max_completion_length=2048,     
            
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            
            beta=0.01,
            use_vllm=False,
            report_to=None 
        )

        log_environment(training_args)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[consistency_reward],
            args=training_args,
            train_dataset=self.dataset,
            processing_class=tokenizer,
            callbacks=[CELPOLoggingCallback(self.metrics_log_path)] 
        )

        logger.info("Starting training...")
        start_time = time.time()
        
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"Training interrupted: {e}")
            raise e
        
        cost_time = time.time() - start_time
        logger.info(f"Training finished. Cost: {timedelta(seconds=int(cost_time))}")
        
        logger.info(f"Saving final model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

if __name__ == "__main__":
    print("DEBUG: execution started")
    model_path = "/root/project/data/xrr/OREAL-7B" 
    
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
    else:
        celpo_trainer = CELPOTrainer(model_path)
        celpo_trainer.train()
