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
    set_seed
)
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from utils import FileIOUtils

# 配置 logger
logger = logging.getLogger(__name__)

# =============================================================================
# 1. 工具函数：环境记录与目录管理
# =============================================================================
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志格式
    log_format = "[%(asctime)s][%(levelname)s][Rank %(process)d] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(os.path.join(output_dir, "train.log"), encoding='utf-8') # [优化] 添加 encoding
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
    
    # 保存配置到本地
    with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding='utf-8') as f:
        json.dump(args.to_dict(), f, indent=4)

# =============================================================================
# 2. 增强版 Reward Function (已优化推理效率)
# =============================================================================
class ConsistencyRewardFunc:
    def __init__(self, model, tokenizer, alpha=2.0, k=0.5, log_file=None):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.k = k
        self.log_file = log_file
        self.__name__ = "consistency_reward" 
        
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def __call__(self, prompts, completions, question_with_hints, **kwargs):
        """
        [优化] 使用 Batch 推理替代逐条循环推理，显著提高 GPU 利用率。
        """
        rewards = []
        trajectories = []
        
        # 1. 准备文本列表
        # Group 1: With Hints + Completion
        inputs_str_1 = [f"{q_h}{c}" for q_h, c in zip(question_with_hints, completions)]
        ctx_strs_1 = question_with_hints
        
        # Group 2: Prompt + Completion
        inputs_str_2 = [f"{p}{c}" for p, c in zip(prompts, completions)]
        ctx_strs_2 = prompts

        device = self.model.device

        # 2. 批量 Tokenize (Batch Tokenization)
        # 注意：需要 padding 才能打包成 tensor，同时不添加 special tokens 以保持拼接逻辑一致
        with torch.no_grad():
            inputs_1_tokens = self.tokenizer(inputs_str_1, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            inputs_2_tokens = self.tokenizer(inputs_str_2, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            
            # 3. 批量推理 (Batch Inference) - 核心加速点
            # 一次 forward pass 处理整个 batch
            outputs_1 = self.model(**inputs_1_tokens)
            outputs_2 = self.model(**inputs_2_tokens)
            
            logits_batch_1 = outputs_1.logits
            logits_batch_2 = outputs_2.logits

        # 4. 后处理计算 (Loop over batch results)
        # 由于每个样本的 context 长度不同，这里的切片计算仍在 CPU/循环中处理，但不再阻塞 GPU 推理
        for i in range(len(prompts)):
            # 获取当前样本的 context 长度 (用于切片)
            # 这里只需 tokenize context 来获取长度，开销很小
            len_ctx_1 = len(self.tokenizer(ctx_strs_1[i], add_special_tokens=False)["input_ids"])
            len_ctx_2 = len(self.tokenizer(ctx_strs_2[i], add_special_tokens=False)["input_ids"])

            # 提取对应的 Logits (去除 padding 带来的影响)
            # inputs_1_tokens['attention_mask'][i] 可以告诉我们实际长度，但这里直接用 slice 逻辑
            # 取出当前样本的有效 logit 行
            curr_logits_1 = logits_batch_1[i]
            curr_logits_2 = logits_batch_2[i]

            # 切片逻辑保持原样
            # slice_1 对应 input_str_1 的预测 (shift 1位: predicting next token)
            slice_1 = curr_logits_1[len_ctx_1 - 1 : -1]
            slice_2 = curr_logits_2[len_ctx_2 - 1 : -1]

            # 对齐长度 (处理 padding 区域或不同生成长度)
            # 注意：由于 input_ids 是 padded 的，我们需要确保只取有效 token 的长度
            # 这里简化逻辑：取 completions 的实际长度进行对齐
            # completion 长度 = total_len - context_len
            completion_len_1 = inputs_1_tokens.input_ids[i].ne(self.tokenizer.pad_token_id).sum().item() - len_ctx_1
            completion_len_2 = inputs_2_tokens.input_ids[i].ne(self.tokenizer.pad_token_id).sum().item() - len_ctx_2
            
            # 取两个生成结果中较短的有效长度
            valid_len = min(completion_len_1, completion_len_2, slice_1.shape[0], slice_2.shape[0])

            if valid_len <= 0:
                rewards.append(0.0)
                kl_val = 0.0
            else:
                slice_1_valid = slice_1[:valid_len]
                slice_2_valid = slice_2[:valid_len]

                p1_probs = F.softmax(slice_1_valid, dim=-1)
                p2_log_probs = F.log_softmax(slice_2_valid, dim=-1)

                kl_val = F.kl_div(p2_log_probs, p1_probs, reduction='batchmean').item()
                reward = self.alpha * torch.exp(torch.tensor(-self.k * kl_val)).item()
                rewards.append(reward)

            # 收集轨迹数据
            if self.log_file:
                trajectories.append({
                    "step_timestamp": datetime.now().isoformat(),
                    "prompt": prompts[i],
                    "hint_context": question_with_hints[i],
                    "completion": completions[i],
                    "kl_divergence": kl_val,
                    "reward": rewards[-1] # 使用刚刚计算的 reward
                })

        # 异步或直接写入轨迹文件
        if self.log_file and len(trajectories) > 0:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for t in trajectories:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            
        return rewards

# =============================================================================
# 3. 自定义 Callback：模仿 xtuner 的日志格式
# =============================================================================
class CELPOLoggingCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """当 Trainer 记录日志时触发 (logging_steps)"""
        if state.is_local_process_zero and logs is not None:
            log_entry = {
                "step": state.global_step,
                "timestamp": datetime.now().isoformat(),
                "epoch": state.epoch,
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
            }
            log_entry.update(logs)
            
            # [优化] 添加 encoding='utf-8'
            with open(self.log_file_path, "a", encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")

# =============================================================================
# 4. 主类
# =============================================================================
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

class CELPOTrainer:
    def __init__(self, model_name: str):
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path)) 
        
        # 1. 创建基于时间戳的目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(project_root, "outputs", "celpo_output", timestamp)
        
        # 2. 设置日志
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
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # [优化] 确保有 pad_token，否则 batch inference 会报错
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # [优化] 设置 padding side 为 right (通常生成类模型 left padding 更好，但这里主要是 forward 计算 logit，right padding 处理切片逻辑较直观)
        # 不过为了不改变原逻辑行为，这里不做强制设定，默认通常是 right
        
        logger.info("Loading model...")
        
        attn_impl = "sdpa" if torch.cuda.is_available() else "eager"
        torch_dtype = torch.float16 # V100 使用 float16
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation=attn_impl
        )

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        
        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}")

        # 实例化 Reward Function
        consistency_reward = ConsistencyRewardFunc(
            model=model, 
            tokenizer=tokenizer, 
            alpha=2.0,
            k=0.5,
            log_file=self.trajectory_log_path 
        )

        # 训练参数配置
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
            gradient_accumulation_steps=16, 
            num_generations=16,             
            max_prompt_length=1904,         
            max_completion_length=4096,     
            
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
        
        trainer.train()
        
        cost_time = time.time() - start_time
        logger.info(f"Training finished. Cost: {timedelta(seconds=int(cost_time))}")
        
        logger.info(f"Saving final model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

if __name__ == "__main__":
    # 请替换为你的模型路径
    model_path = "/root/project/data/xrr/OREAL-7B" 
    celpo_trainer = CELPOTrainer(model_path)
    celpo_trainer.train()
