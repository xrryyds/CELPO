"""
4090 24GB 极致性能版 - CCF A 投稿专用 (提速版)
- 优化: Batch Size=2 (利用剩余显存，速度翻倍)
- 保障: 强制日志刷盘，确保画图数据不丢失
"""
import os
import gc
import time
import json
import random
import re
import datetime
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

# ==================== 环境变量设置 ====================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# [重要] 解决显存碎片的关键配置，对 24GB 显存非常重要
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True' 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# ==================== 日志设置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 配置 (提速修改区) ====================
@dataclass
class GRPOConfig:
    # 1. 模型配置
    model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    
    # 2. 显存与精度
    use_8bit: bool = False    
    use_4bit: bool = False
    gradient_checkpointing: bool = True
    fp16: bool = True
    device: str = "cuda"
    
    # 3. 训练超参数 (提速核心)
    learning_rate: float = 1e-5
    
    # 【修改 1】之前是 1，现在改为 2。
    # 解释: 1.5B 模型较小，24GB 应该能扛住 2*4=8 条并发生成。
    # 如果报错 OOM，请把这里改回 1。
    batch_size: int = 2  
    
    # 【修改 2】之前是 8，现在改为 4。
    # 解释: 保持总的 Effective Batch Size 不变 (2 * 4 = 8)，但更新频率更快。
    gradient_accumulation_steps: int = 4
    
    num_epochs: int = 3
    max_length: int = 1024
    max_new_tokens: int = 512
    
    # 4. GRPO 核心参数
    num_samples_per_prompt: int = 4
    
    temperature: float = 0.9
    beta: float = 0.04 
    max_grad_norm: float = 1.0
    seed: int = 42
    
    # 输出目录
    output_dir: str = f"./outputs/grpo_speed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    save_steps: int = 100
    
    # 【修改 3】保持为 1，确保每个 Step 的数据都存下来画图
    logging_steps: int = 1  
    eval_steps: int = 50
    dataset_name: str = "HuggingFaceH4/MATH-500"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==================== 日志记录器 (确保画图数据) ====================
class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_file = self.log_dir / "train_metrics.jsonl"
        self.eval_log_file = self.log_dir / "eval_metrics.jsonl"
        # 初始化清空
        with open(self.train_log_file, "w") as f: pass
        with open(self.eval_log_file, "w") as f: pass
    
    def log_train(self, step, epoch, total_loss, policy_loss, kl_div, reward):
        data = {
            "step": step, "epoch": epoch, 
            "total_loss": float(total_loss),
            "policy_loss": float(policy_loss), 
            "kl_div": float(kl_div),
            "reward": float(reward), 
            "timestamp": time.time()
        }
        # 强制写盘，防止数据丢失
        with open(self.train_log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
            f.flush()
            os.fsync(f.fileno())
            
    def log_eval(self, step, accuracy, mean_reward):
        data = {"step": step, "accuracy": float(accuracy), "mean_reward": float(mean_reward), "timestamp": time.time()}
        with open(self.eval_log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
            f.flush()
            os.fsync(f.fileno())

# ==================== 数学工具 ====================
def extract_answer(text: str) -> Optional[str]:
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    if matches: return matches[-1].strip()
    patterns = [r'[Tt]he answer is:?\s*([^\n\.]+)', r'[Ff]inal answer:?\s*([^\n\.]+)', r'=\s*([^\n]+)$']
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches: return matches[-1].strip()
    return None

def normalize_answer(answer: str) -> str:
    if answer is None: return ""
    answer = answer.replace(" ", "").lower()
    answer = re.sub(r'\\[a-zA-Z]+', '', answer) 
    answer = re.sub(r'[^0-9a-zA-Z\+\-\*/=\.\,]', '', answer)
    return answer

class MathRewardFunction:
    def __init__(self):
        self.correct_reward = 1.0
        self.wrong_reward = -1.0
        self.format_error_penalty = -0.5
        
    def __call__(self, generated_text: str, reference_solution: str) -> float:
        pred_answer = extract_answer(generated_text)
        ref_answer = extract_answer(reference_solution)
        if pred_answer is None: return self.format_error_penalty
        if normalize_answer(pred_answer) == normalize_answer(ref_answer) and ref_answer:
            return self.correct_reward
        return self.wrong_reward

# ==================== 数据集 ====================
class RealMATHDataset(Dataset):
    def __init__(self, tokenizer, config: GRPOConfig, split='train'):
        self.tokenizer = tokenizer
        logger.info(f"Loading Dataset: {config.dataset_name} ({split})...")
        local_path = "./data/MATH-500"
        dataset = None
        try:
            if os.path.exists(local_path):
                from datasets import load_from_disk
                dataset = load_from_disk(local_path)
            else:
                dataset = load_dataset(config.dataset_name, split='test')
        except Exception as e:
            logger.warning(f"Error loading MATH-500: {e}. Fallback to fake data.")
            dataset = [{"problem": "1+1=?", "solution": "2"}] * 100
        
        if not isinstance(dataset, list):
            dataset = dataset.shuffle(seed=config.seed)
            total_len = len(dataset)
            val_size = max(1, int(total_len * 0.1))
            train_size = total_len - val_size
            if split == 'train': self.data = dataset.select(range(train_size))
            else: self.data = dataset.select(range(train_size, total_len))
        else:
            self.data = dataset 
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'prompt': f"Solve the following math problem step by step.\n\nProblem: {item['problem']}\n\nSolution:",
            'reference_solution': item['solution']
        }

def collate_fn(batch):
    return {
        'prompts': [item['prompt'] for item in batch],
        'reference_solutions': [item['reference_solution'] for item in batch]
    }

# ==================== Trainer ====================
class GRPOTrainer:
    def __init__(self, config: GRPOConfig):
        self.config = config
        set_seed(config.seed)
        self.logger_backend = TrainingLogger(config.output_dir)
        with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
            json.dump(asdict(config), f, indent=2)
            
        logger.info(f"Loading Tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Loading Model (FP16)...")
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        if config.use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.model = get_peft_model(self.model, peft_config)
            
        logger.info("Loading Reference Model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        self.ref_model.eval()
        for p in self.ref_model.parameters(): p.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.reward_fn = MathRewardFunction()
        
        self.train_dataset = RealMATHDataset(self.tokenizer, config, 'train')
        self.eval_dataset = RealMATHDataset(self.tokenizer, config, 'eval')
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        self.global_step = 0

    @torch.no_grad()
    def generate_trajectory_parallel(self, prompts: List[str]):
        self.model.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length).to(self.config.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Generation
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            num_return_sequences=self.config.num_samples_per_prompt,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Reference Log Probs
        ref_outputs = self.ref_model(outputs)
        ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        gen_ids = outputs[:, prompt_len:]
        ref_token_log_probs = ref_log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        
        # Policy Log Probs (Old)
        with torch.no_grad():
            policy_outputs = self.model(outputs)
            policy_logits = policy_outputs.logits[:, prompt_len-1:-1, :]
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            token_log_probs = policy_log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1).sum(dim=1)

        trajectories = []
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        batch_size = len(prompts)
        num_samples = self.config.num_samples_per_prompt
        
        for i in range(batch_size):
            for j in range(num_samples):
                flat_idx = i * num_samples + j
                trajectories.append({
                    'prompt_idx': i,
                    'text': texts[flat_idx],
                    'ids': outputs[flat_idx].cpu(),
                    'prompt_len': prompt_len,
                    'log_prob': token_log_probs[flat_idx].item(),
                    'ref_log_prob': ref_token_log_probs[flat_idx].item()
                })
        
        del inputs, outputs, ref_outputs, ref_logits, policy_outputs, policy_logits, ref_log_probs, policy_log_probs, gen_ids
        torch.cuda.empty_cache()
        return trajectories

    def compute_loss(self, batch_trajectories, advantages):
        self.model.train()
        ids = torch.stack([t['ids'] for t in batch_trajectories]).to(self.config.device)
        prompt_len = batch_trajectories[0]['prompt_len']
        ref_log_probs = torch.tensor([t['ref_log_prob'] for t in batch_trajectories], device=self.config.device)
        advs = torch.tensor(advantages, device=self.config.device)
        
        outputs = self.model(ids)
        logits = outputs.logits[:, prompt_len-1:-1, :]
        gen_ids = ids[:, prompt_len:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        
        kl_div = token_log_probs.detach() - ref_log_probs
        loss = - (advs * token_log_probs - self.config.beta * kl_div)
        return loss.mean(), (-advs * token_log_probs).mean().item(), kl_div.mean().item()

    def train_step(self, batch):
        prompts = batch['prompts']
        solutions = batch['reference_solutions']
        
        trajectories = self.generate_trajectory_parallel(prompts)
        rewards_matrix = np.zeros((len(prompts), self.config.num_samples_per_prompt))
        flat_rewards = []
        
        for idx, traj in enumerate(trajectories):
            p_idx = traj['prompt_idx']
            s_idx = idx % self.config.num_samples_per_prompt
            r = self.reward_fn(traj['text'], solutions[p_idx])
            rewards_matrix[p_idx, s_idx] = r
            flat_rewards.append(r)
            
        mean = rewards_matrix.mean(axis=1, keepdims=True)
        std = rewards_matrix.std(axis=1, keepdims=True) + 1e-8
        advantages = (rewards_matrix - mean) / std
        flat_advantages = advantages.flatten().tolist()
        
        loss, policy_loss, kl_div = self.compute_loss(trajectories, flat_advantages)
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        torch.cuda.empty_cache()
        
        return {"loss": loss.item() * self.config.gradient_accumulation_steps, "policy_loss": policy_loss, "kl_div": kl_div, "reward": np.mean(flat_rewards)}

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        total_reward = 0
        eval_batch_size = 2
        
        logger.info("Evaluating...")
        indices = list(range(min(20, len(self.eval_loader.dataset))))
        eval_subset = [self.eval_loader.dataset[i] for i in indices]
        
        for i in range(0, len(eval_subset), eval_batch_size):
            batch_items = eval_subset[i:i+eval_batch_size]
            prompts = [item['prompt'] for item in batch_items]
            refs = [item['reference_solution'] for item in batch_items]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.config.device)
            try:
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                for j, out_ids in enumerate(outputs):
                    text = self.tokenizer.decode(out_ids[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    r = self.reward_fn(text, refs[j])
                    total_reward += r
                    if r > 0: correct += 1
                    total += 1
            except Exception: continue
        acc = correct / (total + 1e-8)
        mean_r = total_reward / (total + 1e-8)
        logger.info(f"Eval - Acc: {acc:.4f}, Reward: {mean_r:.4f}")
        return acc, mean_r

    def train(self):
        logger.info("="*50)
        logger.info(f"Start Training (Speed Mode). BS: {self.config.batch_size}")
        logger.info("="*50)
        self.optimizer.zero_grad()
        for epoch in range(self.config.num_epochs):
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(progress):
                try:
                    metrics = self.train_step(batch)
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1
                        if self.global_step % self.config.logging_steps == 0:
                            self.logger_backend.log_train(self.global_step, epoch, metrics['loss'], metrics['policy_loss'], metrics['kl_div'], metrics['reward'])
                            progress.set_postfix(loss=f"{metrics['loss']:.2f}", r=f"{metrics['reward']:.2f}")
                        if self.global_step % self.config.eval_steps == 0:
                            acc, mean_r = self.evaluate()
                            self.logger_backend.log_eval(self.global_step, acc, mean_r)
                            self.model.train()
                        if self.global_step % self.config.save_steps == 0:
                            self.model.save_pretrained(os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}"))
                except torch.OutOfMemoryError:
                    logger.error("OOM! Reduce Batch Size to 1.")
                    torch.cuda.empty_cache()
                    continue
        final_path = os.path.join(self.config.output_dir, "final_model")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        logger.info(f"Done. Saved to {final_path}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    config = GRPOConfig()
    trainer = GRPOTrainer(config)
    trainer.train()
