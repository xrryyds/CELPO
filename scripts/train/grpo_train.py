import os
# 设置环境变量，加速下载和显存分配
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True' 
import gc
import json
import logging
from dataclasses import dataclass, asdict
from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
from peft import get_peft_model, LoraConfig, TaskType

# 假设这些是你本地的文件，保持导入不变
from configs import GRPOConfig
from loggers import TrainingLogger
from metric import GRPOMathReward
from data_math import GSM8K, Math_data
from utils import collate_fn

# ==================== 日志设置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Trainer ====================


MY_TOKEN = "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" 
class GRPOTrainer:
    def set_seed(self, seed: int = None):
        seed = seed or self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def __init__(self, config: GRPOConfig, data: Math_data):
        self.config = config
        self.set_seed(config.seed)
        self.data = data
        self.logger_backend = TrainingLogger(config.output_dir)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
            json.dump(asdict(config), f, indent=2)
            
        # ================== 模型与Tokenizer加载 ==================
        logger.info(f"Loading Tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True, padding_side='left', token = MY_TOKEN)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Loading Model (FP16)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name, 
            torch_dtype=torch.float16, 
            device_map="cuda:0", 
            trust_remote_code=True,
            token = MY_TOKEN
        )
        
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        if config.use_lora:
            logger.info("Applying LoRA...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r, 
                lora_alpha=config.lora_alpha, 
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        # 不需要显式加载 ref_model，使用 disable_adapter() 技巧即可
        logger.info("Reference Policy: Implicit (via LoRA disable)")
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.reward_fn = GRPOMathReward()
        
        # ================== 数据集加载 ==================
        self.train_dataset, self.eval_dataset = data.get_dataset()
        logger.info(f"Train Dataset Size: {len(self.train_dataset)}")
        logger.info(f"Eval Dataset Size: {len(self.eval_dataset)}")
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        
        self.global_step = 0
        #用于记录最佳准确率
        self.best_acc = -1.0 

    @torch.no_grad()
    def generate_trajectory_parallel(self, prompts: List[str]):
        """生成轨迹并计算 Reference Log Prob"""
        self.model.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length).to(self.config.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # 1. Policy Model 采样生成
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            num_return_sequences=self.config.num_samples_per_prompt,
            pad_token_id=self.tokenizer.pad_token_id,
            top_p=0.9 
        )
        
        # 2. 准备 Mask (只计算生成部分的 Loss/KL，排除 Padding)
        gen_ids = outputs[:, prompt_len:]
        completion_mask = (gen_ids != self.tokenizer.pad_token_id).int()

        # 3. 计算 Reference Logprobs (通过禁用 Adapter)
        # 这就是不需要 ref_model 的关键，直接用基础模型算
        with self.model.disable_adapter():
            ref_outputs = self.model(outputs)
            # Logits shift: output[t] 预测 input[t+1]
            ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            # Gather 对应 token 的概率
            ref_token_log_probs = ref_log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)
            # Apply Mask
            ref_token_log_probs = ref_token_log_probs * completion_mask 
            ref_sum_log_probs = ref_token_log_probs.sum(dim=1)
        
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
                    'ids': outputs[flat_idx].cpu(), # 保存到 CPU 节省显存
                    'prompt_len': prompt_len,
                    'ref_log_prob': ref_sum_log_probs[flat_idx].item(),
                    'mask': completion_mask[flat_idx].cpu()
                })
        
        # 清理缓存
        del inputs, outputs, ref_outputs, ref_logits
        torch.cuda.empty_cache()
        return trajectories

    def compute_loss(self, batch_trajectories, advantages):
        """计算 GRPO Loss"""
        self.model.train()
        
        # 重新加载数据到 GPU
        ids = torch.stack([t['ids'] for t in batch_trajectories]).to(self.config.device)
        masks = torch.stack([t['mask'] for t in batch_trajectories]).to(self.config.device)
        prompt_len = batch_trajectories[0]['prompt_len']
        ref_log_probs_sum = torch.tensor([t['ref_log_prob'] for t in batch_trajectories], device=self.config.device)
        advs = torch.tensor(advantages, device=self.config.device)
        
        # Forward pass (Policy Model)
        outputs = self.model(ids)
        logits = outputs.logits[:, prompt_len-1:-1, :]
        gen_ids = ids[:, prompt_len:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)
        
        # Apply Mask
        token_log_probs = token_log_probs * masks
        token_log_probs_sum = token_log_probs.sum(dim=1)
        
        # KL Divergence Approximation (log_p - log_ref)
        # 这里的 ref_log_probs_sum 是之前 disable_adapter 算出来的
        kl_div = token_log_probs_sum - ref_log_probs_sum
        
        # GRPO Loss: - (Advantage * LogProb - Beta * KL)
        loss = - (advs * token_log_probs_sum - self.config.beta * kl_div)
        
        return loss.mean(), (-advs * token_log_probs_sum).mean().item(), kl_div.mean().item()

    def train_step(self, batch):
        prompts = batch['prompts']
        solutions = batch['reference_solutions']
        
        # 1. 采样 + 计算 Ref Logprobs
        trajectories = self.generate_trajectory_parallel(prompts)
        
        # 2. 计算 Reward
        rewards_matrix = np.zeros((len(prompts), self.config.num_samples_per_prompt))
        flat_rewards = []
        
        for idx, traj in enumerate(trajectories):
            p_idx = traj['prompt_idx']
            s_idx = idx % self.config.num_samples_per_prompt
            # 计算奖励 (比较生成的 text 和 参考答案)
            r = self.reward_fn(traj['text'], solutions[p_idx])
            rewards_matrix[p_idx, s_idx] = r
            flat_rewards.append(r)
            
        # 3. 计算 Advantage (Group Norm)
        mean = rewards_matrix.mean(axis=1, keepdims=True)
        std = rewards_matrix.std(axis=1, keepdims=True) + 1e-8
        advantages = (rewards_matrix - mean) / std
        flat_advantages = advantages.flatten().tolist()
        
        # 4. 计算 Loss 并反向传播
        loss, policy_loss, kl_div = self.compute_loss(trajectories, flat_advantages)
        
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        # 再次清理，防止显存累积
        torch.cuda.empty_cache()
        
        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps, 
            "policy_loss": policy_loss, 
            "kl_div": kl_div, 
            "reward": np.mean(flat_rewards)
        }

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        total_reward = 0
        eval_batch_size = 2 # 评估时 BatchSize 可以小一点
        
        logger.info("Evaluating...")
        
        # 增加评估样本数，确保 best_acc 可靠 (例如取前50个，或者全部)
        eval_indices = list(range(min(50, len(self.eval_loader.dataset))))
        eval_subset = [self.eval_loader.dataset[i] for i in eval_indices]
        
        for i in range(0, len(eval_subset), eval_batch_size):
            batch_items = eval_subset[i:i+eval_batch_size]
            prompts = [item['prompt'] for item in batch_items]
            refs = [item['reference_solution'] for item in batch_items]
            
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.config.device)
            try:
                with torch.no_grad():
                    # do_sample=False (贪婪搜索) 用于评估确定性能力
                    outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                    
                for j, out_ids in enumerate(outputs):
                    # 解码文本
                    text = self.tokenizer.decode(out_ids[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    r = self.reward_fn(text, refs[j])
                    total_reward += r
                    if r > 0: correct += 1
                    total += 1
            except Exception as e:
                logger.error(f"Eval Error: {e}")
                continue
                
        acc = correct / (total + 1e-8)
        mean_r = total_reward / (total + 1e-8)
        logger.info(f"Eval Result - Acc: {acc:.4f}, Mean Reward: {mean_r:.4f}")
        return acc, mean_r

    def train(self):
        logger.info("="*50)
        logger.info(f"Start Training. Total Epochs: {self.config.num_epochs}")
        logger.info("="*50)
        
        self.optimizer.zero_grad()
        
        for epoch in range(self.config.num_epochs):
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(progress):
                try:
                    # 1. 训练一步
                    metrics = self.train_step(batch)
                    
                    # 2. 梯度累积
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1
                        
                        # Logging
                        if self.global_step % self.config.logging_steps == 0:
                            self.logger_backend.log_train(
                                self.global_step, epoch, metrics['loss'], 
                                metrics['policy_loss'], metrics['kl_div'], metrics['reward']
                            )
                            progress.set_postfix(
                                loss=f"{metrics['loss']:.2f}", 
                                r=f"{metrics['reward']:.2f}",
                                kl=f"{metrics['kl_div']:.2f}"
                            )
                        
                        # Evaluation & Save Best
                        if self.global_step % self.config.eval_steps == 0:
                            acc, mean_r = self.evaluate()
                            self.logger_backend.log_eval(self.global_step, acc, mean_r)
                            
                            # === 关键：保存最佳模型逻辑 ===
                            if acc > self.best_acc:
                                self.best_acc = acc
                                logger.info(f"🔥 New Best Accuracy: {self.best_acc:.4f}! Saving model...")
                                best_path = os.path.join(self.config.output_dir, "best_model")
                                # save_pretrained 只会保存 LoRA 权重，体积很小
                                self.model.save_pretrained(best_path)
                                self.tokenizer.save_pretrained(best_path)
                            # ============================
                            
                            self.model.train() # 记得切回训练模式
                        
                        # Regular Checkpoint
                        if self.global_step % self.config.save_steps == 0:
                            ckpt_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
                            self.model.save_pretrained(ckpt_path)
                            
                except torch.OutOfMemoryError:
                    logger.error("OOM detected! Clearing cache and skipping batch.")
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad() # 安全起见清除梯度
                    continue
                    
        # 训练结束保存最后的模型
        final_path = os.path.join(self.config.output_dir, "final_model")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        logger.info(f"Training Done. Final model saved to {final_path}. Best Acc: {self.best_acc:.4f}")
        

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    
    
    # 加载配置
    config_path = "/home/xrrfolder/CELPO/configs/celpo_train.yaml" 
    config = GRPOConfig.load_yaml(config_path)
    
    print("Loaded Config:", config)
    gsm8k = GSM8K(config)
    trainer = GRPOTrainer(config, gsm8k)
    trainer.train()
