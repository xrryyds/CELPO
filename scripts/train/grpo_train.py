import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True' 
import gc
import json
import logging
from dataclasses import dataclass, asdict
from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from configs import GRPOConfig
import random
from loggers import TrainingLogger
from peft import get_peft_model, LoraConfig, TaskType
from metric import GRPOMathReward
from data_math import Math_500
from utils import collate_fn

# ==================== 环境变量设置 ====================
# ==================== 日志设置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Trainer ====================
class GRPOTrainer:
    def set_seed(self, seed: int = None):
        seed = seed or self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.set_seed(config.seed)
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
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.model = get_peft_model(self.model, peft_config)
            
        logger.info("Loading Reference Model...")
        # self.ref_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        # self.ref_model.eval()
        # for p in self.ref_model.parameters(): p.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.reward_fn = GRPOMathReward()
        
        math_500 = Math_500(config)
        self.train_dataset, self.eval_dataset = math_500.get_dataset()
        
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
            pad_token_id=self.tokenizer.pad_token_id,
            # 加上 top_p 防止采样到极其离谱的词导致 log_prob 极小
            top_p=0.9 
        )
        
        # 关键修正：计算 Mask，一定要排除 Padding Token，否则 KL 和 Loss 会算错
        gen_ids = outputs[:, prompt_len:]
        completion_mask = (gen_ids != self.tokenizer.pad_token_id).int()

        # 优化：使用 disable_adapter 计算 Reference Logits，不需要额外的 ref_model
        with self.model.disable_adapter():
            ref_outputs = self.model(outputs)
            ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            # Gather 并应用 Mask
            ref_token_log_probs = ref_log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)
            ref_token_log_probs = ref_token_log_probs * completion_mask # Masking
            ref_sum_log_probs = ref_token_log_probs.sum(dim=1)
        
        # 预计算 Policy Logits (用于 Experience) - 其实可以跳过，只存 ref 即可，train 时再算 policy
        # 但为了保持逻辑一致，我们只在 Train 阶段重算 Policy

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
                    'ref_log_prob': ref_sum_log_probs[flat_idx].item(),
                    # 必须保存 mask
                    'mask': completion_mask[flat_idx].cpu()
                })
        
        del inputs, outputs, ref_outputs, ref_logits, ref_log_probs, ref_token_log_probs
        torch.cuda.empty_cache()
        return trajectories

    def compute_loss(self, batch_trajectories, advantages):
        self.model.train()
        ids = torch.stack([t['ids'] for t in batch_trajectories]).to(self.config.device)
        masks = torch.stack([t['mask'] for t in batch_trajectories]).to(self.config.device) # 加载 Mask
        prompt_len = batch_trajectories[0]['prompt_len']
        ref_log_probs_sum = torch.tensor([t['ref_log_prob'] for t in batch_trajectories], device=self.config.device)
        advs = torch.tensor(advantages, device=self.config.device)
        
        outputs = self.model(ids)
        logits = outputs.logits[:, prompt_len-1:-1, :]
        gen_ids = ids[:, prompt_len:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)
        
        # 关键修正：应用 Mask
        token_log_probs = token_log_probs * masks
        token_log_probs_sum = token_log_probs.sum(dim=1)
        
        # KL 计算 (Approx: log_p - log_ref)
        kl_div = token_log_probs_sum - ref_log_probs_sum
        
        # 计算 Loss
        # 限制 kl_div 防止数值爆炸 (可选，但在正确 mask 下通常不需要)
        loss = - (advs * token_log_probs_sum - self.config.beta * kl_div)
        
        return loss.mean(), (-advs * token_log_probs_sum).mean().item(), kl_div.mean().item()

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
        eval_batch_size = 2 # 如果显存不够可以调小
        
        logger.info("Evaluating...")
        # 减少评估数量以加快速度
        indices = list(range(min(10, len(self.eval_loader.dataset))))
        eval_subset = [self.eval_loader.dataset[i] for i in indices]
        
        for i in range(0, len(eval_subset), eval_batch_size):
            batch_items = eval_subset[i:i+eval_batch_size]
            prompts = [item['prompt'] for item in batch_items]
            refs = [item['reference_solution'] for item in batch_items]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.config.device)
            try:
                with torch.no_grad():
                    # do_sample=False 用于评估
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
                    raise
        final_path = os.path.join(self.config.output_dir, "final_model")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        logger.info(f"Done. Saved to {final_path}")
        

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    config = GRPOConfig.load_yaml("/home/xrrfolder/CELPO/configs/celpo_train.yaml")
    print(config)
    trainer = GRPOTrainer(config)
    trainer.train()
