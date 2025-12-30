"""
4090优化配置 (24GB显存)
- 使用LoRA: 减少90%参数量
- 混合精度训练: FP16
- 梯度检查点: 减少显存
- 优化batch size和生成长度
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import os
import json
import random
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import re
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 4090优化配置 ====================
@dataclass
class GRPOConfig:
    """针对4090 24GB显存优化的配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen1.5-1.5B"  # 1.5B模型可以跑
    use_lora: bool = True
    lora_r: int = 16  # 增加到16获得更好效果
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # 量化配置 (可选，进一步减少显存)
    use_8bit: bool = False  # 设为True可以跑更大模型
    use_4bit: bool = False
    
    # 训练配置 - 4090优化
    learning_rate: float = 5e-6
    batch_size: int = 2  # 每批2个prompt
    num_epochs: int = 3
    max_length: int = 512
    max_new_tokens: int = 256  # 生成长度
    
    # GRPO配置
    num_samples_per_prompt: int = 4  # 每个prompt生成4个样本
    temperature: float = 0.8
    beta: float = 0.1
    gamma: float = 1.0
    
    # 优化配置
    gradient_accumulation_steps: int = 8  # 梯度累积，有效batch=2*8=16
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True  # 启用梯度检查点
    
    # 混合精度
    fp16: bool = True  # 使用FP16
    
    # 系统配置
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./outputs/grpo_qwen_4090"
    save_steps: int = 200
    logging_steps: int = 10
    
    # 数据配置
    max_samples: int = 500
    train_split: float = 0.9
    num_workers: int = 4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== 工具函数 ====================
def extract_boxed_answer(text: str) -> Optional[str]:
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_final_answer(text: str) -> Optional[str]:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    
    patterns = [
        r'[Tt]he answer is:?\s*([^\n\.]+)',
        r'[Ff]inal answer:?\s*([^\n\.]+)',
        r'=\s*([^\n]+)$',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    
    return None


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    answer = answer.replace(" ", "")
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)
    answer = re.sub(r'[^0-9a-zA-Z\+\-\*/=\.\,]', '', answer)
    return answer.lower()


# ==================== 奖励函数 ====================
class MathRewardFunction:
    def __init__(self):
        self.correct_reward = 10.0
        self.wrong_reward = -1.0
        self.no_answer_penalty = -2.0
        self.length_penalty_coef = 0.001
        
    def __call__(self, generated_text: str, reference_solution: str) -> float:
        pred_answer = extract_final_answer(generated_text)
        ref_answer = extract_final_answer(reference_solution)
        
        if pred_answer is None:
            return self.no_answer_penalty
        
        pred_normalized = normalize_answer(pred_answer)
        ref_normalized = normalize_answer(ref_answer) if ref_answer else ""
        
        if pred_normalized == ref_normalized and ref_normalized:
            reward = self.correct_reward
        else:
            reward = self.wrong_reward
        
        length_penalty = -self.length_penalty_coef * len(generated_text)
        
        format_bonus = 0.0
        if '\\boxed' in generated_text:
            format_bonus += 0.5
        if any(keyword in generated_text.lower() for keyword in ['therefore', 'thus', 'so']):
            format_bonus += 0.3
        
        total_reward = reward + length_penalty + format_bonus
        return total_reward


# ==================== 数据集 ====================
class MATH500Dataset(Dataset):
    def __init__(self, tokenizer, config: GRPOConfig, split='train'):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        logger.info(f"Loading MATH dataset ({split} split)...")
        
        try:
            full_dataset = load_dataset("hendrycks/competition_math", split="test")
        except Exception as e:
            logger.warning(f"Failed to load competition_math: {e}")
            logger.info("Creating dummy dataset for testing...")
            full_dataset = self._create_dummy_dataset()
        
        total_samples = min(config.max_samples, len(full_dataset))
        split_idx = int(total_samples * config.train_split)
        
        if split == 'train':
            self.data = full_dataset.select(range(split_idx))
        else:
            self.data = full_dataset.select(range(split_idx, total_samples))
        
        logger.info(f"Loaded {len(self.data)} samples for {split}")
    
    def _create_dummy_dataset(self):
        from datasets import Dataset as HFDataset
        
        dummy_data = {
            'problem': [
                "What is 2 + 2?",
                "Solve for x: 2x + 3 = 7",
                "What is the square root of 16?",
                "Calculate 5 * 6",
            ] * 125,
            'solution': [
                "2 + 2 = 4. So the answer is \\boxed{4}",
                "2x + 3 = 7, so 2x = 4, therefore x = 2. The answer is \\boxed{2}",
                "The square root of 16 is 4. So \\boxed{4}",
                "5 * 6 = 30. Therefore \\boxed{30}",
            ] * 125,
            'level': ['Level 1'] * 500,
            'type': ['Algebra'] * 500,
        }
        
        return HFDataset.from_dict(dummy_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        problem = item['problem']
        solution = item['solution']
        
        prompt = f"""Solve the following math problem step by step.

Problem: {problem}

Solution:"""
        
        return {
            'prompt': prompt,
            'reference_solution': solution,
            'problem': problem,
        }


def collate_fn(batch):
    return {
        'prompts': [item['prompt'] for item in batch],
        'reference_solutions': [item['reference_solution'] for item in batch],
        'problems': [item['problem'] for item in batch],
    }


# ==================== GRPO训练器 (4090优化) ====================
class GRPOTrainer:
    def __init__(self, config: GRPOConfig):
        self.config = config
        set_seed(config.seed)
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        # 显存监控
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # 加载tokenizer
        logger.info(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 量化配置 (可选)
        quantization_config = None
        if config.use_8bit or config.use_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=config.use_8bit,
                    load_in_4bit=config.use_4bit,
                    bnb_4bit_compute_dtype=torch.float16 if config.use_4bit else None,
                )
                logger.info(f"Using quantization: 8bit={config.use_8bit}, 4bit={config.use_4bit}")
            except:
                logger.warning("BitsAndBytes not available, skipping quantization")
        
        # 加载模型
        logger.info(f"Loading model: {config.model_name}")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        
        if config.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        
        # 梯度检查点
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # 应用LoRA
        if config.use_lora:
            logger.info("Applying LoRA...")
            self.model = self._apply_lora(self.model)
        
        # 参考模型
        logger.info("Loading reference model...")
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # 打印显存使用
        self._print_memory_usage("After model loading")
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        
        # 奖励函数
        self.reward_fn = MathRewardFunction()
        
        # 数据集
        logger.info("Loading datasets...")
        self.train_dataset = MATH500Dataset(self.tokenizer, config, split='train')
        self.eval_dataset = MATH500Dataset(self.tokenizer, config, split='eval')
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # 避免多进程问题
            pin_memory=True,
        )
        
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        
        self.global_step = 0
        self.epoch = 0
        
        logger.info("Trainer initialized successfully!")
    
    def _print_memory_usage(self, prefix=""):
        """打印显存使用情况"""
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"{prefix} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    def _apply_lora(self, model):
        try:
            from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
            
            # 如果使用量化，先准备模型
            if self.config.use_8bit or self.config.use_4bit:
                model = prepare_model_for_kbit_training(model)
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            
            return model
        except ImportError:
            logger.error("PEFT not installed! Install with: pip install peft")
            raise
    
    @torch.no_grad()
    def generate_samples(self, prompts: List[str]) -> Dict:
        """生成样本 - 显存优化版"""
        self.model.eval()
        
        all_samples = []
        all_log_probs = []
        all_ref_log_probs = []
        
        for prompt in prompts:
            samples_for_prompt = []
            log_probs_for_prompt = []
            ref_log_probs_for_prompt = []
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.config.device)
            
            prompt_length = inputs['input_ids'].shape[1]
            
            # 逐个生成样本，避免显存爆炸
            for _ in range(self.config.num_samples_per_prompt):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                generated_ids = outputs[0]
                generated_text = self.tokenizer.decode(
                    generated_ids[prompt_length:],
                    skip_special_tokens=True
                )
                
                # 计算log概率
                model_outputs = self.model(generated_ids.unsqueeze(0))
                logits = model_outputs.logits[0, prompt_length-1:-1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                
                token_log_probs = log_probs.gather(
                    1,
                    generated_ids[prompt_length:].unsqueeze(-1)
                ).squeeze(-1)
                
                total_log_prob = token_log_probs.sum().item()
                
                # 参考模型
                ref_outputs = self.reference_model(generated_ids.unsqueeze(0))
                ref_logits = ref_outputs.logits[0, prompt_length-1:-1, :]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                
                ref_token_log_probs = ref_log_probs.gather(
                    1,
                    generated_ids[prompt_length:].unsqueeze(-1)
                ).squeeze(-1)
                
                ref_total_log_prob = ref_token_log_probs.sum().item()
                
                samples_for_prompt.append({
                    'text': generated_text,
                    'ids': generated_ids,
                    'prompt_length': prompt_length,
                })
                
                log_probs_for_prompt.append(total_log_prob)
                ref_log_probs_for_prompt.append(ref_total_log_prob)
                
                # 清理显存
                del model_outputs, logits, log_probs, ref_outputs, ref_logits, ref_log_probs
                torch.cuda.empty_cache()
            
            all_samples.append(samples_for_prompt)
            all_log_probs.append(log_probs_for_prompt)
            all_ref_log_probs.append(ref_log_probs_for_prompt)
        
        return {
            'samples': all_samples,
            'log_probs': all_log_probs,
            'ref_log_probs': all_ref_log_probs,
        }
    
    def compute_rewards(self, samples: List[List[Dict]], reference_solutions: List[str]) -> List[List[float]]:
        all_rewards = []
        for samples_for_prompt, ref_solution in zip(samples, reference_solutions):
            rewards_for_prompt = []
            for sample in samples_for_prompt:
                reward = self.reward_fn(sample['text'], ref_solution)
                rewards_for_prompt.append(reward)
            all_rewards.append(rewards_for_prompt)
        return all_rewards
    
    def compute_advantages(self, rewards: List[List[float]]) -> List[List[float]]:
        advantages = []
        for group_rewards in rewards:
            group_rewards = np.array(group_rewards)
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards) + 1e-8
            group_advantages = (group_rewards - mean_reward) / std_reward
            group_advantages = group_advantages * self.config.gamma
            advantages.append(group_advantages.tolist())
        return advantages
    
    def compute_loss(
        self,
        samples: List[List[Dict]],
        advantages: List[List[float]],
        old_log_probs: List[List[float]],
        ref_log_probs: List[List[float]],
    ) -> torch.Tensor:
        self.model.train()
        
        total_loss = 0.0
        num_samples = 0
        
        for prompt_samples, prompt_advantages, prompt_old_log_probs, prompt_ref_log_probs in zip(
            samples, advantages, old_log_probs, ref_log_probs
        ):
            for sample, advantage, old_log_prob, ref_log_prob in zip(
                prompt_samples, prompt_advantages, prompt_old_log_probs, prompt_ref_log_probs
            ):
                generated_ids = sample['ids'].unsqueeze(0)
                prompt_length = sample['prompt_length']
                
                model_outputs = self.model(generated_ids)
                logits = model_outputs.logits[0, prompt_length-1:-1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                
                new_token_log_probs = log_probs.gather(
                    1,
                    generated_ids[0, prompt_length:].unsqueeze(-1)
                ).squeeze(-1)
                
                new_log_prob = new_token_log_probs.sum()
                kl_div = new_log_prob - ref_log_prob
                loss = -(advantage * new_log_prob - self.config.beta * kl_div)
                
                total_loss = total_loss + loss
                num_samples += 1
        
        avg_loss = total_loss / num_samples
        return avg_loss
    
    def train_step(self, batch: Dict) -> Dict:
        prompts = batch['prompts']
        reference_solutions = batch['reference_solutions']
        
        generation_outputs = self.generate_samples(prompts)
        rewards = self.compute_rewards(
            generation_outputs['samples'],
            reference_solutions
        )
        advantages = self.compute_advantages(rewards)
        
        loss = self.compute_loss(
            generation_outputs['samples'],
            advantages,
            generation_outputs['log_probs'],
            generation_outputs['ref_log_probs'],
        )
        
        loss.backward()
        
        metrics = {
            'loss': loss.item(),
            'mean_reward': np.mean([np.mean(r) for r in rewards]),
            'max_reward': np.max([np.max(r) for r in rewards]),
            'min_reward': np.min([np.min(r) for r in rewards]),
            'mean_advantage': np.mean([np.mean(a) for a in advantages]),
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        self.model.eval()
        logger.info("Running evaluation...")
        
        eval_metrics = {
            'rewards': [],
            'correct': 0,
            'total': 0,
        }
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            prompts = batch['prompts']
            reference_solutions = batch['reference_solutions']
            
            for prompt, ref_solution in zip(prompts, reference_solutions):
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length,
                ).to(self.config.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                reward = self.reward_fn(generated_text, ref_solution)
                eval_metrics['rewards'].append(reward)
                
                if reward > 5.0:
                    eval_metrics['correct'] += 1
                eval_metrics['total'] += 1
        
        eval_metrics['mean_reward'] = np.mean(eval_metrics['rewards'])
        eval_metrics['accuracy'] = eval_metrics['correct'] / eval_metrics['total']
        
        logger.info(f"Eval - Mean Reward: {eval_metrics['mean_reward']:.4f}, "
                   f"Accuracy: {eval_metrics['accuracy']:.4f}")
        
        return eval_metrics
    
    def save_checkpoint(self, name: str):
        save_path = os.path.join(self.config.output_dir, name)
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {save_path}")
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state': self.optimizer.state_dict(),
        }
        
        torch.save(state, os.path.join(save_path, 'trainer_state.pt'))
        logger.info(f"Checkpoint saved")
    
    def train(self):
        logger.info("=" * 50)
        logger.info("Starting GRPO Training on 4090")
        logger.info("=" * 50)
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"LoRA: r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info("=" * 50)
        
        self._print_memory_usage("Before training")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_metrics = {
                'loss': [],
                'reward': [],
            }
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                
                epoch_metrics['loss'].append(metrics['loss'])
                epoch_metrics['reward'].append(metrics['mean_reward'])
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    if self.global_step % self.config.logging_steps == 0:
                        log_metrics = {
                            'step': self.global_step,
                            'loss': metrics['loss'],
                            'reward': metrics['mean_reward'],
                        }
                        progress_bar.set_postfix(log_metrics)
                    
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
                
                # 每50步清理一次显存
                if step % 50 == 0:
                    torch.cuda.empty_cache()
            
            logger.info(f"\nEpoch {epoch + 1} Summary:")
            logger.info(f"  Average Loss: {np.mean(epoch_metrics['loss']):.4f}")
            logger.info(f"  Average Reward: {np.mean(epoch_metrics['reward']):.4f}")
            
            self._print_memory_usage(f"After epoch {epoch + 1}")
            
            eval_metrics = self.evaluate()
            self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")
        
        self.save_checkpoint("final_model")
        
        logger.info("\n" + "=" * 50)
        logger.info("Training Completed!")
        logger.info("=" * 50)


# ==================== 主函数 ====================
def main():
    """
    4090配置选项:
    
    1. 标准配置 (推荐):
       - 1.5B模型 + LoRA + FP16
       - 显存使用: ~15GB
    
    2. 节省显存:
       - 使用8bit量化
       - 显存使用: ~10GB
    
    3. 最大性能:
       - 增大batch_size到4
       - 显存使用: ~20GB
    """
    
    config = GRPOConfig(
        # 模型配置
        model_name="Qwen/Qwen1.5-1.5B",
        use_lora=True,
        lora_r=16,
        
        # 量化 (可选，节省显存)
        use_8bit=False,  # 改为True可以节省显存
        use_4bit=False,
        
        # 训练配置
        batch_size=2,  # 4090可以设为2-4
        gradient_accumulation_steps=8,
        num_epochs=3,
        
        # GRPO配置
        num_samples_per_prompt=4,
        
        # 优化
        gradient_checkpointing=True,
        fp16=True,
        
        # 数据
        max_samples=500,
        
        output_dir="./outputs/grpo_qwen_4090",
    )
    
    trainer = GRPOTrainer(config)
    trainer.train()
    
    logger.info("\n✅ 训练完成！")
    logger.info(f"模型保存在: {config.output_dir}")


if __name__ == "__main__":
    main()
