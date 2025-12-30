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
import yaml
import argparse
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==================== 配置类 ====================
@dataclass
class GRPOConfig:
    """GRPO 配置类，支持从YAML加载"""
    # 默认值 (如果yaml中缺失会使用这些值)
    model_name: str = "Qwen/Qwen1.5-1.5B"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    use_8bit: bool = False
    use_4bit: bool = False
    
    learning_rate: float = 5e-6
    batch_size: int = 2
    num_epochs: int = 3
    max_length: int = 512
    max_new_tokens: int = 256
    
    num_samples_per_prompt: int = 4
    temperature: float = 0.8
    beta: float = 0.1
    gamma: float = 1.0
    
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    fp16: bool = True
    
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./outputs/grpo_qwen_4090"
    save_steps: int = 200
    logging_steps: int = 10
    
    max_samples: int = 500
    train_split: float = 0.9
    num_workers: int = 4

    @classmethod
    def load_yaml(cls, yaml_path: str):
        """从YAML文件加载并映射到Config类"""
        logger.info(f"Loading config from {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        return cls(
            # Model
            model_name=cfg['model']['name'],
            max_length=cfg['model']['max_length'],
            max_new_tokens=cfg['model']['max_new_tokens'],
            use_8bit=cfg['model']['quantization'].get('use_8bit', False),
            use_4bit=cfg['model']['quantization'].get('use_4bit', False),
            
            # LoRA
            use_lora=cfg['lora']['enable'],
            lora_r=cfg['lora']['r'],
            lora_alpha=cfg['lora']['alpha'],
            lora_dropout=cfg['lora']['dropout'],
            
            # Training
            learning_rate=float(cfg['training']['learning_rate']),
            batch_size=cfg['training']['batch_size'],
            gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
            num_epochs=cfg['training']['num_epochs'],
            max_grad_norm=cfg['training']['max_grad_norm'],
            gradient_checkpointing=cfg['training']['gradient_checkpointing'],
            fp16=cfg['training']['fp16'],
            
            # GRPO
            num_samples_per_prompt=cfg['grpo']['num_samples_per_prompt'],
            temperature=cfg['grpo']['temperature'],
            beta=cfg['grpo']['beta'],
            gamma=cfg['grpo']['gamma'],
            
            # Data
            max_samples=cfg['data']['max_samples'],
            train_split=cfg['data']['train_split'],
            num_workers=cfg['data']['num_workers'],
            
            # System
            seed=cfg['system']['seed'],
            device=cfg['system']['device'],
            output_dir=cfg['system']['output_dir'],
            save_steps=cfg['system']['save_steps'],
            logging_steps=cfg['system']['logging_steps'],
        )

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
        self.correct_reward = 1.0
        self.wrong_reward = -1.0
        self.no_answer_penalty = -1.0
        # 简化奖励，防止数值过大
        
    def __call__(self, generated_text: str, reference_solution: str) -> float:
        pred_answer = extract_final_answer(generated_text)
        ref_answer = extract_final_answer(reference_solution)
        
        if pred_answer is None:
            return self.no_answer_penalty
        
        pred_normalized = normalize_answer(pred_answer)
        ref_normalized = normalize_answer(ref_answer) if ref_answer else ""
        
        if pred_normalized == ref_normalized and ref_normalized:
            return self.correct_reward
        else:
            return self.wrong_reward

# ==================== 数据集 ====================
class MATH500Dataset(Dataset):
    def __init__(self, tokenizer, config: GRPOConfig, split='train'):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        logger.info(f"Loading MATH dataset ({split} split)...")
        try:
            # 尝试加载真实数据集
            full_dataset = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
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
            'problem': ["What is 1 + 1?"] * 500,
            'solution': ["The answer is \\boxed{2}"] * 500,
            'level': ['Level 1'] * 500,
            'type': ['Algebra'] * 500,
        }
        return HFDataset.from_dict(dummy_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # 添加思维链提示模板
        prompt = f"""<|im_start|>user
Solve the following math problem step by step.
Problem: {item['problem']}
<|im_end|>
<|im_start|>assistant
"""
        return {
            'prompt': prompt,
            'reference_solution': item['solution'],
            'problem': item['problem'],
        }

def collate_fn(batch):
    return {
        'prompts': [item['prompt'] for item in batch],
        'reference_solutions': [item['reference_solution'] for item in batch],
        'problems': [item['problem'] for item in batch],
    }

# ==================== GRPO训练器 ====================
class GRPOTrainer:
    def __init__(self, config: GRPOConfig):
        self.config = config
        set_seed(config.seed)
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        # 备份配置
        with open(os.path.join(config.output_dir, 'run_config.yaml'), 'w') as f:
            yaml.dump(config.__dict__, f)
        
        self._print_memory_usage("Init Start")
        
        # 1. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # 2. 量化配置
        quantization_config = None
        if config.use_8bit or config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=config.use_8bit,
                load_in_4bit=config.use_4bit,
                bnb_4bit_compute_dtype=torch.float16 if config.use_4bit else None,
            )
        
        # 3. 加载模型
        logger.info(f"Loading Model: {config.model_name}")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if config.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        # 4. LoRA
        if config.use_lora:
            self.model = self._apply_lora(self.model)
            
        # 5. 参考模型 (冻结)
        logger.info("Loading Ref Model...")
        self.reference_model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False
            
        self._print_memory_usage("Model Loaded")
        
        # 6. 优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        self.reward_fn = MathRewardFunction()
        
        # 7. 数据加载
        self.train_dataset = MATH500Dataset(self.tokenizer, config, split='train')
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0 # Windows下建议0
        )
        
        self.global_step = 0
        
    def _print_memory_usage(self, prefix=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"[{prefix}] Memory: {allocated:.2f} GB")

    def _apply_lora(self, model):
        from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
        if self.config.use_8bit or self.config.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 针对Qwen/Llama
            bias="none",
        )
        return get_peft_model(model, peft_config)

    @torch.no_grad()
    def generate_samples(self, prompts: List[str]) -> Dict:
        """生成样本 - 显存优化版"""
        self.model.eval()
        all_samples, all_log_probs, all_ref_log_probs = [], [], []
        
        for prompt in prompts:
            # 逐个Prompt生成以节省显存
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
            prompt_len = inputs.input_ids.shape[1]
            
            # 批量生成N个回复
            # 输入inputs维度是[1, seq_len]，expand到[N, seq_len]
            input_ids = inputs.input_ids.expand(self.config.num_samples_per_prompt, -1)
            attention_mask = inputs.attention_mask.expand(self.config.num_samples_per_prompt, -1)
            
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            # 处理结果
            samples_list = []
            log_probs_list = []
            ref_log_probs_list = []
            
            for i in range(self.config.num_samples_per_prompt):
                gen_ids = outputs[i]
                text = self.tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=True)
                
                # 计算当前模型的 Log Prob
                # 注意：为了显存，这里是一个个算的，如果显存够大可以batch算
                with torch.no_grad():
                    logits = self.model(gen_ids.unsqueeze(0)).logits[0, prompt_len-1:-1, :]
                    log_prob = F.log_softmax(logits, dim=-1)
                    token_log_prob = log_prob.gather(1, gen_ids[prompt_len:].unsqueeze(-1)).squeeze(-1).sum().item()
                    
                    # 参考模型 Log Prob
                    ref_logits = self.reference_model(gen_ids.unsqueeze(0)).logits[0, prompt_len-1:-1, :]
                    ref_log_prob = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_prob = ref_log_prob.gather(1, gen_ids[prompt_len:].unsqueeze(-1)).squeeze(-1).sum().item()
                
                samples_list.append({'text': text, 'ids': gen_ids, 'prompt_len': prompt_len})
                log_probs_list.append(token_log_prob)
                ref_log_probs_list.append(ref_token_log_prob)
                
            all_samples.append(samples_list)
            all_log_probs.append(log_probs_list)
            all_ref_log_probs.append(ref_log_probs_list)
            
            torch.cuda.empty_cache()
            
        return {'samples': all_samples, 'log_probs': all_log_probs, 'ref_log_probs': all_ref_log_probs}

    def compute_rewards_and_advantages(self, samples, ref_solutions):
        rewards = []
        for prompt_samples, solution in zip(samples, ref_solutions):
            p_rewards = [self.reward_fn(s['text'], solution) for s in prompt_samples]
            rewards.append(p_rewards)
            
        # 组内标准化 (GRPO核心)
        advantages = []
        for r_group in rewards:
            mean = np.mean(r_group)
            std = np.std(r_group) + 1e-8
            adv = [(r - mean) / std for r in r_group]
            advantages.append(adv)
            
        return rewards, advantages

    def compute_loss(self, samples, advantages, old_log_probs, ref_log_probs):
        self.model.train()
        total_loss = 0
        count = 0
        
        # 扁平化处理
        for i in range(len(samples)): # Prompt idx
            for j in range(len(samples[i])): # Sample idx
                sample = samples[i][j]
                adv = advantages[i][j]
                ref_lp = ref_log_probs[i][j]
                
                # 重新前向传播计算梯度
                input_ids = sample['ids'].unsqueeze(0) # [1, seq_len]
                prompt_len = sample['prompt_len']
                
                outputs = self.model(input_ids)
                logits = outputs.logits[0, prompt_len-1:-1, :]
                
                # 计算新 log prob
                log_probs = F.log_softmax(logits, dim=-1)
                target_ids = input_ids[0, prompt_len:]
                token_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
                new_lp = token_log_probs.sum()
                
                # GRPO Loss: -(Adv * Ratio - Beta * KL)
                # 简化版实现 (直接用log_prob差值近似Ratio，当更新步幅小时)
                # 更标准的PPO式实现需要 exp(new - old)
                
                # KL Divergence approx: log(p/ref) = log_p - log_ref
                kl = new_lp - ref_lp 
                
                # 策略梯度
                loss = -(adv * new_lp - self.config.beta * kl)
                
                total_loss += loss
                count += 1
                
        return total_loss / count

    def train(self):
        logger.info("Starting Training...")
        self.model.train()
        
        progress_bar = tqdm(total=self.config.max_samples * self.config.num_epochs // self.config.batch_size)
        
        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(self.train_dataloader):
                
                # 1. 生成 (Evaluation Mode, No Grad)
                gen_out = self.generate_samples(batch['prompts'])
                
                # 2. 计算奖励和优势
                rewards, advantages = self.compute_rewards_and_advantages(
                    gen_out['samples'], batch['reference_solutions']
                )
                
                # 3. 计算Loss并反向传播 (Train Mode)
                loss = self.compute_loss(
                    gen_out['samples'], 
                    advantages, 
                    gen_out['log_probs'], 
                    gen_out['ref_log_probs']
                )
                
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # 4. 优化器步进
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # 日志
                    avg_reward = np.mean([np.mean(r) for r in rewards])
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.3f}",
                        'reward': f"{avg_reward:.3f}"
                    })
                    progress_bar.update(1)
                    
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")

    def save_checkpoint(self, name):
        path = os.path.join(self.config.output_dir, name)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Saved to {path}")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        config = GRPOConfig.load_yaml(args.config)
    else:
        logger.warning(f"Config {args.config} not found, using defaults.")
        config = GRPOConfig()
        
    trainer = GRPOTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
