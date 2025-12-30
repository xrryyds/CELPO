import logging
import yaml
from dataclasses import dataclass
import random
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO 配置类，支持从YAML加载"""
    # 默认值 (如果yaml中缺失会使用这些值)
    model_name: str = "Qwen/Qwen1.5-1.5B"
    
    # lora 配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # 量化配置
    use_8bit: bool = False
    use_4bit: bool = False

    # 训练配置
    learning_rate: float = 5e-6
    batch_size: int = 2
    num_epochs: int = 3
    max_length: int = 512
    max_new_tokens: int = 256

    # GRPO 算法配置
    num_samples_per_prompt: int = 4
    temperature: float = 0.8
    beta: float = 0.1
    gamma: float = 1.0

    # 其他训练配置
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    fp16: bool = True

    # 设备和随机种子配置
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./outputs/grpo_qwen_4090"
    save_steps: int = 200
    logging_steps: int = 10

    # 数据配置
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
