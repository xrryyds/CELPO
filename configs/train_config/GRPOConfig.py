import logging
import yaml
from dataclasses import dataclass
import numpy as np
import datetime  # 补充缺失的导入

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    # 1. 模型配置
    model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # 2. 显存与精度
    use_8bit: bool = False    
    use_4bit: bool = False
    gradient_checkpointing: bool = False
    fp16: bool = True
    device: str = "cuda"
    
    # 3. 训练超参数 (追求速度与稳定)
    learning_rate: float = 1e-6
    batch_size: int = 2
    # 保持总步长等效：原 4*1 -> 现 1*4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 10
    max_length: int = 1024
    max_new_tokens: int = 512
    
    # 4. GRPO 核心参数
    # [建议] 设为 8，GRPO 需要一定数量的样本来计算基线。
    # BS=1 时，1*8 = 8条序列，24GB 显存处理 1.5B 模型完全足够。
    num_samples_per_prompt: int = 4
    temperature: float = 0.9
    beta: float = 0.05
    gamma: float = 1.0
    
    # 5. 补充缺失的字段（解决load_yaml传入报错）
    max_samples: int = 500
    train_split: str = "train"
    num_workers: int = 1
    
    max_grad_norm: float = 1.0
    seed: int = 42
    output_dir: str = f"./outputs/grpo_fix_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_steps: int = 100
    logging_steps: int = 5
    eval_steps: int = 50
    dataset_name: str = "HuggingFaceH4/MATH-500"
    
    thinking_max_tokens: int = 512
    top_p: float = 0.9

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

            # Data（补充缺失字段的传入）
            max_samples=cfg['data']['max_samples'],
            train_split=cfg['data']['train_split'],
            num_workers=cfg['data']['num_workers'],

            seed=cfg['system']['seed'],
            device=cfg['system']['device'],
            output_dir=cfg['system']['output_dir'],
            save_steps=cfg['system']['save_steps'],
            logging_steps=cfg['system']['logging_steps'],
            eval_steps=cfg['system']['eval_steps'], 
            
            thinking_max_tokens = cfg['thinking']['thinking_max_tokens']
            top_p = cfg['training']['top_p']
        )