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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from utils import FileIOUtils

# 配置 logger
logger = logging.getLogger(__name__)

# =============================================================================
# 1. 工具函数
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

# =============================================================================
# 3. Callback
# =============================================================================
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
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Loading model...")
        
        attn_impl = "sdpa" if torch.cuda.is_available() else "eager"
        torch_dtype = torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation=attn_impl
        )

        # [修复 1/3] 使用官方工具函数预处理模型
        # 这会自动处理 LayerNorm 的精度和梯度检查点的兼容性
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        
        # [修复 2/3] 强制转换 LoRA 参数为 float32
        # 这是 V100 + fp16 混合精度的关键，防止 GradScaler 报错
        logger.info("Casting trainable parameters to float32 for mixed precision stability...")
        for name, param in model.named_parameters():
            if "lora" in name or param.requires_grad:
                param.data = param.data.to(torch.float32)

        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}")

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
            # [修复 3/3] 设置 use_reentrant=False 以解决显存和反向传播的兼容性
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
