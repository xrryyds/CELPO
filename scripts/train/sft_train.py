import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GSM8KTrainer:
    def __init__(self, model_name="Qwen/Qwen1.5-1.8B", output_dir="./qwen-gsm8k"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
    def load_and_prepare_data(self):
        """加载并准备 GSM8K 数据集"""
        logger.info("Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main")
        
        def format_prompt(example):
            """格式化为指令-响应格式"""
            prompt = f"""Below is a math word problem. Solve it step by step.

### Problem:
{example['question']}

### Solution:
{example['answer']}"""
            return {"text": prompt}
        
        # 处理数据集
        train_dataset = dataset["train"].map(format_prompt)
        test_dataset = dataset["test"].map(format_prompt)
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def load_model_and_tokenizer(self):
        """加载模型和 tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 设置 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            load_in_8bit=False  # 如果显存不足可设为 True
        )
        
        logger.info("Model loaded successfully")
        
    def setup_lora(self):
        """配置 LoRA"""
        logger.info("Setting up LoRA...")
        
        # LoRA 配置
        lora_config = LoraConfig(
            r=64,  # LoRA 秩
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 应用 LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def tokenize_dataset(self, dataset):
        """Tokenize 数据集"""
        def tokenize_function(examples):
            outputs = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors=None
            )
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized
    
    def train(self, train_dataset, eval_dataset=None):
        """训练模型"""
        logger.info("Starting training...")
        
        # Tokenize 数据
        tokenized_train = self.tokenize_dataset(train_dataset)
        tokenized_eval = self.tokenize_dataset(eval_dataset) if eval_dataset else None
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            weight_decay=0.01,
            fp16=False,
            bf16=True,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_total_limit=2,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            report_to="tensorboard",
            optim="adamw_torch",
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 创建 Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # 开始训练
        trainer.train()
        
        # 保存最终模型
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(f"{self.output_dir}/final")
        self.tokenizer.save_pretrained(f"{self.output_dir}/final")
        
        return trainer

def main():
    # 初始化训练器
    trainer = GSM8KTrainer(
        model_name="Qwen/Qwen1.5-1.8B",
        output_dir="./qwen-1.5b-gsm8k"
    )
    
    # 加载数据
    train_dataset, test_dataset = trainer.load_and_prepare_data()
    
    # 加载模型
    trainer.load_model_and_tokenizer()
    
    # 设置 LoRA
    trainer.setup_lora()
    
    # 训练（使用部分测试集作为验证集）
    eval_dataset = test_dataset.select(range(100))  # 使用 100 个样本验证
    trainer.train(train_dataset, eval_dataset)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
