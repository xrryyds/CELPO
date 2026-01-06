import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM # 引入DataCollator
from configs import SftConfig
from data_math import Math_data, GSM8K
import gc

class SftTrainer:
    def __init__(self, config: SftConfig, data: Math_data):
        self.config = config
        self.output_dir = config.output_dir
        self.model_id = config.model_name
        
        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # 修正 Padding 设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # 某些模型可能需要设置 pad_token_id
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 显式设置 padding_side，防止潜在警告
        self.tokenizer.padding_side = 'right' 

        self.train_dataset, self.eval_dataset = data.get_dataset()
        
        # 2. 加载模型 (增加 device_map 和 flash_attention)
        # 注意：使用 flash_attention_2 需要显卡支持 (Amphere架构以上)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto", 
            attn_implementation="flash_attention_2" # 推荐开启，显存更省，速度更快
        )
            
        self.training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            bf16=config.bf16,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            save_steps=config.save_steps,       
            logging_steps=config.logging_steps,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            num_train_epochs=config.num_train_epochs,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.lr_scheduler_type,
            report_to=config.report_to,
            # 必须开启下列选项以确保 padding 不参与 loss 计算
            gradient_checkpointing=True, 
            group_by_length=True, # 将长度相似的样本分在一组，提高训练效率
        )
        
        self.lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            bias=config.bias,
            task_type=config.task_type,
        )


        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_arguments,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset, # 建议加入验证集
            tokenizer=self.tokenizer,
            peft_config=self.lora_config,
            formatting_func=self.math_formatting_func,
            max_seq_length=config.max_seq_length,
            packing=False,
        )

    def train(self):
        print("开始训练...")
        self.trainer.train()
        save_path = os.path.join(self.output_dir, "final_model")
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"训练完成，模型已保存至: {save_path}")

    def math_formatting_func(self, examples):
        output_texts = []
        for prompt, ref_sol in zip(examples["prompt"], examples["reference_solution"]):
            messages = [
                {"role": "user", "content": prompt},  
                {"role": "assistant", "content": ref_sol}  
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,  
                add_generation_prompt=False  
            )
            output_texts.append(text)
        return output_texts

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    
    config_path = "/home/xrrfolder/CELPO/configs/celpo_train.yaml" 
    config = SftConfig.load_yaml(config_path)
    
    print("Loaded Config:", config)
    
    gsm8k = GSM8K(config) 
    
    trainer = SftTrainer(config, gsm8k)
    trainer.train()

