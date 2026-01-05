import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

if __name__ == "__main__":
    model_id = "Qwen/Qwen2-1.5B-Instruct"
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16
    )

    # 加载数据集
    train_dataset = load_dataset('madroid/bilibot-messages', split="train")

    # 格式化函数
    def formatting_prompts_func(example):
        output_texts = []
        for message in example['messages']: 
            text = tokenizer.apply_chat_template(
                message, 
                tokenize=False, 
                add_generation_prompt=False
            )
            output_texts.append(text)
        return output_texts

    # TrainingArguments
    output_dir = "/home/xrrfolder/CELPO/sft_example/output"
    
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=50,       
        logging_steps=10,
        learning_rate=5e-5,
        max_grad_norm=1.0,
        num_train_epochs=1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # LoRA 配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=2048,
        packing=False,
    )
    
    print("开始训练...")
    trainer.train()
    
    save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"训练完成，模型已保存至: {save_path}")
