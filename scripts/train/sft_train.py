import os
import json
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt 
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    TrainerCallback
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM 
from configs import SftConfig
from data_math import Math_data, GSM8K

# 设置环境
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 功能 1: 自定义回调函数，用于收集数据画图 ---
class SaveMetricsCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.log_history = []
        self.output_dir = output_dir
        self.json_path = os.path.join(output_dir, "training_logs.json")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 记录当前的 step, loss, eval_loss, accuracy 等信息
            log_entry = logs.copy()
            log_entry['step'] = state.global_step
            log_entry['epoch'] = state.epoch
            self.log_history.append(log_entry)
            
            # 实时写入文件，防止程序中断数据丢失
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.log_history, f, indent=4)

# --- 功能 2: 计算准确率 (修复版) ---
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    # SFTTrainer 有时返回 tuple (logits, past_key_values)
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # 【关键修复】
    # 这里的 preds 已经是 preprocess_logits_for_metrics 处理过的 Token ID 了
    # 所以千万不要再做 np.argmax(preds, axis=-1)，否则维度会变成 (Batch,) 导致报错
    pred_ids = preds 
    
    # 【安全措施】强制展平 (Batch, Seq_Len) -> (Batch * Seq_Len)
    # 这样可以忽略 Batch 维度的影响，直接比较所有 Token
    pred_ids = pred_ids.reshape(-1)
    labels = labels.reshape(-1)
    
    # 忽略 label 中的 -100 (padding)
    mask = labels != -100
    
    # 计算准确率
    # 此时 pred_ids[mask] 和 labels[mask] 都是 1维 数组，完全对齐
    correct = (pred_ids[mask] == labels[mask]).sum()
    total = mask.sum()
    
    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy}

# 预处理 Logits 以节省显存 (配合 compute_metrics 使用)
# 这个函数在 GPU 上运行，负责把巨大的 Logits 转成 ID
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

class SftTrainer:
    def __init__(self, config: SftConfig, data: Math_data):
        self.config = config
        self.output_dir = config.output_dir
        self.model_id = config.model_name
        
        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right' 

        train_dataset, eval_dataset = data.get_dataset()
        print(f"训练集样本数: {len(train_dataset.problems)}")
        self.train_dataset = train_dataset.to_hf_dataset()
        self.eval_dataset = eval_dataset.to_hf_dataset()
        
        # 2. 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto", 
        )
            
        # 3. 配置 TrainingArguments
        self.training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            bf16=config.bf16,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            
            # --- 验证与保存策略 ---
            eval_strategy="steps",           
            eval_steps=config.save_steps,    
            save_strategy="steps",           
            save_steps=config.save_steps,
            
            # --- 最佳模型保存机制 ---
            load_best_model_at_end=True,     
            metric_for_best_model="accuracy",
            greater_is_better=True,          
            save_total_limit=2,              
            
            logging_steps=config.logging_steps,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            num_train_epochs=config.num_train_epochs,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.lr_scheduler_type,
            report_to=config.report_to,
            gradient_checkpointing=True, 
            group_by_length=True,
        )
        
        self.lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            bias=config.bias,
            task_type=config.task_type,
        )

        # 4. 初始化 Trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_arguments,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset, 
            tokenizer=self.tokenizer,
            peft_config=self.lora_config,
            formatting_func=self.math_formatting_func,
            max_seq_length=config.max_seq_length,
            packing=False,
            
            # --- 注入指标计算与回调 ---
            compute_metrics=compute_metrics, 
            preprocess_logits_for_metrics=preprocess_logits_for_metrics, 
            callbacks=[SaveMetricsCallback(self.output_dir)]
        )

    def train(self):
        print("开始训练...")
        self.trainer.train() 
        
        save_path = os.path.join(self.output_dir, "best_model_final")
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"训练完成。准确率最高的模型已保存至: {save_path}")
        print(f"绘图数据已保存至: {os.path.join(self.output_dir, 'training_logs.json')}")

    def math_formatting_func(self, examples):
        output_texts = []
        for prompt, ref_sol in zip(examples["prompt"], examples["reference_answer"]):
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

# --- 辅助函数：画图脚本 ---
def plot_training_results(log_file_path):
    if not os.path.exists(log_file_path):
        print("Log file not found.")
        return

    with open(log_file_path, 'r') as f:
        logs = json.load(f)

    steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []
    eval_acc = []

    for entry in logs:
        if 'loss' in entry and 'step' in entry:
            steps.append(entry['step'])
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])
        if 'eval_accuracy' in entry:
            eval_acc.append(entry['eval_accuracy'])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(steps, train_loss, label='Training Loss', color='tab:red', alpha=0.6)
    ax1.plot(eval_steps, eval_loss, label='Validation Loss', color='tab:orange', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    if eval_acc:
        ax2 = ax1.twinx() 
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.plot(eval_steps, eval_acc, label='Validation Accuracy', color='tab:blue', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Training Metrics: Loss & Accuracy')
    fig.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plot_path = log_file_path.replace('.json', '.png')
    plt.savefig(plot_path, dpi=300)
    print(f"图表已保存为: {plot_path}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    
    config_path = "/home/xrrfolder/CELPO/configs/sft.yaml" 

    config = SftConfig.load_yaml(config_path)
    
    print("Loaded Config:", config)
    
    gsm8k = GSM8K() 
    
    trainer = SftTrainer(config, gsm8k)
    trainer.train()
    
    plot_training_results(os.path.join(config.output_dir, "training_logs.json"))
