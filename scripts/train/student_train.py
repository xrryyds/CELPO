import os
import sys
import json
import time
import random
import torch
import torch.nn.functional as F
import transformers
import logging
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass, field
import peft

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    set_seed
)
from transformers.trainer_pt_utils import get_parameter_names
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from prompt import GEN_PROMPT, GEN_HINTS_WIH_ANSWER, GEN_ENHANCE_PROMPT
# ==========================================
# 2. 配置与工具类
# ==========================================

@dataclass
class HintSFTConfig:
    p_hint_start: float = 0.95     
    p_hint_end: float = 0.10       
    hint_loss_weight: float = 4.0  
    debug_sample_steps: int = 50

logger = logging.getLogger(__name__)

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

def log_environment(args, output_dir):
    env_info = OrderedDict()
    env_info["Python"] = sys.version.split()[0]
    env_info["PyTorch"] = torch.__version__
    env_info["Transformers"] = transformers.__version__
    env_info["PEFT"] = peft.__version__
    env_info["CUDA"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
    env_info["GPUs"] = torch.cuda.device_count()
    
    logger.info("*" * 40)
    logger.info("Runtime Environment:")
    for k, v in env_info.items():
        logger.info(f"{k}: {v}")
    logger.info("*" * 40)
    
    with open(os.path.join(output_dir, "training_args.json"), "w", encoding='utf-8') as f:
        json.dump(args.to_dict(), f, indent=4)

# ==========================================
# 3. 核心 Collator (逻辑修正版)
# ==========================================
class HintDropoutCollator:
    def __init__(self, tokenizer, hint_config: HintSFTConfig, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.config = hint_config
        self.max_length = max_length
        self.current_step = 0
        self.total_steps = 1
        
        if self.tokenizer.pad_token_id is None:
             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def set_progress(self, step, total):
        self.current_step = step
        self.total_steps = max(total, 1)

    def get_current_p_hint(self):
        # 线性衰减
        if self.total_steps == 0: return self.config.p_hint_start
        progress = min(self.current_step / self.total_steps, 1.0)
        return self.config.p_hint_start + progress * (self.config.p_hint_end - self.config.p_hint_start)

    def __call__(self, batch):
        p_hint = self.get_current_p_hint()
        
        input_ids_batch = []
        labels_batch = []
        weights_batch = []
        attention_mask_batch = []
        metadata_batch = []

        for item in batch:
            q = item['question']
            b = item['hints']
            c = item['ref_solution']

            use_hint = random.random() < p_hint
        
            if use_hint:
                # --- Mode A: Hint Utilization ---
                # 拼接完整文本
                full_text = GEN_ENHANCE_PROMPT.format(question=q, hints=b) + c
                mode = "with_hint"
                
                # 1. 编码 Full Text
                full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                
                # 2. 找到 Prompt 结束位置 (不计算 Loss)
                prompt_text = GEN_ENHANCE_PROMPT.format(question=q, hints=b)
                len_prompt = len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
                
                weights = [1.0] * len(full_ids)

            else:
                # --- Mode B: Hint Generation ---
                # 拼接完整文本
                # GEN_PROMPT 结尾没有换行，GEN_HINTS_WIH_ANSWER 开头没有换行，手动加一个换行符如果需要
                # 但根据模板定义，GEN_HINTS_WIH_ANSWER 开头是 "# known:"
                # GEN_PROMPT 结尾是 "# Question:\n{question}\n" (通常)
                # 这里的拼接直接相连即可，因为 GEN_HINTS_WIH_ANSWER 自带结构
                full_text = GEN_PROMPT.format(question=q) + GEN_HINTS_WIH_ANSWER.format(hints=b, answer=c)
                mode = "no_hint"
                
                # 1. 编码 Full Text
                full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                
                # 2. 找到 Prompt 结束位置
                prompt_text = GEN_PROMPT.format(question=q)
                len_prompt = len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
                
                # 3. 找到 Hint 结束位置
                # 构造 "Prompt + Hint部分" 的前缀字符串
                # 模板结构: ... # Question:\n{q}\n# known:\n{hints}\n# Answer:...
                # 我们需要构造直到 {hints} 结束的部分
                hint_header = "# known:\n"
                prefix_upto_hint_end = prompt_text + hint_header + b
                
                len_hint_end = len(self.tokenizer(prefix_upto_hint_end, add_special_tokens=False).input_ids)
                
                # 初始化权重
                weights = [1.0] * len(full_ids)
                
                # 应用 Hint 加权
                # 范围: [len_prompt, len_hint_end)
                safe_start = min(len_prompt, len(full_ids))
                safe_end = min(len_hint_end, len(full_ids))
                
                if safe_end > safe_start:
                    for i in range(safe_start, safe_end):
                        weights[i] = self.config.hint_loss_weight

            # --- 通用截断与Mask ---
            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]
                weights = weights[:self.max_length]

            labels = [-100] * len(full_ids)
            
            # 只有 Response 部分计算 Loss (从 len_prompt 开始)
            response_start = min(len_prompt, len(full_ids))
            for i in range(response_start, len(full_ids)):
                labels[i] = full_ids[i]
            
            # Prompt 区域权重强制归零
            for i in range(response_start):
                weights[i] = 0.0

            input_ids_batch.append(torch.tensor(full_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))
            weights_batch.append(torch.tensor(weights, dtype=torch.float))
            attention_mask_batch.append(torch.ones(len(full_ids), dtype=torch.long))
            
            metadata_batch.append({
                "mode": mode, 
                "p_hint": p_hint,
                "raw_text": full_text
            })

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=-100)
        weights = torch.nn.utils.rnn.pad_sequence(weights_batch, batch_first=True, padding_value=0.0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_batch, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_weights": weights, 
            "metadata": metadata_batch
        }

# ... (Trainer 和 Callback 保持不变) ...
class HintSFTTrainer(Trainer):
    def __init__(self, hint_config: HintSFTConfig, snapshot_file: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_config = hint_config
        self.snapshot_file = snapshot_file
        os.makedirs(os.path.dirname(snapshot_file), exist_ok=True)
        with open(snapshot_file, "w", encoding="utf-8") as f:
            f.write("")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        weights = inputs.pop("loss_weights", None) 
        metadata = inputs.pop("metadata", None)  
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.weighted_ce_loss(logits, labels, weights)
        if self.state.global_step % self.hint_config.debug_sample_steps == 0 and self.state.is_local_process_zero:
            self.save_debug_snapshot(metadata, loss.item())
        return (loss, outputs) if return_outputs else loss

    def weighted_ce_loss(self, logits, labels, weights):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_weights = shift_weights.view(-1)
        token_losses = loss_fct(shift_logits, shift_labels)
        token_losses = token_losses * shift_weights
        valid_elements = shift_weights.sum()
        loss = token_losses.sum() / (valid_elements + 1e-8)
        return loss

    def save_debug_snapshot(self, metadata, current_loss):
        if not metadata: return
        sample = metadata[0]
        entry = {
            "step": self.state.global_step,
            "timestamp": datetime.now().isoformat(),
            "loss": current_loss,
            "p_hint": sample["p_hint"],
            "mode": sample["mode"],
            "text_preview": sample["raw_text"][:200] + "..."
        }
        with open(self.snapshot_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

class CurriculumCallback(TrainerCallback):
    def __init__(self, collator, log_file_path):
        self.collator = collator
        self.log_file_path = log_file_path
    def on_step_begin(self, args, state, control, **kwargs):
        self.collator.set_progress(state.global_step, state.max_steps)
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs is not None:
            logs["p_hint"] = self.collator.get_current_p_hint()
            log_entry = {"step": state.global_step, "timestamp": datetime.now().isoformat(), "epoch": state.epoch, **logs}
            with open(self.log_file_path, "a", encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")

# ==========================================
# 4. 验证函数 (Debugging Tool)
# ==========================================
def verify_collator(collator, dataset, tokenizer):
    """
    运行此函数以可视化检查 Collator 的输出是否正确。
    特别是检查 Mode B 下 Hint 是否被正确加权。
    """
    print("\n" + "="*40)
    print(">>> Running Collator Verification")
    print("="*40)
    
    # 强制设置 p_hint = 0.0 以测试 Mode B (生成 Hint)
    original_start = collator.config.p_hint_start
    collator.config.p_hint_start = 0.0
    collator.config.p_hint_end = 0.0
    
    # 取一个样本
    batch_data = [dataset[0]]
    output = collator(batch_data)
    
    input_ids = output['input_ids'][0]
    labels = output['labels'][0]
    weights = output['loss_weights'][0]
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    print(f"\n[Sample Mode]: {output['metadata'][0]['mode']}")
    print(f"[Sample Text]:\n{output['metadata'][0]['raw_text']}")
    print("-" * 20)
    
    # 打印每个 Token 的状态
    print(f"{'Token':<20} | {'Label':<10} | {'Weight':<10}")
    print("-" * 50)
    
    for i, (tok, lbl, w) in enumerate(zip(tokens, labels, weights)):
        # 仅打印非 padding 部分
        if tok == tokenizer.pad_token and i > 10: break
        
        lbl_str = str(lbl.item()) if lbl != -100 else "IGNORE"
        
        # 高亮显示高权重部分 (Hint)
        w_str = f"{w.item():.1f}"
        if w.item() > 1.5:
            w_str += " (HINT!)"
            
        print(f"{tok:<20} | {lbl_str:<10} | {w_str:<10}")

    print("="*40 + "\n")
    
    # 恢复配置
    collator.config.p_hint_start = original_start

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    SEED = 42
    set_seed(SEED)
    model_name_or_path = "/root/project/data/xrr/OREAL-7B" 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = os.getcwd() # 假设当前在根目录运行
    output_dir = os.path.join(project_root, "outputs", "hint_sft", timestamp)
    data_path= os.path.join(project_root, "datasets", "exam", "adv_hints.json")

    metrics_log_path = setup_logging(output_dir)
    snapshot_log_path = os.path.join(output_dir, "debug_snapshots.jsonl")
    
    logger.info(f"Model: {model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists(data_path):
        logger.warning(f"Data file {data_path} not found. Creating dummy data.")
        dummy_data = [
            {"question": "1+1=?", "hints": "Use arithmetic.", "ref_solution": "2"},
            {"question": "Capital of France?", "hints": "It's a city in Europe.", "ref_solution": "Paris"}
        ] * 50
        dataset = Dataset.from_list(dummy_data)
    else:
        dataset = Dataset.from_json(data_path)

    # === 插入验证步骤 ===
    hint_config = HintSFTConfig(p_hint_start=0.9, p_hint_end=0.1, hint_loss_weight=4.0)
    debug_collator = HintDropoutCollator(tokenizer, hint_config, max_length=512)
    verify_collator(debug_collator, dataset, tokenizer)
    # ==================

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if tokenizer.pad_token is None:
        model.config.pad_token_id = model.config.eos_token_id

    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM", bias="none", modules_to_save=["embed_tokens", "lm_head"] 
    )
    model = get_peft_model(model, peft_config)
    for name, param in model.named_parameters():
        if "lora" in name or param.requires_grad:
            param.data = param.data.to(torch.float32)
    model.print_trainable_parameters()

    collator = HintDropoutCollator(tokenizer, hint_config, max_length=1024)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=f"hint_sft_{timestamp}",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        report_to=None
    )
    
    log_environment(training_args, output_dir)

    trainer = HintSFTTrainer(
        hint_config=hint_config,
        snapshot_file=snapshot_log_path,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[CurriculumCallback(collator, metrics_log_path)]
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
