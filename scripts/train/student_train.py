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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from prompt import GEN_PROMPT, GEN_HINTS_WIH_ANSWER, GEN_ENHANCE_PROMPT

# ==========================================
# 2. 配置与工具类 (Updated)
# ==========================================

@dataclass
class HintSFTConfig:
    p_hint_start: float = 0.95     
    p_hint_end: float = 0.10       
    hint_fixed_weight: float = 2.0 # Mode B中Hint的固定权重
    gate_threshold: float = 2.5    # 门控阈值 mu (及格线)
    gate_slope: float = 3.0        # 门控斜率 kappa (敏感度)
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
    # ... (保持不变) ...
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
# 3. 核心 Collator (重构：分离Mask)
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
        if self.total_steps == 0: return self.config.p_hint_start
        progress = min(self.current_step / self.total_steps, 1.0)
        return self.config.p_hint_start + progress * (self.config.p_hint_end - self.config.p_hint_start)

    def __call__(self, batch):
        p_hint = self.get_current_p_hint()
        
        input_ids_batch = []
        labels_batch = []
        hint_masks_batch = []   # 新增：标记Hint部分
        answer_masks_batch = [] # 新增：标记Answer部分
        attention_mask_batch = []
        metadata_batch = []

        for item in batch:
            q = item['question']
            b = item['hints']
            c = item['ref_solution']

            use_hint = random.random() < p_hint
            
            # 初始化Mask列表 (长度之后会截断)
            # 我们先生成完整的token ids，然后再生成对应的mask
            
            if use_hint:
                # --- Mode A: Hint Utilization (q + h -> a) ---
                # Hint在Prompt里，不计算loss
                full_text = GEN_ENHANCE_PROMPT.format(question=q, hints=b) + c
                mode = "with_hint"
                
                full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                
                # 找到Prompt结束位置 (Answer开始位置)
                prompt_text = GEN_ENHANCE_PROMPT.format(question=q, hints=b)
                len_prompt = len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
                
                # 构建 Masks
                # Mode A下，Hint是输入，不计算loss，mask全0
                # Answer是输出，mask为1
                current_hint_mask = [0] * len(full_ids)
                current_answer_mask = [0] * len(full_ids)
                
                # Answer部分
                safe_start = min(len_prompt, len(full_ids))
                for i in range(safe_start, len(full_ids)):
                    current_answer_mask[i] = 1

            else:
                # --- Mode B: Hint Generation (q -> h + a) ---
                full_text = GEN_PROMPT.format(question=q) + GEN_HINTS_WIH_ANSWER.format(hints=b, answer=c)
                mode = "no_hint"
                
                full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                
                prompt_text = GEN_PROMPT.format(question=q)
                len_prompt = len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
                
                # Hint结束位置
                hint_header = "# known:\n"
                prefix_upto_hint_end = prompt_text + hint_header + b
                len_hint_end = len(self.tokenizer(prefix_upto_hint_end, add_special_tokens=False).input_ids)
                
                current_hint_mask = [0] * len(full_ids)
                current_answer_mask = [0] * len(full_ids)
                
                # 标记 Hint 区间
                safe_start_hint = min(len_prompt, len(full_ids))
                safe_end_hint = min(len_hint_end, len(full_ids))
                for i in range(safe_start_hint, safe_end_hint):
                    current_hint_mask[i] = 1
                
                # 标记 Answer 区间 (从Hint结束往后)
                for i in range(safe_end_hint, len(full_ids)):
                    current_answer_mask[i] = 1

            # --- 通用截断 ---
            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]
                current_hint_mask = current_hint_mask[:self.max_length]
                current_answer_mask = current_answer_mask[:self.max_length]

            # 构建 Labels: Prompt部分为-100
            labels = [-100] * len(full_ids)
            # 只有 hint_mask 或 answer_mask 为 1 的地方才需要计算 loss
            for i in range(len(full_ids)):
                if current_hint_mask[i] == 1 or current_answer_mask[i] == 1:
                    labels[i] = full_ids[i]

            input_ids_batch.append(torch.tensor(full_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))
            hint_masks_batch.append(torch.tensor(current_hint_mask, dtype=torch.float))
            answer_masks_batch.append(torch.tensor(current_answer_mask, dtype=torch.float))
            attention_mask_batch.append(torch.ones(len(full_ids), dtype=torch.long))
            
            metadata_batch.append({
                "mode": mode, 
                "p_hint": p_hint,
                "raw_text": full_text
            })

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=-100)
        hint_masks = torch.nn.utils.rnn.pad_sequence(hint_masks_batch, batch_first=True, padding_value=0.0)
        answer_masks = torch.nn.utils.rnn.pad_sequence(answer_masks_batch, batch_first=True, padding_value=0.0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_batch, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "hint_masks": hint_masks,     # Pass to Trainer
            "answer_masks": answer_masks, # Pass to Trainer
            "metadata": metadata_batch
        }

# ==========================================
# 4. Trainer (重构：实现动态门控)
# ==========================================
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
        # 取出掩码
        hint_masks = inputs.pop("hint_masks", None)
        answer_masks = inputs.pop("answer_masks", None)
        metadata = inputs.pop("metadata", None)
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # --- 动态Loss计算开始 ---
        loss, debug_info = self.adaptive_gating_loss(logits, labels, hint_masks, answer_masks)
        # ----------------------

        if self.state.global_step % self.hint_config.debug_sample_steps == 0 and self.state.is_local_process_zero:
            self.save_debug_snapshot(metadata, loss.item(), debug_info)
            
        return (loss, outputs) if return_outputs else loss

    def adaptive_gating_loss(self, logits, labels, hint_masks, answer_masks):
        """
        实现论文中的 Confidence-Aware Adaptive Gating
        """
        # 1. 移位操作 (Next Token Prediction)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_hint_masks = hint_masks[..., 1:].contiguous()
        shift_answer_masks = answer_masks[..., 1:].contiguous()

        # 2. 计算每个Token的Loss (不规约)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # [Batch * Seq_Len]
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # Reshape back to [Batch, Seq_Len]
        token_losses = token_losses.view(shift_labels.shape)

        batch_size = token_losses.size(0)
        total_loss = 0.0
        
        # 用于Debug记录
        weights_log = [] 

        # 3. 逐样本计算 (因为每个样本的Gate权重不同)
        # 这里用向量化操作可能会比较复杂，循环 batch 逻辑更清晰且容易处理 mask sum 为 0 的情况
        for i in range(batch_size):
            # 提取当前样本的数据
            sample_losses = token_losses[i]
            h_mask = shift_hint_masks[i]
            a_mask = shift_answer_masks[i]
            
            # --- 计算 Hint Loss ---
            h_count = h_mask.sum()
            if h_count > 0:
                # Mode B: 存在 Hint
                avg_h_loss = (sample_losses * h_mask).sum() / h_count
                
                # === 核心逻辑：Sigmoid Gating ===
                # w_a = sigmoid( k * (mu - loss_h) )
                # 注意：必须 detach avg_h_loss，防止模型为了降低总loss而故意增大hint loss
                gate_input = self.hint_config.gate_slope * (self.hint_config.gate_threshold - avg_h_loss.detach())
                w_a = torch.sigmoid(gate_input)
                
                # 配置中的 Hint 固定权重 (e.g., 2.0)
                w_h = self.hint_config.hint_fixed_weight
                
                # 计算 Answer Loss
                a_count = a_mask.sum()
                if a_count > 0:
                    avg_a_loss = (sample_losses * a_mask).sum() / a_count
                else:
                    avg_a_loss = 0.0
                
                # 组合 Loss
                sample_final_loss = (w_h * avg_h_loss) + (w_a * avg_a_loss)
                
                weights_log.append(f"B(Gate={w_a:.2f}, H_L={avg_h_loss.item():.2f})")
                
            else:
                # Mode A: 没有 Hint (Hint在Input里)，只有 Answer
                # 标准 SFT Loss
                a_count = a_mask.sum()
                if a_count > 0:
                    avg_a_loss = (sample_losses * a_mask).sum() / a_count
                else:
                    avg_a_loss = 0.0
                    
                sample_final_loss = avg_a_loss # 权重为 1.0
                weights_log.append("A")

            total_loss += sample_final_loss

        return total_loss / batch_size, weights_log

    def save_debug_snapshot(self, metadata, current_loss, debug_info):
        if not metadata: return
        # 记录第一个样本的情况
        sample = metadata[0]
        weight_info = debug_info[0] if debug_info else "N/A"
        entry = {
            "step": self.state.global_step,
            "timestamp": datetime.now().isoformat(),
            "loss": current_loss,
            "p_hint": sample["p_hint"],
            "mode": sample["mode"],
            "gate_info": weight_info, # 记录门控权重
            "text_preview": sample["raw_text"][:200].replace("\n", "\\n") + "..."
        }
        with open(self.snapshot_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ... (CurriculumCallback 保持不变) ...
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
# 5. 验证函数 (Updated for Masks)
# ==========================================
def verify_collator(collator, dataset, tokenizer):
    print("\n" + "="*40)
    print(">>> Running Collator Verification")
    print("="*40)
    
    # 强制设置 Mode B
    original_start = collator.config.p_hint_start
    collator.config.p_hint_start = 0.0
    collator.config.p_hint_end = 0.0
    
    batch_data = [dataset[0]]
    output = collator(batch_data)
    
    input_ids = output['input_ids'][0]
    labels = output['labels'][0]
    h_mask = output['hint_masks'][0]
    a_mask = output['answer_masks'][0]
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    print(f"\n[Sample Mode]: {output['metadata'][0]['mode']}")
    print(f"[Sample Text PREVIEW]:\n{output['metadata'][0]['raw_text'][:100]}...")
    print("-" * 20)
    
    print(f"{'Token':<20} | {'Label':<10} | {'Type':<10}")
    print("-" * 50)
    
    for i, (tok, lbl, hm, am) in enumerate(zip(tokens, labels, h_mask, a_mask)):
        if tok == tokenizer.pad_token and i > 20: break
        
        lbl_str = str(lbl.item()) if lbl != -100 else "IGNORE"
        
        type_str = ""
        if hm == 1: type_str = "HINT"
        elif am == 1: type_str = "ANSWER"
        else: type_str = "PROMPT"
            
        print(f"{tok:<20} | {lbl_str:<10} | {type_str:<10}")

    print("="*40 + "\n")
    collator.config.p_hint_start = original_start

# ==========================================
# 6. Main Execution (保持不变)
# ==========================================
def main():
    SEED = 42
    set_seed(SEED)
    # 修改为你的实际路径
    model_name_or_path = "/root/project/data/xrr/OREAL-7B" 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = os.getcwd()
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

    # === 验证 ===
    # 注意这里 hint_fixed_weight=2.0
    hint_config = HintSFTConfig(
        p_hint_start=0.9, p_hint_end=0.1, 
        hint_fixed_weight=2.0, gate_threshold=2.5, gate_slope=3.0
    )
    debug_collator = HintDropoutCollator(tokenizer, hint_config, max_length=512)
    verify_collator(debug_collator, dataset, tokenizer)
    # ===========

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
