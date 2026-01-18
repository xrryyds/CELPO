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

from prompt import GEN_ENHANCE_PROMPT, GEN_PROMPT, GEN_HINTS_WIH_ANSWER

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

class HintDropoutCollator:
    def __init__(self, tokenizer, hint_config: HintSFTConfig, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.config = hint_config
        self.max_length = max_length
        self.current_step = 0
        self.total_steps = 1
        
        if self.tokenizer.pad_token_id is None:
             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.h_start = tokenizer("<KNOWN>", add_special_tokens=False).input_ids
        self.h_end = tokenizer("</KNOWN>", add_special_tokens=False).input_ids

    def set_progress(self, step, total):
        self.current_step = step
        self.total_steps = max(total, 1)

    def get_current_p_hint(self):
        progress = min(self.current_step / self.total_steps, 1.0)
        return self.config.p_hint_start + progress * (self.config.p_hint_end - self.config.p_hint_start)

    def find_subseq_range(self, seq, sub):
        n, m = len(seq), len(sub)
        if m == 0: return None
        for i in range(n - m + 1):
            if seq[i : i + m] == sub:
                return (i, i + m)
        return None

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
            c = item['']

            use_hint = random.random() < p_hint
        
            if use_hint:
                prompt_str = GEN_ENHANCE_PROMPT.format(question=q, hints=b)
                response_str = c
                mode = "with_hint"
            else:
                prompt_str = GEN_PROMPT.format(question=q)
                response_str = GEN_HINTS_WIH_ANSWER.format(hints=b, answer=c)
                mode = "no_hint"

            p_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
            r_ids = self.tokenizer(response_str, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]

            if len(p_ids) + len(r_ids) > self.max_length:
                print(f"truncate input_ids to max_length!!")
                r_ids = r_ids[:self.max_length - len(p_ids)]
            
            full_ids = p_ids + r_ids
            labels = [-100] * len(p_ids) + r_ids
            weights = [0.0] * len(p_ids) + [1.0] * len(r_ids)

            if mode == "no_hint":
                hs = self.find_subseq_range(r_ids, self.h_start)
                he = self.find_subseq_range(r_ids, self.h_end)

                # 如果找到了完整的标签对，增加中间内容的权重
                if hs and he and he[0] > hs[1]:
                    start_idx = len(p_ids) + hs[0] 
                    end_idx = len(p_ids) + he[1] 
                    
                    for i in range(start_idx, end_idx):
                        if i < len(weights):
                            weights[i] = self.config.hint_loss_weight

            input_ids_batch.append(torch.tensor(full_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))
            weights_batch.append(torch.tensor(weights, dtype=torch.float))
            attention_mask_batch.append(torch.ones(len(full_ids), dtype=torch.long))
            
            metadata_batch.append({
                "mode": mode, 
                "p_hint": p_hint,
                "raw_text": prompt_str + response_str
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

        # 记录 Debug 信息
        if self.state.global_step % self.hint_config.debug_sample_steps == 0 and self.state.is_local_process_zero:
            self.save_debug_snapshot(metadata, loss.item())

        return (loss, outputs) if return_outputs else loss

    def weighted_ce_loss(self, logits, labels, weights):
        # Shift so that tokens < n predict n
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
            
            log_entry = {
                "step": state.global_step,
                "timestamp": datetime.now().isoformat(),
                "epoch": state.epoch,
                **logs
            }
            with open(self.log_file_path, "a", encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")

def main():
    SEED = 42
    set_seed(SEED)
    model_name_or_path = "/root/project/data/xrr/OREAL-7B" 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_root)
    project_root = os.path.dirname(project_root)
    output_dir = os.path.join(project_root, "outputs", "hint_sft", timestamp)
    data_path= os.path.join(project_root, "datasets", "exam", "adv_hints.json")

    metrics_log_path = setup_logging(output_dir)
    snapshot_log_path = os.path.join(output_dir, "debug_snapshots.jsonl")
    
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Output Dir: {output_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "right"

    special_tokens = ["<KNOWN>", "</KNOWN>"]
    if tokenizer.pad_token is None:
        special_tokens.append(tokenizer.eos_token) 
        
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': ["<KNOWN>", "</KNOWN>"]})
    logger.info(f"Added {num_added} special tokens.")

    if not os.path.exists(data_path):
        logger.warning(f"Data file {data_path} not found. Creating dummy data.")
        dummy_data = [
            {"prompt": "1+1=?", "hints": "Use arithmetic.", "response": "2"},
            {"prompt": "Capital of France?", "hints": "It's a city in Europe.", "response": "Paris"}
        ] * 50
        dataset = Dataset.from_list(dummy_data)
    else:
        dataset = Dataset.from_json(data_path)

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    
    for name, param in model.named_parameters():
        if "lora" in name or param.requires_grad:
            param.data = param.data.to(torch.float32)
            
    model.print_trainable_parameters()

    hint_config = HintSFTConfig(
        p_hint_start=0.9, 
        p_hint_end=0.1, 
        hint_loss_weight=4.0
    )
    
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
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        remove_unused_columns=False, 
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
    start_time = time.time()
    
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training interrupted: {e}")
        raise e
    
    cost_time = time.time() - start_time
    logger.info(f"Training finished. Cost: {timedelta(seconds=int(cost_time))}")

    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
