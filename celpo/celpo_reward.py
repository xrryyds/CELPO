import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
import os

class ConsistencyRewardFunc:
    def __init__(self, model, tokenizer, alpha=1.0, k=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.k = k
        self.device = model.device

    def __call__(self, prompts, completions, promptsWithHints, **kwargs):
        rewards = []
        
        for p, c, p_w_h in zip(prompts, completions, promptsWithHints):
            reward = self.compute_single_reward(p, c, p_w_h)
            rewards.append(reward)
            
        return rewards

    def compute_single_reward(self, prompt, promptWithHints, completions):
        input_str_1 = f"{promptWithHints}{completions}"
        input_str_2 = f"{prompt}{completions}"

        ctx_str_1 = promptWithHints
        ctx_str_2 = prompt

        with torch.no_grad():
            tokens_1 = self.tokenizer(input_str_1, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            tokens_2 = self.tokenizer(input_str_2, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            
            len_ctx_1 = len(self.tokenizer(ctx_str_1, add_special_tokens=False)["input_ids"])
            len_ctx_2 = len(self.tokenizer(ctx_str_2, add_special_tokens=False)["input_ids"])

            outputs_1 = self.model(**tokens_1)
            outputs_2 = self.model(**tokens_2)

        logits_1 = outputs_1.logits[0]
        logits_2 = outputs_2.logits[0]

        slice_1 = logits_1[len_ctx_1 - 1 : -1] 
        slice_2 = logits_2[len_ctx_2 - 1 : -1]

        min_len = min(slice_1.shape[0], slice_2.shape[0])
        

        if min_len == 0:
            return 0.0

        slice_1 = slice_1[:min_len]
        slice_2 = slice_2[:min_len]
        
        p1_probs = F.softmax(slice_1, dim=-1)
        p2_log_probs = F.log_softmax(slice_2, dim=-1)

        kl_val = F.kl_div(p2_log_probs, p1_probs, reduction='batchmean').item()

        reward = self.alpha * torch.exp(torch.tensor(-self.k * kl_val)).item()
        
        return reward