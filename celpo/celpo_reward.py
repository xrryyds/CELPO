import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
import os


class ConsistencyRewardFunc:
    def __init__(self, model, tokenizer, alpha=2.0, k=0.5, log_file=None, inference_batch_size=2):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.k = k
        self.log_file = log_file
        self.inference_batch_size = inference_batch_size
        self.__name__ = "consistency_reward" 
        
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def __call__(self, prompts, completions, question_with_hints, **kwargs):
        rewards = []
        trajectories = []
        
        inputs_str_1 = [f"{q_h}{c}" for q_h, c in zip(question_with_hints, completions)]
        inputs_str_2 = [f"{p}{c}" for p, c in zip(prompts, completions)]
        
        device = self.model.device
        total_len = len(prompts)

        # Mini-Batch 推理循环
        for i in range(0, total_len, self.inference_batch_size):
            batch_end = min(i + self.inference_batch_size, total_len)
            
            sub_inputs_1 = inputs_str_1[i:batch_end]
            sub_inputs_2 = inputs_str_2[i:batch_end]
            
            sub_prompts = prompts[i:batch_end]
            sub_hints = question_with_hints[i:batch_end]
            sub_completions = completions[i:batch_end]

            with torch.no_grad():
                inputs_1_tokens = self.tokenizer(sub_inputs_1, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                inputs_2_tokens = self.tokenizer(sub_inputs_2, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                
                outputs_1 = self.model(**inputs_1_tokens)
                outputs_2 = self.model(**inputs_2_tokens)
                
                logits_batch_1 = outputs_1.logits
                logits_batch_2 = outputs_2.logits

            for j in range(len(sub_prompts)):
                ctx_str_1 = sub_hints[j]
                ctx_str_2 = sub_prompts[j]
                
                len_ctx_1 = len(self.tokenizer(ctx_str_1, add_special_tokens=False)["input_ids"])
                len_ctx_2 = len(self.tokenizer(ctx_str_2, add_special_tokens=False)["input_ids"])

                curr_logits_1 = logits_batch_1[j]
                curr_logits_2 = logits_batch_2[j]

                slice_1 = curr_logits_1[len_ctx_1 - 1 : -1]
                slice_2 = curr_logits_2[len_ctx_2 - 1 : -1]

                completion_len_1 = inputs_1_tokens.input_ids[j].ne(self.tokenizer.pad_token_id).sum().item() - len_ctx_1
                completion_len_2 = inputs_2_tokens.input_ids[j].ne(self.tokenizer.pad_token_id).sum().item() - len_ctx_2
                
                valid_len = min(completion_len_1, completion_len_2, slice_1.shape[0], slice_2.shape[0])

                if valid_len <= 0:
                    reward = 0.0
                    kl_val = 0.0
                else:
                    slice_1_valid = slice_1[:valid_len]
                    slice_2_valid = slice_2[:valid_len]

                    p1_probs = F.softmax(slice_1_valid, dim=-1)
                    p2_log_probs = F.log_softmax(slice_2_valid, dim=-1)

                    kl_val = F.kl_div(p2_log_probs, p1_probs, reduction='batchmean').item()
                    reward = self.alpha * torch.exp(torch.tensor(-self.k * kl_val)).item()

                rewards.append(reward)

                if self.log_file:
                    trajectories.append({
                        "step_timestamp": datetime.now().isoformat(),
                        "prompt": sub_prompts[j],
                        "hint_context": sub_hints[j],
                        "completion": sub_completions[j],
                        "kl_divergence": kl_val,
                        "reward": reward
                    })
            
            del logits_batch_1, logits_batch_2, outputs_1, outputs_2
            torch.cuda.empty_cache()

        if self.log_file and len(trajectories) > 0:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for t in trajectories:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            
        return rewards
