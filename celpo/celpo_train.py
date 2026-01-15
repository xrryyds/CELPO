import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from celpo import ConsistencyRewardFunc
from utils import FileIOUtils
import os


def format_dataset(questions, questions_with_hints, ref_solutions, ref_answers):
    data = []
    for q, q_hint, sol, ans in zip(questions, questions_with_hints, ref_solutions, ref_answers):
        entry = {
            "prompt": q,
            "question_with_hints": q_hint,
            "ref_soutions": sol, 
            "ref_answer": ans
        }
        
        data.append(entry)
        
    return Dataset.from_list(data)



class CELPOTrainer:
    def __init__(self, model_name: str):
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path)) 

        self.output_dir = os.path.join(project_root, "outputs", "celpo_output")
        self.model_name = model_name
        self.file_io = FileIOUtils()
        self.file_io.load_question_with_hints()
        question, question_with_hint, ref_solution, ref_answer = self.file_io.parse_hints_exam(self.file_io.question_with_hints)
        self.dataset = format_dataset(question, question_with_hint, ref_solution, ref_answer)
        print(self.dataset[0])
        

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading model...")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            # V100 不支持 FlashAttention2，改为使用 PyTorch 原生的 sdpa 加速
            attn_implementation="sdpa" if torch.cuda.is_available() else "eager"
        )


        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
    

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        consistency_reward = ConsistencyRewardFunc(
            model=model, 
            tokenizer=tokenizer, 
            alpha=2,
            k=0.5       
        )

        training_args = GRPOConfig(
            output_dir=self.output_dir,
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_steps=50,
            bf16=False,
            fp16=True,
            per_device_train_batch_size=4, # 显存敏感，设为 1
            gradient_accumulation_steps=4,
            num_generations=16,             # Group Size (G)
            max_prompt_length=1904,
            max_completion_length=4096,
            beta=0.01,                     # RL KL 惩罚系数
            use_vllm=False,                # 自定义复杂 Reward 不方便用 vLLM，建议 False
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[consistency_reward], # 注册我们的自定义奖励函数
            args=training_args,
            train_dataset=self.dataset,
            processing_class=tokenizer,
        )

        print("Starting training...")
        trainer.train()
        
        # 保存结果
        print(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)

if __name__ == "__main__":
    celpo_trainer = CELPOTrainer("/root/project/data/xrr/OREAL-7B")
    celpo_trainer.train()
