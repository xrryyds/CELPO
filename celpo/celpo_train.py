import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
import os



def get_dummy_dataset():
    """
    构造测试数据集。
    Text_B: 干扰信息或前置条件
    Text_A: 核心指令
    Prompt: 输入给模型的实际 Prompt (通常是 B + A)
    """
    data = [
        {
            "prompt": "Background: It is raining today. Question: What is 1+1? Answer:", 
            "text_b": "Background: It is raining today. ",
            "text_a": "Question: What is 1+1? Answer:"
        },
        {
            "prompt": "System: Be sarcastic. User: Hello. Assistant:", 
            "text_b": "System: Be sarcastic. ",
            "text_a": "User: Hello. Assistant:"
        },
        {
            "prompt": "Context: Python coding. Task: Print hello world. Code:", 
            "text_b": "Context: Python coding. ",
            "text_a": "Task: Print hello world. Code:"
        }
    ] * 20 # 重复多次以构成一个 epoch
    return Dataset.from_list(data)

# =============================================================================
# 3. 主训练流程
# =============================================================================
def train():
    # 配置
    model_name = "Qwen/Qwen2.5-0.5B-Instruct" # 使用小模型演示，可换 1.5B/7B
    output_dir = "outputs/consistency_grpo"
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型 (开启 bf16 以节省显存)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )

    # 3. 配置 LoRA (可选，但推荐)
    # GRPO 显存开销较大，使用 LoRA 可以显著降低显存
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    # 可以在这里显式 wrap，也可以让 Trainer 自动处理
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. 实例化奖励函数
    # 将模型引用传递给奖励函数
    consistency_reward = ConsistencyRewardFunc(
        model=model, 
        tokenizer=tokenizer, 
        alpha=10.0,  # 放大奖励值，便于观察
        k=0.5        # 敏感度
    )

    # 5. 加载数据
    dataset = get_dummy_dataset()

    # 6. 配置 GRPO Trainer
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=50,
        bf16=True,
        per_device_train_batch_size=1, # 显存敏感，设为 1
        gradient_accumulation_steps=4,
        num_generations=4,             # Group Size (G)
        max_prompt_length=128,
        max_completion_length=64,
        beta=0.01,                     # RL KL 惩罚系数
        use_vllm=False,                # 自定义复杂 Reward 不方便用 vLLM，建议 False
    )

    # 7. 启动训练
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[consistency_reward], # 注册我们的自定义奖励函数
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    
    # 保存结果
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    train()
