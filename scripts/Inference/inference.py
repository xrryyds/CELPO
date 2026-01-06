import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
import json
import logging
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
from functools import lru_cache
import time
from peft import PeftModel, LoraConfig, TaskType
from utils.data_utils import extract_answer,normalize_answer

# 导入配置和工具
from configs import GRPOConfig, GRPOConfigInference
from data_math import Math_500, GSM8K
from prompt import QUESTION_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GRPOInference:
    """GRPO模型推理类"""
    
    def __init__(self, config: GRPOConfigInference):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = torch.device(config.device)
        
        logger.info(f"初始化推理器，设备: {self.device}")
        logger.info(f"加载基础模型: {config.base_model}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, 
            trust_remote_code=True, 
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model, 
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map="auto" if self.device.type == 'cuda' else None,
            trust_remote_code=True
        )
        
        # 如果使用LoRA，加载LoRA权重
        if config.use_lora:
            logger.info(f"加载LoRA权重: {config.model_path}")
            try:
                self.model = PeftModel.from_pretrained(
                    self.model, 
                    config.model_path,
                    is_trainable=False  # 推理模式
                )
                logger.info("LoRA权重加载成功")
            except Exception as e:
                logger.warning(f"加载LoRA权重失败: {e}")
                logger.info("尝试直接加载完整模型...")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        config.model_path,
                        torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                        device_map="auto" if self.device.type == 'cuda' else None,
                        trust_remote_code=True
                    )
                    logger.info("完整模型加载成功")
                except Exception as e2:
                    logger.error(f"模型加载失败: {e2}")
                    raise
        
        self.model.eval()
        
    @lru_cache(maxsize=1000)
    def preprocess_prompt(self, problem: str, max_token: int = 512) -> str:
        """预处理问题，添加prompt模板（带缓存优化）"""
        return QUESTION_PROMPT.format(
            max_token=max_token,
            problem_text=problem
        )
    
    def generate_response(self, prompt: str) -> Dict[str, any]:
        """生成单个响应"""
        # 编码输入
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.config.max_length
        ).to(self.device)
        
        # 生成参数
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": self.config.num_return_sequences
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # 解码输出
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        answer = extract_answer(response)
        answer = normalize_answer(answer)
        print("inputs.input_ids.shape:", inputs.input_ids.shape)      # e.g. [1, 50]
        print("outputs.shape:", outputs.shape)                        # e.g. [1, 50] → 说明没生成！
        print("generated_ids:", generated_ids)
        print("generated_ids.shape:", generated_ids.shape)
        return {
            "prompt": prompt,
            "response": response,
            "answer": answer,
            "full_output": response
        }
    
    def batch_inference(self, problems: List[str], max_token: int = 512, batch_size: int = 4) -> List[Dict[str, any]]:
        """批量推理（优化版本）"""
        logger.info(f"开始批量推理，共 {len(problems)} 个问题，批大小: {batch_size}")
        
        results = []
        
        for i in range(0, len(problems), batch_size):
            batch_problems = problems[i:i+batch_size]
            batch_prompts = [self.preprocess_prompt(p, max_token) for p in batch_problems]
            
            try:
                batch_results = self._batch_generate(batch_prompts, batch_problems, i)
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"批量推理第 {i//batch_size+1} 批时出错: {e}")
                for j, problem in enumerate(batch_problems):
                    try:
                        prompt = self.preprocess_prompt(problem, max_token)
                        result = self.generate_response(prompt)
                        result["problem"] = problem
                        result["index"] = i + j
                        results.append(result)
                    except Exception as e2:
                        logger.error(f"单个推理第 {i+j+1} 个问题时出错: {e2}")
                        results.append({
                            "problem": problem,
                            "index": i + j,
                            "error": str(e2),
                            "response": "",
                            "answer": None
                        })
        
        return results
    
    def _batch_generate(self, batch_prompts: List[str], batch_problems: List[str], start_index: int) -> List[Dict[str, any]]:
        inputs = self.tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.config.max_length
        ).to(self.device)
        
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": self.config.num_return_sequences
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        results = []
        for i, (problem, response) in enumerate(zip(batch_problems, responses)):
            answer = extract_answer(response)
            answer = normalize_answer(answer)
            results.append({
                "problem": problem,
                "prompt": batch_prompts[i],
                "response": response,
                "answer": answer,
                "index": start_index + i
            })
        
        return results
    
    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        return {
            "model_path": self.config.model_path,
            "base_model": self.config.base_model,
            "use_lora": self.config.use_lora,
            "device": str(self.device),
            "dtype": str(self.model.dtype),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def clear_cache(self):
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清理")
            
            
        


def main():
    """主函数 - 演示推理用法"""
    
    config = GRPOConfig.load_yaml("/home/xrrfolder/CELPO/configs/celpo_train.yaml")
    inference_config = GRPOConfigInference.load_yaml("/home/xrrfolder/CELPO/configs/inference_config.yaml")
    
    inference = GRPOInference(inference_config)
    model_info = inference.get_model_info()
    print("\n" + "="*60)
    print("模型信息")
    print("="*60)
    for key, value in model_info.items():
        print(f"{key}: {value}")
    
    # math_500 = Math_500(config)
    # test_dataset = math_500.get_test_data()
    gsm8k = GSM8K(config)
    test_dataset = gsm8k.get_test_data()
    ######################################################################
    # 示例1: 单个推理
    print("\n" + "="*50)
    print("示例1: 单个问题推理")
    print("="*50)
    sample_problem =  test_dataset[0]["prompt"]
    ref_answer = test_dataset[0]["reference_solution"]
    result = inference.generate_response(
        inference.preprocess_prompt(sample_problem)
    )
    
    print(f"问题: {sample_problem}")
    print(f"模型回答: {result['response']}")
    print(f"提取的答案: {result['answer']}")
    print(f"参考答案: {ref_answer}")
    ######################################################################
    # 示例2: 批量推理
    # print("\n" + "="*50)
    # print("示例2: 批量推理")
    # print("="*50)
    
    # test_problems = test_dataset[:]["prompt"]
    
    # batch_results = inference.batch_inference(test_problems, batch_size=2)
    
    # for idx, result in enumerate(batch_results):
    #     print(f"\n问题: {result['problem']}")
    #     print(f"答案: {result['answer']}")
    #     print(f"参考: {test_dataset[idx]['reference_solution']}")
    #     if 'error' in result:
    #         print(f"错误: {result['error']}")
    
    ######################################################################
    # 示例3: 在数据集上评估
    # print("\n" + "="*50)
    # print("示例3: 在MATH-500数据集上评估")
    # print("="*50)
    # try:
    #     eval_results = inference.evaluate_on_dataset(
    #         dataset_name="HuggingFaceH4/MATH-500",
    #         split="test",
    #         max_samples=20,  # 限制样本数量以便快速测试
    #         batch_size=4
    #     )
        
    #     print(f"\n评估结果:")
    #     print(f"准确率: {eval_results['accuracy']:.4f}")
    #     print(f"正确: {eval_results['correct']}/{eval_results['total']}")
    #     print(f"评估耗时: {eval_results['eval_time']:.2f}s")
        
    #     # 保存评估结果
    #     inference.save_results(eval_results['results'], "./inference_results.json")
        
    # except Exception as e:
    #     logger.error(f"评估失败: {e}")
    #     print(f"评估失败: {e}")
    
    inference.clear_cache()


if __name__ == "__main__":
    main()