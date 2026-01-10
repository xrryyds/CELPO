# GRPO模型推理使用指南

## 概述

本项目提供了完整的GRPO（Group Relative Policy Optimization）模型推理功能，支持加载训练好的模型进行数学问题求解和评估。

## 文件结构

```
scripts/Inference/
├── inference.py          # 核心推理类
├── run_inference.py      # 命令行入口脚本
└── __init__.py

configs/
├── inference_config.yaml # 推理配置文件
└── ...

outputs/                 # 训练输出目录（包含训练好的模型）
```

## 快速开始

### 1. 基本使用

```bash
# 运行默认示例
python scripts/Inference/run_inference.py

# 使用自定义配置
python scripts/Inference/run_inference.py --config configs/inference_config.yaml
```

### 2. 单个问题推理

```bash
python scripts/Inference/run_inference.py --question "求解方程: x^2 - 5x + 6 = 0"
```

### 3. 批量推理

```bash
# 准备输入文件 (input.txt)
echo "计算 1+2+3+...+100 的和" > input.txt
echo "求圆的面积，半径为5" >> input.txt
echo "解方程: 2x + 3 = 7" >> input.txt

# 批量推理
python scripts/Inference/run_inference.py --batch-input input.txt --output results.json
```

### 4. 交互式推理

```bash
python scripts/Inference/run_inference.py --interactive
```

### 5. 数据集评估

```bash
# 在MATH-500数据集上评估
python scripts/Inference/run_inference.py --evaluate --max-samples 100
```

## 配置说明

### inference_config.yaml

```yaml
# 模型配置
model:
  model_path: "./outputs/grpo_fix_final_model"  # 训练好的模型路径
  base_model: "Qwen/Qwen2-1.5B-Instruct"        # 基础模型
  use_lora: true                                # 是否使用LoRA
  lora:
    r: 16                                       # LoRA秩
    alpha: 32                                   # LoRA Alpha
    dropout: 0.05                               # LoRA Dropout

# 推理参数
inference:
  max_length: 1024                              # 输入最大长度
  max_new_tokens: 512                           # 生成最大长度
  temperature: 0.8                              # 温度参数
  top_p: 0.9                                    # Top-p采样
  top_k: 50                                     # Top-k采样
  num_return_sequences: 1                       # 返回序列数
  device: "cuda"                                # 设备

# 评估配置
evaluation:
  dataset_name: "HuggingFaceH4/MATH-500"        # 数据集名称
  split: "test"                                 # 评估分割
  max_samples: 100                              # 最大样本数

# 输出配置
output:
  results_path: "./inference_results.json"      # 结果保存路径
  save_detailed_log: true                       # 保存详细日志
  log_path: "./inference_log.txt"               # 日志文件路径

# 性能优化
optimization:
  batch_size: 4                                 # 批处理大小
  gradient_checkpointing: false                 # 梯度检查点
  fp16: true                                    # 混合精度
```

## API使用

### 基本推理

```python
from scripts.Inference.inference import GRPOInference, InferenceConfig

# 创建配置
config = InferenceConfig(
    model_path="./outputs/grpo_fix_final_model",
    base_model="Qwen/Qwen2-1.5B-Instruct",
    max_new_tokens=512
)

# 初始化推理器
inference = GRPOInference(config)

# 单个问题推理
problem = "求解方程: x^2 - 5x + 6 = 0"
prompt = inference.preprocess_prompt(problem)
result = inference.generate_response(prompt)

print(f"问题: {problem}")
print(f"答案: {result['answer']}")
print(f"完整回答: {result['response']}")
```

### 批量推理

```python
# 批量推理
problems = [
    "计算 1+2+3+...+100 的和",
    "求圆的面积，半径为5",
    "解方程: 2x + 3 = 7"
]

results = inference.batch_inference(problems, batch_size=2)

for result in results:
    print(f"问题: {result['problem']}")
    print(f"答案: {result['answer']}")
```

### 数据集评估

```python
# 在数据集上评估
eval_results = inference.evaluate_on_dataset(
    dataset_name="HuggingFaceH4/MATH-500",
    split="test",
    max_samples=100,
    batch_size=4
)

print(f"准确率: {eval_results['accuracy']:.4f}")
print(f"正确: {eval_results['correct']}/{eval_results['total']}")
```

## 输出格式

### 单个推理结果

```json
{
  "prompt": "预处理后的问题prompt",
  "response": "模型生成的完整回答",
  "answer": "提取的答案",
  "full_output": "完整输出"
}
```

### 批量推理结果

```json
[
  {
    "problem": "原始问题",
    "prompt": "预处理后的prompt",
    "response": "模型回答",
    "answer": "提取的答案",
    "index": 0
  },
  ...
]
```

### 评估结果

```json
{
  "accuracy": 0.85,
  "correct": 85,
  "total": 100,
  "eval_time": 120.5,
  "results": [
    {
      "index": 0,
      "problem": "问题文本",
      "reference_answer": "标准答案",
      "model_answer": "模型答案",
      "is_correct": true,
      "model_response": "模型回答",
      "reference_answer": "标准解答"
    },
    ...
  ]
}
```

## 性能优化

### 批处理

- 使用 `batch_size` 参数进行批处理推理
- 批处理可以显著提高GPU利用率
- 建议根据显存大小调整批处理大小

### 缓存

- `preprocess_prompt` 方法使用 LRU 缓存
- 避免重复的prompt预处理
- 缓存大小默认为1000

### 内存管理

- 自动清理GPU缓存
- 支持梯度检查点
- 混合精度推理

## 故障排除

### 模型加载失败

1. 检查模型路径是否正确
2. 确认模型文件是否存在
3. 检查LoRA配置是否匹配

```bash
# 验证模型路径
ls -la ./outputs/grpo_fix_final_model/
# 应该包含 config.json, pytorch_model.bin, tokenizer.json 等文件
```

### 显存不足

1. 降低 `batch_size`
2. 启用 `gradient_checkpointing`
3. 使用 `fp16` 混合精度

### 推理结果不准确

1. 检查 `temperature` 参数（建议0.7-0.9）
2. 调整 `max_new_tokens`
3. 验证prompt模板是否正确

## 常见问题

### Q: 如何加载不同的训练模型？

A: 修改配置文件中的 `model_path` 或使用命令行参数：

```bash
python scripts/Inference/run_inference.py --model-path ./path/to/your/model
```

### Q: 如何提高推理速度？

A: 
- 使用批处理 (`batch_size > 1`)
- 启用混合精度 (`fp16: true`)
- 降低 `max_new_tokens`
- 使用更快的采样参数

### Q: 如何自定义prompt模板？

A: 修改 `prompt/prompts.py` 中的 `QUESTION_PROMPT`：

```python
QUESTION_PROMPT = """你的自定义prompt模板"""
```

### Q: 如何处理长文本？

A: 
- 增加 `max_length` 和 `max_new_tokens`
- 注意显存使用情况
- 考虑分段处理

## 开发说明

### 扩展功能

1. **自定义评估指标**: 在 `evaluate_on_dataset` 中添加新的评估逻辑
2. **多模型集成**: 实现模型集成推理
3. **Web服务**: 基于Flask/FastAPI提供API服务

### 代码结构

- `GRPOInference`: 核心推理类
- `InferenceConfig`: 配置数据类
- `run_inference.py`: 命令行接口
- `inference_config.yaml`: 配置文件

## 许可证

本项目遵循与主项目相同的许可证。