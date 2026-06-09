# DeepSeek 系列模型技术报告调研

## 调研日期: 2025-02-25

---

# 一、DeepSeek-V3

## 论文信息
- **标题**: DeepSeek-V3 Technical Report
- **链接**: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- **机构**: DeepSeek AI
- **GitHub**: [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)

---

## 1. 模型架构

### 核心参数
| 规格 | 数值 |
|------|------|
| **总参数量** | 671B |
| **激活参数量** | 37B / token |
| **架构类型** | Mixture of Experts (MoE) |
| **上下文长度** | 128K tokens |

### 架构创新
1. **多头隐式注意力 (MLA)**: Multi-head Latent Attention
2. **DeepSeekMoE 架构**: 高效专家混合设计
3. **无辅助损失负载均衡**: 简化 MoE 训练复杂性
4. **多 Token 预测目标**: 增强模型性能

---

## 2. 预训练

### 数据规模
- **训练数据**: 14.8 万亿高质量多样化 tokens
- **训练稳定性**: 全程无不可恢复的损失峰值或回滚

### 训练效率
- **GPU 小时**: 仅 2.788M H800 GPU 小时
- **成本效益**: 相比同等规模模型显著降低训练成本

---

## 3. 评测结果

### 英文能力

| 基准 | DeepSeek-V3 | Llama-3.1 405B | GPT-4o |
|------|-------------|----------------|--------|
| **MMLU** | 87.1% | 84.4% | - |
| **DROP (F1)** | 89.0 | - | - |
| **HumanEval** | 65.2% | 54.9% | - |

### 数学与推理

| 基准 | DeepSeek-V3 | Qwen2.5 72B |
|------|-------------|-------------|
| **GSM8K** | 89.3% | 88.3% |
| **MATH** | 61.6% | - |

### 代码生成

| 基准 | 得分 |
|------|------|
| **MBPP** | 75.4% |
| **LiveCodeBench** | 19.4% Pass@1 |

### 中文能力

| 基准 | 得分 |
|------|------|
| **C-Eval** | 90.1% |
| **CMath** | 90.7% |
| **C-SimpleQA** | 64.8% |

### 对话评估

| 基准 | DeepSeek-V3 | GPT-4o | Claude-3.5-Sonnet |
|------|-------------|--------|-------------------|
| **AlpacaEval 2.0** | 70.0 | 51.1 | - |
| **Arena-Hard** | 85.5 | - | ~85 |

---

## 4. 关键技术要点

1. **高效 MoE**: 671B 总参数，仅激活 37B，平衡性能与效率
2. **训练稳定**: 无辅助损失策略确保训练过程稳定
3. **成本优势**: 训练成本远低于同等规模模型
4. **开源可用**: 模型权重已在 GitHub 发布

---

# 二、DeepSeek-R1

## 论文信息
- **标题**: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **链接**: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- **发表**: Nature (Volume 645, 2025)
- **GitHub**: [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)

---

## 1. 核心创新

### 训练方法
- **纯强化学习**: 无需人类标注的推理轨迹
- **两阶段训练**:
  1. DeepSeek-R1-Zero: 直接对基座模型应用大规模 RL
  2. DeepSeek-R1: 引入冷启动数据后再进行 RL 训练

### 涌现能力
- 自我验证 (Self-verification)
- 反思机制 (Reflection)
- 长链思维 (Long Chain-of-Thought)
- 动态策略调整

---

## 2. 模型架构

| 规格 | 数值 |
|------|------|
| **基座模型** | DeepSeek-V3-Base |
| **总参数量** | 671B |
| **激活参数量** | 37B (MoE) |
| **上下文长度** | 128K tokens |

---

## 3. 蒸馏版本

DeepSeek-R1 提供六个蒸馏版本：

| 模型 | 参数量 | 基座 |
|------|--------|------|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | Qwen |
| DeepSeek-R1-Distill-Qwen-7B | 7B | Qwen |
| DeepSeek-R1-Distill-Qwen-14B | 14B | Qwen |
| DeepSeek-R1-Distill-Qwen-32B | 32B | Qwen |
| DeepSeek-R1-Distill-Llama-8B | 8B | Llama |
| DeepSeek-R1-Distill-Llama-70B | 70B | Llama |

**亮点**: DeepSeek-R1-Distill-Qwen-32B 在多项基准测试中超越 OpenAI-o1-mini

---

## 4. 评测结果

| 基准 | DeepSeek-R1 | OpenAI-o1 |
|------|-------------|-----------|
| **AIME 2024** | 79.8% | ~80% |
| **MATH-500** | 97.3% | ~96% |
| **GPQA Diamond** (Llama-70B蒸馏) | 65.2% | - |

---

## 5. 使用建议

| 配置项 | 推荐值 |
|--------|--------|
| **Temperature** | 0.5-0.7 (推荐 0.6) |
| **System Prompt** | 完全避免使用 |
| **输出格式** | 强制以思考标签开始 |

---

# 三、技术对比总结

| 特性 | DeepSeek-V3 | DeepSeek-R1 |
|------|-------------|-------------|
| **定位** | 通用大模型 | 推理专用模型 |
| **训练方法** | 预训练 + SFT | 纯 RL 训练 |
| **核心优势** | 高效 MoE、成本低 | 强推理、思维链 |
| **适用场景** | 通用任务 | 数学、编程、STEM |

---

## 资源链接

- **DeepSeek-V3 论文**: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- **DeepSeek-R1 论文**: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- **GitHub**: [github.com/deepseek-ai](https://github.com/deepseek-ai)
