# 推荐系统算法调研报告

> 由 4 个并行 Agent 同时调研生成

---

## 一、召回算法论文

### 1. 协同过滤类

| 算法 | 论文 | 年份 | 会议 | 核心思想 | 适用场景 |
|------|------|------|------|----------|----------|
| BPR | Bayesian Personalized Ranking from Implicit Feedback | 2009 | UAI | 成对比较学习用户偏好排序 | 隐式反馈场景 |
| NeuMF | Neural Collaborative Filtering | 2017 | WWW | 结合矩阵分解和MLP | 需要非线性建模 |
| LightGCN | Simplifying and Powering GCN for Recommendation | 2020 | SIGIR | 简化图卷积，仅保留邻域聚合 | 大规模稀疏数据 |
| NGCF | Neural Graph Collaborative Filtering | 2019 | SIGIR | 二分图上图卷积捕获高阶关系 | 复杂关系建模 |

### 2. 向量召回类

| 算法 | 论文 | 年份 | 会议 | 核心思想 | 适用场景 |
|------|------|------|------|----------|----------|
| DSSM | Deep Structured Semantic Models | 2013 | CIKM | 双塔结构映射到语义空间 | 搜索推荐、文本匹配 |
| YouTube DNN | Deep Neural Networks for YouTube | 2016 | RecSys | 深度网络建模用户历史 | 视频推荐 |
| MIND | Multi-Interest Network with Dynamic Routing | 2019 | KDD | 动态路由捕获多兴趣 | 电商多兴趣建模 |

### 3. 图神经网络类

| 算法 | 论文 | 年份 | 会议 | 核心思想 | 适用场景 |
|------|------|------|------|----------|----------|
| PinSage | Graph CNN for Web-Scale Recommender | 2018 | KDD | 随机游走采样高效训练 | Pinterest大规模推荐 |
| GraphSAGE | Inductive Representation Learning on Graphs | 2017 | NIPS | 归纳式图表示学习 | 冷启动问题 |
| SGL | Self-supervised Graph Learning | 2021 | SIGIR | 图对比学习增强表示 | 数据稀疏场景 |

---

## 二、排序算法论文

### 1. 特征交叉类

| 算法 | 论文 | 年份 | 会议 | 核心创新 | 改进点 |
|------|------|------|------|----------|--------|
| FM | Factorization Machines | 2010 | ICDM | 隐向量内积建模二阶特征交叉 | 处理稀疏特征交叉 |
| FFM | Field-aware FM for CTR | 2016 | RecSys | 每个field独立隐向量 | 考虑field差异 |
| DeepFM | FM-based Neural Network | 2017 | IJCAI | 结合FM和DNN | 无需人工特征工程 |
| xDeepFM | Explicit and Implicit Feature Interactions | 2018 | KDD | CIN网络显式高阶交叉 | 更好建模显式交叉 |
| DCN | Deep & Cross Network | 2017 | ADKDD | Cross网络显式特征交叉 | 有效学习有界度交叉 |

### 2. 注意力机制类

| 算法 | 论文 | 年份 | 会议 | 核心创新 | 改进点 |
|------|------|------|------|----------|--------|
| DIN | Deep Interest Network | 2018 | KDD | 注意力动态聚合历史行为 | 根据候选动态关注历史 |
| DIEN | Deep Interest Evolution Network | 2019 | AAAI | GRU建模兴趣演化 | 考虑兴趣时序变化 |
| AutoInt | Automatic Feature Interaction | 2019 | CIKM | Multi-Head Self-Attention | 自适应学习重要交叉 |

### 3. 多任务学习类

| 算法 | 论文 | 年份 | 会议 | 核心创新 | 改进点 |
|------|------|------|------|----------|--------|
| MMOE | Multi-gate Mixture-of-Experts | 2018 | KDD | 门控专家混合模型 | 自适应选择专家 |
| PLE | Progressive Layered Extraction | 2020 | RecSys | 渐进式分离共享/任务专家 | 减少负迁移 |
| ESMM | Entire Space Multi-Task Model | 2018 | SIGIR | 建模整个样本空间 | 消除样本选择偏差 |

---

## 三、工业界实践案例

### 阿里巴巴
- **DIN/DIEN/MIND**：注意力和多兴趣建模
- **EBR**：基于embedding的召回系统
- **可借鉴**：多兴趣建模、冷启动解决方案

### 字节跳动
- **Monolith**：大规模分布式ML系统，千亿级参数
- **多模态推荐**：视频内容理解
- **可借鉴**：多模态融合、实时性优化

### 美团
- **LBS推荐**：地理位置感知
- **多目标优化**：用户+商家+平台利益平衡
- **可借鉴**：O2O场景设计、时空特征利用

### YouTube/Google
- **Two-Tower架构**：候选生成+排序
- **可借鉴**：双塔设计、负样本采样

### Meta
- **DLRM**：稀疏+稠密特征融合
- **社交推荐**：图建模好友影响力
- **可借鉴**：多任务学习、大规模embedding

---

## 四、2024-2025 最新进展

### 热点方向

#### 1. LLM + 推荐系统 (LLM4Rec)

**技术范式分类** (参考综述 [arXiv:2307.02046](https://arxiv.org/abs/2307.02046)):
- **Pre-training**: LLM 作为特征编码器学习表示
- **Fine-tuning**: 针对推荐任务微调 LLM
- **Prompting**: 通过提示词引导 LLM 执行推荐

| 技术路线 | 代表工作 | 说明 |
|----------|----------|------|
| 预训练+微调 | P5 | 推荐任务统一为文本生成 |
| RAG | RecRAG | 检索增强生成推荐 |
| 指令调优 | InstructRec | 零样本推荐能力 |

#### 2. 生成式推荐

| 方向 | 代表工作 | 说明 |
|------|----------|------|
| 扩散模型 | DiffuRec | 基于扩散过程的序列推荐 |
| 对话推荐 | LLM-based | 多轮对话理解 |
| 生成式检索 | TIGER | Semantic ID 直接生成物品 |

---

### 生成式推荐候选模型 (详细)

#### 基于 LLM 的生成式推荐

| 模型 | 会议 | 核心创新 | 代码 |
|------|------|----------|------|
| **P5** | RecSys 2022 | 统一框架，将所有推荐任务转为 seq2seq 文本生成 | [GitHub](https://github.com/jeykigung/P5) |
| **TALLRec** | RecSys 2023 | 参数高效微调，单卡 3090 可训练 LLaMA-7B，少样本 (<100) 有效 | [GitHub](https://github.com/SAI990323/TALLRec) |
| **TIGER** | NeurIPS 2023 | Semantic ID 生成式检索，首个基于语义ID的推荐模型，改善冷启动 | - |
| **LC-Rec** | ICDE 2024 | 向量量化 + 对齐调优，弥合语言语义与协作语义鸿沟 | [GitHub](https://github.com/RUCAIBox/LC-Rec) |
| **CoLLM** | TKDE 2025 | 外部协作模型映射到 LLM 嵌入空间，冷热场景兼顾 | - |
| **LLaRA** | 2024 | 混合提示 + 课程学习，融合 ID 嵌入与文本特征 | [GitHub](https://github.com/ljy0ustc/LLaRA) |

#### 基于扩散模型的生成式推荐

| 模型 | 会议 | 核心创新 | 代码 |
|------|------|----------|------|
| **DiffuRec** | SIGIR 2023 | 将物品表示为分布而非固定向量，扩散+逆扩散还原目标物品 | [GitHub](https://github.com/WHUIR/DiffuRec) |
| **DreamRec** | 2024 | 结合扩散模型与序列推荐 | - |

#### 跨域/迁移学习模型

| 模型 | 会议 | 核心创新 | 代码 |
|------|------|----------|------|
| **UniSRec** | KDD 2022 | 基于文本描述的通用序列表示，跨域迁移 | [GitHub](https://github.com/RUCAIBox/UniSRec) |
| **VQ-Rec** | WWW 2023 | 文本→代码→表示，向量量化中间表示，跨域微调 | [GitHub](https://github.com/RUCAIBox/VQ-Rec) |

#### 工业级大规模模型

| 模型 | 来源 | 核心创新 | 说明 |
|------|------|----------|------|
| **HSTU** | Meta 2024 | 序列转导器架构，1.5万亿参数，比 FlashAttention2 快 5-15x | 在线 A/B 提升 12.4% |

#### LLM Agent 推荐

| 模型 | 会议 | 核心创新 | 代码 |
|------|------|----------|------|
| **RecMind** | 2023 | Self-Inspiring 算法，LLM 自主 Agent，零样本推荐 | - |

#### 关键技术突破

1. **Semantic ID（语义ID）**
   - 问题：传统 item ID 超出 LLM 词表
   - 方案：为每个物品学习语义化的 codeword 元组
   - 效果：支持端到端生成，改善冷启动

2. **协作语义对齐**
   - 问题：LLM 擅长语言语义，但缺乏协作信号
   - 方案：设计特定调优任务融合两种语义 (LC-Rec)
   - 效果：直接从全量物品集生成推荐

3. **高效微调**
   - LoRA/QLoRA 等参数高效方法
   - 小样本（<100条）即可有效适配
   - 跨域泛化能力强

4. **向量量化 (VQ)**
   - 问题：文本特征过度依赖，跨域迁移困难
   - 方案：文本→离散代码→表示的中间层设计 (VQ-Rec)
   - 效果：减弱文本与表示的紧密绑定，提升迁移性

5. **混合提示 (Hybrid Prompting)**
   - 问题：纯 ID 或纯文本方法各有局限
   - 方案：ID 嵌入 + 文本特征融合，课程学习训练 (LLaRA)
   - 效果：兼顾行为模式和语义理解

---

### 生成式推荐评测数据集

#### 各模型使用数据集

| 模型 | 数据集 | 领域 |
|------|--------|------|
| P5 | Amazon (Beauty, Sports, Toys), Yelp | 电商、本地生活 |
| TALLRec | MovieLens, Amazon Book | 电影、书籍 |
| TIGER | Amazon 多品类 | 电商 |
| LC-Rec | Amazon (Instruments, Arts, Games) | 电商 |
| DiffuRec | Amazon Beauty + 3个其他 | 电商 |

#### 推荐评测数据集

| 数据集 | 规模 | 特点 | 推荐场景 |
|--------|------|------|----------|
| **Amazon Review** | 8283万评分, 935万商品 | 多品类子集 | 通用/生成式 |
| **MovieLens** | 100K/1M/10M 多版本 | 经典基准 | 快速验证 |
| **Yelp** | 699万评论, 15万商家 | 本地商业 | 多模态 |
| **MIND** | 16万文章, 1500万展示 | 新闻推荐 | 序列推荐 |
| **Tenrec** | 500万用户, 1.4亿交互 | 大规模多场景 | 工业级评测 |

#### 常用评测指标

| 指标类型 | 指标名 | 说明 |
|----------|--------|------|
| 排序指标 | HR@K, NDCG@K, MRR | Top-K 排序质量 |
| 分类指标 | AUC, Logloss | CTR 预测 |
| 召回指标 | Recall@K, Precision@K | 召回覆盖率 |

---

### 值得复现的论文 (按优先级)

**高优先级 ⭐⭐⭐⭐⭐**
1. **TALLRec** - 资源要求低，单卡可跑，少样本有效
2. **DiffuRec** - 扩散模型推荐，创新性强
3. **LC-Rec** - ICDE 2024，语义对齐方案
4. **UniSRec** - 跨域迁移，实用性强

**中优先级 ⭐⭐⭐⭐**
1. **P5** - 统一生成框架，理解范式演进
2. **TIGER** - Semantic ID 思路值得学习
3. **VQ-Rec** - 向量量化方案，跨域能力强
4. **LLaRA** - 混合提示，课程学习
5. **CoLLM** - 协作信号融合

**研究参考 ⭐⭐⭐**
1. **HSTU** - Meta 工业级方案，架构设计参考
2. **RecMind** - LLM Agent 推荐，未来方向

### 开源工具推荐

| 工具 | 用途 | 推荐度 |
|------|------|--------|
| RecBole 2.0 | 通用推荐框架，100+算法 | ⭐⭐⭐⭐⭐ |
| FuxiCTR 2.0 | CTR预测专用 | ⭐⭐⭐⭐ |
| Transformers4Rec | Transformer推荐 | ⭐⭐⭐⭐⭐ |

---

## 五、技术发展趋势

1. **从浅层到深层**：矩阵分解 → 深度神经网络
2. **从单一到多元**：单一交互 → 多模态、多行为
3. **从静态到动态**：静态embedding → 序列建模、实时更新
4. **从监督到自监督**：引入对比学习、自监督预训练
5. **从判别到生成**：判别模型 → 扩散模型、LLM生成

---

*调研完成时间：2025-02*
