# CTR 预估算法复现计划

> 目标：快速验证效果，为广告推荐业务选型

---

## 数据集选择

| 数据集 | 规模 | 序列长度 | 用途 |
|--------|------|---------|------|
| **Amazon Electronics** | 550万 | 15-20 | 序列模型验证（短序列） |
| **Taobao UserBehavior** | 1亿 | 101 | 序列模型 + 多任务(CTR+CVR) |

**选择理由**：
- Amazon：DIN/DIEN 论文标配，序列模式明显
- Taobao：长序列 + 多行为(pv/fav/cart/buy)，支持多任务学习

---

## 复现优先级

### 第一阶段：Baseline + 数据准备（2天）

| 优先级 | 算法 | 数据集 | 说明 |
|--------|------|--------|------|
| P0 | **DeepFM** | Amazon | 无序列 Baseline |
| P0 | **DCN v2** | Amazon | 特征交叉 Baseline |

### 第二阶段：序列模型（3-4天）

| 优先级 | 算法 | 数据集 | 说明 |
|--------|------|--------|------|
| P0 | **DIN** | Amazon + Taobao | 序列建模入门，必跑 |
| P0 | **DIEN** | Taobao | 兴趣演化，长序列效果好 |
| P1 | BST | Taobao | Transformer 架构 |

### 第三阶段：多任务学习（2-3天）

| 优先级 | 算法 | 数据集 | 说明 |
|--------|------|--------|------|
| P0 | **MMOE** | Taobao | CTR+CVR 多任务 |
| P1 | ESMM | Taobao | CVR 预估专用 |
| P1 | PLE | Taobao | 减少负迁移 |

### 第四阶段：前沿探索（可选）

| 优先级 | 算法 | 数据集 | 说明 |
|--------|------|--------|------|
| P2 | SIM | Taobao | 超长序列建模 |
| P2 | DSIN | Taobao | 会话级兴趣 |

---

## 评估指标

| 指标 | 说明 | 重要度 |
|------|------|--------|
| **AUC** | 排序能力，最核心指标 | ⭐⭐⭐⭐⭐ |
| **LogLoss** | 预估准确性 | ⭐⭐⭐⭐ |
| GAUC | 分用户AUC，更贴近线上 | ⭐⭐⭐⭐ |
| RelaImpr | 相对提升，对比用 | ⭐⭐⭐ |

---

## 执行计划

### Week 1：数据准备 + Baseline

```
Day 1-2: 环境搭建 + 数据下载
  - [ ] 安装依赖，验证环境可用
  - [ ] 下载 Amazon Electronics 数据集
  - [ ] 下载 Taobao UserBehavior 数据集
  - [ ] 数据预处理和序列构建

Day 3-4: Baseline 模型
  - [ ] Amazon: DeepFM baseline
  - [ ] Amazon: DCN v2 baseline
  - [ ] 记录 AUC/LogLoss
```

### Week 2：序列模型

```
Day 1-2: DIN
  - [ ] Amazon: 跑通 DIN
  - [ ] Taobao: 跑通 DIN
  - [ ] 对比不同序列长度效果

Day 3-4: DIEN
  - [ ] Taobao: 跑通 DIEN
  - [ ] 对比 DIN vs DIEN

Day 5: BST (可选)
  - [ ] Taobao: 跑通 BST
  - [ ] Transformer vs GRU 对比
```

### Week 3：多任务学习

```
Day 1-2: MMOE
  - [ ] Taobao: CTR + CVR 多任务
  - [ ] 对比单任务 vs 多任务

Day 3-4: ESMM / PLE
  - [ ] 跑通 ESMM (CVR 专用)
  - [ ] 跑通 PLE (减少负迁移)

Day 5: 汇总
  - [ ] 整理所有实验结果
  - [ ] 输出选型建议报告
```

---

## 快速运行命令

```bash
# 1. 下载数据集
# Amazon (直接下载)
wget https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Electronics.csv -P data/

# Taobao (需天池账号，手动下载后放入 data/)
# https://tianchi.aliyun.com/dataset/dataDetail?dataId=649

# 2. 跑 Baseline
python scripts/run_baseline.py --model DeepFM --dataset amazon-electronics --save_result

# 3. 跑序列模型
python scripts/run_baseline.py --model DIN --dataset amazon-electronics --save_result
python scripts/run_baseline.py --model DIN --dataset taobao-userbehavior --save_result

# 4. 批量对比
python scripts/run_comparison.py --models DeepFM,DCN,DIN,DIEN --dataset amazon-electronics
```

---

## 预期产出

1. **模型效果对比表**：各模型在 Amazon/Taobao 上的 AUC/LogLoss
2. **序列长度分析**：不同序列长度对效果的影响
3. **多任务对比**：单任务 vs 多任务(CTR+CVR) 效果对比
4. **选型建议**：基于效果和成本的推荐方案

---

## 预期 AUC 参考值

| 模型 | Amazon Electronics | Taobao UserBehavior |
|------|-------------------|---------------------|
| DeepFM | 0.78-0.80 | 0.82-0.84 |
| DCN | 0.79-0.81 | 0.83-0.85 |
| DIN | 0.81-0.82 | 0.84-0.86 |
| DIEN | 0.82-0.83 | 0.85-0.87 |
| MMOE | - | 0.85-0.87 (CTR+CVR) |

---

## 选型决策参考

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 无用户序列 | DeepFM / DCN | 简单高效 |
| 短序列 (≤50) | DIN | Attention 足够 |
| 长序列 (>50) | DIEN / BST | 时序建模更强 |
| 多目标 (CTR+CVR) | MMOE / PLE | 多任务联合训练 |
| 超长序列 (>200) | SIM | 检索式架构 |
