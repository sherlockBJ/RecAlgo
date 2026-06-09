# RecAlgo

推荐系统召回排序算法调研与测试项目。基于 [RecBole](https://recbole.io/) 框架快速复现和对比推荐算法。

## 项目结构

```
RecAlgo/
├── configs/                    # 实验配置
│   ├── models/                 # 模型配置
│   └── datasets/               # 数据集配置
├── scripts/                    # 运行脚本
│   ├── run_baseline.py         # 跑单个模型
│   ├── run_comparison.py       # 批量对比
│   └── download_data.py        # 下载数据集
├── custom_models/              # 自定义模型
│   ├── recall/                 # 召回模型
│   └── rank/                   # 排序模型
├── experiments/                # 实验结果
├── notebooks/                  # 分析笔记本
└── data/                       # 数据目录
```

## 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 下载数据集

```bash
# 查看可用数据集
python scripts/download_data.py --list

# 下载 MovieLens 100K
python scripts/download_data.py --dataset ml-100k
```

### 2. 运行单个模型

```bash
# 运行 BPR 模型
python scripts/run_baseline.py --model BPR --dataset ml-100k

# 运行 LightGCN 并保存结果
python scripts/run_baseline.py --model LightGCN --dataset ml-100k --save_result
```

### 3. 批量对比实验

```bash
# 对比多个召回模型
python scripts/run_comparison.py --models BPR,LightGCN,NeuMF --dataset ml-100k

# 使用预定义模型组
python scripts/run_comparison.py --models recall_cf --dataset ml-100k
python scripts/run_comparison.py --models sequential --dataset ml-100k
```

预定义模型组：
- `recall_cf`: BPR, NeuMF, LightGCN, NGCF, ItemKNN
- `recall_graph`: LightGCN, NGCF, SGL, SimGCL
- `sequential`: GRU4Rec, SASRec, BERT4Rec, Caser, SRGNN
- `ctr`: FM, DeepFM, xDeepFM, DCN, AutoInt

## 支持的模型

### 召回模型 (General Recommendation)
| 模型 | 论文 |
|------|------|
| BPR | BPR: Bayesian Personalized Ranking (UAI 2009) |
| NeuMF | Neural Collaborative Filtering (WWW 2017) |
| LightGCN | LightGCN: Simplifying and Powering Graph Convolution Network (SIGIR 2020) |
| NGCF | Neural Graph Collaborative Filtering (SIGIR 2019) |
| SGL | Self-supervised Graph Learning (SIGIR 2021) |

### 序列推荐 (Sequential Recommendation)
| 模型 | 论文 |
|------|------|
| GRU4Rec | Session-based Recommendations with RNNs (ICLR 2016) |
| SASRec | Self-Attentive Sequential Recommendation (ICDM 2018) |
| BERT4Rec | BERT4Rec: Sequential Recommendation with Bidirectional Encoder (CIKM 2019) |
| Caser | Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding (WSDM 2018) |

### CTR 预估 / 排序 (Context-aware Recommendation)
| 模型 | 论文 |
|------|------|
| FM | Factorization Machines (ICDM 2010) |
| DeepFM | DeepFM: A Factorization-Machine based Neural Network (IJCAI 2017) |
| xDeepFM | xDeepFM: Combining Explicit and Implicit Feature Interactions (KDD 2018) |
| DCN | Deep & Cross Network (ADKDD 2017) |
| AutoInt | AutoInt: Automatic Feature Interaction Learning (CIKM 2019) |

完整模型列表见 [RecBole Model List](https://recbole.io/model_list.html)

## 自定义模型

参考 `custom_models/recall/template.py` 和 `custom_models/rank/template.py` 创建自定义模型。

## 参考资料

- [RecBole 文档](https://recbole.io/)
- [RecBole GitHub](https://github.com/RUCAIBox/RecBole)
- [推荐系统论文集](https://github.com/hongleizhang/RSPapers)
