# PJ1_ADNI — ADNI 三分类（CN / MCI / AD）

基于 [Rootstrap MRI-classifier](https://github.com/rootstrap/MRI-classifier) 的 DenseNet121 3D 模型，在 105 例 ADNI 上做微调。这是 **ADNI 主结果线**（5-fold × 3 seeds，mean acc **0.752**）。UKB 与 ADNI 探索线见同级 `PJ1_UKB/`。

任务说明：预测每个受试者的诊断标签（CN / MCI / AD），提交格式为 CSV：`ID,Pre`。

**代码**：[CS50029-ComputationalNeuroscience / PJ1 / PJ1_ADNI](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ1/PJ1_ADNI)  
**大文件**：[ModelScope · sSzHox/PJ_ADNI_UKB](https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB)

### 本目录：从哪里获取什么

| 标记 | 来源 | 路径（相对 `PJ1/`） | 说明 |
|------|------|---------------------|------|
| GitHub | [PJ1/PJ1_ADNI](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ1/PJ1_ADNI) | `configs/`、`scripts/`、`src/`、`tests/` | 配置、训练/推理/预处理脚本、核心库 |
| GitHub | 同上 | `README.md`、`PJ1_计算神经学.md`、`requirements*.txt` | 文档与依赖 |
| ModelScope | [PJ_ADNI_UKB](https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB) | `data/ADNI_data_105cases/` | 原始 NIfTI + CSV（与 `PJ1_UKB` 共用 `data/`） |
| ModelScope | 同上 | `PJ1_ADNI/models/` | Rootstrap 预训练 `86_acc_model.pth` |
| ModelScope | 同上 | `PJ1_ADNI/outputs/rootstrap_adni_finetune_data_aug_seed3/` | **15 个**微调 ckpt + `metrics.json` / `pred.csv` 等 |
| 本地生成 | WSL + FSL | `PJ1_ADNI/dataset/processed_rootstrap/` | 由 `scripts/preprocess.py` 从 `data/` 生成，**不上传** GitHub / ModelScope |

clone 后拉取（在 `PJ1/` 根目录）：

```bash
git clone https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience.git
cd CS50029-ComputationalNeuroscience/PJ1
pip install modelscope
python scripts/download_modelscope.py
```

预训练权重也可自行下载（与 ModelScope 中 `models/` 相同）：HuggingFace `rootstrap-org/Alzheimer-Classifier-Demo`。

---

## 目录结构

```text
PJ1/
├── data/                            [ModelScope] 共享原始数据
│   └── ADNI_data_105cases/
├── scripts/                         [GitHub] 统一测试集入口
├── PJ1_ADNI/                        [GitHub] 本目录代码
│   ├── configs/                     [GitHub]
│   ├── scripts/                     [GitHub]
│   ├── src/                         [GitHub]
│   ├── dataset/processed_rootstrap/ [本地] 预处理 .npy
│   ├── models/                      [ModelScope] 预训练权重
│   └── outputs/                     [ModelScope] 微调结果
└── PJ1_UKB/                         [GitHub 代码 + ModelScope 权重]
```

原始 NIfTI 与 CSV 放在 `PJ1/data/`，不在 `PJ1_ADNI/data/` 内。默认路径为 `../data/ADNI_data_105cases/ADNI_data`。

---

## 环境准备

```bash
cd PJ1_ADNI
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements-gpu.txt   # 或 requirements.txt（CPU）
```

**FSL** 为预处理必需（BET、FLIRT）：

```bash
export FSLDIR=$HOME/fsl
source $FSLDIR/etc/fslconf/fsl.sh
bet --help   # 验证可用
```

预训练权重默认路径 `models/Alzheimer-Classifier-Demo/86_acc_model.pth`（ModelScope 已包含；亦可 HuggingFace `rootstrap-org/Alzheimer-Classifier-Demo`）。

可选环境变量：

```bash
export PJ1_DATA_ROOT=/path/to/PJ1/data   # 覆盖默认 ../data/
```

---

## 训练流程（105 例）

### 1. 预处理

```bash
python scripts/preprocess.py
# 默认读取 ../data/ADNI_data_105cases/ADNI_data
# 产出 dataset/processed_rootstrap/ADNI/images/*.npy
```

### 2. 五折微调（3 seeds）

```bash
python scripts/train.py --config configs/rootstrap_adni_finetune_data_aug_seed3.yaml
```

输出目录 `outputs/rootstrap_adni_finetune_data_aug_seed3/` 包含：

- `seed_{42,2024,3407}_fold_{0-4}.pt` — 15 个 checkpoint
- `metrics.json` — 各 fold 指标与均值
- `pred.csv` — OOF 集成预测
- `train_log.csv` — 训练日志

仅跑预训练 baseline（不微调）：

```bash
python scripts/train.py --mode baseline
```

---

## 测试集推理

### 方式 A：原始测试包（目录或 tar.gz）

```bash
python scripts/eval.py --dataset /path/to/test_ADNI.tar.gz
# 自动解压、预处理、15-checkpoint 集成
# 输出 outputs/<dataset_name>/pred.csv
```

### 方式 B：本地已预处理测试集

```bash
# 预处理新测试数据（--input 可为任意路径，或共享 data 下的目录）
python scripts/preprocess_test.py --input ../data/test_ADNI_105/ADNI_data --name TEST_105

# 集成推理
python scripts/eval_test.py --data TEST_105
# 若 ../data/TEST_105/labels.csv 存在，会额外输出 test_metrics.json
```

提交文件格式：`ID,Pre`（Pre 为 CN / MCI / AD）。

---

## 脚本索引

| 脚本 | 用途 |
|------|------|
| `scripts/preprocess.py` | 105 例训练集预处理 |
| `scripts/train.py` | 微调或 baseline 评估 |
| `scripts/eval.py` | 新测试包（tar/目录）端到端推理 |
| `scripts/preprocess_test.py` | 测试集预处理 |
| `scripts/eval_test.py` | 已预处理测试集集成推理 |

---

## 已保存的最优结果

`outputs/rootstrap_adni_finetune_data_aug_seed3/metrics.json`：

| 指标 | 值 |
|------|-----|
| mean acc | 0.752 |
| mean balanced_acc | 0.752 |
| mean macro_f1 | 0.745 |

---

## 测试

```bash
pytest tests/ -q
```

---

## 与 PJ1_UKB 的关系

- `PJ1_UKB` 中保留了 ADNI 的 `mri_classifier` 与 `sfcn_v4` 探索线（acc ~0.70 / 0.67），权重在 ModelScope `PJ1_UKB/outputs/ADNI/`。
- **本目录** 的 Rootstrap 微调结果更好，作为 ADNI 主结果独立维护。

## 官方测试集

见 [`../scripts/README.md`](../scripts/README.md)：`preprocess_test.py` / `eval_test.py`（`--pipeline adni_rootstrap`）。
