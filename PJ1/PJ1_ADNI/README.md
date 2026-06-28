# PJ1_ADNI — ADNI 三分类（CN / MCI / AD）

基于 [Rootstrap MRI-classifier](https://github.com/rootstrap/MRI-classifier) 的 DenseNet121 3D 模型，在 105 例 ADNI 上微调。**ADNI 主结果线**（5-fold × 3 seeds，mean acc **0.752**）。UKB 与 ADNI 探索线在同级 `PJ1_UKB/`。

提交格式：`ID,Pre`（CN / MCI / AD）。

| | 链接 |
|---|------|
| **代码** | [GitHub · PJ1/PJ1_ADNI](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ1/PJ1_ADNI) |
| **大文件** | [ModelScope · sSzHox/PJ_ADNI_UKB](https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB) |

---

## GitHub / ModelScope / 本地 分工

| 路径（相对 `PJ1/`） | GitHub | ModelScope | 本地 WSL |
|---------------------|:------:|:----------:|:--------:|
| `PJ1_ADNI/configs/`、`src/`、`scripts/`、`tests/` | ✅ | — | — |
| `data/ADNI_data_105cases/` | — | ✅ | — |
| `data/ADNI_test20_release/` | — | ✅ | — |
| `PJ1_ADNI/models/`（预训练 `86_acc_model.pth`） | — | ✅ | — |
| `PJ1_ADNI/outputs/rootstrap_adni_finetune_data_aug_seed3/`（15 ckpt） | — | ✅ | — |
| `PJ1_ADNI/outputs/test/<name>/adni_rootstrap/`（test20 提交） | — | ✅ | 可本地重跑 |
| `PJ1_ADNI/dataset/processed_rootstrap/`（训练 & test 预处理） | — | — | ✅ **必须本地 FSL** |

原始 NIfTI 在 **`PJ1/data/`**（不在 `PJ1_ADNI/data/`）。默认训练集路径：`data/ADNI_data_105cases/ADNI_data/`。

---

## 目录结构

```text
PJ1/
├── data/                                      [ModelScope]
│   ├── ADNI_data_105cases/                    训练 105 例
│   └── ADNI_test20_release/                   官方 test20 原始
├── scripts/                                   [GitHub] test20 统一 preprocess/eval
├── PJ1_ADNI/
│   ├── configs/  scripts/  src/  tests/       [GitHub]
│   ├── models/                                [ModelScope]
│   ├── outputs/
│   │   ├── rootstrap_adni_finetune_data_aug_seed3/   [ModelScope] 训练 ckpt
│   │   └── test/ADNI_test20/adni_rootstrap/          [ModelScope] test20 提交
│   └── dataset/processed_rootstrap/           [本地 WSL] 预处理 .nii.gz + metadata
└── PJ1_UKB/                                   见 PJ1_UKB/README.md
```

---

## 环境

```bash
cd PJ1_ADNI
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt   # 或 requirements.txt

export FSLDIR=$HOME/fsl
source $FSLDIR/etc/fslconf/fsl.sh
```

大文件在 `PJ1/` 根目录下载：`python scripts/download_modelscope.py`

---

## 105 例训练（复现）

### 1. 预处理 → 本地 only

```bash
python scripts/preprocess.py
# 读 ../data/ADNI_data_105cases/ADNI_data
# 写 dataset/processed_rootstrap/ADNI/
```

### 2. 微调

```bash
python scripts/train.py --config configs/rootstrap_adni_finetune_data_aug_seed3.yaml
```

产出 `outputs/rootstrap_adni_finetune_data_aug_seed3/`：`seed_*_fold_*.pt`（15 个）、`metrics.json`、`pred.csv`（OOF）。

| 指标 | 值 |
|------|-----|
| mean acc | 0.752 |
| mean macro_f1 | 0.745 |

---

## 官方 test20

**不要**用本目录旧式单线脚本；统一在 **`PJ1/` 根目录**：

```bash
cd ../   # 到 PJ1/

# 预处理（WSL + FSL）→ 写 PJ1_ADNI/dataset/processed_rootstrap/ADNI_test20/
python scripts/preprocess_test.py --pipeline adni_rootstrap \
  --name ADNI_test20 --raw ADNI_test20_release/ADNI_test20_release --jobs 4

# 推理（15 ckpt logit-sum ensemble）
python scripts/eval_test.py --pipeline adni_rootstrap \
  --name ADNI_test20 --raw ADNI_test20_release/ADNI_test20_release
```

| 产物 | 路径 |
|------|------|
| 预处理 | `dataset/processed_rootstrap/ADNI_test20/`（**仅本地**） |
| 预测 + 提交 | `outputs/test/ADNI_test20/adni_rootstrap/pred.csv`、`ADNI_submission_filled.csv`（**ModelScope 已有一份**） |

仅从 ModelScope 取结果、跳过本地推理：

```bash
python scripts/download_modelscope.py --target test20
# 含 data/*_test20_release/ 与 PJ1_ADNI/outputs/test/
```

---

## 本目录脚本（105 例 / 打包推理）

| 脚本 | 用途 |
|------|------|
| `scripts/preprocess.py` | 105 例训练集预处理 |
| `scripts/train.py` | 微调 / baseline |
| `scripts/eval.py` | 新测试 tar/目录 端到端 |
| `scripts/eval_test.py` | 供 `PJ1/scripts/eval_runners.py` 调用的集成推理实现 |

**test20** 预处理与评测入口：`PJ1/scripts/preprocess_test.py`、`PJ1/scripts/eval_test.py`（见 [../README.md](../README.md)）。

---

## 与 PJ1_UKB

- `PJ1_UKB/outputs/ADNI/mri_classifier/`、`sfcn_v4/` 为探索线（acc ~0.70 / 0.67），ModelScope 有权重与 test20 提交。
- **课程 ADNI 主提交**用本目录 Rootstrap test20 结果。

```bash
pytest tests/ -q
```
