# PJ1 — UKB 年龄/性别 + ADNI 三分类

| | 链接 |
|---|------|
| **代码（GitHub）** | [tree/main/PJ1](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ1) |
| **数据与权重（ModelScope）** | [sSzHox/PJ_ADNI_UKB](https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB) |

- **GitHub**：`scripts/`、`PJ1_UKB/`、`PJ1_ADNI/` 下全部代码与配置  
- **ModelScope**：`data/` 原始影像、`checkpoints/` 预训练、各线 `outputs/` 微调权重  
- **本地生成**：`processed/`、`dataset/processed_rootstrap/`（WSL + FSL 预处理，不在上述两处）

首次使用（训练集复现）：

```bash
git clone https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience.git
cd CS50029-ComputationalNeuroscience/PJ1
pip install modelscope
python scripts/download_modelscope.py
```

---

## 官方测试集：放进 `data/` 之后怎么做

把课程下发的测试包解压到 **`PJ1/data/<名称>/`**（结构与训练集类似：UKB 含 `image_T1_raw/` + CSV；ADNI 含按 ID 分文件夹的 NIfTI + `labels.csv` 或 `selected_*_info.csv`）。

以下命令均在 **`PJ1/`** 根目录执行；`--name` 与 `--raw` 使用同一目录名（例如 `TEST_ADNI`）。

### 阶段 A — 预处理（WSL，需 FSL）

```bash
# ADNI：四条预处理线一次跑完（Rootstrap + mri + sfcn + 共用 data）
python scripts/preprocess_test.py --pipeline all --name TEST_ADNI --raw TEST_ADNI --jobs 4

# UKB：只跑 SFCN 一条
python scripts/preprocess_test.py --pipeline ukb_sfcn --name TEST_UKB --raw TEST_UKB --jobs 8
```

预处理产物位置：

| `--pipeline` | 写入目录 |
|--------------|----------|
| `ukb_sfcn` | `PJ1_UKB/processed/UKB_sfcn_new/<name>/` |
| `adni_rootstrap` | `PJ1_ADNI/dataset/processed_rootstrap/<name>/` |
| `adni_mri_classifier` | `PJ1_UKB/processed/ADNI_mri_classifier/<name>/` |
| `adni_sfcn_v4` | `PJ1_UKB/processed/ADNI_sfcn_v4/<name>/` |

### 阶段 B — 推理 / 评测（GPU）

```bash
# 六任务全开：UKB(both/onlyage/onlysex) + ADNI 三线
python scripts/eval_test.py --pipeline all --name TEST_ADNI --raw TEST_ADNI

# 单跑示例
python scripts/eval_test.py --pipeline ukb_sfcn --task both --name TEST_UKB --raw TEST_UKB
python scripts/eval_test.py --pipeline adni_rootstrap --name TEST_ADNI --raw TEST_ADNI
```

### 结果在哪看

| 任务 | 提交文件 | 有标签时的指标 |
|------|----------|----------------|
| UKB `both` / `onlyage` / `onlysex` | `PJ1_UKB/outputs/test/<name>/ukb_sfcn_<task>/pred.csv` | 同目录 `test_metrics.json`（MAE / sex acc） |
| ADNI Rootstrap ⭐ | `PJ1_ADNI/outputs/test/<name>/adni_rootstrap/pred.csv` | `test_metrics.json` |
| ADNI mri_classifier | `PJ1_UKB/outputs/test/<name>/adni_mri_classifier/pred.csv` | `test_metrics.json` |
| ADNI sfcn_v4 | `PJ1_UKB/outputs/test/<name>/adni_sfcn_v4/pred.csv` | `test_metrics.json` |

ADNI 提交列：`ID,Pre`（`CN` / `MCI` / `AD`）。UKB：`ID,Age` 和/或 `ID,Sex`（由 `--task` 决定）。

默认使用已训练好的最优权重（见 ModelScope `outputs/`）；可 `--checkpoint-dir` 覆盖。

---

## 更多文档

| 文档 | 内容 |
|------|------|
| [scripts/README.md](scripts/README.md) | 测试集参数说明、`--pipeline` / `--task` |
| [PJ1_UKB/README.md](PJ1_UKB/README.md) | UKB SFCN 与 ADNI 探索线训练 |
| [PJ1_ADNI/README.md](PJ1_ADNI/README.md) | ADNI Rootstrap 主结果（acc ~0.75） |
