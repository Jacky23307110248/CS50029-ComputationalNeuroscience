# PJ1 测试集统一入口

**大文件**：[sSzHox/PJ_ADNI_UKB](https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB) · 下载：`python scripts/download_modelscope.py`（在 `PJ1/` 根目录）

含训练集与 **官方 test20**（`data/*_test20_release/`）。仅 test20：`python scripts/download_modelscope.py --target test20`

官方测试集放出后，将原始数据放入 `PJ1/data/<测试集名>/`，用本目录脚本完成**预处理（WSL）**与**评测（GPU）**两阶段。

## 四条预处理线 + UKB 三任务

| `--pipeline` | 预处理产出 | 默认权重 |
|--------------|-----------|----------|
| `ukb_sfcn` | `PJ1_UKB/processed/UKB_sfcn_new/<name>/` | `outputs/UKB/sfcn/20260606_12*`（按 task） |
| `adni_rootstrap` | `PJ1_ADNI/dataset/processed_rootstrap/<name>/` | `outputs/rootstrap_adni_finetune_data_aug_seed3/` |
| `adni_mri_classifier` | `PJ1_UKB/processed/ADNI_mri_classifier/<name>/` | `outputs/ADNI/mri_classifier/kfold/` |
| `adni_sfcn_v4` | `PJ1_UKB/processed/ADNI_sfcn_v4/<name>/` | `outputs/ADNI/sfcn_v4/kfold/` |

`--pipeline all`：一次跑 1 条 UKB + 3 条 ADNI 预处理。

评测 `--pipeline all`：跑 6 个任务（UKB `both` / `onlyage` / `onlysex` + 3 条 ADNI）。

## 阶段 A：预处理（WSL，需 FSL）

```bash
cd PJ1
# ADNI 测试集 — 四条线全部预处理
python scripts/preprocess_test.py --pipeline all --name TEST_ADNI --raw TEST_ADNI --jobs 4

# 仅 UKB
python scripts/preprocess_test.py --pipeline ukb_sfcn --name TEST_UKB --raw TEST_UKB --jobs 8
```

`--raw` 可为 `PJ1/data/` 下的目录名或绝对路径。目录结构应与训练集类似：

- UKB：`image_T1_raw/<eid>/T1.nii.gz` + `selected_100_age_sex.csv`（含 `eid,age,sex`）
- ADNI：`<eid>/*.nii` + `labels.csv` 或 `selected_ADNI_105_info.csv`（含 `eid/ID,label`）

## 阶段 B：评测（GPU）

```bash
cd PJ1
# 六线全开
python scripts/eval_test.py --pipeline all --name TEST_ADNI --raw TEST_ADNI

# 单线 + UKB 子任务
python scripts/eval_test.py --pipeline ukb_sfcn --task onlyage --name TEST_UKB --raw TEST_UKB

# 覆盖权重目录
python scripts/eval_test.py --pipeline adni_rootstrap --name TEST_ADNI \
  --checkpoint-dir PJ1_ADNI/outputs/rootstrap_adni_finetune_data_aug_seed3
```

产出：

- `PJ1_UKB/outputs/test/<name>/ukb_sfcn_<task>/pred.csv` + `test_metrics.json`
- `PJ1_ADNI/outputs/test/<name>/adni_rootstrap/pred.csv` + `test_metrics.json`
- `PJ1_UKB/outputs/test/<name>/adni_mri_classifier/` …
- `PJ1_UKB/outputs/test/<name>/adni_sfcn_v4/` …

有标签时自动计算指标；ADNI 提交列为 `ID,Pre`（CN/MCI/AD）。

## 环境变量

```bash
export PJ1_DATA_ROOT=/path/to/PJ1/data   # 可选
```
