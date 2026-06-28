# PJ1_UKB — UKB 年龄/性别 + ADNI 探索线

CS50029 课程项目。本目录：**UKB SFCN**（年龄/性别）与 **ADNI 探索线**（`mri_classifier`、`sfcn_v4`）。ADNI **主结果** Rootstrap 在同级 [`PJ1_ADNI/`](../PJ1_ADNI/README.md)（acc ~0.75）。

| | 链接 |
|---|------|
| **代码** | [GitHub · PJ1/PJ1_UKB](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ1/PJ1_UKB) |
| **大文件** | [ModelScope · sSzHox/PJ_ADNI_UKB](https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB) |

---

## GitHub / ModelScope / 本地 分工

| 路径（相对 `PJ1/`） | GitHub | ModelScope | 本地 WSL |
|---------------------|:------:|:----------:|:--------:|
| `PJ1_UKB/configs/`、`src/`、`scripts/`、`tests/` | ✅ | — | — |
| `data/UKB_T1_100cases/`、`data/ADNI_data_105cases/` | — | ✅ | — |
| `data/*_test20_release/` | — | ✅ | — |
| `PJ1_UKB/checkpoints/`（SFCN / mri 预训练） | — | ✅ | — |
| `PJ1_UKB/outputs/UKB/sfcn/20260606_*`（三任务微调） | — | ✅ | — |
| `PJ1_UKB/outputs/ADNI/mri_classifier/`、`sfcn_v4/` | — | ✅ | — |
| `PJ1_UKB/outputs/test/`（test20 pred + submission） | — | ✅ | 可本地重跑 |
| `PJ1_UKB/processed/`（`.npz` 预处理） | — | — | ✅ **必须本地 FSL** |

共享原始数据在 **`PJ1/data/`**（不在 `PJ1_UKB/data/`）。

```bash
export PJ1_DATA_ROOT=/path/to/PJ1/data   # 可选
```

---

## 目录结构

```text
PJ1/
├── data/                                      [ModelScope]
├── scripts/                                   [GitHub] test20 统一入口
├── PJ1_UKB/
│   ├── configs/  scripts/  src/  tests/       [GitHub]
│   ├── checkpoints/                           [ModelScope]
│   ├── processed/                             [本地 WSL] UKB/ADNI npz
│   └── outputs/
│       ├── UKB/sfcn/20260606_*/               [ModelScope] 训练
│       ├── ADNI/mri_classifier/  sfcn_v4/   [ModelScope] 训练
│       └── test/                              [ModelScope] test20 提交
└── PJ1_ADNI/                                  Rootstrap 主结果
```

---

## 环境

**Windows（无 GPU）**：`pip install -r requirements-local.txt`，可跑 `verify_data.py`。

**WSL / Linux（预处理 + 训练）**：

```bash
cd PJ1_UKB
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt

python scripts/check_fsl_env.py    # 必须 exit 0
# 无 FSL 时: python scripts/setup_fsl_official.py
```

大文件：`cd ../ && python scripts/download_modelscope.py`

---

## 已验证训练管线（105/100 例）

| 任务 | 配置 | ModelScope 权重 |
|------|------|-----------------|
| UKB both / onlyage / onlysex | `configs/ukb_sfcn.yaml` | `outputs/UKB/sfcn/20260606_12*` |
| ADNI mri_classifier | `configs/adni_mri_classifier.yaml` | `outputs/ADNI/mri_classifier/` |
| ADNI sfcn_v4 | `configs/adni_sfcn_v4.yaml` | `outputs/ADNI/sfcn_v4/` |

训练前预处理（**本地 only**，示例）：

```bash
python scripts/preprocess_ukb_sfcn_new.py --config configs/preprocess_sfcn_new.yaml --jobs 8
python scripts/preprocess_mri_classifier_adni.py --jobs 8
python scripts/preprocess_sfcn_adni.py --preprocess-version v4 --jobs 4
python scripts/verify_processed.py --config configs/ukb_sfcn.yaml --dataset ukb
```

---

## 官方 test20

在 **`PJ1/` 根目录** 统一处理（不要单独改本目录路径）：

```bash
cd ../

# 预处理四条线中的 UKB + 两条 ADNI 探索线（rootstrap 写 PJ1_ADNI/）
python scripts/preprocess_test.py --pipeline all \
  --name ADNI_test20 --raw ADNI_test20_release/ADNI_test20_release --jobs 4

python scripts/preprocess_test.py --pipeline ukb_sfcn \
  --name UKB_test20 --raw UKB_test20_release/UKB_test20_release --jobs 4

# 评测：UKB 三任务 + ADNI 探索线（六任务全开加 adni_rootstrap 见 PJ1 README）
python scripts/eval_test.py --pipeline all --name ADNI_test20 \
  --raw ADNI_test20_release/ADNI_test20_release
python scripts/eval_test.py --pipeline ukb_sfcn --task both \
  --name UKB_test20 --raw UKB_test20_release/UKB_test20_release
```

| 内容 | 本地路径 | ModelScope |
|------|----------|------------|
| UKB 预处理 | `processed/UKB_sfcn_new/UKB_test20/` | ❌ |
| ADNI 探索线预处理 | `processed/ADNI_mri_classifier/ADNI_test20/` 等 | ❌ |
| UKB 提交 | `outputs/test/UKB_test20/ukb_sfcn_*/UKB_submission_filled_*.csv` | ✅ |
| ADNI 探索提交 | `outputs/test/ADNI_test20/adni_mri_classifier/`、`adni_sfcn_v4/` | ✅ |

校验：`python scripts/verify_processed.py --ukb-name UKB_test20 --adni-name ADNI_test20`  
提交：`python scripts/verify_test20_outputs.py`

---

## 本目录脚本索引

| 脚本 | 用途 |
|------|------|
| `verify_data.py` | 原始 T1 与 CSV 一致性 |
| `verify_processed.py` | 训练集 npz 检查（与 `PJ1/scripts/verify_processed.py` test20 版不同） |
| `check_fsl_env.py` / `setup_fsl_official.py` | FSL |
| `preprocess_ukb_sfcn_new.py` 等 | 105/100 例训练预处理 |
| `train_ukb_sfcn.py`、`train_adni_mri.py`、`train_adni_sfcn.py` | 训练 |
| `infer_sfcn_test.py`、`infer_adni_sfcn.py` | 105 例 holdout 推理 |
| `download_*_weights.py` | 拉公开预训练（ModelScope 也有一份） |

**test20** 统一入口：[../scripts/README.md](../scripts/README.md)、[../README.md](../README.md)。

---

## 备注

- UKB 最优 MAE ~2.47：`outputs/UKB/sfcn/20260606_12*`（ModelScope 已同步）。
- ADNI 主结果请用 `PJ1_ADNI` Rootstrap；本目录两条线为对比探索。
- `both` 与 `onlyage`+`onlysex` 在 test20 上几乎一致；提交 UKB 用 `--task both` 即可。

```bash
pytest tests/ -q
```
