# PJ1 — UKB 年龄/性别 + ADNI 三分类

| | 链接 |
|---|------|
| **代码（GitHub）** | [tree/main/PJ1](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ1) |
| **数据与权重（ModelScope）** | [sSzHox/PJ_ADNI_UKB](https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB) |

---

## 三处分别有什么

| 内容 | GitHub | ModelScope | 本地 WSL/FSL 生成 |
|------|:------:|:----------:|:-----------------:|
| 代码、配置、`scripts/` | ✅ | — | — |
| **课程提交 CSV（test20 填好表）** | ✅ **`Submission/`** | ✅ `outputs/test/` | 可重跑后复制更新 |
| 原始 NIfTI + CSV（105/100 训练集） | — | ✅ `data/` | — |
| 官方 test20 原始包（**空 template，勿交**） | — | ✅ `data/*_test20_release/` | — |
| 预训练 / 微调 checkpoint | — | ✅ `checkpoints/`、`models/`、`outputs/`（训练） | — |
| 预处理中间产物（`.npz` / rootstrap `.nii.gz`） | — | — | ✅ 必须本地跑 |
| 运行日志、`.venv` | — | — | 本地 |

**ModelScope 不含**：`PJ1_UKB/processed/`、`PJ1_ADNI/dataset/processed_rootstrap/`。从原始影像到可推理格式，需在 **WSL + FSL** 本地执行 `preprocess_test.py`（或训练用 `preprocess*.py`）。

---

## ⭐ 课程提交（test20，交这两个 CSV）

**交 `Submission/` 下文件，不要交 `data/` 里的空 template。**

| 任务 | 提交文件（相对 `PJ1/`） | 格式 | 来源 |
|------|-------------------------|------|------|
| **ADNI test20** | **`Submission/ADNI_submission_filled.csv`** | `eid,label`（CN/MCI/AD） | Rootstrap 15 模型 logit-sum 集成 |
| **UKB test20** | **`Submission/UKB_submission_filled_both.csv`** | `eid,age,sex` | SFCN `both` 任务 |

与 `eval_test.py` 产出一致；源文件分别为：

- `PJ1_ADNI/outputs/test/ADNI_test20/adni_rootstrap/ADNI_submission_filled.csv`
- `PJ1_UKB/outputs/test/UKB_test20/ukb_sfcn_both/UKB_submission_filled_both.csv`

重跑评测后请同步复制到 `Submission/` 再 push GitHub：

```cmd
copy PJ1_ADNI\outputs\test\ADNI_test20\adni_rootstrap\ADNI_submission_filled.csv Submission\ADNI_submission_filled.csv
copy PJ1_UKB\outputs\test\UKB_test20\ukb_sfcn_both\UKB_submission_filled_both.csv Submission\UKB_submission_filled_both.csv
```

勿交：`data/ADNI_test20_release/.../ADNI_submission_template.csv`、`data/UKB_test20_release/.../UKB_submission_template.csv`（列为空）。勿交 ADNI 探索线（`mri_classifier` / `sfcn_v4`）的 CSV。

---

## 快速开始

```bash
git clone https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience.git
cd CS50029-ComputationalNeuroscience/PJ1
pip install modelscope
python scripts/download_modelscope.py          # 全部大文件
# python scripts/download_modelscope.py --target test20   # 仅 test20 原始 + 提交结果
```

上传维护（维护者）：

```cmd
cd PJ1
set MODELSCOPE_API_TOKEN=ms-xxx
python scripts\verify_data_layout.py --require-test20
python scripts\verify_test20_outputs.py
python scripts\upload_modelscope.py
```

全量 upload 会跳过远端已存在的 105/100 与权重，**只补传**本地新增或变更（如 test20）。

---

## 官方 test20 流程

test20 原始数据在 ModelScope 路径（与本地一致）：

```text
data/UKB_test20_release/UKB_test20_release/
  images/<eid>/T1.nii.gz
  UKB_submission_template.csv

data/ADNI_test20_release/ADNI_test20_release/
  images/<eid>/T1.nii(.gz)
  ADNI_submission_template.csv
```

已填好的提交文件亦在 ModelScope `PJ1_*/outputs/test/`；**GitHub 课程交表见 [`Submission/`](Submission/)**。

```text
Submission/                              ← 课程提交（GitHub）⭐
  ADNI_submission_filled.csv
  UKB_submission_filled_both.csv

data/.../UKB_submission_template.csv     ← 空 template，勿交
data/.../ADNI_submission_template.csv
```

评测产出目录（与 Submission 同步）：

```text
PJ1_UKB/outputs/test/UKB_test20/ukb_sfcn_both/
PJ1_ADNI/outputs/test/ADNI_test20/adni_rootstrap/
PJ1_UKB/outputs/test/ADNI_test20/adni_{mri_classifier,sfcn_v4}/   # 探索线，勿交课程
```

以下命令均在 **`PJ1/`** 根目录；`--name` 为产物子目录名（默认 `UKB_test20` / `ADNI_test20`），`--raw` 为 release **内层目录**（相对 `data/` 或绝对路径）。

### 阶段 A — 预处理（WSL，需 FSL）【不在 ModelScope】

```bash
cd /mnt/d/.../PJ1
export FSLDIR=$HOME/fsl    # 按实际安装修改
source $FSLDIR/etc/fslconf/fsl.sh

python scripts/preprocess_test.py --pipeline ukb_sfcn \
  --name UKB_test20 --raw UKB_test20_release/UKB_test20_release --jobs 4

python scripts/preprocess_test.py --pipeline all \
  --name ADNI_test20 --raw ADNI_test20_release/ADNI_test20_release --jobs 4
```

| `--pipeline` | 本地预处理产出 |
|--------------|----------------|
| `ukb_sfcn` | `PJ1_UKB/processed/UKB_sfcn_new/<name>/` |
| `adni_rootstrap` | `PJ1_ADNI/dataset/processed_rootstrap/<name>/` |
| `adni_mri_classifier` | `PJ1_UKB/processed/ADNI_mri_classifier/<name>/` |
| `adni_sfcn_v4` | `PJ1_UKB/processed/ADNI_sfcn_v4/<name>/` |

校验：`python scripts/verify_processed.py --ukb-name UKB_test20 --adni-name ADNI_test20`

### 阶段 B — 推理（GPU）

权重默认从 ModelScope 已下载的 `outputs/` 读取；预处理读阶段 A 本地产物。

```bash
python scripts/eval_test.py --pipeline ukb_sfcn --task both \
  --name UKB_test20 --raw UKB_test20_release/UKB_test20_release

python scripts/eval_test.py --pipeline adni_rootstrap \
  --name ADNI_test20 --raw ADNI_test20_release/ADNI_test20_release

python scripts/eval_test.py --pipeline all \
  --name ADNI_test20 --raw ADNI_test20_release/ADNI_test20_release
```

校验提交 CSV：`python scripts/verify_test20_outputs.py`

### test20 产出一览

| 任务 | 本地路径（`<name>` 如 `ADNI_test20`） | ModelScope 同步 |
|------|--------------------------------------|-----------------|
| UKB `both` / `onlyage` / `onlysex` | `PJ1_UKB/outputs/test/<name>/ukb_sfcn_*/` | ✅ |
| ADNI Rootstrap ⭐ | `PJ1_ADNI/outputs/test/<name>/adni_rootstrap/` | ✅ |
| ADNI mri_classifier | `PJ1_UKB/outputs/test/<name>/adni_mri_classifier/` | ✅ |
| ADNI sfcn_v4 | `PJ1_UKB/outputs/test/<name>/adni_sfcn_v4/` | ✅ |

test20 无标签时只填 submission template，不算 `test_metrics.json`。

---

## 子项目

| 目录 | 说明 |
|------|------|
| [PJ1_ADNI/README.md](PJ1_ADNI/README.md) | ADNI Rootstrap 主结果（105 例训练 + test20） |
| [PJ1_UKB/README.md](PJ1_UKB/README.md) | UKB SFCN + ADNI 探索线 |
| [scripts/README.md](scripts/README.md) | `--pipeline` / `--task` 参数 |

训练集复现见各子目录 README；test20 统一走本页与 `scripts/` 入口。
