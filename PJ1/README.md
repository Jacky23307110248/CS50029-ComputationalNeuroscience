# PJ1 — UKB 年龄/性别 + ADNI 三分类

| | 链接 |
|---|------|
| **代码（GitHub）** | [tree/main/PJ1](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ1) |
| **数据与权重（ModelScope）** | [sSzHox/PJ_ADNI_UKB](https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB) |

- **GitHub**：`scripts/`、`PJ1_UKB/`、`PJ1_ADNI/` 下全部代码与配置  
- **ModelScope**：`data/` 训练集 + **官方 test20**、`checkpoints/` 预训练、各线 `outputs/` 微调权重  
- **本地生成**：`processed/`、`dataset/processed_rootstrap/`（WSL + FSL 预处理，不在上述两处）

首次使用（训练集复现）：

```bash
git clone https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience.git
cd CS50029-ComputationalNeuroscience/PJ1
pip install modelscope
python scripts/download_modelscope.py
```

---

## 官方测试集 test20

**ModelScope 已含**（与训练集同在 `data/` 下，路径与本地一致）：

```
data/UKB_test20_release/UKB_test20_release/
data/ADNI_test20_release/ADNI_test20_release/
```

仅拉 test20：`python scripts/download_modelscope.py --target test20`

若从课程包解压，也放到 `PJ1/data/`，目录结构为：

```
data/UKB_test20_release/UKB_test20_release/
  images/<eid>/T1.nii.gz
  UKB_submission_template.csv      # eid,age,sex（空列待填）

data/ADNI_test20_release/ADNI_test20_release/
  images/<eid>/T1.nii(.gz)
  ADNI_submission_template.csv     # eid,label（空列待填）
```

以下命令均在 **`PJ1/`** 根目录执行；`--name` 为预处理产物子目录名，`--raw` 指向 **release 内层目录**（含 `images/` 与 template CSV）。

### 阶段 A — 预处理（WSL，需 FSL）

在 WSL 中进入项目并加载 FSL：

```bash
cd /mnt/d/大三下/计算神经学/CS50029-ComputationalNeuroscience/PJ1
export FSLDIR=/usr/local/fsl   # 按你的 FSL 安装路径修改
source $FSLDIR/etc/fslconf/fsl.sh

# UKB 官方 20 例：SFCN 预处理
python scripts/preprocess_test.py --pipeline ukb_sfcn \
  --name UKB_test20 \
  --raw UKB_test20_release/UKB_test20_release \
  --jobs 4

# ADNI 官方 20 例：四条预处理线一次跑完
python scripts/preprocess_test.py --pipeline all \
  --name ADNI_test20 \
  --raw ADNI_test20_release/ADNI_test20_release \
  --jobs 4
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
# UKB 提交：务必 --task both，会自动填 UKB_submission_*_filled.csv
python scripts/eval_test.py --pipeline ukb_sfcn --task both \
  --name UKB_test20 --raw UKB_test20_release/UKB_test20_release

# ADNI 主结果线 + 探索线（有 GPU 时）
python scripts/eval_test.py --pipeline adni_rootstrap \
  --name ADNI_test20 --raw ADNI_test20_release/ADNI_test20_release

# 六任务全开
python scripts/eval_test.py --pipeline all \
  --name ADNI_test20 --raw ADNI_test20_release/ADNI_test20_release
```

### 结果在哪看

| 任务 | 提交文件 | 有标签时的指标 |
|------|----------|----------------|
| UKB `both` | `PJ1_UKB/outputs/test/<name>/ukb_sfcn_both/pred.csv` + `UKB_submission_*_filled.csv` | 同目录 `test_metrics.json` |
| ADNI Rootstrap ⭐ | `PJ1_ADNI/outputs/test/<name>/adni_rootstrap/pred.csv` + `ADNI_submission_*_filled.csv` | `test_metrics.json` |
| ADNI mri_classifier | `PJ1_UKB/outputs/test/<name>/adni_mri_classifier/` | `test_metrics.json` |
| ADNI sfcn_v4 | `PJ1_UKB/outputs/test/<name>/adni_sfcn_v4/` | `test_metrics.json` |

官方测试集 CSV 中 age/sex/label 为空时，脚本只做推理并填写 submission template，不计算指标。

默认使用已训练好的最优权重（见 ModelScope `outputs/`）；可 `--checkpoint-dir` 覆盖。

---

## 更多文档

| 文档 | 内容 |
|------|------|
| [scripts/README.md](scripts/README.md) | 测试集参数说明、`--pipeline` / `--task` |
| [PJ1_UKB/README.md](PJ1_UKB/README.md) | UKB SFCN 与 ADNI 探索线训练 |
| [PJ1_ADNI/README.md](PJ1_ADNI/README.md) | ADNI Rootstrap 主结果（acc ~0.75） |
