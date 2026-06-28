# PJ1_UKB — UKB 年龄/性别 + ADNI 探索线（mri_classifier / sfcn_v4）

CS50029 计算神经学课程项目。本子目录包含 **UKB SFCN** 与 **ADNI 两条对比探索线**（主结果 ADNI Rootstrap 在同级 `PJ1_ADNI/`）。

**代码**：[CS50029-ComputationalNeuroscience / PJ1 / PJ1_UKB](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ1/PJ1_UKB)  
**大文件**：[ModelScope · sSzHox/PJ_ADNI_UKB](https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB)（与 GitHub 路径对齐，见下表）

### 本目录：从哪里获取什么

| 标记 | 来源 | 路径（相对 `PJ1/`） | 说明 |
|------|------|---------------------|------|
| GitHub | [PJ1/PJ1_UKB](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ1/PJ1_UKB) | `configs/`、`scripts/`、`src/`、`tests/` | 配置、训练/推理/预处理脚本、核心库 |
| GitHub | 同上 | `README.md`、`requirements-*.txt` | 文档与依赖清单 |
| GitHub | 同上 | `scripts/download_*.py` 等 | 从 HuggingFace / GitHub 拉**公开预训练**的脚本（权重本体在 ModelScope 已有一份） |
| ModelScope | [PJ_ADNI_UKB](https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB) | `data/UKB_T1_100cases/`、`data/ADNI_data_105cases/`、`data/*_test20_release/` | 共享原始 NIfTI + CSV（与 `PJ1_ADNI` 共用 `data/`） |
| ModelScope | 同上 | `PJ1_UKB/checkpoints/` | SFCN 与 mri_classifier **预训练**权重 |
| ModelScope | 同上 | `PJ1_UKB/outputs/UKB/sfcn/20260606_*` | UKB 三任务微调结果（`both` / `onlyage` / `onlysex`） |
| ModelScope | 同上 | `PJ1_UKB/outputs/ADNI/mri_classifier/` | ADNI DenseNet121 五折 ckpt + 摘要 |
| ModelScope | 同上 | `PJ1_UKB/outputs/ADNI/sfcn_v4/` | ADNI SFCN v4 五折 + final + 提交 CSV |
| 本地生成 | WSL + FSL | `PJ1_UKB/processed/` | 由 `preprocess_*.py` 从 `data/` 生成，**不上传** GitHub / ModelScope |

clone 后一键拉取大文件（在 `PJ1/` 根目录）：

```bash
git clone https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience.git
cd CS50029-ComputationalNeuroscience/PJ1
pip install modelscope
python scripts/download_modelscope.py
```

仅本目录相关时可：`python scripts/download_modelscope.py --target weights`（仍需上层 `data/` 做预处理时再 `--target data`）。

---

已验证管线：

| 任务 | 方法 | 配置 | 输出目录 |
|------|------|------|----------|
| UKB 年龄/性别 | SFCN 微调（both / onlyage / onlysex） | `configs/ukb_sfcn.yaml` | `outputs/UKB/sfcn/` |
| ADNI CN/MCI/AD | Rootstrap DenseNet121 | `configs/adni_mri_classifier.yaml` | `outputs/ADNI/mri_classifier/` |
| ADNI CN/MCI/AD | SFCN v4 分类 | `configs/adni_sfcn_v4.yaml` | `outputs/ADNI/sfcn_v4/` |

---

## 目录结构

```text
PJ1/
├── data/                    [ModelScope] 共享原始数据
│   ├── UKB_T1_100cases/
│   └── ADNI_data_105cases/
├── scripts/                 [GitHub] 统一测试集 + ModelScope 上下传
├── PJ1_UKB/                 [GitHub] 本目录代码
│   ├── configs/             [GitHub]
│   ├── scripts/             [GitHub]
│   ├── src/                 [GitHub]
│   ├── processed/           [本地] 预处理 npz
│   ├── checkpoints/         [ModelScope]
│   └── outputs/             [ModelScope]
└── PJ1_ADNI/                [GitHub 代码 + ModelScope 权重]
```

`PJ1_UKB` 内不再存放 `data/`；默认从 `../data/`（即 `PJ1/data/`）读取原始数据。可通过环境变量覆盖：

```bash
export PJ1_ROOT=/path/to/PJ1_UKB      # 项目根
export PJ1_DATA_ROOT=/path/to/PJ1/data  # 可选，显式指定共享数据目录
```

---

## 环境准备

### 本机（Windows，无 GPU）

```powershell
cd PJ1_UKB
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements-local.txt

python scripts/verify_data.py --dataset ukb
python scripts/verify_data.py --dataset adni
```

将整包上传到 GPU 服务器时保持 `PJ1/data/` 与 `PJ1_UKB/` 的相对位置不变。可通过环境变量指定项目根目录：

```bash
export PJ1_ROOT=/path/to/PJ1_UKB
```

### 显卡机（Linux / WSL，训练与预处理）

```bash
cd $PJ1_ROOT
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
```

**FSL 为预处理必需**（BET、FLIRT、MNI 模板）：

```bash
# 无 conda 时推荐
python scripts/setup_fsl_official.py
export FSLDIR=$HOME/fsl
source $FSLDIR/etc/fslconf/fsl.sh
python scripts/check_fsl_env.py   # 必须 exit 0
```

有 conda 时可选：`conda install -c conda-forge fsl -y`，然后 `export FSLDIR=$CONDA_PREFIX`。

**SwanLab**（可选，训练日志）：

```bash
pip install swanlab   # 已含在 requirements-gpu.txt
swanlab login
python scripts/test_swanlab_connection.py
```

---

## UKB SFCN 流程

### 1. 下载预训练权重

```bash
python scripts/download_sfcn_weights.py
# 需要 checkpoints/run_20190719_00_epoch_best_mae.p
#      checkpoints/run_20191008_00_epoch_last.p
```

### 2. 预处理

```bash
python scripts/preprocess_ukb_sfcn_new.py --config configs/preprocess_sfcn_new.yaml --jobs 8 --force
# 产出: processed/UKB_sfcn_new/{eid}.npz  (160×192×160)
```

### 3. 验证

```bash
python scripts/verify_processed.py --config configs/ukb_sfcn.yaml --dataset ukb
python scripts/qc_screen.py --dataset ukb   # 可选
```

### 4. 五折训练

三种任务各跑一次（最优结果对应 `20260606_12*` 批次）：

```bash
# 联合 age + sex
python scripts/train_ukb_sfcn.py --config configs/ukb_sfcn.yaml --sfcn_task both

# 仅年龄
python scripts/train_ukb_sfcn.py --config configs/ukb_sfcn.yaml --sfcn_task onlyage

# 仅性别
python scripts/train_ukb_sfcn.py --config configs/ukb_sfcn.yaml --sfcn_task onlysex
```

输出目录示例：`outputs/UKB/sfcn/20260606_120652_onlyage/`

常用参数：`--fold 0`（只跑单折）、`--no-swanlab`、`--epochs 60`、`--run_stamp myrun`

### 5. 测试集推理（五折集成）

```bash
python scripts/infer_sfcn_test.py \
  --kfold_dir outputs/UKB/sfcn/20260606_121355_both \
  --ids_csv /path/to/test_ids.csv \
  --output outputs/UKB/sfcn/test_submission.csv
```

若 onlyage / onlysex 分任务提交，可用 `merge_ukb_predictions.py` 合并 CSV。

---

## ADNI 流程

### MRI-classifier（Rootstrap DenseNet121）

```bash
python scripts/download_mri_classifier_weights.py
python scripts/preprocess_mri_classifier_adni.py --jobs 8 --force
python scripts/verify_processed.py --config configs/adni_mri_classifier.yaml --dataset adni
python scripts/train_adni_mri.py --config configs/adni_mri_classifier.yaml --swanlab
```

### SFCN v4（daomuyang/ADNI 对齐）

```bash
python scripts/download_sfcn_weights.py
python scripts/preprocess_sfcn_adni.py --preprocess-version v4 --jobs 4 --force
python scripts/verify_processed.py --config configs/adni_sfcn_v4.yaml --dataset adni
python scripts/train_adni_sfcn.py --config configs/adni_sfcn_v4.yaml --swanlab
```

K-fold + final holdout 由配置 `train.run_final_after_cv` 控制。推理：

```bash
python scripts/infer_adni_sfcn.py --config configs/adni_sfcn_v4.yaml \
  --kfold_dir outputs/ADNI/sfcn_v4/kfold
```

查看 GitHub 对齐划分：

```bash
python scripts/print_adni_github_splits.py --config configs/adni_sfcn_v4.yaml
```

---

## 脚本索引

| 脚本 | 用途 |
|------|------|
| `verify_data.py` | 检查原始 T1 与 CSV 是否匹配 |
| `verify_processed.py` | 检查 npz 是否存在且版本一致 |
| `check_fsl_env.py` | 检查 FSL 环境 |
| `setup_fsl_official.py` | 安装 FSL |
| `qc_screen.py` | 预处理 QC 异常筛查 |
| `preprocess_ukb_sfcn_new.py` | UKB SFCN 预处理 |
| `train_ukb_sfcn.py` | UKB SFCN 五折训练 |
| `infer_sfcn_test.py` | UKB 测试集五折集成推理 |
| `merge_ukb_predictions.py` | 合并分任务预测 CSV |
| `preprocess_mri_classifier_adni.py` | ADNI MRI-classifier 预处理 |
| `train_adni_mri.py` | ADNI DenseNet121 五折训练 |
| `preprocess_sfcn_adni.py` | ADNI SFCN 预处理 |
| `train_adni_sfcn.py` | ADNI SFCN 五折 + final |
| `infer_adni_sfcn.py` | ADNI SFCN 集成推理 |
| `download_sfcn_weights.py` | 下载 SFCN 预训练权重 |
| `download_mri_classifier_weights.py` | 下载 Rootstrap 预训练权重 |
| `test_swanlab_connection.py` | SwanLab 连通性测试 |

---

## 测试

```bash
pytest tests/ -q
```

---

## 备注

- UKB 最优 SFCN 结果（MAE ~2.47）在 `outputs/UKB/sfcn/20260606_12*` 三个目录。
- ADNI 更优的 Rootstrap 完整实验在独立项目 `PJ1_ADNI/`（acc ~0.75），本仓库保留 `mri_classifier` 与 `sfcn_v4` 作为对比探索。
- 预处理耗时较长，建议在 GPU 服务器上并行运行（`--jobs`）；训练阶段只读取 `processed/` 下的 npz。
- **官方测试集**：见 [`../scripts/README.md`](../scripts/README.md) 统一入口 `preprocess_test.py` / `eval_test.py`。
