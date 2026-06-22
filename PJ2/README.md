# PJ2 — MRI T1 2D U-Net 去噪

计算神经学课程 **PJ2 图像去噪**：对 T1 MRI 轴位 slice 做 2D U-Net 监督去噪（输入 noisy，监督 clean）。

本文档说明项目结构、数据流、命令入口与报告取数位置。具体实验数值请从输出 JSON 或 SwanLab 读取，不在此硬编码。

**代码与日志**：[CS50029-ComputationalNeuroscience](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience) · PJ2 子目录：[tree/main/PJ2](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ2)

**大文件（checkpoint / predictions）**：[PJ-denoise](https://modelscope.cn/datasets/sSzHox/PJ-denoise)（公开数据集）

**训练记录（SwanLab）**：[PJ2/overview](https://swanlab.cn/@23307110248JackyH/PJ2/overview)

### 下载来源一览

| 来源 | 链接 | 包含内容 |
|------|------|----------|
| **GitHub** | […/tree/main/PJ2](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ2) | 全部 `scripts/`、`src/`、`config.yaml`；`report/`（LaTeX + PDF）；`figures/`；`swanlog/run-*/`（SwanLab 本地 run）；`outputs/test_metrics.json`、`outputs/metrics/*_steps.json`、`outputs/eval_run1/metrics/` 与 `test_metrics.json` 等轻量产物 |
| **ModelScope** | [sSzHox/PJ-denoise](https://modelscope.cn/datasets/sSzHox/PJ-denoise) | 仅大文件：`outputs/checkpoints/`（约 4 个 `.pt`）、`outputs/predictions/`（60 个 test `{caseid}.npz`）、`outputs/eval_run1/predictions/`（60 个 Run1 `{caseid}.npz`） |
| **SwanLab** | [PJ2/overview](https://swanlab.cn/@23307110248JackyH/PJ2/overview) | 训练曲线、超参快照、验证样例图等交互式记录（云端；本地副本见 GitHub `swanlog/`） |
| **本地自备** | 课程数据 / 自行生成 | `dataset/` 原始 NIfTI；`data/` 含 `processed/` 与 `slice_cache/`（`preprocess.py` + `materialize_slices.py`） |

clone 命令：

```bash
git clone https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience.git
cd CS50029-ComputationalNeuroscience/PJ2
```

ModelScope 下载见 **§3.2**；合并后本地目录树与下文 **§2** 一致。

| 标记 | 含义 |
|------|------|
| `[GitHub]` | 已提交至 GitHub 仓库，clone 后即可获得（含 `swanlog/`、`outputs/metrics/` 等） |
| `[ModelScope]` | 公开数据集 [sSzHox/PJ-denoise](https://modelscope.cn/datasets/sSzHox/PJ-denoise) 可下载，路径与本地 `PJ2/` 下一致 |
| `[本地]` | 未上传 GitHub / ModelScope，需从课程数据自行放置，或由脚本本地生成 |

仓库根目录另有 `cn_project/`（与 `PJ2/dataset/` 同源的课程原始 NIfTI），为 `[本地]`。复现时将数据放入 `PJ2/dataset/`，再从 GitHub 取代码与日志、从 ModelScope 取 checkpoint / predictions，即可拼合完整工程。

---

## 1. 端到端流程

```text
dataset/                          原始 NIfTI
    ↓  scripts/preprocess.py
data/processed/{train,val,test}/  按 case 的 npz + split/meta
    ↓  scripts/materialize_slices.py
data/slice_cache/                 train/val memmap 切片缓存（本地 data/ 的一部分）
    ↓  scripts/train.py            （GPU；日志 Data: memmap）
outputs/checkpoints/ + SwanLab
    ↓  scripts/evaluate.py         （GPU/CPU）
outputs/test_metrics.json + predictions/
    ↓  scripts/visualize.py        （CPU，三列对比图）
outputs/figures/
    ↓  scripts/plot_training.py    （CPU，报告专用曲线/误差图）
figures/  →  report/report.tex
```

**说明**

- Test 集评估直接读 `data/processed/test/*.npz`，**不需要**对 test 做 `materialize_slices`。
- 本实验训练/验证读 **`data/slice_cache/`**（memmap）；完整本地 `data/` 应同时包含 `processed/` 与 `slice_cache/`。
- 所有命令均在 **项目根目录 `PJ2/`** 下执行：`python scripts/<name>.py`
- 可通过环境变量 **`PJ_DENOISE_ROOT`** 覆盖项目根路径（见 `src/paths.py`）。

---

## 2. 目录结构（完整本地布局）

```text
# 仓库根 CS50029-ComputationalNeuroscience/
cn_project/                     [本地] 课程原始 NIfTI（未上传）；可拷贝至 PJ2/dataset/

PJ2/
├── config.yaml                 [GitHub]  全局超参、路径、SwanLab 开关
├── requirements-cpu.txt        [GitHub]  本地：预处理 / 可视化 / 报告出图
├── requirements-gpu.txt        [GitHub]  服务器：在已有 PyTorch 上追加依赖
├── README.md                   [GitHub]
│
├── dataset/                    [本地]  原始 NIfTI（未上传）；见上方 cn_project/ 或课程数据包
│   └── cn_project_t1_noise2/
│       ├── manifest.csv
│       └── {caseid}/T1_noisy.nii.gz, T1_clean.nii.gz
│
├── data/                       [本地]  预处理与训练 I/O 缓存（整目录未上传）
│   ├── processed/              # preprocess.py
│   │   ├── train|val|test/*.npz
│   │   ├── split.json
│   │   ├── slice_index.csv
│   │   └── meta.json
│   └── slice_cache/            # materialize_slices.py；本实验训练用 memmap
│       ├── meta.json
│       └── {train,val}/noisy.dat, clean.dat, index.csv
│
├── outputs/
│   ├── checkpoints/            [ModelScope]  best.pt / last.pt 等（GitHub 未含）
│   ├── predictions/            [ModelScope]  evaluate 默认预测 npz，60 case（GitHub 未含）
│   ├── test_metrics.json       [GitHub]  evaluate 默认汇总指标
│   ├── train_history.json      [GitHub]  训练完整结束后写入（若存在）
│   ├── figures/                [GitHub]  visualize 输出：{caseid}_triplet.png
│   ├── metrics/                [GitHub]  已导出的 SwanLab step 对齐 JSON（供 plot_training）
│   └── eval_run1/              [GitHub]  示例：独立 evaluate 结果目录
│       ├── metrics/
│       ├── test_metrics.json
│       └── predictions/        [ModelScope]  Run1 预测 npz，60 case（GitHub 未含）
│
├── figures/                    [GitHub]  plot_training 输出：P1_losses.png 等
├── report/                     [GitHub]  LaTeX 报告（report.tex）
├── swanlog/                    [GitHub]  SwanLab 本地 run 缓存（已同步至 GitHub）
│   └── run-*/                  # 含 run-*.swanlab、files/config.yaml、debug 等
│
├── scripts/                    [GitHub]  可执行入口
│   ├── preprocess.py
│   ├── materialize_slices.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── plot_training.py
│   └── upload_modelscope.py    # 维护者：上传 checkpoint / predictions 至 ModelScope
│
└── src/                        [GitHub]  库代码
    ├── paths.py
    ├── config.py
    ├── model.py
    ├── losses.py
    ├── metrics.py
    ├── ssim_ops.py
    ├── dataset.py
    ├── slice_cache.py
    ├── dataloader_utils.py
    ├── training_log.py
    ├── eval_runner.py
    ├── preprocess_utils.py
    └── swanlab_config.py
```

**拼合完整工程（不重新训练）的典型顺序**：

1. `git clone` GitHub 仓库，进入 `PJ2/`（获得代码、`swanlog/`、`outputs/metrics/`、`test_metrics.json` 等）
2. 从 ModelScope 下载大文件到本地 `PJ2/outputs/`（见 §3）
3. 自行准备 `dataset/`，运行 `preprocess.py` → `materialize_slices.py`，使本地 `data/` 含 `processed/` 与 `slice_cache/`
4. 运行 `visualize.py` / `plot_training.py` 等下游脚本

**若需从头复现训练**：在上述 `data/` 就绪后，从 ModelScope 取 checkpoint 可跳过训练；否则执行 `train.py`（默认 `prefer_slice_cache: true`，读 `data/slice_cache/`）。

---

## 3. ModelScope 数据集（大文件）

**页面**：[https://modelscope.cn/datasets/sSzHox/PJ-denoise](https://modelscope.cn/datasets/sSzHox/PJ-denoise)（公开）

GitHub 因体积与 `.gitignore` 未包含 checkpoint 与 predictions；上述内容托管在 ModelScope，**远端目录结构与本地 `PJ2/` 下路径一致**，下载后合并到 clone 出来的 `PJ2/` 即可。

### 3.1 ModelScope 上的结构

```text
sSzHox/PJ-denoise/
└── outputs/
    ├── checkpoints/              # 约 4 个文件：best.pt、last.pt 等训练权重
    ├── predictions/                # 60 个 {caseid}.npz（默认 test 评估）
    └── eval_run1/
        └── predictions/            # 60 个 {caseid}.npz（Run1 独立 evaluate）
```

**每个 `{caseid}.npz` 含**：`noisy`, `clean`, `pred`, `slice_indices`（与 `evaluate.py` 输出相同）。

**不在 ModelScope 上、需从其他来源获取的内容**：

| 内容 | 来源 |
|------|------|
| 代码、配置、报告、figures | [GitHub](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ2) |
| SwanLab 本地 run（`swanlog/run-*/`） | **GitHub**（已同步，含 `.swanlab` 与 config 快照） |
| `outputs/metrics/*_steps.json` | **GitHub**（供 `plot_training.py` 画损失曲线） |
| `outputs/test_metrics.json` 等轻量指标 | **GitHub** |
| 原始 NIfTI | **本地** `dataset/`（课程数据） |
| `data/processed/`、`data/slice_cache/` | **本地** `data/`（`preprocess.py` + `materialize_slices.py`；未上传 GitHub / ModelScope） |

### 3.2 下载到本地

安装：`pip install modelscope`

```python
from modelscope.hub.snapshot_download import snapshot_download

# 下载整个数据集到缓存目录
cache_dir = snapshot_download("sSzHox/PJ-denoise", repo_type="dataset")
print(cache_dir)  # 记下路径，将其中的 outputs/ 合并到 PJ2/outputs/
```

合并示例（PowerShell，按实际 cache 路径修改）：

```powershell
cd "path\to\PJ2"
Copy-Item -Recurse -Force "path\to\cache\sSzHox\PJ-denoise\outputs\checkpoints" "outputs\"
Copy-Item -Recurse -Force "path\to\cache\sSzHox\PJ-denoise\outputs\predictions" "outputs\"
Copy-Item -Recurse -Force "path\to\cache\sSzHox\PJ-denoise\outputs\eval_run1\predictions" "outputs\eval_run1\"
```

下载完成后可直接：`evaluate.py` 跳过（已有 predictions）、`visualize.py`、`plot_training.py`（P3 仍需本地 `data/processed/test/`）。

### 3.3 上传至 ModelScope（维护者）

脚本：`scripts/upload_modelscope.py` — 将 `.gitignore` 中排除的三处目录上传到 [PJ-denoise](https://modelscope.cn/datasets/sSzHox/PJ-denoise)。

```bash
pip install modelscope

# 预览
python scripts/upload_modelscope.py --dry-run

# 上传（Token 见 https://modelscope.cn/my/myaccesstoken ）
python scripts/upload_modelscope.py --token "ms-你的SDK令牌"
# 或
export MODELSCOPE_API_TOKEN="ms-你的SDK令牌"
python scripts/upload_modelscope.py
```

可选参数：`--repo-id sSzHox/PJ-denoise`、`--commit-message "..."`。

---

## 4. 脚本说明

| 脚本 | 环境 | 作用 |
|------|------|------|
| `preprocess.py` | CPU | 原始 NIfTI → `data/processed/` |
| `materialize_slices.py` | CPU | `processed` → `data/slice_cache/`（训练 I/O 加速） |
| `train.py` | GPU | 训练 U-Net；SwanLab + checkpoint |
| `evaluate.py` | GPU/CPU | 加载 checkpoint，在 **test** 上推理并写指标 |
| `visualize.py` | CPU | 从 predictions 画 Noisy \| Clean \| Pred 三列图 |
| `plot_training.py` | CPU | 从已导出的 metrics JSON 生成报告用训练曲线与误差分析图 |
| `upload_modelscope.py` | CPU | 维护者：将 checkpoint / predictions 上传至 ModelScope |

### 常用命令

```bash
# 预处理
python scripts/preprocess.py

# 训练前缓存（推荐）
python scripts/materialize_slices.py

# 训练
python scripts/train.py
python scripts/train.py --resume outputs/checkpoints/last.pt
python scripts/train.py --no-swanlab
python scripts/train.py --no-test

# 评估
python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt
python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt \
  --output-dir outputs/eval_run1

# 简单定性对比图
python scripts/visualize.py
python scripts/visualize.py --predictions outputs/eval_run1/predictions \
  --metrics outputs/eval_run1/test_metrics.json --max_cases 6

# 报告专用图（需 outputs/metrics/ 下已有 *_steps.json）
python scripts/plot_training.py
```

---

## 5. 配置（`config.yaml`）

| 区块 | 内容 |
|------|------|
| `data.*` | 原始/处理后路径、`slice_cache_root`、`prefer_slice_cache` |
| `preprocess.*` | 轴位、脑 mask、归一化分位数、pad 尺寸、划分比例 |
| `train.*` | batch、epoch、lr、loss 权重、early stop、DataLoader 选项 |
| `model.*` | U-Net `in/out_channels`、`base_channels` |
| `inference.batch_size` | evaluate 时 volume 内 slice batch |
| `swanlab.*` | 实验名、日志频率、训练后是否跑 test |

读取方式：`src/config.py` → `load_config()`；训练时也会写入 checkpoint 的 `config` 字段。

---

## 6. 核心模块

### 模型与损失

| 文件 | 说明 |
|------|------|
| `src/model.py` | `UNet2D`：单通道 in/out，sigmoid 输出 [0,1] |
| `src/losses.py` | `DenoiseLoss` = L1 + SSIM loss |
| `src/ssim_ops.py` | SSIM 卷积，供 loss 与 metrics 共用 |

### 数据

| 组件 | 训练/验证 | 测试 evaluate |
|------|-----------|---------------|
| Dataset | `DenoiseSliceDataset` | `DenoiseVolumeDataset` |
| 优先数据源 | `MemmapSliceDataset`（slice_cache） | `data/processed/test/*.npz` |
| 回退 | npz + `CaseGroupedBatchSampler` | — |
| 增强 | GPU 随机水平翻转 | 无 |

索引文件：`split.json`（case 划分）、`slice_index.csv`（每 case slice 数）、`meta.json`（规模与预处理参数）。

### 训练（`scripts/train.py`）

- 每 epoch 结束写 `last.pt`；验证指标创新高时写 `best.pt`
- Early stopping：连续 `early_stop_patience` 个 epoch 无提升则停
- 续训：`--resume last.pt` 恢复模型与优化器；**不会**恢复历史 best 追踪（续训前建议备份 `best.pt`）
- 启动时会打印 `Data: memmap` 或 `npz`，表明是否命中 slice cache

### 评估（`scripts/evaluate.py` + `src/eval_runner.py`）

- 逐 test case 滑窗推理，unpad 到原始 slice 尺寸
- 预测 npz 含 `noisy`, `clean`, `pred`, `slice_indices`
- 汇总指标写入 `test_metrics.json`

### 可视化

**`visualize.py`**：读 `test_metrics.json` 的 `per_case`，按 PSNR 选最好/最差 case，取中间 slice 画三列 PNG → `outputs/figures/`。

**`plot_training.py`**：读 `outputs/metrics/` 与 `outputs/eval_run1/metrics/` 中已导出的 `*_steps.json` / `*_metrics.json`，生成：

| 输出 | 用途 |
|------|------|
| `figures/P1_losses.png` | 训练/验证损失曲线（Epoch 1 batch 级 + Epoch 2–36 epoch 级） |
| `figures/P3_error_analysis.png` | 最佳/最差 test case 逐切片误差分析 |

脚本内 run ID 与路径为写死常量；若重新训练产生新 run，需同步修改 `plot_training.py` 中的路径，并确保对应 `*_steps.json` 存在（可从 SwanLab 导出或保留历史 JSON）。

---

## 7. 报告取数

### 测试集定量结果（主报）

**文件**：`outputs/test_metrics.json`（或 `outputs/eval_*/test_metrics.json`）

| 字段 | 含义 | 建议 |
|------|------|------|
| `mean_case_psnr` | case 级 PSNR 平均 | **主报 PSNR** |
| `mean_case_ssim` | case 级 SSIM 平均 | **主报 SSIM** |
| `per_case[]` | 每 case 的 `caseid`, `psnr`, `ssim` | 最好/最差分析 |
| `mean_slice_psnr` / `mean_slice_ssim` | batch 汇总 | 参考，不建议主报 |

验证集最优指标也可从 checkpoint 读取：

```python
import torch
ckpt = torch.load("outputs/checkpoints/best.pt", map_location="cpu", weights_only=False)
# ckpt["epoch"], ckpt["val_metrics"]["psnr"], ckpt["val_metrics"]["ssim"]
```

### 训练曲线

| 来源 | 内容 |
|------|------|
| SwanLab 网页 | epoch 级 `val/psnr`, `val/ssim`, `train/loss` 等；[PJ2/overview](https://swanlab.cn/@23307110248JackyH/PJ2/overview) |
| `swanlog/run-*/` | **GitHub 已含**；本地 run 元数据与 config 快照 |
| `outputs/train_history.json` | 训练完整结束后的 epoch 汇总 |
| `outputs/metrics/*_steps.json` | **GitHub 已含**；step 对齐 JSON，供 `plot_training.py` |

多次中断续训会产生多个 SwanLab run；拼接 epoch 曲线时按 epoch 编号对齐，batch 曲线需注意 step offset。

### 定性图

| 来源 | 内容 |
|------|------|
| `figures/P1_losses.png`, `P3_error_analysis.png` | `plot_training.py` → `report/report.tex` |
| `outputs/figures/{caseid}_triplet.png` | `visualize.py` |
| SwanLab `samples/denoise` | 训练中验证集样例 strip |

### 数据规模

| 文件 | 内容 |
|------|------|
| `data/processed/meta.json` | case 数、各 split slice 总数 |
| `data/processed/split.json` | train/val/test case 列表 |
| `dataset/.../manifest.csv` | 原始 case 清单 |

---

## 8. SwanLab 日志要点

**训练记录**：[https://swanlab.cn/@23307110248JackyH/PJ2/overview](https://swanlab.cn/@23307110248JackyH/PJ2/overview)

`swanlog/` 目录**已同步至 GitHub**（不在 ModelScope 上），clone 后可直接查看各 run 的 `files/config.yaml`、`run-*.swanlab` 等；完整交互式曲线以 SwanLab 云端为准。

关闭方式：`--no-swanlab` 或未配置 API Key（见 `src/swanlab_config.py`）。

| 曲线类型 | 横轴 | 含义 |
|----------|------|------|
| `batch/train_*` | `global_step` | 每 `log_batch_every` 个训练 batch 一个点 |
| `train/*`, `val/*` | epoch 编号 | 每个 epoch 结束一个点 |

Checkpoint **仅在 epoch 结束**写入；中断时未跑完的 epoch 不会出现在 `last.pt`。

**epoch 级常用指标**：`train/loss`, `val/loss`, `val/psnr`, `val/ssim`, `best/val_psnr`, `train/early_stopped`

**test 指标**（训练结束后若开启）：`test/mean_case_psnr`, `test/mean_case_ssim` — 与 `test_metrics.json` 同源，**主报 case 级均值**。

完整指标列表见 `scripts/train.py` 与 `src/training_log.py` 中的 `flatten_epoch_metrics`。

---

## 9. 依赖与环境

| 文件 | 用途 |
|------|------|
| `requirements-cpu.txt` | 本地 preprocess / materialize / visualize / plot_training |
| `requirements-gpu.txt` | 服务器在 **已有 PyTorch** 上追加；不含 torch 重装 |
| `pip install modelscope` | 仅 `upload_modelscope.py` 或从 ModelScope 下载大文件时需要 |

GPU 训练需 CUDA。`evaluate.py` 与可视化脚本可在 CPU 运行；`plot_training.py` 的 P3 部分需加载 checkpoint，建议已安装 PyTorch。

---

## 10. 复现检查清单

1. **GitHub**：`git clone https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience.git`，进入 `PJ2/`（代码 + `swanlog/` + 指标 JSON + report）
2. **ModelScope**：从 [PJ-denoise](https://modelscope.cn/datasets/sSzHox/PJ-denoise) 下载 `outputs/checkpoints/`、`outputs/predictions/`、`outputs/eval_run1/predictions/` 合并到本地（§3.2）
3. **课程数据与 `data/`**：NIfTI 放入 `dataset/` → `preprocess.py` → `materialize_slices.py`（本地 `data/` 含 `processed/` + `slice_cache/`）
4. 训练 I/O：启动 `train.py` 时应见 `Data: memmap`（读 `data/slice_cache/`）
5. 超参：`config.yaml` 与 SwanLab / `swanlog/run-*/files/config.yaml`
6. 定量结果：`outputs/test_metrics.json`（GitHub）；或本地 `evaluate.py` / ModelScope predictions 复算
7. 训练曲线：`plot_training.py` 或 SwanLab；`*_steps.json` 已在 GitHub
8. 误差分析：`plot_training.py` → `figures/P3_error_analysis.png`（需 ModelScope checkpoint + 本地 `data/processed/test/`）
9. 报告编译：`report/report.tex` + `figures/*.png`

---

## 11. 速查

| 需求 | 位置 |
|------|------|
| 代码 / 报告 / swanlog | [GitHub …/PJ2](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ2) |
| checkpoint / predictions | [ModelScope PJ-denoise](https://modelscope.cn/datasets/sSzHox/PJ-denoise) |
| 训练记录 SwanLab | [PJ2/overview](https://swanlab.cn/@23307110248JackyH/PJ2/overview) |
| 上传大文件 | `scripts/upload_modelscope.py` |
| 改超参 | `config.yaml` |
| SwanLab 项目 / Key | `src/swanlab_config.py` |
| SwanLab 本地 run | `swanlog/run-*/`（GitHub） |
| 项目根路径 | `src/paths.py` 或 `PJ_DENOISE_ROOT` |
| 预处理 / memmap 缓存 | `data/processed/`、`data/slice_cache/`（本地 `data/`） |
| 最佳/最后权重 | `outputs/checkpoints/best.pt`（ModelScope） |
| 测试指标 | `outputs/test_metrics.json`（GitHub） |
| 报告图 | `figures/` |
| 简单对比图 | `outputs/figures/` |
| LaTeX 报告 | `report/report.tex` |
