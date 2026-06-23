# CS50029 Computational Neuroscience

复旦大学计算神经学课程项目仓库。

## 内容

| 目录 | 说明 |
|------|------|
| [`PJ1/`](PJ1/) | UKB 年龄/性别（SFCN）+ ADNI 三分类（Rootstrap 主结果 + 探索线） |
| [`PJ2/`](PJ2/) | MRI T1 2D U-Net 图像去噪（代码、报告、训练日志等） |

## 快速链接

### PJ1

| 资源 | 链接 |
|------|------|
| 详细文档 | [PJ1/README.md](PJ1/README.md) |
| 代码（GitHub） | [tree/main/PJ1](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ1) |
| 数据与权重（ModelScope） | [sSzHox/PJ_ADNI_UKB](https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB) |

**测试集**：原始包放入 `PJ1/data/<名称>/` → `python scripts/preprocess_test.py`（WSL）→ `python scripts/eval_test.py`（GPU）→ 结果在 `PJ1_UKB/outputs/test/`、`PJ1_ADNI/outputs/test/`。

### PJ2

| 资源 | 链接 |
|------|------|
| 详细文档 | [PJ2/README.md](PJ2/README.md) |
| 代码（GitHub） | [tree/main/PJ2](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ2) |
| 模型权重与预测（ModelScope） | [sSzHox/PJ-denoise](https://modelscope.cn/datasets/sSzHox/PJ-denoise) |
| 训练记录（SwanLab） | [PJ2/overview](https://swanlab.cn/@23307110248JackyH/PJ2/overview) |

## 复现概要

**PJ1**

```bash
git clone https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience.git
cd CS50029-ComputationalNeuroscience/PJ1
pip install modelscope && python scripts/download_modelscope.py
```

GitHub：代码；ModelScope：`data/`、预训练与微调权重。预处理中间产物在本地生成。

**PJ2**

```bash
git clone https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience.git
cd CS50029-ComputationalNeuroscience/PJ2
```

GitHub：代码与报告；ModelScope：`outputs/checkpoints/`、`outputs/predictions/`；课程 NIfTI 与 `data/` 需自行准备（见 [PJ2/README.md](PJ2/README.md)）。
