# CS50029 Computational Neuroscience

复旦大学计算神经学课程项目仓库。

## 内容

| 目录 | 说明 |
|------|------|
| [`PJ2/`](PJ2/) | MRI T1 2D U-Net 图像去噪（代码、报告、训练日志等） |

其他 PJ 若后续加入，将并列放在本仓库根目录下。

## 快速链接

| 资源 | 链接 |
|------|------|
| PJ2 详细文档 | [PJ2/README.md](PJ2/README.md) |
| 代码（GitHub） | [tree/main/PJ2](https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience/tree/main/PJ2) |
| 模型权重与预测（ModelScope） | [sSzHox/PJ-denoise](https://modelscope.cn/datasets/sSzHox/PJ-denoise) |
| 训练记录（SwanLab） | [PJ2/overview](https://swanlab.cn/@23307110248JackyH/PJ2/overview) |

## 复现 PJ2（概要）

```bash
git clone https://github.com/Jacky23307110248/CS50029-ComputationalNeuroscience.git
cd CS50029-ComputationalNeuroscience/PJ2
```

1. 从 GitHub 获得代码、报告、`swanlog/` 与指标 JSON  
2. 从 ModelScope 下载 `outputs/checkpoints/`、`outputs/predictions/` 等到本地 `PJ2/outputs/`  
3. 自行准备课程 NIfTI 与 `data/`（见 [PJ2/README.md](PJ2/README.md)）

大体积原始数据与预处理产物不在 GitHub 上；checkpoint / predictions 在 ModelScope 公开数据集。
