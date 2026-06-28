# PJ1 测试集统一入口

**大文件**：[sSzHox/PJ_ADNI_UKB](https://modelscope.cn/datasets/sSzHox/PJ_ADNI_UKB)

```bash
cd PJ1
python scripts/download_modelscope.py              # 全部
python scripts/download_modelscope.py --target test20   # 原始 test20 + outputs/test 提交
```

| 内容 | ModelScope | 本地 WSL 预处理 |
|------|:----------:|:---------------:|
| `data/*_test20_release/` 原始影像 | ✅ | — |
| `PJ1_*/outputs/test/` 填好 submission | ✅ | 可重跑 |
| `PJ1_UKB/processed/`、`PJ1_ADNI/dataset/processed_rootstrap/` | ❌ | ✅ |

---

## 管线与默认权重

| `--pipeline` | 本地预处理产出 | 默认 checkpoint |
|--------------|----------------|-----------------|
| `ukb_sfcn` | `PJ1_UKB/processed/UKB_sfcn_new/<name>/` | `outputs/UKB/sfcn/20260606_12*` |
| `adni_rootstrap` | `PJ1_ADNI/dataset/processed_rootstrap/<name>/` | `outputs/rootstrap_adni_finetune_data_aug_seed3/` |
| `adni_mri_classifier` | `PJ1_UKB/processed/ADNI_mri_classifier/<name>/` | `outputs/ADNI/mri_classifier/kfold/` |
| `adni_sfcn_v4` | `PJ1_UKB/processed/ADNI_sfcn_v4/<name>/` | `outputs/ADNI/sfcn_v4/kfold/` |

`--pipeline all`（预处理）：1×UKB + 3×ADNI。  
`--pipeline all`（评测）：UKB 三任务 + 3×ADNI（不含 rootstrap 时需单独加 `--pipeline adni_rootstrap`；根目录 `eval_test.py --pipeline all` 含六线+rootstrap 见主 README）。

---

## 阶段 A：预处理（WSL + FSL）

```bash
cd PJ1
python scripts/preprocess_test.py --pipeline all \
  --name ADNI_test20 --raw ADNI_test20_release/ADNI_test20_release --jobs 4

python scripts/preprocess_test.py --pipeline ukb_sfcn \
  --name UKB_test20 --raw UKB_test20_release/UKB_test20_release --jobs 4
```

校验：`python scripts/verify_processed.py --ukb-name UKB_test20 --adni-name ADNI_test20`

---

## 阶段 B：评测（GPU）

```bash
python scripts/eval_test.py --pipeline ukb_sfcn --task both \
  --name UKB_test20 --raw UKB_test20_release/UKB_test20_release

python scripts/eval_test.py --pipeline adni_rootstrap \
  --name ADNI_test20 --raw ADNI_test20_release/ADNI_test20_release

python scripts/eval_test.py --pipeline all \
  --name ADNI_test20 --raw ADNI_test20_release/ADNI_test20_release
```

校验提交：`python scripts/verify_test20_outputs.py`

产出目录：`PJ1_UKB/outputs/test/<name>/`、`PJ1_ADNI/outputs/test/<name>/`（ModelScope 可下载）。

---

## 维护脚本

| 脚本 | 用途 |
|------|------|
| `upload_modelscope.py` | 上传大文件（远端已有则 skip） |
| `download_modelscope.py` | 下载 |
| `verify_data_layout.py` | 检查 `data/` 目录布局 |
| `verify_processed.py` | 检查四条预处理线 |
| `verify_test20_outputs.py` | 检查 submission CSV |

```bash
export PJ1_DATA_ROOT=/path/to/PJ1/data   # 可选
```
