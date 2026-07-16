# Mini-VLO / Semantic-Motion 代码提交版说明

> COMP7705 工程报告配套代码包。书面报告见 `report/report-template.pdf`（单独提交，不包含在本代码包内）。

## 1. 项目概述

本仓库实现 **Semantic-Motion** 端到端工程管线，涵盖：

| 模块 | 目录 | 功能 |
| --- | --- | --- |
| Video2Tasks（推荐主路径） | `video2tasks/`, `run_video2tasks.py` | 视频任务分段与语义标注 |
| 多视角语义感知 | `src/semantic_motion/`, `run_video_task.py` | Fixed/Ego/Fused 视角融合 |
| 语言增强 + Module C | `src/module_c/`, `run_generate_filter.py` | 文本改写与多模态诊断 |
| 静态场景评测 | `src/evaluator.py`, `run_eval.py` | 30 场景结构化理解基准 |
| Module D 渲染 | `module_d/render_pipeline.py` | Blender 动作可视化（需本地 Blender） |

## 2. 环境配置

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-video2tasks.txt

cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
```

依赖 Python 3.10+。调用 VLM 需有效的 DashScope API Key；本地单元测试无需 API Key。

## 3. 快速验证（无需 API Key）

```bash
python -m pytest tests/ -q
# 预期：33 passed
```

## 4. 推荐演示命令（需 API Key）

### 4.1 Video2Tasks 单视频分段

```bash
python run_video2tasks.py \
  --video demos/video2tasks_compare/hang_towel.mp4 \
  --model qwen3-vl-flash
```

### 4.2 LIBERO Goal 多视角融合

```bash
python run_video2tasks.py \
  --manifest data/libero_goal/processed/manifest.json \
  --sample-id put_the_bowl_on_the_plate_demo_0 \
  --view-mode fused \
  --model qwen3-vl-flash
```

### 4.3 静态 30 场景评测

```bash
python run_eval.py
```

### 4.4 Module D 渲染（需 Blender 4.x）

```bash
blender --background --python module_d/render_pipeline.py -- \
  --input module_d/motions/drop_from_roof.fbx \
  --output module_d/output_videos/
```

## 5. 代码包内容

### 已包含

- **核心源码**：`src/`, `video2tasks/`, 全部 `run_*.py` 入口
- **工具与测试**：`tools/`, `tests/`, `configs/`
- **可复现小样例数据**：
  - `benchmark/` — 30 个合成静态场景
  - `demos/` — 演示视频
  - `data/gold/` — 人工标注候选包
  - `data/libero_goal/processed/` — LIBERO Goal 三任务各 1 demo
  - `data/wgo_bench/subset/`, `homer_subset/`, `galaxea_subset/` — WGO 小子集
  - `data/wgo_bench/full/manifest.json` — 完整基准清单（仅元数据）
  - `module_d/motions/` — Module D 示例动作
- **示例结果**：`results/examples/` — 代表性 JSON 输出

### 刻意排除（体积或敏感原因）

| 路径 | 原因 |
| --- | --- |
| `.env`, `.venv/` | API 密钥与本地环境 |
| `data/libero_goal/raw/` (~1.3 GB) | 原始 HDF5，可用 `tools/prepare_libero_goal_samples.py` 重新导出 |
| `data/wgo_bench/full/videos/` (~1.3 GB) | 完整 WGO 视频，可用 `tools/prepare_wgo_bench.py` 下载 |
| `report/` | 书面报告单独提交 |
| `docs/*.pdf`, `docs/*.docx` | 参考文献原文，运行代码不需要 |
| `module_d/output_videos/` | 渲染输出，可本地复现 |
| `results/` 全量日志 | 仅保留 `results/examples/` 摘要 |

## 6. 目录结构

```
mini-vlo/
├── SUBMISSION.md              ← 本文件
├── README.md                  ← 完整使用文档
├── requirements.txt
├── requirements-video2tasks.txt
├── .env.example
├── run_video2tasks.py         ← 推荐主入口
├── run_video_task.py
├── run_generate_filter.py
├── run_semantic_motion.py
├── run_eval.py
├── compare_video2tasks.py
├── src/                       ← Semantic-Motion 核心库
├── video2tasks/               ←  vendored Video2Tasks (MIT)
├── module_d/                  ← Blender 渲染管线
├── tools/                     ← 数据准备与评测脚本
├── tests/                     ← 33 个单元测试
├── configs/
├── benchmark/
├── demos/
├── data/
└── results/examples/
```

## 7. 重新生成大数据集

```bash
# LIBERO Goal 原始 HDF5 → 配对视频
python tools/prepare_libero_goal_samples.py \
  --raw-dir /path/to/libero_goal_raw \
  --output-dir data/libero_goal/processed

# WGO-Bench 子集或完整集
python tools/prepare_wgo_bench.py --subset homer --max-samples 3
python tools/prepare_wgo_bench.py --subset full --max-samples 100
```

## 8. 与 GitHub 仓库的关系

- 在线仓库：https://github.com/MarkfuGod/mini-vlo
- 本代码提交版由 `tools/package_code_submission.sh` 从仓库根目录打包生成
- 提交版为**快照归档**，不含 `.git/` 历史；完整开发历史请克隆 GitHub 仓库

## 9. 联系人

如有复现问题，请参考 `README.md` 各模块说明，或查阅 `report/report-template.pdf` 第 5 章 System Integration 部分。
