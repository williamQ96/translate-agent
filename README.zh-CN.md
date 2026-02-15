# Translate Agent（中文说明）

面向长文档（英文 -> 中文）的本地分块翻译流水线，内置审计与迭代重写质量闭环。

## 文档语言

- 英文版：`README.md`
- 中文版：`README.zh-CN.md`

## 项目概述

Translate Agent 适用于 OCR 噪声较多、篇幅较长、需要术语一致性与多轮质量提升的文本。

核心思路：

1. 先将原文组织为稳定的分块文件（chunk-first）。
2. 分块翻译时结合术语表与 RAG 上下文。
3. 第一轮翻译尽量“风格中性”，不做强风格化。
4. 风格注入仅在润色/审校阶段生效。
5. 进行分块级审计与迭代重写。
6. 最终再聚合为整本 Markdown。

## 当前端到端流程

`src.pipeline` 默认会顺序执行以下阶段：

1. OCR / 输入接入（支持 `.pdf`、`.md`、`.txt` 或 OCR 目录）
2. 原文分块组织（`data/output/source_chunks`）
3. 术语抽取/复用（`data/glossary.json`）
4. RAG 索引（ChromaDB）
5. 分块翻译（`translator -> reviewer/polisher`）
6. 聚合输出（`data/output/*_translated.md`）
7. 分块审计（`src.audit`）
8. 迭代重写循环（`src.rewrite_audit_loop`）

## 当前状态快照（来自现有日志与产物）

以下状态基于当前工作区：

- 流水线日志：`data/output/MinerU_processed_pipeline.log`
- 活跃重写运行目录：`data/output/rewrites/rewrite_loop_run_20260213_174529`
- 循环状态文件：`data/output/rewrites/rewrite_loop_run_20260213_174529/loop_state.json`

已观测状态：

1. 循环推进到 `next_loop=11`。
2. 已锁定块数：`22/52`，目标分数为 `9`。
3. 历史中最近完成轮次为 `loop 10`。
4. 最近一轮包含 `16` 个人工关注块，以及 `7` 个被守护策略拒绝的候选重写。

## 质量控制机制

1. 原文块与译文块严格对齐（`source_chunks` + `chunks`）。
2. 审计以原文分块为依据，不依赖整本聚合文本做唯一参照。
3. 低置信样本会打 `needs_human_attention` 标记，进入人工关注通道。
4. 重写采用“守护式接受策略”，降低多轮中分数回退风险。
5. 每轮保留证据目录（`loop1`、`loop2`...），包括：
   - 本轮审计报告副本
   - 本轮接受的重写块
   - 本轮被拒绝的候选重写

## 常用命令

从 OCR 目录执行完整流程（推荐）：

```bash
python -m src.pipeline --source "data/input/MinerU_processed" --style "能看懂，保持原作风格，中文本土化"
```

使用中性润色风格（不弹风格输入）：

```bash
python -m src.pipeline --source "data/input/MinerU_processed" --no-style-prompt
```

关闭翻译后的质量循环（仅翻译+聚合）：

```bash
python -m src.pipeline --source "data/input/MinerU_processed" --no-quality-loop
```

在 pipeline 内调整重写循环参数：

```bash
python -m src.pipeline \
  --source "data/input/MinerU_processed" \
  --loop-target-score 9 \
  --loop-max-loops 30 \
  --loop-acceptance-min-delta 1
```

允许自动重写人工关注块：

```bash
python -m src.pipeline --source "data/input/MinerU_processed" --loop-rewrite-human-attention
```

需要手动拆分执行时：

```bash
python -m src.audit --source-chunks-dir "data/output/source_chunks" --chunks-dir "data/output/chunks"
python -m src.rewrite_audit_loop --source-chunks-dir "data/output/source_chunks" --chunks-dir "data/output/chunks"
```

## 关键输出路径

1. 原文分块：`data/output/source_chunks/chunk_XXX.md`
2. 首轮译文分块：`data/output/chunks/chunk_XXX.md`
3. 首轮聚合译文：`data/output/*_translated.md`
4. 重写运行目录：`data/output/rewrites/rewrite_loop_run_YYYYMMDD_HHMMSS`
5. 每次重写运行的最终聚合译文：`.../rewritten_translated.md`
6. 流水线日志：`data/output/*_pipeline.log`

## 依赖

1. 安装 Python 依赖：

```bash
pip install -r requirements.txt
```

2. 在 `config.yaml` 中配置 Ollama/模型（默认 `qwen3:30b`）。

