# Translate Agent（中文说明）

面向长篇书籍翻译（英文 -> 中文）的本地化流水线，采用分块优先（chunk-first）与迭代质控闭环。

![logo_icon_text](img/logo_icon_texted.png)

## 当前已实现能力

1. 端到端流程：`translate -> audit -> rewrite -> re-audit -> assemble`
2. 双模型路由：
   - 默认快路径：`qwen3:8b`
   - 难块升级：`qwen3:30b`
3. 混合检索（Hybrid Retrieval）：
   - 稠密检索 + 词法检索 + 融合
   - 术语表联动
4. 分块审计与锁定：
   - 达标块锁定，不再重复改写
   - 低价值候选改写自动拒绝
5. 工程脚本完善：
   - 小规模烟雾测试
   - 交付导出
   - 审计 JSON 转可读报告/表格
   - 产物归档到 legacy
   - 权限诊断

![pipeline](img/pipeline%20figure.png)

## 当前流水线

`src.pipeline` 默认执行：

1. OCR/输入接入（`.pdf`、`.md`、`.txt` 或 OCR 目录）
2. 原文分块组织（`data/output/source_chunks`）
3. 术语抽取/复用（`data/glossary.json`）
4. RAG 索引
5. 分块翻译与润色
6. 聚合输出（`data/output/*_translated.md`）
7. 分块审计（`src.audit`）
8. 迭代重写循环（`src.rewrite_audit_loop`）

## Interactive GUI（规划中）

交互式 GUI 目前是规划项（尚未发布），目标包括：

1. 启动/暂停/续跑流水线
2. 实时显示每轮分数、锁定块、耗时
3. 人工关注块（human-attention）审阅面板
4. 一键导出交付包（整书 + 最新 chunk）

## Beads 集成计划（规划中）

`beads` 评估定位为“可选记忆/追踪层”，不替代当前 RAG 主链路。

预期用途：

1. 持久化每个 chunk 的改写决策与审计结果
2. 记录人工修订结论与术语决策
3. 复用历史 hard chunk 经验

约束：

1. 采用 fail-open 设计（beads 不可用时，主流水线仍可运行）

## 领域定位（基于证据）

结论：当前项目在工程工作流上有差异化，但不宜直接宣称“全球独创/行业第一”。

原因：

1. 文档级 AI 翻译产品已成熟存在（DeepL 文档翻译、Google Docs 翻译、Amazon/KDP 翻译功能）。
2. 该赛道竞争活跃。
3. 我们的核心差异化在于：
   - chunk-to-chunk 源文审计
   - 可锁定的迭代重写闭环
   - 本地优先双模型路由
   - 产物可追踪与可复现

建议对外表述：

1. “面向出版级长文翻译的开源工程化质量闭环方案”
2. 避免在缺乏基准评测时使用“全球首创/绝对领先”

## 目录结构（发布友好）

保持 root 简洁，运行产物集中放到 `data/`：

1. `src/`：核心代码
2. `scripts/`：运维与调试脚本
3. `config/`：配置文件（例如 `config/magic-pdf.json`）
4. `data/input/`：输入源文件
5. `data/output/`：运行产物（chunks/audits/rewrites/logs/smoke）
6. `data/output/legacy_data/`：历史批次归档

## 常用命令

运行完整流水线：

```bash
python -m src.pipeline --source "data/input/MinerU_processed" --style "能看懂，保持原作风格，中文本土化"
```

3-4 chunk 小规模实测：

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test_4chunks.ps1 -ChunkIds "1,2,3,4" -MaxLoops 5
```

归档当前产物到 legacy：

```bash
python scripts/archive_output_artifacts.py --run-name "work_YYYYMMDD_HHMMSS" --include-ocr
```

审计 JSON 转中文可读报告 + CSV：

```bash
python scripts/audit_json_to_cn_report.py --input "path/to/audit_loop_XX.json"
```

## 依赖

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 拉取 Ollama 模型：

```bash
ollama pull qwen3:8b
ollama pull qwen3:30b
```

## 参考链接

1. Beads：https://github.com/steveyegge/beads
2. Beads 文档：https://beads.ignition.dev/
3. DeepL 文档翻译：https://support.deepl.com/hc/en-us/articles/360020698639-Translate-documents
4. Google Docs 翻译：https://support.google.com/docs/answer/187189
5. Amazon KDP 翻译功能：https://kdp.amazon.com/en_US/help/topic/GTH4C7FLRNCXSWJW
