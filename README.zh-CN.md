# Translate Agent（中文说明）

面向长篇书籍翻译（英文 -> 中文）的本地优先流水线，采用分块优先（chunk-first）与迭代质控闭环。

![logo_icon_text](img/logo_icon_texted.png)

## 当前能力

1. 端到端流程：`translate -> audit -> rewrite -> re-audit -> assemble`
2. 双模型路由：
   - 默认快路径：`qwen3-8b`
   - 难块升级：`qwen3-32b`
3. 检索增强：
   - 混合检索（dense + lexical + fusion）
   - 术语表联动
4. 分块质量控制：
   - 达标块锁定，不重复重写
   - 候选重写先做门禁再落盘
5. 交付能力：
   - 生成整书 markdown
   - 导出原文 chunks / 最新译文 chunks
   - 生成中文可读审计报告与 CSV

![pipeline](img/pipeline%20figure.png)

## 流水线阶段

`src.pipeline` 默认执行：

1. OCR/输入接入（`.pdf`、`.md`、`.txt` 或 OCR 目录）
2. 原文分块组织（`data/output/source_chunks`）
3. 术语抽取/复用（`data/glossary.json`）
4. RAG 索引
5. 分块翻译与润色
6. 聚合初稿
7. 分块审计（`src.audit`）
8. 迭代重写循环（`src.rewrite_audit_loop`）

## 目录约定（清理后）

1. `src/`：核心代码
2. `scripts/`：运维、诊断、导出脚本
3. `config/`：配置文件
4. `img/`：README 图片资源
5. `data/`：运行数据与产物（本地使用，默认不纳入 git）

## 常用命令

运行完整流水线：

```bash
python -m src.pipeline --source "data/input/MinerU_processed" --style "能看懂，保持原作风格，中文本土化"
```

小规模 smoke test：

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test_4chunks.ps1 -ChunkIds "1,2,3,4" -MaxLoops 5
```

生成交付包（整书 + chunks + 审计可读报告 + glossary）：

```bash
python scripts/build_delivery_package.py --run-dir "data/output/rewrites/rewrite_loop_run_YYYYMMDD_HHMMSS"
```

审计 JSON 转中文可读报告 + CSV：

```bash
python scripts/audit_json_to_cn_report.py --input "path/to/audit_*.json"
```

## 运行后端（默认 LM Studio）

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 启动 LM Studio OpenAI 兼容服务（默认 `http://127.0.0.1:1234/v1`）
3. 保证 `config.yaml` 的模型名与 `/v1/models` 输出一致

可选检查：

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\check_lmstudio_models.ps1
```

## 仓库卫生策略

1. `data/` 下运行产物与中间件缓存默认不提交
2. 交付通过 `scripts/build_delivery_package.py` 导出，不靠提交产物文件
3. 根目录保持轻量，只保留源码、配置与文档

