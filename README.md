# Knowledge Base RAG System

本地知识库 RAG 系统，基于 FastAPI + Ollama + Qdrant + MySQL，支持混合检索、多轮对话、结构化输出与可解释来源。

## 技术栈

| 组件 | 技术 |
|------|------|
| API 框架 | FastAPI |
| LLM | Ollama (qwen3:8b) |
| Embedding | Ollama (nomic-embed-text, 768 维) |
| 向量检索 | Qdrant |
| 关键词检索 | MySQL FULLTEXT + BM25 |
| 数据库 | MySQL |
| 前端 | 简易 HTML 演示页面 |

## 目录结构

```
knowledge_base/
├── app/
│   ├── agent/              # LangChain Agent 模式
│   ├── api.py              # FastAPI 入口
│   ├── config.py           # 全局配置（环境变量）
│   ├── db/                 # MySQL 连接、schema、repository
│   ├── ingestion/          # 文档解析 pipeline
│   │   ├── detectors.py    # 文件类型检测
│   │   ├── loaders.py      # 文档加载
│   │   ├── parsers/        # PDF / DOCX / TXT 解析
│   │   └── pipeline.py     # 导入流程编排
│   ├── main.py             # CLI 入口
│   ├── models.py           # Pydantic 请求/响应模型
│   ├── services/           # 核心业务逻辑
│   │   ├── retrieval_service.py  # 混合检索（向量 + 关键词）
│   │   ├── qa_service.py        # RAG 问答
│   │   ├── chunk_service.py      # chunk 管理
│   │   ├── document_service.py   # 文档管理
│   │   ├── llm_service.py        # Ollama LLM 调用
│   │   └── vector_store.py       # Qdrant 接口
│   └── tools/              # LangChain Tools（搜索/摘要/导入/索引...）
├── data/                   # 示例文档
├── evals/                  # 评测系统
│   ├── datasets/           # 评测数据集
│   ├── configs/            # 评测配置
│   ├── reports/            # 评测报告
│   ├── scripts/            # 评测工具脚本
│   └── utils/              # 评测公共组件
├── logs/                   # 运行日志
├── .env                     # 环境变量配置
└── knowledge.db             # SQLite 会话存储
```

## 核心能力

### 混合检索

向量检索与关键词检索加权融合：

```
score = 0.45 × embedding + 0.20 × keyword + 0.15 × title + 0.10 × section + 0.10 × BM25
```

- **邻接 chunk 扩展**：命中后自动扩展相邻 chunk 补充上下文
- **去重**：文本相似度阈值 0.82，超过则去重
- **重排序**：多路召回后按权重合并打分

### 双模式问答

| 模式 | 接口 | 说明 |
|------|------|------|
| 直接 RAG | `POST /ask` | 直接检索知识库生成答案 |
| Agent 模式 | `POST /agent/ask` | LangChain Agent，可自主调用多种工具 |

### 文档导入流程

```
文件检测 → 内容解析 → 块切分 → 生成 embedding → 存入 MySQL + Qdrant → 建立索引
```

支持格式：PDF、DOCX、TXT

### 文档解析架构

详见 [docs/parser_architecture.md](docs/parser_architecture.md)。

文档解析支持多策略自动选择（PDF）、质量评分、噪声清洗、表格保留、段落重建等能力。

---

## 快速开始

### 1. 环境准备

```bash
# 克隆后安装依赖
pip install -r requirements.txt

# 配置环境变量 (.env)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
MYSQL_HOST=127.0.0.1
MYSQL_DATABASE=knowledge_base
QDRANT_URL=http://localhost:6333
```

### 2. 初始化数据库

```bash
python -m app.main init-db
```

### 3. 导入文档

```bash
# 导入文件夹
python -m app.main import data/

# 导入单个文件
python -m app.main import-file data/fastapi.md
```

### 4. 启动服务

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### 5. 访问接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/ask` | POST | 直接 RAG 问答 |
| `/agent/ask` | POST | Agent 模式问答 |
| `/agent/ask/stream` | POST | Agent 模式流式输出 |
| `/import/file` | POST | 导入单个文档 |
| `/import/folder` | POST | 导入文件夹 |
| `/index` | POST | 为文档建立索引 |
| `/summary` | POST | 文档摘要 |
| `/chat/session` | POST | 创建会话 |
| `/chat/{session_id}` | GET | 获取会话历史 |
| `/demo` | GET | 演示页面 |

### 6. CLI 交互

```bash
# 交互式聊天
python -m app.main chat

# 直接提问
python -m app.main ask "这个项目支持哪些文档格式？"

# 查看文档列表
python -m app.main list-docs
```

---

## 评测系统

详见 [evals/README.md](evals/README.md)

### 评测指标

**检索层**：Hit@K、Recall@5、MRR

**回答层**：exact / partial / wrong / refuse_correct / refuse_wrong / clarify_correct / clarify_wrong

### 运行评测

```bash
# 生成 baseline
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_baseline.json

# 回归对比
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/your_config.yaml \
    --output evals/reports/run_exp.json

python evals/scripts/compare_runs.py \
    --base evals/reports/run_baseline.json \
    --new evals/reports/run_exp.json
```

### Gold 标注工作流

```bash
# AI 辅助推荐 gold
python evals/scripts/suggest_gold.py \
    --report evals/reports/run_v2.json \
    --only-need-labeling \
    --output evals/reports/suggestions.json

# 查看候选 Chunk
python evals/scripts/review_candidates.py \
    --report evals/reports/run_v2.json \
    --only-need-labeling

# 回填标注到数据集
python evals/scripts/apply_gold.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --case kb_0001 \
    --gold-chunk 1 --gold-chunk 10 \
    --label-status labeled_chunk
```
