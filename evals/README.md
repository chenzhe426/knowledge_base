# Evals – Knowledge Base Evaluation System

## Purpose

这套评测系统为本地知识库项目提供**可重复的参数回归测试**能力，覆盖：

1. **检索层** – 评测混合检索的 chunk 命中率和 MRR
2. **回答层** – 评测最终答案的正确性（exact / partial / refuse / clarify）
3. **回归测试** – 对比两次运行的指标 diff，快速发现退化
4. **SFT 数据兼容** – 评测样本的数据格式可直接导出为 SFT 训练 messages

---

## Directory Structure

```
evals/
├── README.md                  # 本文件
├── datasets/
│   └── kb_eval_seed.jsonl     # 种子评测数据（10 条样本）
├── configs/
│   └── baseline.yaml          # 评测默认配置
├── reports/                   # 生成的报告放这里
├── scripts/
│   ├── run_eval.py            # 评测运行脚本
│   ├── compare_runs.py        # 两次运行对比脚本
│   ├── review_candidates.py   # 查看候选 Chunk，辅助人工标注
│   ├── apply_gold.py          # 将 gold 标注回填到数据集
│   ├── suggest_gold.py        # AI 辅助推荐 gold 标注
│   └── export_sft.py          # 导出 SFT messages 格式
└── utils/
    ├── __init__.py
    ├── dataset.py             # 数据 schema 定义 + 加载/校验工具
    ├── adapters.py            # 适配器：统一内部调用和 API 调用
    ├── scorer.py              # 检索评分 + 回答评分逻辑
    └── report.py             # JSON / Markdown 报告生成
```

---

## Data Format

每条样本是 JSONL 的一行，主要字段：

| 字段 | 说明 |
|------|------|
| `id` | 样本唯一 ID |
| `task_type` | `factoid` / `yesno` / `refuse` / `clarify` |
| `question.user_query` | 用户原始问题 |
| `question.conversation_history` | 多轮对话历史 |
| `retrieval.label_status` | 标注状态（见下表） |
| `retrieval.gold_chunk_ids` | 标准 Chunk ID 列表（用于检索评测） |
| `retrieval.gold_doc_ids` | 兜底用 Doc ID 列表 |
| `answer.gold_answer` | 标准答案 |
| `answer.must_include` | 答案必须包含的关键词/短语 |
| `answer.must_not_include` | 答案禁止包含的关键词/短语 |
| `evaluation.expected_behavior` | `answer` / `refuse` / `clarify` |
| `supervision.sft_messages_*` | SFT 训练格式，直接导出 |

### retrieval.label_status 标注状态

| 状态 | 说明 | 参与检索评测 | 参与回答评测 |
|------|------|-------------|-------------|
| `unlabeled` | 尚未补充 retrieval gold | 否 | 是 |
| `labeled_doc` | 已有 `gold_doc_ids` | 是（doc-level） | 是 |
| `labeled_chunk` | 已有 `gold_chunk_ids` | 是（chunk-level） | 是 |
| `unanswerable` | 不可回答/拒答评测样本 | 否 | 是（refuse_correct评判） |

### 第一阶段工作流（推荐顺序）

**当前阶段**：所有种子样本 `label_status` 为 `unlabeled` 或 `unanswerable`，只评测回答质量，检索指标标注为 skipped。

**后续步骤**：
1. 导入相关文档到知识库
2. 补充 `retrieval.gold_chunk_ids`（或 `gold_doc_ids`），将 `label_status` 改为 `labeled_chunk`
3. 重新运行 `run_eval.py`，激活检索指标统计

### 如何新增样本

1. 按上述 schema 构造一条 JSON 对象
2. 以追加模式写入 `evals/datasets/kb_eval_seed.jsonl`（每行一条）
3. 确保 `id` 不重复
4. 明确设置 `retrieval.label_status`（默认为 `unlabeled`）

---

## 运行评测

### 第一次运行（生成 baseline）

```bash
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_baseline.json
```

输出：
- `evals/reports/run_baseline.json`   – 详细 JSON 报告
- `evals/reports/run_baseline.md`     – 可读的 Markdown 报告

### 回归测试（改动参数或代码后）

```bash
# 运行新配置
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/your_config.yaml \
    --output evals/reports/run_exp.json

# 对比 baseline vs 新运行
python evals/scripts/compare_runs.py \
    --base evals/reports/run_baseline.json \
    --new evals/reports/run_exp.json
```

对比报告输出到 `evals/reports/compare_run_baseline_vs_run_exp.json` 和 `.md`。

---

## 配置说明（baseline.yaml）

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `mode` | `internal`（直接调 Python）或 `api`（HTTP） | `internal` |
| `top_k` | 检索返回的 chunk 数量 | `5` |
| `normalize_text` | 答案比较前是否归一化 | `true` |
| `use_doc_level_fallback` | chunk ID 匹配失败时是否用 doc ID 兜底 | `true` |
| `api.base_url` | API 模式下的 FastAPI 地址 | `http://127.0.0.1:8000` |

---

## 评测指标说明

### 检索层

| 指标 | 说明 |
|------|------|
| Hit@K | Top-K 结果中是否命中任意一个 gold chunk |
| Recall@5 | Top-5 命中的 gold chunk 占总数的比例 |
| MRR | 首个命中位置的倒数均值 |

**匹配规则**：
1. 优先用 `chunk_id` 匹配 `gold_chunk_ids`
2. 若 `chunk_id` 未提供且 `use_doc_level_fallback=true`，则用 `document_id` 匹配 `gold_doc_ids`

### 回答层

| Label | 条件 |
|-------|------|
| `exact` | 归一化后与 gold_answer 完全一致（或 token 重叠 ≥85%） |
| `partial` | `must_include` 满足 ≥50% 且无 `must_not_include` 违规 |
| `wrong` | 既不是 exact 也不是 partial |
| `refuse_correct` | `expected_behavior=refuse` 且答案含拒答语义（无证据/无法确认/未看到…） |
| `refuse_wrong` | `expected_behavior=refuse` 但给出了确定性答案 |
| `clarify_correct` | `expected_behavior=clarify` 且答案含澄清意图 |
| `clarify_wrong` | `expected_behavior=clarify` 但未澄清 |

---

## SFT 数据导出

评测样本中的 `supervision` 字段已经是 SFT messages 格式。示例导出脚本（可按需实现）：

```python
import json
from pathlib import Path
from evals.utils.dataset import load_dataset

samples = load_dataset("evals/datasets/kb_eval_seed.jsonl")
for s in samples:
    print(json.dumps(s.supervision.sft_messages_with_context or s.supervision.sft_messages_no_context))
```

---

## 如何补 Gold 标注（增量工作流）

### 第一阶段现状

当前 `kb_eval_seed.jsonl` 中大多数样本 `label_status=unlabeled`（无检索 gold），
`kb_0003` 为 `unanswerable`。回答评测已激活，检索评测跳过。

### 标注步骤

**1. 运行评测，生成报告**

```bash
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_debug.json
```

**2. AI 辅助建议 Gold（可选）**

```bash
# 对所有需要标注的样本让 LLM 推荐 gold
python evals/scripts/suggest_gold.py \
    --report evals/reports/run_debug.json \
    --only-need-labeling

# 只看某一条
python evals/scripts/suggest_gold.py \
    --report evals/reports/run_debug.json --case kb_0001

# 将建议保存为 JSON 供后续参考
python evals/scripts/suggest_gold.py \
    --report evals/reports/run_debug.json \
    --only-need-labeling \
    --output evals/reports/suggestions.json
```

AI 会分析每条样本的候选 chunks，判断哪些 chunk 明确包含答案，给出 `labeled_chunk` / `labeled_doc` / `skip` 建议及理由。**所有建议仅供参考，需人工确认后回填。**

> **报告格式说明**：`suggest_gold.py` 支持新旧两种报告格式。
> - **新格式**（含 `retrieved_chunks` + text preview）：更新后的 `report.py` 生成，分析质量最高。
> - **旧格式**（只有 `retrieved_chunk_ids` 扁平列表）：旧版 `run_debug.json`，只能看到 chunk ID，AI 判断受限，但仍会输出可用建议。
> - 如看到 `⚠️ IDs only` 提示，说明需要重新运行一次 `run_eval.py` 获得新格式报告，以获得更准确的分析。


**3. 查看候选 Chunk（人工确认 AI 建议）**

```bash
# 查看所有需要标注的样本
python evals/scripts/review_candidates.py \
    --report evals/reports/run_debug.json \
    --only-need-labeling

# 只看某一条
python evals/scripts/review_candidates.py \
    --report evals/reports/run_debug.json --case kb_0001

# Markdown 格式（方便复制到文档）
python evals/scripts/review_candidates.py \
    --report evals/reports/run_debug.json --only-unlabeled --markdown
```

`review_candidates.py` 输出包括：
- `retrieved_chunk_ids` / `retrieved_doc_ids`（检索返回的）
- `gold_chunk_ids` / `gold_doc_ids`（已有的，标为 `unlabeled` 时为空）
- `content_preview`（每条 chunk 的前 400 字摘要）
- `final_answer` / `answer_label`（当前回答情况）

**4. 回填 Gold 到数据集**

```bash
# Chunk 级标注（知道具体 chunk_id）
python evals/scripts/apply_gold.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --case kb_0001 \
    --gold-chunk 1 --gold-chunk 10 --gold-chunk 2 \
    --label-status labeled_chunk

# Doc 级标注（只知道文档 doc_id）
python evals/scripts/apply_gold.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --case kb_0002 \
    --gold-doc 3 \
    --label-status labeled_doc

# 标记为不可回答（Gold 允许为空）
python evals/scripts/apply_gold.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --case kb_0003 \
    --label-status unanswerable

# 恢复为 unlabeled（清空 gold）
python evals/scripts/apply_gold.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --case kb_0001 \
    --label-status unlabeled --clear-gold

# 预览修改内容（不写入）
python evals/scripts/apply_gold.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --case kb_0001 --gold-chunk 1 \
    --label-status labeled_chunk --dry-run
```

> 自动生成 `.jsonl.bak` 备份文件（可用 `--no-backup` 禁用）。

**5. 重新运行评测，验证检索指标**

```bash
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_after_labeling.json
```

查看报告确认：
- `retrieval_labeled_cases` 已增加
- `retrieval` 指标（Hit@K / MRR）不再为 `skipped`
- 如有 `hit_at_5=true`，说明检索命中 gold

### 标注状态说明

| label_status | 检索评测 | 回答评测 | gold 要求 |
|---|---|---|---|
| `unlabeled` | 跳过 | 是 | 可为空 |
| `labeled_doc` | 是（doc 级） | 是 | 至少一个 `gold_doc_ids` |
| `labeled_chunk` | 是（chunk 级） | 是 | 至少一个 `gold_chunk_ids` |
| `unanswerable` | 跳过 | 是（refuse评判） | 可为空 |

---

## 设计原则

1. **规则优先** – 当前所有评分逻辑都是规则-based，无 LLM 依赖，便于 CI 集成
2. **零侵入主链路** – 所有调用通过 `EvalAdapter` 封装，不修改原有服务代码
3. **未来兼容** – schema 预留了 `sft_messages_with_context` 等字段，可直接扩展到 LLM judge 或更大规模数据集
4. **可读报告** – JSON 报告供程序消费，Markdown 报告供人工审查

---

## 扩展方向

- [x] `export_sft.py` – 把评测样本批量导出为标准 SFT 数据集格式
- [ ] LLM Judge 模式 – 对 `partial` 和 `wrong` 样本做二次评判
- [ ] 对话评测 – 支持多轮 conversation eval
- [ ] 不可回答评测 – 专门评测系统拒答质量
- [ ] pytest 集成 – 把 `run_eval.py` 接入 CI regression suite
