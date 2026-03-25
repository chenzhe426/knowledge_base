# Evaluation Report

**Run ID**: `run_20260325_124219_b2c050`
**Dataset**: kb_eval_seed.jsonl
**Generated**: 2026-03-25T12:43:40.495625+00:00

---

## Summary

| Metric | Value |
|--------|-------|
| Total samples | 10 |
| Retrieval labeled cases | 3 |
| Retrieval skipped (unlabeled / unanswerable) | 7 |

### Retrieval  _(only over labeled cases)_

| Metric | Value |
|--------|-------|
| Hit@1 | 0.6667 |
| Hit@3 | 0.6667 |
| Hit@5 | 0.6667 |
| Recall@5 | 0.6667 |
| MRR | 0.6667 |

### Answer

| Label | Count |
|-------|-------|
| exact | 0 |
| partial | 4 |
| wrong | 4 |
| refuse_correct | 1 |
| refuse_wrong | 0 |
| clarify_correct | 0 |
| clarify_wrong | 1 |

---

## Per-task-type Breakdown

| Task Type | Total | exact | partial | wrong |
|-----------|-------|-------|---------|-------|
| clarify | 1 | 0 | 0 | 0 |
| factoid | 8 | 0 | 4 | 4 |
| yesno | 1 | 0 | 0 | 0 |

---

## Retrieval Skipped Cases  _(no gold labels)_

- `[kb_0002]` label_status=unlabeled | 混合检索是怎么合并向量检索和关键词检索结果的？
- `[kb_0003]` label_status=unanswerable | 系统能否导入 Excel 文件？
- `[kb_0005]` label_status=unlabeled | 怎么导入文档？
- `[kb_0007]` label_status=unlabeled | 文档切块时如何保证章节上下文不被打断？
- `[kb_0008]` label_status=unlabeled | 那它的向量维度是多少？
- `[kb_0009]` label_status=unlabeled | 检索结果去重的阈值是多少？
- `[kb_0010]` label_status=unlabeled | 文档导入的整体流程是什么？

## Failure Cases  _(labeled cases only)_

### [kb_0001] 这个项目支持导入哪些文档格式？

- Labels: retrieval_miss
- Retrieved: [1, 10, 2, 9, 11]
- Gold: []
- Answer: 知识库中提到该项目当前支持导入的文档格式包括：PDF、Word（docx）以及纯文本。此外，还支持本地文件夹导入，后续可扩展支持网页/研报链接导入。

### [kb_0006] Agent 当前有哪些可用工具？

- Labels: answer=wrong
- Retrieved: [17, 16, 18]
- Gold: [17]
- Answer: Agent 当前可用工具仅包含以下两个：
1. `search_knowledge_base`（知识库检索）
2. `summarize_document`（文档摘要）

其他工具如 `get_chat_history` 虽然定义了 Lan

---

## Retrieval Low-Confidence Samples  _(labeled cases)_

- `[kb_0001]` MRR=0.0 | 这个项目支持导入哪些文档格式？

---
_Report generated at 2026-03-25T12:43:40.495625+00:00_