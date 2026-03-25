# Evaluation Report

**Run ID**: `run_20260325_112447_a5bff4`
**Dataset**: kb_eval_seed.jsonl
**Generated**: 2026-03-25T11:26:08.674213+00:00

---

## Summary

| Metric | Value |
|--------|-------|
| Total samples | 10 |
| Retrieval labeled cases | 0 |
| Retrieval skipped (unlabeled / unanswerable) | 10 |

### Retrieval  _(only over labeled cases)_

| Metric | Value |
|--------|-------|
| Hit@1 | 0.0 |
| Hit@3 | 0.0 |
| Hit@5 | 0.0 |
| Recall@5 | 0.0 |
| MRR | 0.0 |

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

- `[kb_0001]` label_status=unlabeled | 这个项目支持导入哪些文档格式？
- `[kb_0002]` label_status=unlabeled | 混合检索是怎么合并向量检索和关键词检索结果的？
- `[kb_0003]` label_status=unanswerable | 系统能否导入 Excel 文件？
- `[kb_0004]` label_status=unlabeled | /ask 和 /agent/ask 接口有什么区别？
- `[kb_0005]` label_status=unlabeled | 怎么导入文档？
- `[kb_0006]` label_status=unlabeled | Agent 当前有哪些可用工具？
- `[kb_0007]` label_status=unlabeled | 文档切块时如何保证章节上下文不被打断？
- `[kb_0008]` label_status=unlabeled | 那它的向量维度是多少？
- `[kb_0009]` label_status=unlabeled | 检索结果去重的阈值是多少？
- `[kb_0010]` label_status=unlabeled | 文档导入的整体流程是什么？

## Failure Cases  _(none)_

---

## Retrieval Low-Confidence Samples  _(labeled cases)_

_none_

---
_Report generated at 2026-03-25T11:26:08.674213+00:00_