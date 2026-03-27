# Evaluation Report

**Run ID**: `run_20260327_061235_39a775`
**Dataset**: kb_eval_seed.jsonl
**Generated**: 2026-03-27T06:13:42.144019+00:00

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
| Hit@1 | 0.0 |
| Hit@3 | 0.0 |
| Hit@5 | 0.0 |
| Recall@5 | 0.0 |
| MRR | 0.0 |
| Page strict hit@5 | 0.0 |
| Page relaxed hit@5 | 0.0 |
| Section hit@5 | 0.0 |
| Evidence semantic hit@5 | 0.0 |
| Evidence lexical hit@5 | 0.0 |

### Answer

| Label | Count |
|-------|-------|
| exact | 0 |
| partial | 2 |
| wrong | 6 |
| refuse_correct | 1 |
| refuse_wrong | 0 |
| clarify_correct | 0 |
| clarify_wrong | 1 |

### V4 Pipeline (answer verifier + self-refine)

| Metric | Value |
|--------|-------|
| Answer supported rate | 0.0 |
| Answer refined rate | 0.0 |
| Numeric verifier pass rate | 0.0 |
| Citation adequate rate | 0.0 |

### V4 Fallback Rates (lower is better)

| Metric | Value |
|--------|-------|
| Verifier fallback rate | 0.0 |
| Refine fallback rate | 0.0 |
| LLM rerank fallback rate | 0.0 |

### LLM Reranker

| Metric | Value |
|--------|-------|
| LLM rerank hit@5 | 0.0 |

---

## Per-task-type Breakdown

| Task Type | Total | exact | partial | wrong |
|-----------|-------|-------|---------|-------|
| clarify | 1 | 0 | 0 | 0 |
| factoid | 8 | 0 | 2 | 6 |
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

- Labels: answer=wrong, retrieval_miss
- Retrieved: [3290, 3303, 3750, 3723, 3712]
- Gold: []
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: Insufficient information. The provided context does not mention any document formats supported for import. The content f

### [kb_0004] /ask 和 /agent/ask 接口有什么区别？

- Labels: retrieval_miss
- Retrieved: [1278, 3286, 1283, 3267, 5014]
- Gold: [12]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: The provided financial document contexts (AMD_2022_10K, AMERICANEXPRESS_2022_10K, BOEING_2022_10K) contain no informatio

### [kb_0006] Agent 当前有哪些可用工具？

- Labels: answer=wrong, retrieval_miss
- Retrieved: [5402, 5436, 5433, 5401, 5403]
- Gold: [17]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: The provided context does not mention any tools available to the Agent. Insufficient information is available to answer 

---

## Retrieval Low-Confidence Samples  _(labeled cases)_

- `[kb_0001]` MRR=0.0 | 这个项目支持导入哪些文档格式？
- `[kb_0004]` MRR=0.0 | /ask 和 /agent/ask 接口有什么区别？
- `[kb_0006]` MRR=0.0 | Agent 当前有哪些可用工具？

---
_Report generated at 2026-03-27T06:13:42.144019+00:00_