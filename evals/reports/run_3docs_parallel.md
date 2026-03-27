# Evaluation Report

**Run ID**: `run_20260327_092106_56b313`
**Dataset**: financebench_v1_subset_3docs_eval.jsonl
**Generated**: 2026-03-27T09:22:51.570287+00:00

---

## Summary

| Metric | Value |
|--------|-------|
| Total samples | 15 |
| Retrieval labeled cases | 15 |
| Retrieval skipped (unlabeled / unanswerable) | 0 |

### Retrieval  _(only over labeled cases)_

| Metric | Value |
|--------|-------|
| Hit@1 | 0.2 |
| Hit@3 | 0.2 |
| Hit@5 | 0.3333 |
| Recall@5 | 0.0889 |
| MRR | 0.2333 |
| Page strict hit@5 | 0.0 |
| Page relaxed hit@5 | 0.0 |
| Section hit@5 | 0.0 |
| Evidence semantic hit@5 | 0.0 |
| Evidence lexical hit@5 | 0.0 |

### Answer

| Label | Count |
|-------|-------|
| exact | 0 |
| partial | 11 |
| wrong | 4 |
| refuse_correct | 0 |
| refuse_wrong | 0 |
| clarify_correct | 0 |
| clarify_wrong | 0 |

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
| factoid | 15 | 0 | 11 | 4 |

---

## Failure Cases  _(labeled cases only)_

### [financebench_id_00222] Does AMD have a reasonably healthy liquidity profile based o

- Labels: retrieval_miss
- Retrieved: [3599, 4404, 631, 3835, 4403]
- Gold: []
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: The knowledge base provided does not include AMD's quick ratio or any liquidity metrics for FY22. Therefore, it is insuf

### [financebench_id_01279] Among operations, investing, and financing activities, which

- Labels: retrieval_miss
- Retrieved: [671, 684, 670, 672, 683]
- Gold: [50, 79, 567]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: The net cash provided by operating activities brought in the most cash flow for AMD in FY22, totaling **$3.6 billion**. 

### [financebench_id_00476] Which debt securities are registered to trade on a national 

- Labels: retrieval_miss
- Retrieved: [4921, 4920, 4922, 4239, 4605]
- Gold: [1415, 3793, 5048, 5088, 5089]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: The provided context does not mention American Express or any debt securities registered under its name. The content per

### [financebench_id_01028] What are the geographies that American Express primarily ope

- Labels: retrieval_miss
- Retrieved: [3890, 3876, 3882, 4306, 3889]
- Gold: [143, 2940, 2941, 2942, 3894]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: American Express primarily operates in the United States (with significant presence in New York, Arizona, Delaware, and 

### [financebench_id_00723] Does AMEX have an improving operating margin profile as of 2

- Labels: retrieval_miss
- Retrieved: [546, 636, 545, 547, 635]
- Gold: [2756, 4404, 4410]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: AMD's operating margin worsened in 2022, declining from 22.2% (2021: $3.648B operating income / $16.434B net revenue) to

### [financebench_id_00720] What drove gross margin change as of the FY2022 for American

- Labels: retrieval_miss
- Retrieved: [546, 651, 545, 547, 4436]
- Gold: [650, 2630]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: Insufficient information: The provided context does not contain any data related to American Express's gross margin or f

### [financebench_id_01351] How much has the effective tax rate of American Express chan

- Labels: answer=wrong, retrieval_miss
- Retrieved: [664, 663, 665, 4681, 813]
- Gold: [2119]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: The effective tax rate for American Express is not discussed in the provided contexts, which reference AMD (Advanced Mic

### [financebench_id_00517] Are there any product categories / service categories that r

- Labels: answer=wrong
- Retrieved: [831, 845, 742, 2101, 830]
- Gold: [2037, 2078, 2086, 2101, 2875]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: The provided knowledge base does not contain any information about Boeing's revenue categories or their proportionate sh

### [financebench_id_01091] Has Boeing reported any materially important ongoing legal b

- Labels: retrieval_miss
- Retrieved: [4093, 1041, 1058, 4096, 4092]
- Gold: [4121, 4945]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: Boeing has not reported any specific, materially important ongoing legal battles in FY2022. The 2022 10-K mentions risks

### [financebench_id_00678] Does Boeing have an improving gross margin profile as of FY2

- Labels: answer=wrong
- Retrieved: [546, 651, 545, 547, 650]
- Gold: [546, 650, 2630, 4442, 5512, 5513]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: The provided context exclusively discusses AMD's financials, not Boeing. There is no information in the knowledge base a

### [financebench_id_01290] Who are the primary customers of Boeing as of FY2022?

- Labels: answer=wrong, retrieval_miss
- Retrieved: [4225, 4964, 4224, 4226, 4235]
- Gold: [4001, 4328, 4333, 4715, 4963, 4966, 4983, 5512, 5513]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: Boeing's primary customers as of FY2022 were U.S. customers, which accounted for $17,144 million in revenue (74% of tota

### [financebench_id_00464] Is Boeing's business subject to cyclicality?

- Labels: retrieval_miss
- Retrieved: [807, 3745, 806, 808, 3355]
- Gold: [212, 1660, 2214, 3996, 4230]
- Section hit@5: None
- Evidence semantic hit@5: None
- Evidence lexical hit@5: None
- Answer: The provided knowledge base contexts exclusively reference AMD and American Express 2022 10-K filings, containing no inf

---

## Retrieval Low-Confidence Samples  _(labeled cases)_

- `[financebench_id_00222]` MRR=0.0 | Does AMD have a reasonably healthy liquidity profile based o
- `[financebench_id_00995]` MRR=0.25 | What are the major products and services that AMD sells as o
- `[financebench_id_01279]` MRR=0.0 | Among operations, investing, and financing activities, which
- `[financebench_id_00476]` MRR=0.0 | Which debt securities are registered to trade on a national 
- `[financebench_id_01028]` MRR=0.0 | What are the geographies that American Express primarily ope
- `[financebench_id_00723]` MRR=0.0 | Does AMEX have an improving operating margin profile as of 2
- `[financebench_id_00720]` MRR=0.0 | What drove gross margin change as of the FY2022 for American
- `[financebench_id_01351]` MRR=0.0 | How much has the effective tax rate of American Express chan
- `[financebench_id_00517]` MRR=0.25 | Are there any product categories / service categories that r
- `[financebench_id_01091]` MRR=0.0 | Has Boeing reported any materially important ongoing legal b
- `[financebench_id_01290]` MRR=0.0 | Who are the primary customers of Boeing as of FY2022?
- `[financebench_id_00464]` MRR=0.0 | Is Boeing's business subject to cyclicality?

---
_Report generated at 2026-03-27T09:22:51.570287+00:00_