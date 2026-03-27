# Evaluation Report

**Run ID**: `run_20260326_143629_95d62c`
**Dataset**: financebench_v1_subset_3docs_eval.jsonl
**Generated**: 2026-03-26T14:40:04.707639+00:00

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
| Hit@1 | 0.0 |
| Hit@3 | 0.0 |
| Hit@5 | 0.0 |
| Recall@5 | 0.0 |
| MRR | 0.0 |

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

---

## Per-task-type Breakdown

| Task Type | Total | exact | partial | wrong |
|-----------|-------|-------|---------|-------|
| factoid | 15 | 0 | 11 | 4 |

---

## Failure Cases  _(labeled cases only)_

### [financebench_id_00222] Does AMD have a reasonably healthy liquidity profile based o

- Labels: retrieval_miss
- Retrieved: [618, 869, 478, 459, 483, 719, 671, 692, 617, 619, 868, 870, 477, 479, 458, 460, 482, 718, 670, 691]
- Gold: []
- Answer: AMD's quick ratio for FY22 is **10.68**, calculated using cash and cash equivalents ($4,835M), short-term investments ($

### [financebench_id_00995] What are the major products and services that AMD sells as o

- Labels: retrieval_miss
- Retrieved: [459, 898, 567, 389, 736, 835, 901, 458, 460, 897, 899, 566, 568, 900, 388, 390, 391, 735, 737, 834]
- Gold: []
- Answer: AMD’s major products and services as of FY22 include x86 microprocessors (CPUs), graphics processing units (GPUs), accel

### [financebench_id_01198] What drove revenue change as of the FY22 for AMD?

- Labels: retrieval_miss
- Retrieved: [546, 643, 545, 547, 544, 642, 644, 869, 239, 481, 732, 362, 868, 870, 238, 240, 480, 731, 733, 361]
- Gold: []
- Answer: The 44% increase in FY22 net revenue was driven by:
- **+64% Data Center segment revenue** from higher EPYC™ server proc

### [financebench_id_00917] What drove operating margin change as of the FY22 for AMD? I

- Labels: retrieval_miss
- Retrieved: [546, 650, 643, 545, 547, 869, 898, 239, 649, 651, 459, 362, 642, 644, 868, 870, 897, 899, 238, 458]
- Gold: []
- Answer: The context does not explicitly state operating margin percentages for FY22, but it explains that the **decrease in oper

### [financebench_id_01279] Among operations, investing, and financing activities, which

- Labels: retrieval_miss
- Retrieved: [671, 239, 684, 483, 730, 869, 670, 672, 692, 238, 240, 683, 685, 482, 484, 729, 731, 732, 868, 870]
- Gold: []
- Answer: The cash flow from operations brought in the most cash for AMD in FY22, with **$3.6 billion** (Source 3). Financing acti

### [financebench_id_00476] Which debt securities are registered to trade on a national 

- Labels: retrieval_miss
- Retrieved: [4107, 2119, 2479, 4223, 2180, 2897, 4493, 4106, 4108, 2118, 2120, 2478, 2480, 4222, 4224, 2179, 2181, 2477, 2896, 4492]
- Gold: []
- Answer: The provided context does not specify which debt securities are registered to trade on a national securities exchange un

### [financebench_id_01028] What are the geographies that American Express primarily ope

- Labels: retrieval_miss
- Retrieved: [3882, 3890, 3888, 2119, 2952, 1688, 3881, 3883, 3889, 3891, 2205, 2366, 3887, 2118, 2120, 2951, 2953, 1687, 2204, 2365]
- Gold: []
- Answer: American Express primarily operates in the United States, Europe (UK, France, Spain, Netherlands, Sweden, Italy), Asia (

### [financebench_id_00723] Does AMEX have an improving operating margin profile as of 2

- Labels: retrieval_miss
- Retrieved: [650, 546, 636, 254, 4414, 4349, 649, 651, 545, 547, 869, 239, 635, 637, 253, 255, 4413, 4415, 868, 238]
- Gold: []
- Answer: The knowledge base does not contain any information about AMEX (American Express) operating margins or financial perform

### [financebench_id_00720] What drove gross margin change as of the FY2022 for American

- Labels: retrieval_miss
- Retrieved: [546, 2479, 643, 2119, 3882, 2180, 1889, 545, 547, 2478, 2480, 642, 644, 2118, 2120, 3881, 3883, 2179, 2181, 1888]
- Gold: []
- Answer: The provided knowledge base does not contain any information about American Express's gross margin changes for FY2022 or

### [financebench_id_01351] How much has the effective tax rate of American Express chan

- Labels: answer=wrong, retrieval_miss
- Retrieved: [2479, 3882, 2180, 3890, 3888, 1866, 2119, 2885, 2478, 2480, 3881, 3883, 2179, 2181, 3889, 3891, 3887, 1865, 2118, 2884]
- Gold: []
- Answer: The provided context does not include specific figures for American Express' effective tax rate in FY2021 or FY2022, nor

### [financebench_id_00517] Are there any product categories / service categories that r

- Labels: answer=wrong, retrieval_miss
- Retrieved: [650, 546, 4349, 649, 651, 545, 547, 489, 544, 845, 743, 869, 4348, 4350, 488, 490, 846, 844, 742, 868]
- Gold: []
- Answer: The provided knowledge base does not include detailed revenue breakdowns by product or service categories for Boeing's F

### [financebench_id_01091] Has Boeing reported any materially important ongoing legal b

- Labels: retrieval_miss
- Retrieved: [4133, 4093, 5512, 5171, 4401, 5167, 4414, 5066, 5050, 4092, 4094, 5513, 5511, 5170, 5172, 4400, 5168, 4413, 5067, 5049]
- Gold: []
- Answer: Boeing disclosed involvement in "a number of legal proceedings" in FY2022 (Source 1) and noted that legal contingencies 

### [financebench_id_00678] Does Boeing have an improving gross margin profile as of FY2

- Labels: answer=wrong, retrieval_miss
- Retrieved: [546, 4996, 650, 4349, 671, 545, 547, 459, 869, 254, 4995, 4997, 649, 651, 670, 672, 458, 460, 868, 253]
- Gold: []
- Answer: The context does not provide Boeing's gross margin figures or trends for FY2022. While Boeing’s 10-K mentions revenue re

### [financebench_id_01290] Who are the primary customers of Boeing as of FY2022?

- Labels: answer=wrong, retrieval_miss
- Retrieved: [4225, 5512, 4983, 5050, 4964, 5167, 4224, 5076, 5513, 5511, 4352, 4982, 4406, 5049, 4963, 4965, 5168, 5075, 5077, 4351]
- Gold: []
- Answer: The primary customers of Boeing as of FY2022 were U.S. customers, which accounted for $19,662 million in revenue, compar

### [financebench_id_00464] Is Boeing's business subject to cyclicality?

- Labels: retrieval_miss
- Retrieved: [3355, 807, 5167, 4006, 5513, 5426, 5050, 5089, 3745, 3354, 3356, 806, 808, 5168, 4005, 5512, 5425, 5049, 5088, 3744]
- Gold: []
- Answer: The provided context does not explicitly address whether Boeing's business is subject to cyclicality. While the document

---

## Retrieval Low-Confidence Samples  _(labeled cases)_

- `[financebench_id_00222]` MRR=0.0 | Does AMD have a reasonably healthy liquidity profile based o
- `[financebench_id_00995]` MRR=0.0 | What are the major products and services that AMD sells as o
- `[financebench_id_01198]` MRR=0.0 | What drove revenue change as of the FY22 for AMD?
- `[financebench_id_00917]` MRR=0.0 | What drove operating margin change as of the FY22 for AMD? I
- `[financebench_id_01279]` MRR=0.0 | Among operations, investing, and financing activities, which
- `[financebench_id_00476]` MRR=0.0 | Which debt securities are registered to trade on a national 
- `[financebench_id_01028]` MRR=0.0 | What are the geographies that American Express primarily ope
- `[financebench_id_00723]` MRR=0.0 | Does AMEX have an improving operating margin profile as of 2
- `[financebench_id_00720]` MRR=0.0 | What drove gross margin change as of the FY2022 for American
- `[financebench_id_01351]` MRR=0.0 | How much has the effective tax rate of American Express chan
- `[financebench_id_00517]` MRR=0.0 | Are there any product categories / service categories that r
- `[financebench_id_01091]` MRR=0.0 | Has Boeing reported any materially important ongoing legal b
- `[financebench_id_00678]` MRR=0.0 | Does Boeing have an improving gross margin profile as of FY2
- `[financebench_id_01290]` MRR=0.0 | Who are the primary customers of Boeing as of FY2022?
- `[financebench_id_00464]` MRR=0.0 | Is Boeing's business subject to cyclicality?

---
_Report generated at 2026-03-26T14:40:04.707639+00:00_