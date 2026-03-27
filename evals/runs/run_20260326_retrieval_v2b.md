# Evaluation Report

**Run ID**: `run_20260326_144523_5a0603`
**Dataset**: financebench_v1_subset_3docs_eval.jsonl
**Generated**: 2026-03-26T14:49:02.705743+00:00

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
| partial | 12 |
| wrong | 3 |
| refuse_correct | 0 |
| refuse_wrong | 0 |
| clarify_correct | 0 |
| clarify_wrong | 0 |

---

## Per-task-type Breakdown

| Task Type | Total | exact | partial | wrong |
|-----------|-------|-------|---------|-------|
| factoid | 15 | 0 | 12 | 3 |

---

## Failure Cases  _(labeled cases only)_

### [financebench_id_00222] Does AMD have a reasonably healthy liquidity profile based o

- Labels: retrieval_miss
- Retrieved: [671, 459, 869, 193, 389, 898, 766, 670, 672, 458, 460, 868, 870, 192, 194, 388, 897, 899, 765, 669]
- Gold: []
- Answer: The quick ratio is not explicitly stated in the context, but it can be inferred. AMD's cash and cash equivalents were $2

### [financebench_id_00995] What are the major products and services that AMD sells as o

- Labels: retrieval_miss
- Retrieved: [459, 898, 567, 389, 736, 835, 901, 458, 460, 897, 899, 566, 568, 900, 388, 390, 391, 735, 737, 834]
- Gold: []
- Answer: AMD’s major products and services as of FY22 include:
- **Hardware Products**: x86 microprocessors (CPUs), graphics proc

### [financebench_id_01198] What drove revenue change as of the FY22 for AMD?

- Labels: retrieval_miss
- Retrieved: [545, 646, 671, 820, 544, 546, 869, 647, 648, 481, 748, 1277, 670, 672, 819, 821, 868, 870, 480, 747]
- Gold: []
- Answer: The 44% increase in AMD’s FY22 revenue ($23.6B vs. $16.4B in 2021) was driven by:
- **Positive drivers**: +64% Data Cent

### [financebench_id_00917] What drove operating margin change as of the FY22 for AMD? I

- Labels: retrieval_miss
- Retrieved: [546, 636, 459, 671, 898, 869, 545, 547, 390, 635, 637, 458, 460, 670, 672, 897, 899, 868, 389, 900]
- Gold: []
- Answer: The operating margin change for AMD in FY22 was driven by **higher revenue** (+$809 million, or 23% YoY growth in Data C

### [financebench_id_01279] Among operations, investing, and financing activities, which

- Labels: retrieval_miss
- Retrieved: [671, 692, 4349, 684, 4361, 731, 670, 672, 691, 693, 4348, 4350, 478, 683, 685, 483, 730, 732, 477, 479]
- Gold: []
- Answer: Operating activities brought in the most cash for AMD in FY22, with **$3.6 billion** in net cash provided, compared to *

### [financebench_id_00476] Which debt securities are registered to trade on a national 

- Labels: retrieval_miss
- Retrieved: [2119, 2479, 2180, 2116, 2170, 2897, 3876, 2118, 2120, 2478, 2480, 2179, 2181, 2115, 2117, 2169, 2171, 2896, 2898, 3875]
- Gold: []
- Answer: The provided context does not specify which debt securities are registered to trade on a national securities exchange un

### [financebench_id_01028] What are the geographies that American Express primarily ope

- Labels: retrieval_miss
- Retrieved: [3882, 3890, 3888, 2119, 2952, 1688, 3881, 3883, 3889, 3891, 2205, 2366, 3887, 2118, 2120, 2951, 2953, 1687, 2204, 2365]
- Gold: []
- Answer: American Express primarily operates in the following geographies as of 2022: **India, Thailand, China, the United States

### [financebench_id_00723] Does AMEX have an improving operating margin profile as of 2

- Labels: retrieval_miss
- Retrieved: [546, 636, 639, 2180, 254, 459, 869, 832, 545, 547, 635, 637, 640, 641, 253, 255, 458, 460, 868, 831]
- Gold: []
- Answer: Insufficient information. The provided context does not include any data on American Express's (AMEX) operating margins 

### [financebench_id_00720] What drove gross margin change as of the FY2022 for American

- Labels: retrieval_miss
- Retrieved: [546, 2479, 651, 2180, 2116, 2119, 3882, 671, 545, 547, 2478, 2480, 650, 652, 2179, 2181, 2115, 2117, 2118, 670]
- Gold: []
- Answer: The context provided does not contain information about American Express's gross margin changes for FY2022. The only gro

### [financebench_id_01351] How much has the effective tax rate of American Express chan

- Labels: answer=wrong, retrieval_miss
- Retrieved: [3882, 1866, 3890, 3888, 2119, 2952, 2885, 3881, 3883, 3843, 1865, 1867, 3889, 3891, 3887, 2118, 2120, 2951, 2884, 3842]
- Gold: []
- Answer: The context provided does not include specific figures for American Express' effective tax rate for FY2021 or FY2022, no

### [financebench_id_00517] Are there any product categories / service categories that r

- Labels: retrieval_miss
- Retrieved: [4844, 4349, 671, 5512, 4333, 4493, 4721, 4843, 4845, 4348, 4350, 670, 672, 4351, 5513, 5511, 4332, 4492, 4722, 4720]
- Gold: []
- Answer: Yes, Boeing's **Boeing Global Services (BGS)** represented **26% of total company revenues** in FY2022, exceeding the 20

### [financebench_id_01091] Has Boeing reported any materially important ongoing legal b

- Labels: retrieval_miss
- Retrieved: [4133, 4093, 5512, 5171, 4401, 5167, 4414, 5066, 5050, 4092, 4094, 5513, 5511, 5170, 5172, 4400, 5168, 4413, 5067, 5049]
- Gold: []
- Answer: Boeing disclosed ongoing legal proceedings in FY2022 (referenced in Note 21 to Consolidated Financial Statements) but di

### [financebench_id_00678] Does Boeing have an improving gross margin profile as of FY2

- Labels: answer=wrong, retrieval_miss
- Retrieved: [546, 4404, 651, 671, 4349, 459, 545, 547, 869, 4403, 4405, 650, 652, 670, 672, 4348, 4350, 4351, 458, 868]
- Gold: []
- Answer: The knowledge base does not provide specific gross margin figures or trends for Boeing in FY2022. While Boeing’s 2022 10

### [financebench_id_01290] Who are the primary customers of Boeing as of FY2022?

- Labels: answer=wrong, retrieval_miss
- Retrieved: [4225, 5512, 4983, 5050, 4964, 5167, 4224, 5076, 5513, 5511, 4352, 4982, 4406, 5049, 4963, 4965, 5168, 5075, 5077, 4351]
- Gold: []
- Answer: As of FY2022, Boeing’s primary customers are U.S. customers, which accounted for $17,144 million (or $17.144 billion) in

### [financebench_id_00464] Is Boeing's business subject to cyclicality?

- Labels: retrieval_miss
- Retrieved: [3355, 807, 5167, 4006, 5513, 5426, 5050, 5089, 3745, 3354, 3356, 806, 808, 5168, 4005, 5512, 5425, 5049, 5088, 3744]
- Gold: []
- Answer: The provided context does not address whether Boeing's business is subject to cyclicality. The documents focus on specif

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
_Report generated at 2026-03-26T14:49:02.705743+00:00_