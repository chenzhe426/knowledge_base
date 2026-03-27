# Evaluation Report

**Run ID**: `run_20260326_151702_58a6b3`
**Dataset**: financebench_v1_subset_3docs_eval.jsonl
**Generated**: 2026-03-26T15:20:37.053188+00:00

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
- Retrieved: [671, 670, 672, 459, 869, 193, 389, 669, 898, 766, 458, 460, 868, 870, 192, 194, 388, 897, 899, 765]
- Gold: []
- Answer: AMD's quick ratio for FY22 is **4.54x**, calculated as ($2,366M cash + $1,582M short-term investments + $299M accounts r

### [financebench_id_00995] What are the major products and services that AMD sells as o

- Labels: retrieval_miss
- Retrieved: [459, 898, 567, 389, 591, 736, 835, 458, 460, 897, 899, 566, 568, 900, 388, 390, 590, 592, 735, 834]
- Gold: []
- Answer: AMD’s major products and services as of FY22 include x86 microprocessors (CPUs), graphics processing units (GPUs), accel

### [financebench_id_01198] What drove revenue change as of the FY22 for AMD?

- Labels: retrieval_miss
- Retrieved: [545, 646, 671, 544, 546, 647, 648, 670, 672, 820, 869, 481, 748, 1277, 819, 821, 868, 870, 480, 747]
- Gold: []
- Answer: The 44% increase in AMD’s FY22 net revenue ($23.6B vs. $16.4B in 2021) was driven by:
- **Positive drivers**: A 64% rise

### [financebench_id_00917] What drove operating margin change as of the FY22 for AMD? I

- Labels: retrieval_miss
- Retrieved: [546, 636, 671, 545, 547, 635, 637, 670, 672, 459, 898, 869, 390, 458, 460, 897, 899, 868, 389, 900]
- Gold: []
- Answer: The operating margin decline in FY22 was driven by **amortization of intangible assets from the Xilinx acquisition**, wh

### [financebench_id_01279] Among operations, investing, and financing activities, which

- Labels: retrieval_miss
- Retrieved: [671, 684, 4361, 670, 672, 692, 4349, 683, 685, 4362, 731, 691, 693, 4348, 478, 483, 730, 732, 477, 479]
- Gold: []
- Answer: The cash flow activity that brought in the most cash for AMD in FY22 was **operating activities**, which provided **$3.6

### [financebench_id_00476] Which debt securities are registered to trade on a national 

- Labels: retrieval_miss
- Retrieved: [2119, 2479, 2180, 2116, 2170, 2897, 3876, 2118, 2120, 2478, 2480, 2179, 2181, 2115, 2117, 2169, 2171, 2896, 2898, 3875]
- Gold: []
- Answer: The provided knowledge base does not specify which debt securities are registered to trade on a national securities exch

### [financebench_id_01028] What are the geographies that American Express primarily ope

- Labels: retrieval_miss
- Retrieved: [3882, 3890, 3888, 2119, 2952, 2205, 3881, 3883, 1688, 3889, 3891, 2057, 3887, 2118, 2120, 2951, 2953, 2204, 1687, 2056]
- Gold: []
- Answer: American Express primarily operates in the following geographies as of 2022: Australia, Argentina, Brazil, Canada, China

### [financebench_id_00723] Does AMEX have an improving operating margin profile as of 2

- Labels: retrieval_miss
- Retrieved: [546, 636, 639, 545, 547, 635, 637, 640, 641, 2180, 2107, 254, 459, 869, 2181, 2179, 253, 255, 458, 460]
- Gold: []
- Answer: AMD's total operating margin worsened in 2022. Total operating income fell to $1.264 billion (5.35% of net revenue) in 2

### [financebench_id_00720] What drove gross margin change as of the FY2022 for American

- Labels: retrieval_miss
- Retrieved: [546, 651, 671, 545, 547, 4436, 2479, 2180, 4412, 652, 650, 670, 672, 4404, 4435, 4437, 2478, 4411, 4405, 4403]
- Gold: []
- Answer: The provided knowledge base does not contain any information about American Express or its financial metrics, including 

### [financebench_id_01351] How much has the effective tax rate of American Express chan

- Labels: answer=wrong, retrieval_miss
- Retrieved: [3882, 1866, 3890, 3888, 2119, 2952, 3881, 3883, 1865, 1867, 2885, 3889, 3891, 3887, 3843, 2118, 2120, 2951, 2884, 3842]
- Gold: []
- Answer: Insufficient information. The provided context does not include specific effective tax rate figures for FY2021 or FY2022

### [financebench_id_00517] Are there any product categories / service categories that r

- Labels: retrieval_miss
- Retrieved: [671, 4333, 4844, 670, 672, 545, 898, 4332, 4334, 459, 742, 669, 4843, 4845, 544, 546, 897, 899, 458, 741]
- Gold: []
- Answer: The 737, 747, 767, 777, 777X, and 787 programs collectively represented **26% of total company revenues** for FY2022. Ho

### [financebench_id_01091] Has Boeing reported any materially important ongoing legal b

- Labels: retrieval_miss
- Retrieved: [4133, 4093, 5512, 4401, 4428, 5171, 4473, 5167, 4092, 4094, 4421, 5513, 5511, 4400, 4402, 4427, 4429, 5170, 4472, 5168]
- Gold: []
- Answer: Boeing disclosed ongoing legal proceedings and contingent obligations in FY2022, including "a number of legal proceeding

### [financebench_id_00678] Does Boeing have an improving gross margin profile as of FY2

- Labels: answer=wrong, retrieval_miss
- Retrieved: [546, 671, 651, 545, 547, 4404, 4410, 670, 672, 4349, 650, 652, 4364, 459, 4403, 4405, 4409, 4411, 4348, 458]
- Gold: []
- Answer: Insufficient information. The provided context exclusively references AMD's 2022 financials and does not include any dat

### [financebench_id_01290] Who are the primary customers of Boeing as of FY2022?

- Labels: answer=wrong, retrieval_miss
- Retrieved: [4225, 5512, 4983, 4352, 4358, 4172, 5050, 4200, 4224, 4226, 5513, 5511, 4982, 4351, 4353, 4357, 4359, 4171, 5049, 4198]
- Gold: []
- Answer: The context does not explicitly list Boeing's primary customers for FY2022. However, it highlights that **Chinese custom

### [financebench_id_00464] Is Boeing's business subject to cyclicality?

- Labels: retrieval_miss
- Retrieved: [3355, 5089, 4364, 4040, 1965, 1962, 1957, 3745, 3354, 3356, 5088, 5090, 4363, 4365, 1964, 1966, 1961, 1963, 1956, 3744]
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
_Report generated at 2026-03-26T15:20:37.053188+00:00_