# Evaluation Report

**Run ID**: `run_20260326_145327_93703a`
**Dataset**: financebench_v1_subset_3docs_eval.jsonl
**Generated**: 2026-03-26T14:57:05.954466+00:00

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
- Retrieved: [671, 459, 4404, 869, 193, 389, 898, 766, 719, 780, 4349, 567, 4844, 670, 672, 4493, 458, 460, 901, 4721, 4403, 4405, 868, 870, 192, 194, 388, 390, 897, 899, 765, 767, 669, 1277, 900, 522, 718, 779, 781, 4348, 4350, 566, 568, 4843, 4845, 4492, 4494, 4720, 1278, 521]
- Gold: []
- Answer: The quick ratio is not explicitly mentioned in the context, but it can be inferred from the balance sheet data. AMD's ca

### [financebench_id_00995] What are the major products and services that AMD sells as o

- Labels: retrieval_miss
- Retrieved: [4145, 458, 898, 541, 4234, 4142, 567, 4144, 4146, 457, 459, 460, 897, 899, 540, 542, 4233, 4235, 4141, 4143, 566, 4431, 4329, 4513, 4455, 4226, 4293, 4375, 4183, 3997, 5418, 4430, 4432, 4328, 4330, 4512, 4514, 4454, 4456, 4225, 4227, 4294, 4292, 4374, 4376, 4236, 4182, 3996, 5419, 5417]
- Gold: []
- Answer: AMD's major products and services as of FY22 include **Field Programmable Gate Arrays (FPGAs)**, **Adaptive SoCs**, and 

### [financebench_id_01198] What drove revenue change as of the FY22 for AMD?

- Labels: retrieval_miss
- Retrieved: [545, 646, 671, 4173, 820, 742, 544, 546, 869, 647, 648, 481, 4349, 1277, 239, 4536, 670, 672, 4172, 4174, 317, 79, 21, 719, 819, 821, 741, 743, 868, 870, 1206, 1185, 480, 482, 4348, 4350, 1278, 238, 240, 4535, 4537, 316, 318, 78, 80, 20, 22, 718, 1205, 1184]
- Gold: []
- Answer: The revenue increase for AMD in FY22 was driven by:
- **+64% Data Center segment revenue** from higher EPYC™ server proc

### [financebench_id_00917] What drove operating margin change as of the FY22 for AMD? I

- Labels: retrieval_miss
- Retrieved: [546, 636, 639, 459, 898, 4414, 4169, 869, 4412, 4410, 4308, 832, 545, 547, 390, 317, 635, 637, 640, 641, 458, 460, 897, 899, 719, 901, 4413, 4415, 4168, 4170, 1277, 761, 868, 870, 4411, 4409, 4307, 1007, 831, 833, 389, 391, 900, 316, 318, 718, 902, 1278, 760, 1006]
- Gold: []
- Answer: The operating margin change for AMD in FY22 was driven by:
- **Positive drivers**: Higher revenue in the Data Center seg

### [financebench_id_01279] Among operations, investing, and financing activities, which

- Labels: retrieval_miss
- Retrieved: [671, 692, 4349, 4361, 4492, 239, 731, 670, 672, 691, 693, 673, 934, 4487, 4926, 4348, 4350, 478, 4362, 483, 4491, 4493, 238, 240, 4482, 869, 1023, 4205, 719, 730, 732, 933, 935, 4486, 4488, 4925, 4927, 4351, 729, 477, 479, 4494, 482, 484, 4481, 4483, 868, 1022, 4206, 718]
- Gold: []
- Answer: The net cash provided by investing activities was **$2 billion** in FY22, which is the highest among the three categorie

### [financebench_id_00476] Which debt securities are registered to trade on a national 

- Labels: retrieval_miss
- Retrieved: [2119, 2479, 2180, 2116, 2170, 2897, 3876, 2125, 2107, 3891, 2952, 2948, 2114, 2118, 2120, 3882, 2520, 2481, 2217, 2161, 1474, 2478, 2480, 2179, 2181, 2115, 2117, 2169, 2171, 2896, 2898, 2172, 3875, 3877, 2124, 2126, 2108, 3890, 3892, 2951, 2953, 2947, 2949, 2113, 3881, 2521, 2519, 2216, 2162, 1473]
- Gold: []
- Answer: The provided context does not specify which debt securities are registered to trade on a national securities exchange un

### [financebench_id_01028] What are the geographies that American Express primarily ope

- Labels: retrieval_miss
- Retrieved: [3355, 3882, 3890, 3751, 3888, 3335, 2119, 2952, 1688, 3881, 3883, 3889, 3891, 2205, 2366, 2114, 3383, 3843, 3887, 2391, 1675, 2479, 1696, 1607, 2897, 2118, 2120, 2951, 2953, 1687, 1689, 2204, 2206, 2365, 2367, 2113, 2115, 3382, 3384, 3842, 3844, 2390, 2392, 1674, 1676, 2478, 2480, 1695, 1606, 2896]
- Gold: []
- Answer: American Express primarily operates in the following geographies as of 2022: **India, Thailand, China, Australia, New Ze

### [financebench_id_00723] Does AMEX have an improving operating margin profile as of 2

- Labels: retrieval_miss
- Retrieved: [546, 636, 639, 2180, 2107, 254, 3501, 2481, 2125, 2170, 3886, 869, 2116, 832, 545, 547, 953, 635, 637, 459, 640, 641, 2181, 2179, 239, 898, 2108, 253, 255, 3502, 2480, 2124, 2126, 2169, 2171, 2172, 3885, 3887, 868, 870, 2115, 2117, 831, 833, 952, 954, 458, 238, 897, 899]
- Gold: []
- Answer: Insufficient information. The provided context does not include AMEX's operating margin data for 2022 or any comparative

### [financebench_id_00720] What drove gross margin change as of the FY2022 for American

- Labels: retrieval_miss
- Retrieved: [546, 2479, 651, 2180, 3501, 2116, 2119, 3882, 2170, 2125, 3890, 2481, 671, 3876, 2107, 2952, 545, 547, 2478, 2480, 459, 869, 650, 652, 2181, 2179, 3502, 2115, 2117, 2118, 2120, 3881, 3883, 2169, 2171, 2124, 2126, 2172, 3889, 3891, 670, 672, 2114, 3875, 3877, 2108, 2951, 2953, 458, 460]
- Gold: []
- Answer: The provided context does not contain information about gross margin changes for American Express in FY2022. The only gr

### [financebench_id_01351] How much has the effective tax rate of American Express chan

- Labels: answer=wrong, retrieval_miss
- Retrieved: [3818, 3882, 1866, 3890, 3888, 2119, 3448, 3362, 2952, 3678, 3731, 3373, 3395, 2885, 3456, 3881, 3883, 3843, 1852, 1865, 1867, 2897, 1790, 3889, 3891, 2114, 3887, 3317, 3225, 2269, 2118, 2120, 2951, 2953, 2884, 2886, 3842, 3844, 1851, 1853, 2896, 2898, 1789, 1791, 2113, 2115, 3316, 3318, 3224, 2268]
- Gold: []
- Answer: The context provided does not include specific figures for American Express' effective tax rate for FY2021 or FY2022, no

### [financebench_id_00517] Are there any product categories / service categories that r

- Labels: retrieval_miss
- Retrieved: [4844, 4349, 5512, 4473, 4333, 4532, 5502, 4493, 4721, 4964, 4980, 4036, 4843, 4845, 4348, 4350, 4482, 4406, 4570, 4172, 5444, 4351, 5513, 5511, 4472, 4474, 4332, 4334, 4531, 4533, 5501, 5503, 4492, 4494, 4722, 4720, 4963, 4965, 4979, 4981, 4035, 4037, 4481, 4483, 4405, 4407, 4569, 4171, 5445, 5443]
- Gold: []
- Answer: Yes, Boeing's 737 family and other aircraft categories (including 747, 767, 777, 777X, and 787) represented **26% of tot

### [financebench_id_01091] Has Boeing reported any materially important ongoing legal b

- Labels: retrieval_miss
- Retrieved: [4133, 4093, 5512, 5315, 5171, 4401, 5167, 4414, 5066, 5050, 4092, 4094, 5460, 5513, 5511, 4261, 4428, 5314, 5316, 5150, 5199, 4473, 4985, 5060, 5475, 4556, 5533, 5170, 5172, 4400, 4402, 5168, 4413, 4415, 5067, 5049, 5459, 5461, 4260, 4427, 4429, 5151, 5149, 5200, 4472, 4474, 4984, 4986, 5474, 4554]
- Gold: []
- Answer: Boeing disclosed involvement in "a number of legal proceedings" in FY2022 (Source 1) and noted "significant contingent o

### [financebench_id_00678] Does Boeing have an improving gross margin profile as of FY2

- Labels: answer=wrong, retrieval_miss
- Retrieved: [546, 4436, 4429, 651, 4404, 671, 4349, 345, 4996, 459, 545, 547, 4493, 869, 4721, 4435, 4437, 4428, 4430, 898, 389, 650, 652, 953, 719, 4403, 4405, 670, 672, 4348, 4350, 344, 346, 4995, 4997, 4351, 458, 460, 4492, 4494, 868, 870, 4722, 4720, 897, 899, 388, 952, 718, 900]
- Gold: []
- Answer: Insufficient information to determine if Boeing's gross margin profile improved in FY2022. The context does not provide 

### [financebench_id_01290] Who are the primary customers of Boeing as of FY2022?

- Labels: answer=wrong, retrieval_miss
- Retrieved: [4225, 4235, 5512, 4145, 4983, 5050, 4963, 5167, 4224, 4226, 4234, 4236, 4233, 5076, 5055, 5513, 5511, 4144, 4146, 4352, 4982, 4406, 5171, 4340, 4358, 5070, 4716, 4172, 5049, 4962, 4964, 4965, 5168, 5077, 5075, 5056, 5054, 4351, 4353, 4405, 4407, 5170, 5172, 4339, 4341, 4357, 4359, 4715, 4717, 4171]
- Gold: []
- Answer: The provided context does not explicitly list Boeing's primary customers as of FY2022. However, it mentions that **Chine

### [financebench_id_00464] Is Boeing's business subject to cyclicality?

- Labels: retrieval_miss
- Retrieved: [254, 3355, 807, 3664, 253, 255, 3354, 3356, 806, 808, 494, 1862, 3745, 1965, 1957, 3826, 1691, 2274, 1962, 482, 3832, 696, 1606, 1650, 3663, 3665, 493, 495, 1861, 1863, 3744, 3746, 1964, 1966, 1956, 1958, 1864, 3825, 3827, 1690, 1692, 2273, 2275, 1961, 1963, 481, 483, 3831, 695, 1605]
- Gold: []
- Answer: The provided knowledge base documents do not mention Boeing or its business cyclicality. The context includes discussion

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
_Report generated at 2026-03-26T14:57:05.954466+00:00_