# Ground-truth labelling scheme

How we give every mined rule a reference label (0-3) so the LLM's diagnosis can be scored.

## 1. Why we need an automatic scheme

The paper's ground truth comes from experts who labelled each rule. We cannot do that for several
hundred rules per run. The LBNL dataset gives us something the paper did not have: every fault file
is faulty for the **whole year**, and `ChillerPlant.csv` is the same plant without any fault.
So a rule's behaviour in the fault-free file tells us whether it describes normal operation or
something the fault created.

## 2. Label set (same as the paper)

| Label | Meaning | Present in our data? |
|---|---|---|
| 0 | Invalid rule or normal behaviour | Yes |
| 1 | Design anomaly | **No** (no design fault in the dataset) |
| 2 | Operational anomaly (wrong control or operation) | Yes |
| 3 | Performance degradation (equipment efficiency loss) | Yes |

A rule is **positive** (an anomaly rule, `validity = true`) when its label is 1, 2 or 3.

## 3. Which files we use

One file per fault type, at the highest severity, so the fault signal is strongest:

| Fault type | File | Category | Root variables |
|---|---|---|---|
| Chiller 1 supply-temperature sensor bias | `chiller_bias_+2` | 2 | `CHL_SW_TEMP_1` |
| Cooling tower 1 supply-temperature sensor bias | `coolingtower_bias_+2` | 2 | `CT_SW_TEMP_1` |
| Secondary-loop DP sensor bias | `secondary_chilled_water_pressure_bias_+20` | 2 | `CWL_SEC_DP` |
| Three-way valve leakage | `bypass_leakage_75` | 2 | `TWV_CTRL` |
| Three-way valve stuck | `bypass_stuck_75` | 2 | `TWV_CTRL` |
| Condenser PI tuning | `coolingtower_PI` | 2 | `CT_FAN_SPD_CTRL_1..3` |
| Cooling tower 1 fouling | `coolingtower_fouling_065` | 3 | `CT_SW_TEMP_1`, `CT_APPROACH_1` |
| Chiller fouling (undocumented) | `chiller_fouling_065` | 3 | `CHL_COP_1..3`, `CHL_CDDT_1..3` |

Use the exact file names from `data/raw`; the sign/format of the severity suffix may differ.

Category reasoning:
- **Sensor biases are operational (2)**: the equipment is healthy, but the controller acts on a wrong
  reading, so the plant runs at the wrong operating point.
- **Valve faults and PI tuning are operational (2)**: control or actuation is wrong.
- **Fouling is performance degradation (3)**: the heat exchanger itself has lost capacity.

`chiller_fouling` is not in the official documentation. We include it, flag it in the report, and
also report results without it.

## 4. Labelling rule

Every rule `r` mined from fault file `F` is compared with the fault-free transactions `T0`.
Both files are discretised with the **same breakpoints** (fitted on the fault-free file), so items
mean the same thing in both.

Quantities:
- `conf_F(r)`: the rule's confidence in the fault file (from mining).
- `supp_0(r)`, `conf_0(r)`: support and confidence of the same rule computed on `T0`.
  If the antecedent never occurs in `T0`, the rule is treated as new.
- `delta = |conf_F(r) - conf_0(r)|`.
- **Affected set** `A(F)`: the fault's root variables plus every node within 2 hops of them in the
  association graph (`k_hop_neighbourhood`, same k as mining).

Decision:

| Condition | Label |
|---|---|
| `supp_0 >= 0.02` and `delta < 0.10` (rule holds just as well without the fault) | **0** (normal) |
| (`supp_0 < 0.02` **or** `delta >= 0.20`) **and** the rule uses at least one variable in `A(F)` | **category of F** (2 or 3) |
| (`supp_0 < 0.02` **or** `delta >= 0.20`) but no variable in `A(F)` | **ambiguous**, excluded |
| `0.10 <= delta < 0.20` | **ambiguous**, excluded |

`0.02` is the paper's support threshold θs. The 0.10 and 0.20 margins are our choices; the gap
between them keeps borderline rules out of the evaluation instead of forcing a label on them.
We report how many rules were excluded.

Why the affected-set check: a fault running for a whole year changes many things slightly. A new rule
that has nothing to do with the faulty component is more likely a side effect or noise than a
diagnosis of the fault, so we do not count it either way.

new (supp_0 < 0.02 and (conf_0 undefined or delta >= 0.10)) or delta >= 0.20, and the rule uses at least one variable in A(F) → category of F

## 5. Checking the automatic labels by hand

Before the main experiments (and **before looking at LLM results**, so the thresholds are not tuned
to the model):

1. Draw 50 rules, stratified by automatic label (normal / anomaly / ambiguous).
2. Two team members label them independently, without seeing the automatic label, using the paper's
   definitions plus the physics in `knowledge_base/`.
3. Report Cohen's kappa between the two humans and between the humans and the automatic labels.
4. If agreement with the automatic labels is poor (kappa < 0.6), adjust the margins once, document
   the change, and freeze them.

## 6. What the LLM must never see

The fault file name, `supp_0`, `conf_0`, `delta` and the affected set are used **only** to build the
ground truth. The prompt contains the rule, its support/confidence/lift in the mined file, and the
retrieved knowledge (which contains no dataset-specific fault descriptions). Otherwise the diagnosis
would leak the answer.

## 7. Limitations (for the report)

- Category 1 cannot be evaluated; we cover 3 of the paper's 4 categories.
- Labels come from data differences, not expert judgement; the hand check in section 5 measures how
  much that matters.
- Each fault file contains one fault type, so every anomaly rule in a file gets the same category.
  The task is easier than in a real building where several faults overlap.
- Rules excluded as ambiguous are not scored; we report their number per file.
