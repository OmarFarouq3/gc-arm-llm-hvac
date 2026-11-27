"""Tests for src/improvement.py on toy data with a known answer.
Run: python tests/test_improvement.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import pandas as pd
from src import improvement as im

ts, oa_bin = im.toy_transactions()
confounded = {"antecedent": ("CHL_POW_1=high",), "consequent": ("CT_FAN_SPD_1=high",)}
genuine = {"antecedent": ("CHL_DT_1=low",), "consequent": ("CHL_COP_1=low",)}

# 1. stratified_lift: hand-checked counts on a tiny example
tiny = [{"A", "B", "T=hot"}, {"A", "B", "T=hot"}, {"A", "T=hot"}, {"B", "T=hot"},
        {"A", "T=cold"}, {"T=cold"}, {"B", "T=cold"}, {"T=cold"}]
r = {"antecedent": ("A",), "consequent": ("B",)}
t = im.stratified_lift(r, tiny, "T", min_count=1)
assert t.loc["T=hot", "n"] == 4 and t.loc["T=hot", "n_ac"] == 2
assert abs(t.loc["T=hot", "lift"] - (2 * 4) / (3 * 3)) < 1e-12
assert t.loc["T=cold", "n_ac"] == 0 and t.loc["T=cold", "lift"] == 0
# bins given as an aligned list give the same answer as a variable name
t2 = im.stratified_lift(r, tiny, ["hot"] * 4 + ["cold"] * 4, min_count=1)
assert list(t2["lift"]) == list(t["lift"])
# small bins are marked invalid with the default min_count
assert not im.stratified_lift(r, tiny, "T")["valid"].any()

# 2. confounded toy rule: pooled lift well above 1, ~1 inside each temperature bin
s = im.stratified_lift(confounded, ts, "OA_TEMP")
assert len(s) == 5 and s["valid"].all()
assert ((s["lift"] - 1).abs() < 0.2).all(), s
f = im.flag_confounded(confounded, ts, "OA_TEMP")
assert f["pooled_lift"] > 1.15 and f["confounded"], f

# 3. genuine rule: association survives stratification, not flagged
g = im.flag_confounded(genuine, ts, "OA_TEMP")
assert g["pooled_lift"] > 1.5 and g["stratified_lift"] > 1.5 and not g["confounded"], g

# 4. a rule with no association at all is not flagged as confounded
none = {"antecedent": ("CHL_DT_1=low",), "consequent": ("CT_FAN_SPD_1=high",)}
n = im.flag_confounded(none, ts, "OA_TEMP")
assert not n["confounded"] and n["reason"] == "no pooled association", n

# 5. not enough data in any bin -> not flagged, reason given
x = im.flag_confounded(confounded, ts[:40], "OA_TEMP", min_count=100)
assert not x["confounded"] and x["reason"] == "no bin has enough data"

# 6. majority_vote: lists, dicts, missing labels, ties
v = im.majority_vote([[1, 0, 2, 3], [1, 3, 2, 0], [0, 0, None, None]])
assert v["label"].tolist() == [1, 0, 2, 3]       # rule 3 ties 3 vs 0 -> earliest run's label
assert abs(v.loc[0, "agreement"] - 2 / 3) < 1e-12 and v.loc[2, "votes"] == 2
assert im.majority_vote([[3], [0]], tie_break="none").loc[0, "label"] is None
d = im.majority_vote([{"r1": 2, "r2": 1}, {"r1": 2}, {"r1": 0, "r2": float("nan")}])
assert d.loc["r1", "label"] == 2 and d.loc["r2", "label"] == 1 and d.loc["r2", "votes"] == 1
e = im.majority_vote([[None], [None]])
assert e.loc[0, "label"] is None and math.isnan(e.loc[0, "agreement"])

# 7. temperature_bins: equal-frequency
b = pd.Series(im.temperature_bins(range(100), 5)).value_counts()
assert len(b) == 5 and (b == 20).all()

print("all improvement tests passed")
