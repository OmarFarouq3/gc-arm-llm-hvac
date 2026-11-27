"""Tests for src/baseline_threshold.py on synthetic data with a known answer.
Run: python tests/test_baseline_threshold.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src import baseline_threshold as bt

S = bt.load_settings()
rng = np.random.default_rng(0)
n = 1000
idx = pd.date_range("2018-01-01", periods=n, freq="15min")


def plant(n_chl_on=1.0):
    """Healthy plant at this site's real ranges, chiller 1 and tower 1 running: COP ~2.5, delta T ~10 F, approach ~8 F."""
    df = pd.DataFrame(index=idx)
    df["CWL_PRI_SW_TEMPSPT"] = 44.0
    df["CHL_SW_TEMP_1"] = 44.0 + rng.normal(0, 0.3, n)
    df["CHL_RW_TEMP_1"] = df["CHL_SW_TEMP_1"] + 10 + rng.normal(0, 0.5, n)
    df["CHL_CW_FLOW_1"] = 1200.0
    df["CHL_POW_1"] = 1200 * 10 / 24 * 3.517 / 2.5
    df["CHL_STA_1"] = n_chl_on
    df["OA_TEMP_WB"] = 70 + rng.normal(0, 1, n)
    df["CT_SW_TEMP_1"] = df["OA_TEMP_WB"] + 8 + rng.normal(0, 0.5, n)
    df["CT_STA_1"] = 1.0
    return df


# 1. derived metrics have the expected values
d = bt.add_baseline_metrics(plant())
assert abs(d["CHL_COP_1"].median() - 2.5) < 0.1
assert abs(d["CHL_DT_1"].median() - 10) < 0.2
assert abs(d["CT_APPROACH_1"].median() - 8) < 0.2
assert abs(d["CHL_SPT_DEV_1"].median()) < 0.1

# 2. a healthy plant raises (almost) no flags with the plant limits ...
ff = plant()
assert bt.flag_rows(ff, S)["any"].mean() < 0.01
# ... whereas textbook limits (COP < 4, approach > 10 would be fine) flag every healthy row
textbook = {**S, "limits": [{**l, "low": 4.0} if l["name"] == "low_cop" else l for l in S["limits"]]}
assert bt.flag_rows(ff, textbook)["low_cop"].mean() > 0.99

# 2b. each of the four limits catches its own fault and nothing else
cases = {
    "low_cop":       lambda d: d.assign(CHL_POW_1=d["CHL_POW_1"] * 2.5 / 1.6),          # COP 2.5 -> 1.6
    "low_delta_t":   lambda d: d.assign(CHL_RW_TEMP_1=d["CHL_SW_TEMP_1"] + 2.5,          # delta T 10 -> 2.5 F
                                        CHL_CW_FLOW_1=1200.0 * 10 / 2.5),                # same load, so COP unchanged
    "setpoint_dev":  lambda d: d.assign(CHL_SW_TEMP_1=d["CHL_SW_TEMP_1"] + 3,            # supply 3 F above set-point
                                        CHL_RW_TEMP_1=d["CHL_RW_TEMP_1"] + 3),
    "high_approach": lambda d: d.assign(CT_SW_TEMP_1=d["CT_SW_TEMP_1"] + 5),             # approach 8 -> 13 F
}
for name, fault in cases.items():
    f = bt.flag_rows(fault(plant()), S)
    assert f[name].mean() > 0.95, (name, f.mean())
    others = [c for c in f.columns if c not in (name, "any")]
    assert f[others].to_numpy().mean() < 0.01, (name, f.mean())

# 2c. check_limits: healthy medians sit inside the limits, ~0 % of healthy rows beyond them
cl = bt.check_limits(ff, S).set_index("limit")
assert abs(cl.loc["low_cop", "median"] - 2.5) < 0.1 and abs(cl.loc["high_approach", "median"] - 8) < 0.2
assert (cl["healthy beyond (%)"] < 1).all(), cl

# 3. tower fouling (approach +6 F -> 14 F) is caught by high_approach only
foul = plant()
foul["CT_SW_TEMP_1"] += 6
f = bt.flag_rows(foul, S)
assert f["high_approach"].mean() > 0.95 and f["low_cop"].mean() < 0.01

# 4. a stopped unit is never flagged
off = plant(n_chl_on=0.0)
off["CHL_POW_1"] = 0.0
assert bt.flag_rows(off, S)[["low_cop", "low_delta_t", "setpoint_dev"]].to_numpy().sum() == 0

# 5. persistence: isolated one-row spikes are dropped when min_consecutive = 3
spikes = plant()
spikes.iloc[::50, spikes.columns.get_loc("CT_SW_TEMP_1")] += 10
assert bt.flag_rows(spikes, S)["high_approach"].sum() >= n // 50 - 1
assert bt.flag_rows(spikes, {**S, "min_consecutive": 3})["high_approach"].sum() == 0

# 6. evaluate_rows: fault-free ~0 %, fouling ~100 %
t = bt.evaluate_rows(ff, {"fouling": foul}, S)
assert t.loc[0, "flagged (%)"] < 1 and t.loc[1, "flagged (%)"] > 95

# 7. interval parsing and rule-level classification (paper Table 3 setting)
assert bt.parse_interval("<4") == (-np.inf, 4.0)
assert bt.parse_interval(">=10.5") == (10.5, np.inf)
assert bt.parse_interval("[3.6,8)") == (3.6, 8.0)
assert bt.parse_interval("on") is None
rules = [
    {"antecedent": ("CHL_DT_1=<3.6",), "consequent": ("CHL_POW_1=<100",)},                 # paper Case 2 -> 1
    {"antecedent": ("CT_APPROACH_1=[11.5,14)", "CT_FAN_SPD_1=>=0.8"), "consequent": ("CHL_COP_1=[2,3)",)},  # -> 1
    {"antecedent": ("CHL_COP_1=[2.2,2.8)",), "consequent": ("OA_TEMP=[60,80)",)},          # normal for this plant -> 0
    {"antecedent": ("CHL_COP_1=[1.5,2.5)",), "consequent": ("CHL_STA_1=on",)},             # straddles 2.0 -> 0
    {"antecedent": ("CHL_SPT_DEV_1=>=2.5",), "consequent": ("CHL_POW_1=>=300",)},           # -> 1
    {"antecedent": ("CHL_SPT_DEV_1=<-3",), "consequent": ("CHL_POW_1=>=300",)},             # -> 1
]
pred = bt.classify_rules(rules, S)
assert pred["anomaly"].tolist() == [1, 1, 0, 0, 1, 1], pred
assert pred.loc[1, "alarms"] == "high_approach"
# a COP interval that is only low by textbook standards is normal here
assert bt.classify_rule({"antecedent": ("CHL_COP_1=<3",), "consequent": ()}, S)["anomaly"] == 0
assert bt.classify_rule({"antecedent": ("CHL_COP_1=<1.8",), "consequent": ()}, S)["anomaly"] == 1

# 8. detection metrics: ambiguous truth dropped, LLM labels 0-3 binarised
m = bt.detection_metrics([2, 3, 0, 0, -1, None], [1, 0, 0, 2, 1, 1])
assert m["n"] == 4 and m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
assert abs(m["detection_accuracy"] - 0.5) < 1e-9 and abs(m["far"] - 0.5) < 1e-9

print("all baseline_threshold tests passed")
