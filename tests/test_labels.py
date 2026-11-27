"""Tests for src/labels.py on toy data.

Run: python tests/test_labels.py   (or: pytest tests/test_labels.py)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.labels import (FAULTS, affected_set, fault_from_filename, label_rule,  # noqa: E402
                        label_rules, load_rules, rule_stats)

SETTINGS = {"new_support": 0.02, "stable_delta": 0.10, "changed_delta": 0.20, "hops": 2}

# Fault-free transactions T0 (100 rows):
#   40 x {A=x, B=y}, 10 x {A=x, B=z}  -> conf(A=x -> B=y) = 0.8, support 0.4
#   25 x {E=lo, F=hi}, 25 x {E=lo, F=lo} -> conf(E=lo -> F=hi) = 0.5, support 0.25
#   C=hi never occurs
T0 = ([{"A=x", "B=y"}] * 40 + [{"A=x", "B=z"}] * 10
      + [{"E=lo", "F=hi"}] * 25 + [{"E=lo", "F=lo"}] * 25)

AFFECTED = {"C", "E", "F"}      # toy affected set
CATEGORY = 3


def rule(ante, cons, conf):
    return {"antecedent": ante, "consequent": cons, "support": 0.3, "confidence": conf, "lift": 1.2}


def test_rule_stats():
    s = rule_stats(rule(("A=x",), ("B=y",), 0.8), T0)
    assert abs(s["support"] - 0.40) < 1e-9 and abs(s["confidence"] - 0.80) < 1e-9
    s = rule_stats(rule(("C=hi",), ("D=lo",), 0.9), T0)
    assert s["support"] == 0.0 and s["confidence"] is None       # antecedent never occurs


def test_normal_rule():
    r = label_rule(rule(("A=x",), ("B=y",), 0.82), T0, AFFECTED, CATEGORY, SETTINGS)
    assert r["status"] == "normal" and r["label"] == 0
    assert abs(r["delta"] - 0.02) < 1e-9


def test_anomaly_new_rule():
    r = label_rule(rule(("C=hi",), ("F=hi",), 0.90), T0, AFFECTED, CATEGORY, SETTINGS)
    assert r["status"] == "anomaly" and r["label"] == CATEGORY
    assert r["conf_0"] is None and r["delta"] is None


def test_anomaly_confidence_changed():
    r = label_rule(rule(("E=lo",), ("F=hi",), 0.90), T0, AFFECTED, CATEGORY, SETTINGS)
    assert abs(r["delta"] - 0.40) < 1e-9
    assert r["status"] == "anomaly" and r["label"] == CATEGORY


def test_ambiguous_middle_band():
    r = label_rule(rule(("E=lo",), ("F=hi",), 0.65), T0, AFFECTED, CATEGORY, SETTINGS)
    assert abs(r["delta"] - 0.15) < 1e-9
    assert r["status"] == "ambiguous" and r["label"] is None


def test_ambiguous_changed_but_not_affected():
    r = label_rule(rule(("E=lo",), ("F=hi",), 0.90), T0, {"X", "Y"}, CATEGORY, SETTINGS)
    assert r["status"] == "ambiguous" and r["label"] is None


def test_boundaries():
    # delta exactly 0.10 is not "stable"; delta exactly 0.20 counts as "changed"
    r = label_rule(rule(("E=lo",), ("F=hi",), 0.60), T0, AFFECTED, CATEGORY, SETTINGS)
    assert r["status"] == "ambiguous"
    r = label_rule(rule(("E=lo",), ("F=hi",), 0.70), T0, AFFECTED, CATEGORY, SETTINGS)
    assert r["status"] == "anomaly"


def test_json_lists_become_sorted_tuples():
    rules = [{"antecedent": ["B=y", "A=x"], "consequent": ["F=hi"],
              "support": 0.1, "confidence": 0.5, "lift": 1.0}]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "rules.json")
        with open(p, "w") as f:
            json.dump(rules, f)
        loaded = load_rules(p)
    assert loaded[0]["antecedent"] == ("A=x", "B=y")
    assert loaded[0]["consequent"] == ("F=hi",)
    # label_rule also accepts lists directly
    r = label_rule(rules[0], T0, AFFECTED, CATEGORY, SETTINGS)
    assert isinstance(r["antecedent"], tuple)


def test_affected_set_uses_graph():
    a = affected_set("coolingtower_fouling", k=2)
    assert {"CT_SW_TEMP_1", "CT_APPROACH_1"} <= a        # roots
    assert "CT_FLOW_1" in a and "OA_TEMP_WB" in a        # 1 hop
    assert "CHL_SW_TEMP_1" not in a                      # far away
    assert affected_set("coolingtower_fouling", k=0) == {"CT_SW_TEMP_1", "CT_APPROACH_1"}


def test_label_rules_counts():
    rules = [rule(["A=x"], ["B=y"], 0.80),                       # normal
             rule(["CT_SW_TEMP_1=high"], ["B=y"], 0.90),         # new + affected -> anomaly
             rule(["E=lo"], ["F=hi"], 0.65)]                     # middle band -> ambiguous
    labelled, counts = label_rules(rules, T0, "coolingtower_fouling", settings=SETTINGS)
    assert counts == {"normal": 1, "anomaly": 1, "ambiguous": 1}
    assert labelled[1]["label"] == FAULTS["coolingtower_fouling"][0] == 3


def test_fault_from_filename():
    assert fault_from_filename("ChillerPlant_coolingtower_fouling_065.parquet") == "coolingtower_fouling"
    assert fault_from_filename("ChillerPlant_coolingtower_bias_2") == "coolingtower_bias"
    assert fault_from_filename("ChillerPlant_coolingtower_PI.parquet") == "coolingtower_PI"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
