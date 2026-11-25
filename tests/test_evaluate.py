"""Tests for src/evaluate.py.
Run: python tests/test_evaluate.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import numpy as np
from src import evaluate as ev


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def from_confusion(cm):
    """Rebuild y_true / y_pred lists from a confusion matrix (rows = true)."""
    yt, yp = [], []
    for t in range(cm.shape[0]):
        for p in range(cm.shape[1]):
            yt += [t] * int(cm[t, p])
            yp += [p] * int(cm[t, p])
    return yt, yp


# 1. The paper's Experiment 1 confusion matrix (ChatGPT-4o, run 1): 296 rules.
PAPER_CM = np.array([[183, 2, 10, 6],
                     [0, 20, 2, 0],
                     [3, 1, 62, 0],
                     [0, 0, 3, 4]])
yt, yp = from_confusion(PAPER_CM)
m = ev.paper_metrics(yt, yp)
assert m["n"] == 296
assert close(m["accuracy"], 269 / 296)               # 90.88 %
assert (m["tp"], m["fp"], m["fn"]) == (92, 18, 3)
assert close(m["far"], 18 / 110)
assert close(m["mar"], 3 / 95)
assert (ev.confusion(yt, yp) == PAPER_CM).all()

# 2. Per-class report and macro-F1 skip a class that never occurs.
yt2 = [0, 0, 2, 2, 3, 3]
yp2 = [0, 2, 2, 2, 3, 0]
r = ev.per_class_report(yt2, yp2)
assert close(r[2]["precision"], 2 / 3) and close(r[2]["recall"], 1.0)
assert r[1]["support"] == 0                          # no design rules
f1_0, f1_2, f1_3 = 0.5, 0.8, 2 / 3
assert close(r["macro_f1"], (f1_0 + f1_2 + f1_3) / 3)

# 3. Ambiguous ground truth (None / -1) is ignored.
m3 = ev.paper_metrics([0, None, 2, -1], [0, 3, 2, 3])
assert m3["n"] == 2 and close(m3["accuracy"], 1.0)

# 4. LLM output -> label.
assert ev.to_prediction({"validity": True, "label": 2}) == 2
assert ev.to_prediction({"validity": False, "label": 2}) == 0
assert ev.to_prediction({"validity": True, "label": "3"}) == 3
assert ev.to_prediction({"validity": True, "label": 7}) == 0
assert ev.to_prediction(None) == 0
assert ev.count_failures([None, {"label": 2}, {"label": 9}, {"label": "x"}]) == 3

# 5. Run agreement.
same = [[0, 2, 3, 0]] * 3
a = ev.run_agreement(same)
assert a["all_agree"] == 1.0 and a["mean_pairwise"] == 1.0 and close(a["fleiss_kappa"], 1.0)
mixed = [[0, 2, 3, 0], [0, 2, 0, 0], [0, 3, 3, 0]]
a = ev.run_agreement(mixed)
assert close(a["all_agree"], 0.5)                    # rules 0 and 3 agree
assert close(a["mean_pairwise"], (0.75 + 0.75 + 0.5) / 3)
assert a["fleiss_kappa"] < 1.0

# 6. Mean +/- std over runs, and the printed summary.
res = ev.evaluate_runs(yt2, [yp2, yt2, yp2])
acc = [4 / 6, 1.0, 4 / 6]
assert close(res["summary"]["accuracy"]["mean"], np.mean(acc))
assert close(res["summary"]["accuracy"]["std"], np.std(acc, ddof=1))
text = ev.format_summary(res)
assert "Accuracy" in text and "Fleiss" in text
print(text)

# 7. No anomalies predicted -> FAR is undefined, not a crash.
m7 = ev.paper_metrics([0, 2], [0, 0])
assert math.isnan(m7["far"]) and m7["mar"] == 1.0

print("All evaluation tests passed.")
