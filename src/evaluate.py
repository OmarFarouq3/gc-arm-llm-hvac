"""Evaluation metrics for the LLM rule diagnosis.

Paper (Deng et al., 2026, Section 4): accuracy, FAR, MAR and consistency.
    FAR = FP / (TP + FP)   (the paper's definition; this is 1 - precision)
    MAR = FN / (TP + FN)   (1 - recall)
Positive = anomaly rule (label 1, 2 or 3); negative = label 0.

Our additions: per-class precision/recall/F1, macro-F1, confusion matrix,
mean +/- std over runs, and run-to-run agreement (all-agree rate,
mean pairwise agreement, Fleiss' kappa).

Ground-truth labels come from report/labelling_scheme.md. Rules labelled
ambiguous (None or -1) are dropped before scoring.

Owner: Omar
"""

from itertools import combinations

import numpy as np

LABELS = (0, 1, 2, 3)
NAMES = {0: "normal/invalid", 1: "design", 2: "operational", 3: "degradation"}


# ---------------------------------------------------------------------------
# Turning LLM outputs into labels
# ---------------------------------------------------------------------------

def to_prediction(output):
    """Map one LLM output to a label 0-3.

    output: dict like {"validity": bool, "label": int, "reason": str},
            or None if the answer could not be parsed.

    Rules:
      - unparsable answer -> 0 (counted separately by count_failures)
      - validity false    -> 0, whatever the label says
      - otherwise         -> the label, if it is 0-3, else 0
    """
    if not isinstance(output, dict):
        return 0
    if not output.get("validity", False):
        return 0
    try:
        label = int(output.get("label", 0))
    except (TypeError, ValueError):
        return 0
    return label if label in LABELS else 0


def count_failures(outputs):
    """Number of outputs that were not a usable dict with a valid label."""
    bad = 0
    for o in outputs:
        if not isinstance(o, dict):
            bad += 1
            continue
        try:
            if int(o.get("label", -1)) not in LABELS:
                bad += 1
        except (TypeError, ValueError):
            bad += 1
    return bad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(y_true, y_pred):
    """Drop rules whose ground truth is ambiguous (None or -1)."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} predicted")
    keep = [i for i, t in enumerate(y_true) if t is not None and t != -1]
    yt = np.array([int(y_true[i]) for i in keep], dtype=int)
    yp = np.array([int(y_pred[i]) for i in keep], dtype=int)
    return yt, yp


def _div(a, b):
    return float(a) / b if b else float("nan")


def confusion(y_true, y_pred, labels=LABELS):
    """Confusion matrix; rows = true label, columns = predicted label."""
    yt, yp = _clean(y_true, y_pred)
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(yt, yp):
        cm[idx[t], idx[p]] += 1
    return cm


# ---------------------------------------------------------------------------
# Paper metrics
# ---------------------------------------------------------------------------

def paper_metrics(y_true, y_pred):
    """Accuracy (4-class), FAR and MAR (anomaly vs normal), as in the paper."""
    yt, yp = _clean(y_true, y_pred)
    pos_t, pos_p = yt != 0, yp != 0
    tp = int(np.sum(pos_t & pos_p))
    fp = int(np.sum(~pos_t & pos_p))
    fn = int(np.sum(pos_t & ~pos_p))
    return {
        "n": int(len(yt)),
        "accuracy": _div(np.sum(yt == yp), len(yt)),
        "far": _div(fp, tp + fp),
        "mar": _div(fn, tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# Our additions
# ---------------------------------------------------------------------------

def per_class_report(y_true, y_pred, labels=LABELS):
    """Precision, recall, F1 and support per class, plus macro-F1.

    Macro-F1 averages only over classes that occur in the ground truth,
    so the missing design class (1) does not count as a zero.
    """
    cm = confusion(y_true, y_pred, labels)
    report = {}
    f1s = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        precision = _div(tp, cm[:, i].sum())
        recall = _div(tp, cm[i, :].sum())
        if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
            f1 = 0.0 if cm[i, :].sum() else float("nan")
        else:
            f1 = 2 * precision * recall / (precision + recall)
        report[lab] = {
            "name": NAMES.get(lab, str(lab)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
        }
        if cm[i, :].sum() > 0:
            f1s.append(f1)
    report["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    report["confusion"] = cm
    return report


def run_agreement(predictions_by_run, labels=LABELS):
    """How consistent the LLM is across repeated runs on the same rules.

    predictions_by_run: list of runs, each a list of labels (same rule order).

    Returns:
      all_agree      share of rules where every run gave the same label
                     (we use this as the paper's "consistency")
      mean_pairwise  average share of rules on which two runs agree
      fleiss_kappa   agreement corrected for chance
    """
    runs = [list(r) for r in predictions_by_run]
    if len(runs) < 2:
        raise ValueError("need at least 2 runs")
    n_rules = len(runs[0])
    if any(len(r) != n_rules for r in runs):
        raise ValueError("all runs must cover the same rules")
    m = np.array(runs, dtype=int)            # shape (runs, rules)
    n_raters = m.shape[0]

    all_agree = float(np.mean(np.all(m == m[0], axis=0)))
    pairwise = [np.mean(m[a] == m[b]) for a, b in combinations(range(n_raters), 2)]

    # Fleiss' kappa: counts[i, j] = how many runs gave rule i label j
    counts = np.stack([(m == lab).sum(axis=0) for lab in labels], axis=1)
    p_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / (n_rules * n_raters)
    p_e = np.sum(p_j ** 2)
    kappa = 1.0 if np.isclose(p_e, 1.0) else float((p_bar - p_e) / (1 - p_e))

    return {
        "all_agree": all_agree,
        "mean_pairwise": float(np.mean(pairwise)),
        "fleiss_kappa": kappa,
    }


def evaluate_runs(y_true, predictions_by_run):
    """Score every run, then report mean and std like the paper (e.g. 89.08 +/- 1.70)."""
    per_run = []
    for preds in predictions_by_run:
        m = paper_metrics(y_true, preds)
        m["macro_f1"] = per_class_report(y_true, preds)["macro_f1"]
        per_run.append(m)

    summary = {}
    for key in ("accuracy", "far", "mar", "macro_f1"):
        values = np.array([r[key] for r in per_run], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    # Agreement is measured on rules that have a ground-truth label.
    keep = [i for i, t in enumerate(y_true) if t is not None and t != -1]
    kept_runs = [[run[i] for i in keep] for run in predictions_by_run]
    agreement = run_agreement(kept_runs) if len(kept_runs) > 1 else None

    return {"per_run": per_run, "summary": summary, "agreement": agreement}


def format_summary(result):
    """One-line-per-metric text, percentages like the paper's tables."""
    lines = []
    for key, label in (("accuracy", "Accuracy"), ("far", "FAR"),
                       ("mar", "MAR"), ("macro_f1", "Macro-F1")):
        s = result["summary"][key]
        lines.append(f"{label:<9} {100 * s['mean']:6.2f} +/- {100 * s['std']:.2f} %")
    if result["agreement"]:
        a = result["agreement"]
        lines.append(f"{'Consist.':<9} {100 * a['all_agree']:6.2f} %  "
                     f"(pairwise {100 * a['mean_pairwise']:.2f} %, "
                     f"Fleiss kappa {a['fleiss_kappa']:.3f})")
    return "\n".join(lines)
