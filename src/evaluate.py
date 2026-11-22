"""Metrics.

Paper: Eq. 22-25 (accuracy, FAR = FP/(TP+FP), MAR = FN/(TP+FN), consistency).
Our additions: macro-F1, per-class precision/recall, run-to-run agreement.

Owner: Member D
"""


def paper_metrics(y_true, y_pred):
    raise NotImplementedError


def per_class_report(y_true, y_pred):
    raise NotImplementedError


def run_agreement(predictions_by_run):
    raise NotImplementedError
