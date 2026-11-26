"""Improvement step: weed out rules that only hold because of outdoor temperature, and
combine LLM labels from several runs.

Many mined HVAC rules link two variables that both follow the weather (e.g. "high
chiller power => high tower fan speed"). Their pooled lift is well above 1, but inside a
narrow outdoor-temperature band the link disappears. Such rules are flagged as likely
confounded before they reach the LLM.

    stratified_lift(rule, transactions, bins)   lift of a rule inside each outdoor-temperature bin
    flag_confounded(rule, transactions, bins)   lift ~ 1 inside the bins, though not pooled -> confounded
    majority_vote(runs)                          one label per rule from several LLM runs

Rule format is the team's: {"antecedent": ("VAR=interval", ...), "consequent": (...)}.
A transaction is a set of such items (one per 15-min row).
`bins` is either a sequence of bin labels aligned with the transactions, or a variable
name such as "OA_TEMP": the transaction's item for that variable is then its bin.

Settings in config.yaml -> improvement (n_bins, vote_runs).
Run: python src/improvement.py   (toy demo)   |   python tests/test_improvement.py

Owner: Member C
"""
from __future__ import annotations

from collections import Counter
from typing import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd

LIFT_TOL = 0.1       # our choice: |lift - 1| <= 0.1 counts as "no association"
MIN_BIN_COUNT = 5    # our choice: a bin needs >= 5 rows with the antecedent and with the consequent


def _bin_of(transaction: Iterable[str], var: str) -> str | None:
    for item in transaction:
        if item.split("=", 1)[0] == var:
            return item
    return None


def _bin_labels(transactions: Sequence[Iterable[str]], bins) -> list:
    if isinstance(bins, str):
        return [_bin_of(t, bins) for t in transactions]
    bins = list(bins)
    if len(bins) != len(transactions):
        raise ValueError(f"{len(bins)} bin labels for {len(transactions)} transactions")
    return bins


def temperature_bins(oa_temp, n_bins: int = 5) -> list:
    """Equal-frequency bins of outdoor temperature (config: improvement.n_bins)."""
    return list(pd.qcut(pd.Series(oa_temp), q=n_bins, duplicates="drop", precision=1).astype(str))


def _counts(rule: dict, transactions: Sequence[set]) -> dict:
    a, c = set(rule["antecedent"]), set(rule["consequent"])
    n = len(transactions)
    n_a = sum(a <= t for t in transactions)
    n_c = sum(c <= t for t in transactions)
    n_ac = sum((a | c) <= t for t in transactions)
    lift = (n_ac * n) / (n_a * n_c) if n_a and n_c else float("nan")
    return {"n": n, "n_a": n_a, "n_c": n_c, "n_ac": n_ac,
            "support": n_ac / n if n else float("nan"),
            "confidence": n_ac / n_a if n_a else float("nan"), "lift": lift}


def stratified_lift(rule: dict, transactions: Sequence[Iterable[str]], bins,
                    min_count: int = MIN_BIN_COUNT) -> pd.DataFrame:
    """One row per bin: n, n_a, n_c, n_ac, support, confidence, lift, valid.

    `valid` is False when the bin has fewer than `min_count` rows with the antecedent or
    with the consequent: its lift is too noisy to judge (or undefined).
    """
    ts = [set(t) for t in transactions]
    labels = _bin_labels(ts, bins)
    groups: dict = {}
    for t, b in zip(ts, labels):
        groups.setdefault(b, []).append(t)
    rows = []
    for b, members in groups.items():
        c = _counts(rule, members)
        c["valid"] = c["n_a"] >= min_count and c["n_c"] >= min_count
        rows.append({"bin": b, **c})
    return pd.DataFrame(rows).set_index("bin")


def flag_confounded(rule: dict, transactions: Sequence[Iterable[str]], bins,
                    tol: float = LIFT_TOL, min_count: int = MIN_BIN_COUNT) -> dict:
    """Is the rule likely confounded by outdoor temperature?

    Flagged when the pooled lift is away from 1 (|lift - 1| > tol) but the stratified lift,
    the average of the within-bin lifts weighted by how often the antecedent occurs in each
    valid bin, is about 1 (|lift - 1| <= tol). A rule whose pooled lift is already ~1 is not
    flagged: it shows no association to explain.

    Returns {'confounded', 'pooled_lift', 'stratified_lift', 'valid_bins', 'reason'}.
    """
    ts = [set(t) for t in transactions]
    pooled = _counts(rule, ts)["lift"]
    table = stratified_lift(rule, ts, bins, min_count)
    valid = table[table["valid"]]
    out = {"confounded": False, "pooled_lift": pooled, "stratified_lift": float("nan"),
           "valid_bins": len(valid)}
    if np.isnan(pooled) or abs(pooled - 1) <= tol:
        return {**out, "reason": "no pooled association"}
    if valid.empty:
        return {**out, "reason": "no bin has enough data"}
    strat = float(np.average(valid["lift"], weights=valid["n_a"]))
    out["stratified_lift"] = strat
    if abs(strat - 1) <= tol:
        return {**out, "confounded": True, "reason": "lift ~ 1 within outdoor-temperature bins"}
    return {**out, "reason": "association holds within bins"}


def majority_vote(runs: Sequence, tie_break: str = "first") -> pd.DataFrame:
    """Combine labels from several runs into one label per rule.

    `runs` is a list of runs; each run is a sequence of labels (position = rule) or a
    dict {rule_id: label}. Missing labels (None, NaN, or a rule absent from a run) are
    ignored, e.g. when the LLM's answer could not be parsed.

    Ties: tie_break="first" keeps the tied label given by the earliest run (deterministic,
    and run 1 is the paper's baseline setting); tie_break="none" returns None.
    Returns a DataFrame indexed by rule with 'label', 'agreement' (share of valid votes for
    the winner) and 'votes'.
    """
    if tie_break not in ("first", "none"):
        raise ValueError("tie_break must be 'first' or 'none'")
    dicts = [dict(r) if isinstance(r, dict) else dict(enumerate(r)) for r in runs]
    keys: list[Hashable] = []
    for d in dicts:
        keys += [k for k in d if k not in keys]

    def ok(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    rows = []
    for k in keys:
        votes = [d[k] for d in dicts if k in d and ok(d[k])]
        if not votes:
            rows.append({"rule": k, "label": None, "agreement": float("nan"), "votes": 0})
            continue
        counts = Counter(votes)
        top = max(counts.values())
        tied = [v for v in counts if counts[v] == top]
        if len(tied) == 1:
            label = tied[0]
        elif tie_break == "first":
            label = next(v for v in votes if v in tied)
        else:
            label = None
        rows.append({"rule": k, "label": label, "agreement": top / len(votes), "votes": len(votes)})
    return pd.DataFrame(rows).set_index("rule")


# ---------------------------------------------------------------------------
# Toy demo
# ---------------------------------------------------------------------------

def toy_transactions(n: int = 4000, seed: int = 42):
    """Toy plant: outdoor temperature drives chiller power and tower fan speed (confounded
    pair); low delta T really does lower COP inside every temperature band (genuine rule)."""
    rng = np.random.default_rng(seed)
    oa = rng.uniform(50, 95, n)
    hot = (oa - 50) / 45
    pow_hi = rng.random(n) < 0.1 + 0.8 * hot
    fan_hi = rng.random(n) < 0.1 + 0.8 * hot          # independent of power given oa
    dt_lo = rng.random(n) < 0.3
    cop_lo = rng.random(n) < np.where(dt_lo, 0.7, 0.2)
    oa_bin = temperature_bins(oa, 5)
    ts = [{f"OA_TEMP={b}",
           f"CHL_POW_1={'high' if p else 'low'}", f"CT_FAN_SPD_1={'high' if f else 'low'}",
           f"CHL_DT_1={'low' if d else 'ok'}", f"CHL_COP_1={'low' if c else 'ok'}"}
          for b, p, f, d, c in zip(oa_bin, pow_hi, fan_hi, dt_lo, cop_lo)]
    return ts, oa_bin


if __name__ == "__main__":
    ts, oa_bin = toy_transactions()
    rules = {
        "power => fan": {"antecedent": ("CHL_POW_1=high",), "consequent": ("CT_FAN_SPD_1=high",)},
        "low dT => low COP": {"antecedent": ("CHL_DT_1=low",), "consequent": ("CHL_COP_1=low",)},
    }
    for name, r in rules.items():
        print(name)
        print(stratified_lift(r, ts, "OA_TEMP")[["n", "n_a", "lift", "valid"]].round(2).to_string())
        print(flag_confounded(r, ts, "OA_TEMP"), "\n")
    print(majority_vote([[1, 0, 2], [1, 3, 2], [0, 0, None]]))
