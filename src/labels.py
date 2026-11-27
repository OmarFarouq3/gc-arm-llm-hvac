"""Ground-truth labels for mined rules (report/labelling_scheme.md).

Every rule mined from a fault file F is compared with the fault-free transactions T0
(discretised with the same breakpoints):

    supp_0, conf_0 = support and confidence of the rule on T0
    delta          = |conf_F - conf_0|            (None if the antecedent never occurs in T0)
    A(F)           = root variables of F plus every node within `hops` hops in the graph

    new              supp_0 < new_support and (conf_0 is None or delta >= stable_delta)
    normal (0)       supp_0 >= new_support and delta < stable_delta
    anomaly (cat F)  (new or delta >= changed_delta) and the rule uses A(F)
    ambiguous (None) everything else (excluded from the evaluation), including a rare
                     rule whose confidence is unchanged (supp_0 < new_support, delta < stable_delta)

Thresholds come from config.yaml -> labelling.
The LLM never sees supp_0, conf_0, delta, the fault name or the affected set.

Owner: Member D (written by Member A)
"""
from __future__ import annotations

import json
from pathlib import Path

try:                                  # works both as `src.labels` and from inside src/
    from src.graph import build_hvac_graph, k_hop_neighbourhood
except ImportError:                   # pragma: no cover
    from graph import build_hvac_graph, k_hop_neighbourhood


# ---------------------------------------------------------------------------
# Fault table (labelling_scheme.md, section 3)
# ---------------------------------------------------------------------------
# fault type -> (category, root variables)
FAULTS = {
    "chiller_bias": (2, ("CHL_SW_TEMP_1",)),
    "coolingtower_bias": (2, ("CT_SW_TEMP_1",)),
    "secondary_chilled_water_pressure_bias": (2, ("CWL_SEC_DP",)),
    "bypass_leakage": (2, ("TWV_CTRL",)),
    "bypass_stuck": (2, ("TWV_CTRL",)),
    "coolingtower_PI": (2, ("CT_FAN_SPD_CTRL_1", "CT_FAN_SPD_CTRL_2", "CT_FAN_SPD_CTRL_3")),
    "coolingtower_fouling": (3, ("CT_SW_TEMP_1", "CT_APPROACH_1")),
    # chiller_fouling_065 fouls chiller 1: median CHL_COP_1 2.48 -> 1.48 vs the fault-free
    # year, while CHL_COP_2 (2.30 -> 2.12) and CHL_COP_3 (2.05 -> 2.27) barely move
    "chiller_fouling": (3, ("CHL_COP_1", "CHL_CDDT_1")),
}

# fault type -> file used for labelling (highest severity), names as in data/processed
FAULT_FILES = {
    "chiller_bias": "ChillerPlant_chiller_bias_2",
    "coolingtower_bias": "ChillerPlant_coolingtower_bias_2",
    "secondary_chilled_water_pressure_bias": "ChillerPlant_secondary_chilled_water_pressure_bias_020",
    "bypass_leakage": "ChillerPlant_bypass_leakage_075",
    "bypass_stuck": "ChillerPlant_bypass_stuck_075",
    "coolingtower_PI": "ChillerPlant_coolingtower_PI",
    "coolingtower_fouling": "ChillerPlant_coolingtower_fouling_065",
    "chiller_fouling": "ChillerPlant_chiller_fouling_065",   # undocumented: report with and without
}

DEFAULTS = {"new_support": 0.02, "stable_delta": 0.10, "changed_delta": 0.20, "hops": 2}
_EPS = 1e-12      # so that a delta of exactly 0.10 / 0.20 is not lost to float rounding


def load_settings(config_path: str | Path = "config.yaml") -> dict:
    """Thresholds from config.yaml -> labelling (defaults if missing)."""
    s = dict(DEFAULTS)
    p = Path(config_path)
    if p.exists():
        import yaml
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        s.update(cfg.get("labelling") or {})
    return s


def fault_from_filename(name: str) -> str:
    """'ChillerPlant_coolingtower_fouling_065.parquet' -> 'coolingtower_fouling'."""
    stem = Path(str(name)).stem
    for fault in sorted(FAULTS, key=len, reverse=True):   # longest match first
        if fault in stem:
            return fault
    raise KeyError(f"No fault type in file name: {name}")


# ---------------------------------------------------------------------------
# Rule helpers
# ---------------------------------------------------------------------------
def _items(x) -> tuple:
    """Sorted tuple of items from a tuple, a JSON list, a set or a single string.
    'A=x AND B=y' strings (as in the LLM test CSVs) are split too."""
    if x is None:
        return ()
    if isinstance(x, str):
        x = [part.strip() for part in x.split(" AND ")] if x.strip() else []
    return tuple(sorted(x))


def normalise_rule(rule: dict) -> dict:
    """Copy of `rule` with antecedent and consequent as sorted tuples
    (rules saved to JSON come back with lists)."""
    out = dict(rule)
    out["antecedent"] = _items(rule.get("antecedent"))
    out["consequent"] = _items(rule.get("consequent"))
    return out


def load_rules(path: str | Path) -> list[dict]:
    """Read rules saved as JSON and turn the lists back into sorted tuples."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rules" in data:
        data = data["rules"]
    return [normalise_rule(r) for r in data]


def rule_variables(rule: dict) -> set[str]:
    """Variable names used by a rule ('CT_SW_TEMP_1=high' -> 'CT_SW_TEMP_1')."""
    items = _items(rule.get("antecedent")) + _items(rule.get("consequent"))
    return {item.split("=", 1)[0] for item in items}


# ---------------------------------------------------------------------------
# Affected set
# ---------------------------------------------------------------------------
def affected_set(fault: str, G=None, k: int | None = None) -> set[str]:
    """Root variables of `fault` plus every node within k hops of them."""
    if fault not in FAULTS:
        raise KeyError(f"Unknown fault type: {fault}. Known: {sorted(FAULTS)}")
    G = build_hvac_graph() if G is None else G
    k = load_settings()["hops"] if k is None else int(k)
    _, roots = FAULTS[fault]
    affected = set(roots)
    for root in roots:
        if root in G:
            affected |= k_hop_neighbourhood(G, root, k)
    return affected


# ---------------------------------------------------------------------------
# Support / confidence on given transactions
# ---------------------------------------------------------------------------
def rule_stats(rule: dict, transactions) -> dict:
    """Support and confidence of `rule` on `transactions` (iterables of items).
    Confidence is None if the antecedent never occurs."""
    ante = frozenset(_items(rule.get("antecedent")))
    both = ante | frozenset(_items(rule.get("consequent")))
    n = n_ante = n_both = 0
    for t in transactions:
        t = t if isinstance(t, (set, frozenset)) else frozenset(t)
        n += 1
        if ante <= t:
            n_ante += 1
            if both <= t:
                n_both += 1
    support = n_both / n if n else 0.0
    confidence = n_both / n_ante if n_ante else None
    return {"support": support, "confidence": confidence}


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
def label_rule(rule: dict, transactions_0, affected: set[str], category: int,
               settings: dict | None = None) -> dict:
    """Return a copy of `rule` with supp_0, conf_0, delta, status and label added.

    status: 'normal' (label 0), 'anomaly' (label = category) or 'ambiguous' (label None).
    `rule["confidence"]` is the confidence in the fault file (from mining).
    """
    s = settings or load_settings()
    out = normalise_rule(rule)
    stats = rule_stats(out, transactions_0)
    supp_0, conf_0 = stats["support"], stats["confidence"]
    delta = None if conf_0 is None else abs(float(out["confidence"]) - conf_0)

    low_support = supp_0 < s["new_support"] - _EPS
    stable = delta is not None and delta < s["stable_delta"] - _EPS
    # "new" = rare without the fault AND not just as reliable there; a rare rule whose
    # confidence is unchanged is not evidence of the fault -> ambiguous
    is_new = low_support and not stable
    changed = is_new or (delta is not None and delta >= s["changed_delta"] - _EPS)
    uses_affected = bool(rule_variables(out) & set(affected))

    if not low_support and stable:
        status, label = "normal", 0
    elif changed and uses_affected:
        status, label = "anomaly", int(category)
    else:
        status, label = "ambiguous", None

    out.update(supp_0=supp_0, conf_0=conf_0, delta=delta, status=status, label=label)
    return out


def label_rules(rules, transactions_0, fault: str, G=None,
                settings: dict | None = None) -> tuple[list[dict], dict]:
    """Label every rule mined from the file of `fault`.

    Returns (labelled rules, counts) where counts has keys normal / anomaly / ambiguous.
    """
    s = settings or load_settings()
    category, _ = FAULTS[fault]
    affected = affected_set(fault, G, s["hops"])
    t0 = [t if isinstance(t, frozenset) else frozenset(t) for t in transactions_0]
    labelled = [label_rule(r, t0, affected, category, s) for r in rules]
    counts = {"normal": 0, "anomaly": 0, "ambiguous": 0}
    for r in labelled:
        counts[r["status"]] += 1
    return labelled, counts
