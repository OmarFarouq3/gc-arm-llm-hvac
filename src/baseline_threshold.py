"""Threshold baseline: the "conventional engineering method" of the paper's Table 3.

Paper (Deng et al., 2026, Section 4.4): a threshold method of the kind used in BMS
"classifies a rule as anomalous if its antecedent or consequent falls into predefined
extreme intervals (e.g. protection limits or alarm thresholds from design manuals)".
It only says normal / anomalous, so it is not diagnostic (no category).

The COP and approach limits follow this plant's measured operating ranges, not textbook
values (textbook COP 4-6 would flag almost every row of a plant that normally runs at ~2.5).
Four fixed limits (values in config.yaml -> baseline.limits, units as in the LBNL data):
    CHL_COP_i      chiller COP               low  if < 2.0       (this plant: normal COP ~2.5)
    CT_APPROACH_i  tower approach (degF)     high if > 9.8       (this plant: approach >= 9.8 in 7% of healthy tower-1 windows, 31% with 65% fouling)
    CHL_DT_i       chilled-water delta T     low  if < 3.6       (paper Case 2: delta T < 2 degC)
    CHL_SPT_DEV_i  supply temp - set-point   |dev| > 2.0         (knowledge base: normal within ~1 degF)

The same limits are used at two levels:
1. Rows  (`flag_rows`, `evaluate_rows`): flag every time step where a running unit breaks
   a limit. Fault-free file -> false-alarm rate; each fault file -> detection rate.
   This is how a BMS alarm would behave.
2. Rules (`classify_rule`, `classify_rules`): a mined rule is anomalous (1) if any of its
   items is an interval that lies entirely inside an alarm region, e.g. "CHL_COP_1=<4" or
   "CT_APPROACH_1=[10.5,14)". This is the paper's Table 3 setting and is scored against the
   same ground-truth labels as the LLM (`detection_metrics`).

Run: python -m src.baseline_threshold   (needs data/processed/*.parquet or .pkl)

Owner: Member C
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "min_consecutive": 1,   # our choice: rows in a row a limit must be broken before a row is flagged
    "limits": [
        {"name": "low_cop",       "pattern": r"^CHL_COP_\d$",     "low": 2.0},
        {"name": "high_approach", "pattern": r"^CT_APPROACH_\d$", "high": 9.8},
        {"name": "low_delta_t",   "pattern": r"^CHL_DT_\d$",      "low": 3.6},
        {"name": "setpoint_dev",  "pattern": r"^CHL_SPT_DEV_\d$", "abs": 2.0},
    ],
}

# Fault files used for evaluation (report/labelling_scheme.md, section 3).
FAULT_FILES = [
    "ChillerPlant_chiller_bias_+2",
    "ChillerPlant_coolingtower_bias_+2",
    "ChillerPlant_secondary_chilled_water_pressure_bias_+20",
    "ChillerPlant_bypass_leakage_75",
    "ChillerPlant_bypass_stuck_75",
    "ChillerPlant_coolingtower_PI",
    "ChillerPlant_coolingtower_fouling_065",
    "ChillerPlant_chiller_fouling_065",
]

TONS_TO_KW = 3.517


def load_settings(config_path=ROOT / "config.yaml") -> dict:
    s = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    p = Path(config_path)
    if p.exists():
        import yaml
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        s.update(cfg.get("baseline") or {})
    return s


# ---------------------------------------------------------------------------
# Derived metrics (only the ones the baseline needs; skipped if already present)
# ---------------------------------------------------------------------------

def add_baseline_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add CHL_DT_i, CHL_COP_i, CT_APPROACH_i and CHL_SPT_DEV_i (i = 1..3) where possible."""
    df = df.copy()
    for i in (1, 2, 3):
        sw, rw, flow, pw = (f"CHL_SW_TEMP_{i}", f"CHL_RW_TEMP_{i}", f"CHL_CW_FLOW_{i}", f"CHL_POW_{i}")
        if f"CHL_DT_{i}" not in df and {sw, rw} <= set(df.columns):
            df[f"CHL_DT_{i}"] = df[rw] - df[sw]
        if f"CHL_COP_{i}" not in df and {sw, rw, flow, pw} <= set(df.columns):
            cooling_kw = df[flow] * (df[rw] - df[sw]) / 24.0 * TONS_TO_KW
            df[f"CHL_COP_{i}"] = cooling_kw / df[pw].where(df[pw] > 1.0)   # avoid dividing by ~0 kW
        if f"CT_APPROACH_{i}" not in df and {f"CT_SW_TEMP_{i}", "OA_TEMP_WB"} <= set(df.columns):
            df[f"CT_APPROACH_{i}"] = df[f"CT_SW_TEMP_{i}"] - df["OA_TEMP_WB"]
        if f"CHL_SPT_DEV_{i}" not in df and {sw, "CWL_PRI_SW_TEMPSPT"} <= set(df.columns):
            df[f"CHL_SPT_DEV_{i}"] = df[sw] - df["CWL_PRI_SW_TEMPSPT"]
    return df


def _running(df: pd.DataFrame, column: str) -> pd.Series:
    """True where the unit behind `column` is on (CHL_* -> CHL_STA_i, CT_* -> CT_STA_i)."""
    m = re.search(r"_(\d)$", column)
    if m:
        prefix = "CT" if column.startswith("CT_") else "CHL"
        sta = f"{prefix}_STA_{m.group(1)}"
        if sta in df:
            return df[sta] > 0.5
    return pd.Series(True, index=df.index)


def _breaks(x: pd.Series, limit: dict) -> pd.Series:
    out = pd.Series(False, index=x.index)
    if "low" in limit:
        out |= x < limit["low"]
    if "high" in limit:
        out |= x > limit["high"]
    if "abs" in limit:
        out |= x.abs() > limit["abs"]
    return out & x.notna()


def _persistent(flag: pd.Series, n: int) -> pd.Series:
    """Keep only runs of at least n consecutive True values."""
    if n <= 1:
        return flag
    run_id = (flag != flag.shift()).cumsum()
    run_len = flag.groupby(run_id).transform("size")
    return flag & (run_len >= n)


def check_limits(fault_free: pd.DataFrame, s: dict | None = None, q: float = 0.01) -> pd.DataFrame:
    """Compare each limit with the fault-free file's own range (running units only).

    Gives the median, the q / 1-q quantiles and the share of healthy rows beyond the limit,
    so the fixed limits can be checked against the plant's real data before Table 3.
    """
    s = s or load_settings()
    df = add_baseline_metrics(fault_free)
    rows = []
    for limit in s["limits"]:
        cols = [c for c in df.columns if re.match(limit["pattern"], c)]
        vals = pd.concat([df[c][_running(df, c)] for c in cols]) if cols else pd.Series(dtype=float)
        vals = vals.dropna()
        if vals.empty:
            rows.append({"limit": limit["name"], "n": 0})
            continue
        broken = _breaks(vals, limit)
        rows.append({"limit": limit["name"], "n": len(vals), "median": round(vals.median(), 2),
                     f"q{q:g}": round(vals.quantile(q), 2), f"q{1 - q:g}": round(vals.quantile(1 - q), 2),
                     "threshold": {k: limit[k] for k in ("low", "high", "abs") if k in limit},
                     "healthy beyond (%)": round(100 * broken.mean(), 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Row level
# ---------------------------------------------------------------------------

def flag_rows(df: pd.DataFrame, s: dict | None = None) -> pd.DataFrame:
    """One boolean column per limit (any unit) plus `any`. Only running units are checked."""
    s = s or load_settings()
    df = add_baseline_metrics(df)
    out = pd.DataFrame(index=df.index)
    for limit in s["limits"]:
        cols = [c for c in df.columns if re.match(limit["pattern"], c)]
        flag = pd.Series(False, index=df.index)
        for c in cols:
            flag |= _breaks(df[c], limit) & _running(df, c)
        out[limit["name"]] = _persistent(flag, int(s.get("min_consecutive", 1)))
    out["any"] = out.any(axis=1)
    return out


def evaluate_rows(fault_free: pd.DataFrame, faults: dict[str, pd.DataFrame], s: dict | None = None) -> pd.DataFrame:
    """Flag rate per file and per limit.

    Every fault in the LBNL data is active for the whole year, so in a fault file every
    row is "faulty": its flag rate is the detection rate. In the fault-free file every
    flag is a false alarm.
    """
    s = s or load_settings()
    rows = []
    for name, df in {"fault-free": fault_free, **faults}.items():
        f = flag_rows(df, s)
        r = {"file": name, "rows": len(f)}
        r.update({f"{c} (%)": round(100 * f[c].mean(), 2) for c in f.columns})
        rows.append(r)
    table = pd.DataFrame(rows).rename(columns={"any (%)": "flagged (%)"})
    ff_rate = table.loc[0, "flagged (%)"]
    table["minus fault-free (pp)"] = (table["flagged (%)"] - ff_rate).round(2)
    return table


# ---------------------------------------------------------------------------
# 2. Rule level (paper Table 3)
# ---------------------------------------------------------------------------

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def parse_interval(value: str) -> tuple[float, float] | None:
    """'<4' -> (-inf, 4); '>=10' -> (10, inf); '[3.6,8)' -> (3.6, 8). Other values -> None."""
    v = value.strip().replace(" ", "")
    if m := re.fullmatch(rf"<({_NUM})", v):
        return -math.inf, float(m.group(1))
    if m := re.fullmatch(rf">=?({_NUM})", v):
        return float(m.group(1)), math.inf
    if m := re.fullmatch(rf"[\[(]({_NUM}),({_NUM})[\])]", v):
        return float(m.group(1)), float(m.group(2))
    return None


def item_alarm(item: str, s: dict | None = None) -> str | None:
    """Name of the limit an item lies entirely beyond, or None."""
    s = s or load_settings()
    if "=" not in item:
        return None
    var, value = item.split("=", 1)
    iv = parse_interval(value)
    if iv is None:
        return None
    lo, hi = iv
    for limit in s["limits"]:
        if not re.match(limit["pattern"], var):
            continue
        if "low" in limit and hi <= limit["low"]:
            return limit["name"]
        if "high" in limit and lo >= limit["high"]:
            return limit["name"]
        if "abs" in limit and (hi <= -limit["abs"] or lo >= limit["abs"]):
            return limit["name"]
    return None


def classify_rule(rule: dict, s: dict | None = None) -> dict:
    """{'anomaly': 0/1, 'alarms': 'low_cop; ...'} for a rule in the team's rule format."""
    items = list(rule.get("antecedent", ())) + list(rule.get("consequent", ()))
    alarms = sorted({a for a in (item_alarm(i, s) for i in items) if a})
    return {"anomaly": int(bool(alarms)), "alarms": "; ".join(alarms)}


def classify_rules(rules: list[dict], s: dict | None = None) -> pd.DataFrame:
    s = s or load_settings()
    rows = []
    for k, r in enumerate(rules):
        c = classify_rule(r, s)
        rows.append({"rule_id": k, "antecedent": " & ".join(r["antecedent"]),
                     "consequent": " & ".join(r["consequent"]), **c})
    return pd.DataFrame(rows)


def detection_metrics(y_true, y_pred) -> dict:
    """Binary detection accuracy, FAR and MAR (paper Eq. 23-24) for Table 3.

    Works for the baseline (0/1) and for LLM labels (0-3): anything non-zero is "anomaly".
    Ground-truth values None or -1 (ambiguous) are dropped.
    """
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and t != -1]
    yt = np.array([int(t) != 0 for t, _ in pairs])
    yp = np.array([int(p) != 0 for _, p in pairs])
    tp, fp, fn = int(np.sum(yt & yp)), int(np.sum(~yt & yp)), int(np.sum(yt & ~yp))
    n = len(pairs)

    def div(a, b):
        return a / b if b else float("nan")

    return {"n": n, "detection_accuracy": div(int(np.sum(yt == yp)), n),
            "far": div(fp, tp + fp), "mar": div(fn, tp + fn), "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

def _load_processed(stem: str) -> pd.DataFrame | None:
    for ext, reader in [(".parquet", pd.read_parquet), (".pkl", pd.read_pickle)]:
        p = ROOT / "data" / "processed" / f"{stem}{ext}"
        if p.exists():
            return reader(p)
    return None


def to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Same resampling as notebooks/04_discretise.ipynb: 15-min means, status = on if on most of the time."""
    if isinstance(df.index, pd.DatetimeIndex) and df.index.to_series().diff().median() < pd.Timedelta("15min"):
        df = df.resample("15min").mean()
        sta = [c for c in df.columns if re.search(r"_STA_\d$", c)]
        df[sta] = (df[sta] > 0.5).astype(float)
    return df


def main():
    s = load_settings()
    ff = _load_processed("ChillerPlant")
    if ff is None:
        print("data/processed/ChillerPlant.parquet (or .pkl) not found - run the preprocessing notebook first")
        return
    faults = {}
    for stem in FAULT_FILES:
        df = _load_processed(stem)
        if df is None:
            print(f"missing {stem}, skipped")
            continue
        faults[stem.replace("ChillerPlant_", "")] = to_15min(df)
    print(check_limits(to_15min(ff), s).to_string(index=False), "\n")
    table = evaluate_rows(to_15min(ff), faults, s)
    out = ROOT / "results" / "tables" / "baseline_threshold_rows.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
