"""
Discretisation: continuous readings -> interval items such as "CHL_COP_1=[4,6)".

Paper: Section 3.3, step 2 and Eq. 7 (Deng et al., 2026).

1. Design breakpoints: cut points from engineering knowledge (set-points, limits,
   normal ranges). See DESIGN_BREAKPOINTS below and data/dataset_notes.md.
2. Entropy refinement: inside every design interval, a split s is accepted when
       p_L * H_L + p_R * H_R < (1 - eta) * H_parent,       eta = 1.5 %   (paper)
   and the refinement is applied recursively to the two halves.

Choices the paper leaves open (document them in the report):
- Eq. 7 is the classic SUPERVISED entropy split, so it needs a target class. We use an
  efficiency band, not a fault label: plant COP cut into low / medium / high (tertiles of
  the fault-free year). Splits are therefore kept only where they help explain efficiency.
  (Measuring entropy on the values themselves was tested and rejected: halving any range
  always cuts entropy by far more than 1.5 %, so the threshold never stops a split.)
- Candidate split points: the `n_grid` quantiles of the variable.
- Stopping rule: every interval keeps >= `min_frac` of the samples, and a variable gets at
  most `max_bins` intervals.
- Variables with no design breakpoint and no accepted entropy split fall back to tertiles
  (low / medium / high), so every variable still yields informative items.

Intervals are fitted on the FAULT-FREE file only and then applied to every file,
so an item means the same thing in all files (needed for the ground-truth comparison).

Owner: Member C (implemented with the team lead)
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings (overridden by config.yaml -> discretise: ...)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "eta": 0.015,      # paper: minimum relative entropy reduction for a split
    "min_frac": 0.05,  # our choice: each interval keeps >= 5 % of the samples
    "max_bins": 6,     # our choice: at most 6 intervals per variable
    "n_grid": 40,      # our choice: candidate split points = 40 quantiles
    "target": "PLANT_COP",   # our choice: efficiency band used as the entropy target
    "target_bins": 3,        # low / medium / high (tertiles of the fault-free year)
    "decimals": 1,     # how breakpoints are rounded in item names
}

# Design breakpoints (units as in the LBNL data: degF, kW, 0-1).
# Patterns are regular expressions matched against column names.
DESIGN_BREAKPOINTS = {
    r"^(CHL_SW_TEMP_\d|CWL_PRI_SW_TEMP|CWL_SEC_SW_TEMP)$": [42.0, 52.0],  # set-point reset range
    r"^OA_TEMP$": [60.0, 80.0],                     # outdoor range that drives the reset
    r"^(CT_SW_TEMP_\d|CDWL_SW_TEMP)$": [60.0],      # minimum tower leaving water temperature
    r"^CT_APPROACH_\d$": [8.0],                     # design approach (wet-bulb + 8 degF)
    r"^CHL_DT_\d$": [3.6],                          # paper's "delta T < 2 degC" rule (= 3.6 degF)
    r"^CHL_COP_\d$": [4.0, 6.0],                    # paper's normal COP range
}

# Columns that are on/off (0/1) rather than continuous.
BINARY_PATTERN = re.compile(r"_STA(_\d+)?$")


def load_settings(config_path="config.yaml") -> dict:
    s = dict(DEFAULTS)
    try:
        import yaml
        from pathlib import Path
        p = Path(config_path)
        if p.exists():
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            s.update(cfg.get("discretise") or {})
    except Exception as e:  # a config problem should not stop the pipeline
        print(f"[discretise] could not read {config_path}: {e}; using defaults")
    return s


def design_breakpoints(column: str) -> list[float]:
    for pattern, bps in DESIGN_BREAKPOINTS.items():
        if re.match(pattern, column):
            return list(bps)
    return []


# ---------------------------------------------------------------------------
# Entropy refinement (supervised by an efficiency band)
# ---------------------------------------------------------------------------
def _entropy(labels: np.ndarray) -> float:
    """Shannon entropy (bits) of a set of class labels."""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def efficiency_band(df: pd.DataFrame, s: dict) -> tuple[pd.Series, list[float]]:
    """Target classes 0..target_bins-1 from the target column (tertiles by default)."""
    t = df[s["target"]]
    qs = np.linspace(0, 1, s["target_bins"] + 1)[1:-1]
    edges = sorted(set(np.nanquantile(t.dropna(), qs).round(6)))
    band = pd.Series(np.digitize(t, edges), index=df.index).where(t.notna())
    return band, edges


def refine(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float], s: dict,
           n_total: int) -> list[float]:
    """Entropy-refined breakpoints strictly inside one design interval (paper Eq. 7).
    x: variable values, y: target class of the same rows (no NaN)."""
    lo_b, hi_b = bounds
    inside = (x >= lo_b) & (x < hi_b)
    min_n = max(1, int(math.ceil(s["min_frac"] * n_total)))
    found = []

    def recurse(xv, yv):
        if len(xv) < 2 * min_n:
            return
        h_parent = _entropy(yv)
        if h_parent == 0:
            return
        cands = np.unique(np.quantile(xv, np.linspace(0, 1, s["n_grid"] + 1)[1:-1]))
        best = None
        for c in cands:
            left = xv < c
            nl = left.sum(); nr = len(xv) - nl
            if nl < min_n or nr < min_n:
                continue
            h_cond = (nl * _entropy(yv[left]) + nr * _entropy(yv[~left])) / len(xv)
            if best is None or h_cond < best[1]:
                best = (float(c), h_cond)
        if best and best[1] < (1 - s["eta"]) * h_parent:          # paper Eq. 7
            c = best[0]
            found.append(c)
            recurse(xv[xv < c], yv[xv < c])
            recurse(xv[xv >= c], yv[xv >= c])

    recurse(x[inside], y[inside])
    return sorted(found)


def _limit_bins(breaks: list[float], design: list[float], values: np.ndarray,
                max_bins: int) -> list[float]:
    """Keep at most max_bins - 1 inner breakpoints. Design breakpoints are always kept;
    entropy breakpoints that split off the smallest share of samples are dropped first."""
    breaks = sorted(set(breaks))
    while len(breaks) > max_bins - 1:
        removable = [b for b in breaks if b not in design]
        if not removable:
            break
        def share(b):  # samples in the smaller neighbouring interval if b is removed
            i = breaks.index(b)
            lo = breaks[i - 1] if i > 0 else -np.inf
            hi = breaks[i + 1] if i + 1 < len(breaks) else np.inf
            left = ((values >= lo) & (values < b)).mean()
            right = ((values >= b) & (values < hi)).mean()
            return min(left, right)
        breaks.remove(min(removable, key=share))
    return breaks


# ---------------------------------------------------------------------------
# Fitting and applying
# ---------------------------------------------------------------------------
def fit_breakpoints(df: pd.DataFrame, columns: list[str] | None = None,
                    s: dict | None = None) -> dict:
    """Learn breakpoints on the FAULT-FREE data. Returns {column: spec}, where spec is
    {"type": "binary"} or {"type": "interval", "breaks": [...], "design": [...], "method": ...}."""
    s = s or load_settings()
    columns = columns or list(df.select_dtypes("number").columns)
    has_target = s.get("target") in df
    band = efficiency_band(df, s)[0] if has_target else None
    specs = {}
    for c in columns:
        col = df[c]
        ok = col.notna().to_numpy()
        x = col.to_numpy(dtype="float64")[ok]
        if len(x) == 0:
            continue                                   # e.g. a chiller that never runs
        if BINARY_PATTERN.search(c):
            specs[c] = {"type": "binary"}
            continue
        if x.max() - x.min() < 1e-9:
            continue                                   # constant column: no information
        design = [b for b in design_breakpoints(c) if x.min() < b < x.max()]
        breaks, method = list(design), "design" if design else ""
        # entropy refinement against the efficiency band (not for the target itself)
        if has_target and c != s["target"]:
            both = ok & band.notna().to_numpy()
            xb = col.to_numpy(dtype="float64")[both]
            yb = band.to_numpy()[both].astype(int)
            edges = [-np.inf] + design + [np.inf]
            ent = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                ent += refine(xb, yb, (lo, hi), s, len(xb))
            if ent:
                breaks += ent
                method = (method + " + entropy").strip(" +")
        if not breaks:                                 # fallback: tertiles
            breaks = list(np.unique(np.quantile(x, [1 / 3, 2 / 3])))
            method = "tertiles"
        breaks = _limit_bins(breaks, design, x, s["max_bins"])
        specs[c] = {"type": "interval",
                    "breaks": sorted(set(round(b, s["decimals"]) for b in breaks)),
                    "design": design, "method": method}
    return specs


def _label(lo: float, hi: float) -> str:
    def f(v):
        return f"{v:g}"
    if lo == -np.inf:
        return f"<{f(hi)}"
    if hi == np.inf:
        return f">={f(lo)}"
    return f"[{f(lo)},{f(hi)})"


def discretise(df: pd.DataFrame, specs: dict) -> pd.DataFrame:
    """Map every value to its interval label. Missing values stay missing (no item)."""
    out = {}
    for c, spec in specs.items():
        if c not in df:
            continue
        x = df[c]
        if spec["type"] == "binary":
            out[c] = x.map(lambda v: np.nan if pd.isna(v) else ("on" if v > 0.5 else "off"))
            continue
        edges = [-np.inf] + list(spec["breaks"]) + [np.inf]
        labels = [_label(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
        out[c] = pd.cut(x, bins=edges, labels=labels, right=False).astype(object)
    return pd.DataFrame(out, index=df.index)


def to_transactions(df_discrete: pd.DataFrame) -> list[set[str]]:
    """One transaction per row: {"VARIABLE=interval", ...}. Missing values are skipped."""
    cols = list(df_discrete.columns)
    rows = df_discrete.to_numpy(dtype=object)
    return [{f"{c}={v}" for c, v in zip(cols, row) if isinstance(v, str)} for row in rows]


def describe(specs: dict, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Readable summary: breakpoints per variable, which came from design limits,
    and (if df is given) the share of samples in each interval."""
    rows = []
    for c, spec in specs.items():
        if spec["type"] == "binary":
            rows.append({"variable": c, "method": "binary", "breakpoints": "on / off", "shares": ""})
            continue
        shares = ""
        if df is not None and c in df:
            d = discretise(df[[c]], {c: spec})[c].value_counts(normalize=True, sort=False)
            shares = ", ".join(f"{k}: {v:.0%}" for k, v in d.items())
        rows.append({"variable": c, "method": spec.get("method", ""),
                     "breakpoints": spec["breaks"], "shares": shares})
    return pd.DataFrame(rows)
