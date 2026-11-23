"""
Preprocessing for the LBNL simulated chiller plant data (Member A).

Steps
-----
1. Load one CSV (1-minute data, 3 chillers / 3 cooling towers).
2. Frozen-sensor removal: a measured value that does not change for
   >= FROZEN_MINUTES while its equipment is running is treated as stuck.
3. 3-sigma outlier removal: limits (mean +/- 3 std) are computed on the
   FAULT-FREE file only, using running samples only, and then applied to
   every file. (Computing them inside a fault file would delete the fault.)
4. Derived metrics, with the team's agreed names:
      CHL_DT_i, CHL_COOL_i, CHL_COP_i, CHL_CDDT_i, CT_APPROACH_i,
      PLANT_POW, PLANT_COP
5. Save the cleaned table to data/processed/.

Column names follow the LBNL data inventory (e.g. CHL_RW_TEMP_1, CHL_POW_1);
see data/dataset_notes.md. Paper: Section 3.3 step 1 and Eq. 1-5.

Owner: Member A
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Settings (overridden by config.yaml -> preprocess: ..., if present)
# ----------------------------------------------------------------------------
DEFAULTS = {
    "n_chillers": 3,
    "n_towers": 3,
    "frozen_window": 12,       # consecutive identical readings while running -> stuck
                               # (config.yaml preprocess.frozen_window; 1-minute data)
    "frozen_tol": 1e-6,        # "unchanged" means |difference| <= this
    "sigma": 3.0,              # outlier limit in standard deviations
    "max_flag_frac": 0.20,     # if a check would remove more than this share of a
                               # column's running values, skip that check for that
                               # column (it is behaviour, not a sensor fault)
    "min_chiller_kw": 1.0,     # below this the chiller is treated as off for COP
    "min_plant_kw": 1.0,
    "min_cool_kw": 50.0,       # COP only when real cooling is delivered (~10% of a
                               # chiller's ~560 kW peak); at near-zero load COP is noise
    "sigma_by_month": True,    # 3-sigma limits per calendar month, so normal summer
                               # peaks are not treated as outliers
    "outliers_on_fault_files": False,  # 3-sigma step on fault files would delete the
                               # fault signal itself; frozen check still runs
    "frozen_exclude": ["OA_TEMP", "OA_TEMP_WB"],  # hourly weather: flat for 60 min
    "tiny": 1e-9,              # |x| below this is simulation noise (e.g. 3.5e-18) -> 0
    "raw_dir": "data/raw",
    "processed_dir": "data/processed",
    "fault_free_file": "ChillerPlant.csv",
}

BTU_PER_H_PER_KW = 3412.14
GPM_DEGF_TO_BTUH = 500.0       # water: 8.33 lb/gal * 60 min/h * 1 Btu/(lb.F)


def load_settings(config_path: str | Path = "config.yaml") -> dict:
    s = dict(DEFAULTS)
    p = Path(config_path)
    if p.exists():
        try:
            import yaml
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            s.update((cfg.get("preprocess") or {}))
            d = cfg.get("data") or {}
            s["raw_dir"] = d.get("raw_dir", s["raw_dir"])
            s["processed_dir"] = d.get("processed_dir", s["processed_dir"])
        except Exception as e:  # config problems should not stop the notebook
            print(f"[preprocess] could not read {p}: {e}; using defaults")
    return s


def data_dir(default: str = "data/raw") -> Path:
    """DATA_DIR from .env (or the environment); falls back to data.raw_dir."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return Path(os.environ.get("DATA_DIR") or default)


def list_files(folder: Path | None = None) -> list[Path]:
    folder = Path(folder or data_dir())
    # skip macOS zip junk (__MACOSX/, ._ChillerPlant.csv)
    files = sorted(f for f in folder.rglob("*.csv")
                   if "__MACOSX" not in f.parts and not f.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {folder}. Check DATA_DIR in .env")
    return files


def find_fault_free(files: list[Path], name: str = "ChillerPlant.csv") -> Path:
    for f in files:
        if f.name == name:
            return f
    pat = re.compile(r"fault[\s_-]*free|faultfree|normal|baseline", re.I)
    hits = [f for f in files if pat.search(f.name)]
    if not hits:
        raise ValueError("Could not spot the fault-free file by name. Pass it explicitly, "
                         "e.g. FAULT_FREE = files[0]")
    return hits[0]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    """Read one file, set a DatetimeIndex, store numbers as float32."""
    df = pd.read_csv(path, low_memory=False)
    ts_col = next((c for c in df.columns
                   if re.search(r"time|date", c, re.I) or df[c].dtype == object), None)
    if ts_col is not None:
        idx = pd.to_datetime(df[ts_col], errors="coerce")
        if idx.notna().mean() > 0.95:
            df = df.drop(columns=[ts_col])
            df.index = idx
        else:
            ts_col = None
    if ts_col is None:
        # The files are 1-minute data for one year with the first hour removed.
        print(f"[preprocess] {path.name}: no timestamp column found; "
              "building a 1-minute index from Jan 1, 01:00")
        df.index = pd.date_range("2023-01-01 01:00", periods=len(df), freq="min")
    df.index.name = "time"
    num = df.select_dtypes("number").columns
    df[num] = df[num].astype("float32")
    # simulation noise such as 3.5e-18 -> exactly 0
    df[num] = df[num].mask(df[num].abs() < DEFAULTS["tiny"], 0.0)
    return df


# ----------------------------------------------------------------------------
# Which columns are sensors, and when is their equipment running?
# ----------------------------------------------------------------------------
SENSOR_PAT = re.compile(r"(TEMP|FLOW|POW|LOAD|_DP)(_\d+)?$")
NOT_SENSOR_PAT = re.compile(r"SPT|_STA(_\d+)?$|CTRL|SPD")


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if SENSOR_PAT.search(c) and not NOT_SENSOR_PAT.search(c)]


def plant_on(df: pd.DataFrame) -> pd.Series:
    """True where at least one chiller is on."""
    on = pd.Series(False, index=df.index)
    for c in df.columns:
        if re.fullmatch(r"CHL_STA_\d+", c):
            on |= df[c] > 0.5
    return on


def running_mask(df: pd.DataFrame, col: str, _cache={}) -> pd.Series:
    """True where the equipment that owns `col` is running."""
    m = re.search(r"_(\d+)$", col)
    key = (id(df), len(df))
    if key not in _cache:
        _cache.clear()
        _cache[key] = plant_on(df)
    plant = _cache[key]
    if m:
        i = m.group(1)
        if col.startswith("CHL_") and f"CHL_STA_{i}" in df:
            return df[f"CHL_STA_{i}"] > 0.5
        if col.startswith("CT_") and f"CT_STA_{i}" in df:
            return df[f"CT_STA_{i}"] > 0.5
        if col.startswith("CWL_SEC_PM") and f"CWL_SEC_PM_STA_{i}" in df:
            return df[f"CWL_SEC_PM_STA_{i}"] > 0.5
    return plant


def _run_lengths(changed: np.ndarray) -> np.ndarray:
    """Length of the constant run each sample belongs to."""
    grp = np.cumsum(changed)
    counts = np.bincount(grp)
    return counts[grp]


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def frozen_flags(df: pd.DataFrame, cols: list[str], s: dict) -> tuple[pd.DataFrame, dict]:
    flags = pd.DataFrame(False, index=df.index, columns=cols)
    skipped = {}
    for c in cols:
        if c in s.get("frozen_exclude", []):
            continue
        run = running_mask(df, c).to_numpy()
        x = df[c].to_numpy(dtype="float64")
        changed = np.r_[True, np.abs(np.diff(x)) > s["frozen_tol"]] | np.r_[True, run[1:] != run[:-1]]
        stuck = (_run_lengths(changed) >= s["frozen_window"]) & run
        frac = stuck.sum() / max(run.sum(), 1)
        if frac > s["max_flag_frac"]:
            skipped[c] = round(float(frac), 3)
            continue
        flags[c] = stuck
    return flags, skipped


def fit_limits(df_fault_free: pd.DataFrame, s: dict) -> pd.DataFrame:
    """mean +/- sigma*std per sensor (and per calendar month if sigma_by_month),
    from running samples of the fault-free file. Index: (sensor, month); month 0 = all year."""
    rows = []
    for c in sensor_columns(df_fault_free):
        x = df_fault_free.loc[running_mask(df_fault_free, c), c].astype("float64")
        if len(x) < 100:
            continue
        groups = x.groupby(x.index.month) if s.get("sigma_by_month") else [(0, x)]
        for m, xm in groups:
            if len(xm) < 100:
                continue
            mu, sd = xm.mean(), xm.std()
            rows.append({"sensor": c, "month": int(m), "mean": mu, "std": sd,
                         "low": mu - s["sigma"] * sd, "high": mu + s["sigma"] * sd})
    return pd.DataFrame(rows).set_index(["sensor", "month"])


def outlier_flags(df: pd.DataFrame, cols: list[str], limits: pd.DataFrame,
                  s: dict) -> tuple[pd.DataFrame, dict]:
    flags = pd.DataFrame(False, index=df.index, columns=cols)
    skipped = {}
    sensors = set(limits.index.get_level_values(0))
    by_month = (limits.index.get_level_values(1) != 0).any()
    months = df.index.month if by_month else np.zeros(len(df), dtype=int)
    for c in cols:
        if c not in sensors:
            continue
        lim = limits.loc[c]
        lo = pd.Series(months, index=df.index).map(lim["low"])
        hi = pd.Series(months, index=df.index).map(lim["high"])
        sd = pd.Series(months, index=df.index).map(lim["std"])
        run = running_mask(df, c) & lo.notna() & (sd > 0)
        out = run & ((df[c] < lo) | (df[c] > hi))
        frac = out.sum() / max(run.sum(), 1)
        if frac > s["max_flag_frac"]:
            skipped[c] = round(float(frac), 3)
            continue
        flags[c] = out
    return flags, skipped


def clean(df: pd.DataFrame, limits: pd.DataFrame, s: dict,
          fault_free: bool = True) -> tuple[pd.DataFrame, dict]:
    """Drop rows with a stuck or out-of-range sensor value. Returns (clean_df, report).
    On fault files the 3-sigma step runs only if outliers_on_fault_files is True."""
    cols = sensor_columns(df)
    n0 = len(df)
    fz, fz_skip = frozen_flags(df, cols, s)
    bad_fz = fz.any(axis=1)
    df1 = df.loc[~bad_fz]
    if fault_free or s.get("outliers_on_fault_files", False):
        ol, ol_skip = outlier_flags(df1, cols, limits, s)
    else:
        ol, ol_skip = pd.DataFrame(False, index=df1.index, columns=cols), {}
    bad_ol = ol.any(axis=1)
    df2 = df1.loc[~bad_ol]
    report = {
        "rows_raw": n0,
        "rows_after_frozen": len(df1),
        "rows_after_outliers": len(df2),
        "frozen_by_column": fz.sum().loc[lambda x: x > 0].sort_values(ascending=False),
        "outliers_by_column": ol.sum().loc[lambda x: x > 0].sort_values(ascending=False),
        "frozen_check_skipped": fz_skip,
        "outlier_check_skipped": ol_skip,
        "outlier_step_run": bool(fault_free or s.get("outliers_on_fault_files", False)),
    }
    return df2.copy(), report


# ----------------------------------------------------------------------------
# Derived metrics (team-agreed names)
# ----------------------------------------------------------------------------
def add_derived(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    out = df.copy()
    cool_total = pd.Series(0.0, index=out.index)
    for i in range(1, s["n_chillers"] + 1):
        sta = out[f"CHL_STA_{i}"] > 0.5
        dt = out[f"CHL_RW_TEMP_{i}"] - out[f"CHL_SW_TEMP_{i}"]                     # F
        cool = out[f"CHL_CW_FLOW_{i}"] * dt * GPM_DEGF_TO_BTUH / BTU_PER_H_PER_KW  # kW
        pw = out[f"CHL_POW_{i}"]
        # temperature differences only mean something while the chiller runs
        out[f"CHL_DT_{i}"] = dt.where(sta)
        out[f"CHL_COOL_{i}"] = cool.where(sta, 0.0)
        out[f"CHL_COP_{i}"] = (cool / pw).where(sta & (pw > s["min_chiller_kw"])
                                                & (cool >= s["min_cool_kw"]))
        # condenser water heats up through the chiller: leaving - entering
        out[f"CHL_CDDT_{i}"] = (out[f"CHL_SWCD_TEMP_{i}"] - out[f"CHL_RWCD_TEMP_{i}"]).where(sta)
        cool_total += cool.where(sta, 0.0).fillna(0.0)
    for i in range(1, s["n_towers"] + 1):
        ct_on = out[f"CT_STA_{i}"] > 0.5 if f"CT_STA_{i}" in out else True
        out[f"CT_APPROACH_{i}"] = (out[f"CT_SW_TEMP_{i}"] - out["OA_TEMP_WB"]).where(ct_on)
    pow_cols = [c for c in out.columns
                if re.fullmatch(r"(CHL_POW|CT_POW|CDWL_PM_POW|CWL_PRI_PM_POW|CWL_SEC_PM_POW)_\d+", c)]
    out["PLANT_POW"] = out[pow_cols].sum(axis=1)
    any_on = out[[f"CHL_STA_{i}" for i in range(1, s["n_chillers"] + 1)]].max(axis=1) > 0.5
    # dataset_notes.md: plant COP = building load (CWL_SEC_LOAD, W -> kW) / plant power
    load_kw = out["CWL_SEC_LOAD"] / 1000.0 if "CWL_SEC_LOAD" in out else cool_total
    out["PLANT_COP"] = (load_kw / out["PLANT_POW"]).where(any_on & (out["PLANT_POW"] > s["min_plant_kw"])
                                                          & (load_kw >= s["min_cool_kw"]))
    return out


# ----------------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------------
def save(df: pd.DataFrame, name: str, folder: str | Path = "data/processed") -> Path:
    """Parquet if pyarrow is installed, otherwise pickle."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        p = folder / f"{name}.parquet"
        df.to_parquet(p)
    except ImportError:
        p = folder / f"{name}.pkl"
        df.to_pickle(p)
        print("[preprocess] pyarrow not installed: saved as .pkl instead of .parquet")
    return p


def process_file(path: Path, limits: pd.DataFrame, s: dict, out_dir=None):
    out_dir = out_dir or s.get("processed_dir", "data/processed")
    raw = load_csv(path)
    is_ff = Path(path).name == s.get("fault_free_file", "ChillerPlant.csv")
    cleaned, report = clean(raw, limits, s, fault_free=is_ff)
    full = add_derived(cleaned, s)
    saved = save(full, Path(path).stem, out_dir)
    return raw, full, report, saved


# ----------------------------------------------------------------------------
# Names from the original stub (kept so other code can call them)
# ----------------------------------------------------------------------------
def load_raw(path):
    return load_csv(Path(path))


def remove_frozen_sensors(df, window=None):
    s = dict(load_settings())
    if window is not None:
        s["frozen_window"] = window
    flags, _ = frozen_flags(df, sensor_columns(df), s)
    return df.loc[~flags.any(axis=1)]


def remove_outliers(df, sigma=3, limits=None):
    """limits: from fit_limits(fault_free_df). If None, they are fitted on df itself."""
    s = dict(load_settings())
    s["sigma"] = sigma
    limits = fit_limits(df, s) if limits is None else limits
    flags, _ = outlier_flags(df, sensor_columns(df), limits, s)
    return df.loc[~flags.any(axis=1)]


def add_derived_metrics(df):
    return add_derived(df, load_settings())


if __name__ == "__main__":
    s = load_settings()
    files = list_files(data_dir(s["raw_dir"]))
    ff = find_fault_free(files, s["fault_free_file"])
    ff_raw = load_csv(ff)
    limits = fit_limits(ff_raw, s)
    for f in files:
        _, full, rep, saved = process_file(f, limits, s)
        print(f"{f.name}: {rep['rows_raw']} -> {rep['rows_after_outliers']} rows, saved {saved}")
