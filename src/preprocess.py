"""
Preprocessing for the LBNL simulated chiller plant data (Member A).

Every file (fault-free and the 23 fault files) goes through exactly the same steps,
so that the ground-truth comparison is not biased:

1. Load one CSV (1-minute data, 3 chillers / 3 cooling towers).
2. Frozen-sensor removal (preprocess.frozen_removal): a measured value that does not
   change for >= frozen_window minutes while its equipment is running is treated as
   stuck. Hourly weather inputs (OA_TEMP, OA_TEMP_WB) are excluded.
3. 3-sigma outlier removal (preprocess.outlier_removal, default OFF): the data are
   simulated, so there are no sensor errors to remove, and on fault files the test
   removes the fault signal itself. Kept as an option for the report.
4. Resample to 15-minute means (preprocess.resample). Status columns become 1 if the
   equipment was on for at least half of the window. Windows with too few valid
   minutes (preprocess.min_coverage) are dropped.
5. Keep only rows where at least one chiller is on.
6. Derived metrics, with the team's agreed names:
      CHL_DT_i, CHL_COOL_i, CHL_COP_i, CHL_CDDT_i, CT_APPROACH_i,
      PLANT_POW, PLANT_COP
7. Save the cleaned table to data/processed/<file>.parquet.

Column names follow the LBNL data inventory (see data/dataset_notes.md).
Paper: Section 3.3 step 1 and Eq. 1-5.

Owner: Member A
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Defaults; every key can be overridden in config.yaml under `preprocess:`
DEFAULTS = {
    "n_chillers": 3,
    "n_towers": 3,
    "frozen_removal": False,
    "frozen_window": 60,          # minutes unchanged while running -> stuck
    "frozen_tol": 1e-6,
    "frozen_exclude": ["OA_TEMP", "OA_TEMP_WB"],
    "outlier_removal": False,     # 3-sigma; applied to ALL files if switched on
    "sigma": 3.0,
    "sigma_by_month": True,
    "max_flag_frac": 0.20,        # skip a check for a column if it would flag more than this
    "resample": "15min",
    "min_coverage": 0.67,         # share of valid 1-minute rows needed in a window
    "chiller_on_only": True,
    "min_chiller_kw": 1.0,
    "min_cool_kw": 50.0,          # COP only when real cooling is delivered
    "min_plant_kw": 1.0,
    "delta_t_scale": 1.0,         # 1.8 if temperature differences turn out to be in degC
    "tiny": 1e-9,                 # |x| below this is simulation noise (e.g. 3.5e-18) -> 0
    "raw_dir": "data/raw",
    "processed_dir": "data/processed",
    "fault_free_file": "ChillerPlant.csv",
}

BTU_PER_H_PER_KW = 3412.14
GPM_DEGF_TO_BTUH = 500.0          # water: 8.33 lb/gal * 60 min/h * 1 Btu/(lb.F)


# ----------------------------------------------------------------------------
# Settings and files
# ----------------------------------------------------------------------------
def load_settings(config_path: str | Path = "config.yaml") -> dict:
    s = dict(DEFAULTS)
    p = Path(config_path)
    if p.exists():
        try:
            import yaml
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            s.update(cfg.get("preprocess") or {})
            d = cfg.get("data") or {}
            s["raw_dir"] = d.get("raw_dir", s["raw_dir"])
            s["processed_dir"] = d.get("processed_dir", s["processed_dir"])
        except Exception as e:
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
    files = sorted(f for f in folder.rglob("*.csv")
                   if "__MACOSX" not in f.parts and not f.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {folder}. Check DATA_DIR in .env")
    return files


def find_fault_free(files: list[Path], name: str = "ChillerPlant.csv") -> Path:
    for f in files:
        if f.name == name:
            return f
    raise ValueError(f"{name} not found; set preprocess.fault_free_file in config.yaml")


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
        print(f"[preprocess] {Path(path).name}: no timestamp column; building a 1-minute index")
        df.index = pd.date_range("2018-01-01 01:00", periods=len(df), freq="min")
    df.index.name = "time"
    num = df.select_dtypes("number").columns
    df[num] = df[num].astype("float32")
    df[num] = df[num].mask(df[num].abs() < DEFAULTS["tiny"], 0.0)

    # The LBNL CSVs have the two weather columns swapped: OA_TEMP holds wet-bulb
    # and OA_TEMP_WB holds dry-bulb (WB > DB in 99% of rows; the tower set-point
    # equals OA_TEMP + 8 degF). Wet-bulb can never exceed dry-bulb, so if the
    # "wet-bulb" column is the higher one in most rows, swap the names.
    if {"OA_TEMP", "OA_TEMP_WB"} <= set(df.columns):
        if (df["OA_TEMP_WB"] > df["OA_TEMP"]).mean() > 0.5:
            df = df.rename(columns={"OA_TEMP": "OA_TEMP_WB", "OA_TEMP_WB": "OA_TEMP"})
    return df


# ----------------------------------------------------------------------------
# Sensors and running masks
# ----------------------------------------------------------------------------
SENSOR_PAT = re.compile(r"(TEMP|FLOW|POW|LOAD|_DP)(_\d+)?$")
NOT_SENSOR_PAT = re.compile(r"SPT|_STA(_\d+)?$|CTRL|SPD")
STATUS_PAT = re.compile(r"_STA(_\d+)?$")


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if SENSOR_PAT.search(c) and not NOT_SENSOR_PAT.search(c)]


def plant_on(df: pd.DataFrame) -> pd.Series:
    """True where at least one chiller is on."""
    on = pd.Series(False, index=df.index)
    for c in df.columns:
        if re.fullmatch(r"CHL_STA_\d+", c):
            on |= df[c] > 0.5
    return on


def running_mask(df: pd.DataFrame, col: str, plant: pd.Series | None = None) -> pd.Series:
    """True where the equipment that owns `col` is running."""
    m = re.search(r"_(\d+)$", col)
    if m:
        i = m.group(1)
        for prefix, sta in (("CHL_", f"CHL_STA_{i}"), ("CT_", f"CT_STA_{i}"),
                            ("CWL_SEC_PM", f"CWL_SEC_PM_STA_{i}")):
            if col.startswith(prefix) and sta in df:
                return df[sta] > 0.5
    return plant if plant is not None else plant_on(df)


def _run_lengths(changed: np.ndarray) -> np.ndarray:
    grp = np.cumsum(changed)
    return np.bincount(grp)[grp]


# ----------------------------------------------------------------------------
# Cleaning steps
# ----------------------------------------------------------------------------
def frozen_flags(df: pd.DataFrame, cols: list[str], s: dict) -> tuple[pd.DataFrame, dict]:
    flags = pd.DataFrame(False, index=df.index, columns=cols)
    skipped = {}
    plant = plant_on(df)
    for c in cols:
        if c in s["frozen_exclude"]:
            continue
        run = running_mask(df, c, plant).to_numpy()
        x = df[c].to_numpy(dtype="float64")
        changed = (np.r_[True, np.abs(np.diff(x)) > s["frozen_tol"]]
                   | np.r_[True, run[1:] != run[:-1]])
        stuck = (_run_lengths(changed) >= s["frozen_window"]) & run
        frac = stuck.sum() / max(run.sum(), 1)
        if frac > s["max_flag_frac"]:          # constant by design (e.g. fixed-speed pump)
            skipped[c] = round(float(frac), 3)
            continue
        flags[c] = stuck
    return flags, skipped


def fit_limits(df_fault_free: pd.DataFrame, s: dict) -> pd.DataFrame:
    """mean +/- sigma*std per sensor (per calendar month if sigma_by_month), from running
    samples of the fault-free file. Index (sensor, month); month 0 = whole year."""
    rows = []
    plant = plant_on(df_fault_free)
    for c in sensor_columns(df_fault_free):
        x = df_fault_free.loc[running_mask(df_fault_free, c, plant), c].astype("float64")
        groups = x.groupby(x.index.month) if s["sigma_by_month"] else [(0, x)]
        for m, xm in groups:
            if len(xm) >= 100:
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
    months = pd.Series(df.index.month if by_month else 0, index=df.index)
    plant = plant_on(df)
    for c in cols:
        if c not in sensors:
            continue
        lim = limits.loc[c]
        lo, hi, sd = months.map(lim["low"]), months.map(lim["high"]), months.map(lim["std"])
        run = running_mask(df, c, plant) & lo.notna() & (sd > 0)
        out = run & ((df[c] < lo) | (df[c] > hi))
        frac = out.sum() / max(run.sum(), 1)
        if frac > s["max_flag_frac"]:
            skipped[c] = round(float(frac), 3)
            continue
        flags[c] = out
    return flags, skipped


def resample(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    """15-minute means. Status columns: 1 if on for at least half of the window.
    Windows with fewer than min_coverage valid minutes are dropped."""
    rule = s["resample"]
    if not rule:
        return df
    minutes = pd.Timedelta(rule) / pd.Timedelta("1min")
    num = df.select_dtypes("number")
    out = num.resample(rule).mean()
    status = [c for c in out.columns if STATUS_PAT.search(c)]
    out[status] = (out[status] >= 0.5).astype("float32")
    count = num.iloc[:, 0].resample(rule).count()
    out = out.loc[count >= s["min_coverage"] * minutes]
    return out.astype("float32")


def clean(df: pd.DataFrame, limits: pd.DataFrame | None, s: dict) -> tuple[pd.DataFrame, dict]:
    """Same steps for every file. Returns (clean 15-minute table, report)."""
    cols = sensor_columns(df)
    rep = {"rows_raw": len(df)}
    if s["frozen_removal"]:
        fz, fz_skip = frozen_flags(df, cols, s)
        df = df.loc[~fz.any(axis=1)]
        rep["frozen_by_column"] = fz.sum().loc[lambda x: x > 0].sort_values(ascending=False)
        rep["frozen_check_skipped"] = fz_skip
    rep["rows_after_frozen"] = len(df)
    if s["outlier_removal"] and limits is not None:
        ol, ol_skip = outlier_flags(df, cols, limits, s)
        df = df.loc[~ol.any(axis=1)]
        rep["outliers_by_column"] = ol.sum().loc[lambda x: x > 0].sort_values(ascending=False)
        rep["outlier_check_skipped"] = ol_skip
    rep["rows_after_outliers"] = len(df)
    df = resample(df, s)
    rep["rows_15min"] = len(df)
    if s["chiller_on_only"]:
        df = df.loc[plant_on(df)]
    rep["rows_chiller_on"] = len(df)
    return df.copy(), rep


# ----------------------------------------------------------------------------
# Derived metrics (team-agreed names)
# ----------------------------------------------------------------------------
def add_derived(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    out = df.copy()
    k = s["delta_t_scale"]
    cool_total = pd.Series(0.0, index=out.index)
    for i in range(1, s["n_chillers"] + 1):
        sta = out[f"CHL_STA_{i}"] > 0.5
        dt = (out[f"CHL_RW_TEMP_{i}"] - out[f"CHL_SW_TEMP_{i}"]) * k               # F
        cool = out[f"CHL_CW_FLOW_{i}"] * dt * GPM_DEGF_TO_BTUH / BTU_PER_H_PER_KW  # kW
        pw = out[f"CHL_POW_{i}"]
        out[f"CHL_DT_{i}"] = dt.where(sta)
        out[f"CHL_COOL_{i}"] = cool.where(sta, 0.0)
        out[f"CHL_COP_{i}"] = (cool / pw).where(sta & (pw > s["min_chiller_kw"])
                                                & (cool >= s["min_cool_kw"]))
        # condenser water heats up through the chiller: leaving - entering
        out[f"CHL_CDDT_{i}"] = ((out[f"CHL_SWCD_TEMP_{i}"] - out[f"CHL_RWCD_TEMP_{i}"]) * k).where(sta)
        cool_total += cool.where(sta, 0.0).fillna(0.0)
    for i in range(1, s["n_towers"] + 1):
        # only when the tower really works: status on, water flowing and fan running
        on = out[f"CT_STA_{i}"] > 0.5 if f"CT_STA_{i}" in out else pd.Series(True, index=out.index)
        if f"CT_FLOW_{i}" in out:
            on &= out[f"CT_FLOW_{i}"] > 0
        if f"CT_POW_{i}" in out:
            on &= out[f"CT_POW_{i}"] > 0
        out[f"CT_APPROACH_{i}"] = (out[f"CT_SW_TEMP_{i}"] - out["OA_TEMP_WB"]).where(on)
    pow_cols = [c for c in out.columns
                if re.fullmatch(r"(CHL_POW|CT_POW|CDWL_PM_POW|CWL_PRI_PM_POW|CWL_SEC_PM_POW)_\d+", c)]
    out["PLANT_POW"] = out[pow_cols].sum(axis=1)
    any_on = plant_on(out)
    # dataset_notes.md: plant COP = building load (CWL_SEC_LOAD, W -> kW) / plant power
    load_kw = out["CWL_SEC_LOAD"] / 1000.0 if "CWL_SEC_LOAD" in out else cool_total
    out["PLANT_COP"] = (load_kw / out["PLANT_POW"]).where(
        any_on & (out["PLANT_POW"] > s["min_plant_kw"]) & (load_kw >= s["min_cool_kw"]))
    return out


def energy_balance(df: pd.DataFrame, s: dict) -> pd.Series:
    """sum(CHL_COOL_i) / (CWL_SEC_LOAD/1000) at real load; should be close to 1."""
    cool = sum(df[f"CHL_COOL_{i}"] for i in range(1, s["n_chillers"] + 1))
    load = df["CWL_SEC_LOAD"] / 1000.0
    ok = load >= s["min_cool_kw"]
    return (cool[ok] / load[ok]).describe()


# ----------------------------------------------------------------------------
# Saving and running
# ----------------------------------------------------------------------------
def save(df: pd.DataFrame, name: str, folder: str | Path = "data/processed") -> Path:
    """Parquet (pyarrow is in requirements.txt)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{name}.parquet"
    df.to_parquet(p)
    return p


def process_file(path: Path, limits: pd.DataFrame | None, s: dict, out_dir=None):
    out_dir = out_dir or s["processed_dir"]
    raw = load_csv(path)
    cleaned, report = clean(raw, limits, s)
    full = add_derived(cleaned, s)
    saved = save(full, Path(path).stem, out_dir)
    return raw, full, report, saved


def process_all(s: dict | None = None) -> pd.DataFrame:
    s = s or load_settings()
    files = list_files(data_dir(s["raw_dir"]))
    limits = None
    if s["outlier_removal"]:
        limits = fit_limits(load_csv(find_fault_free(files, s["fault_free_file"])), s)
    rows = []
    for f in files:
        _, full, rep, saved = process_file(f, limits, s)
        rows.append({"file": f.name, **{k: v for k, v in rep.items() if k.startswith("rows")}})
        print(f"{f.name}: {rep['rows_raw']} -> {rep['rows_chiller_on']} rows, saved {saved}")
    return pd.DataFrame(rows)


# Names from the original stub (kept so other code can call them)
def load_raw(path):
    return load_csv(Path(path))


def remove_frozen_sensors(df, window=None):
    s = load_settings()
    if window is not None:
        s["frozen_window"] = window
    flags, _ = frozen_flags(df, sensor_columns(df), s)
    return df.loc[~flags.any(axis=1)]


def remove_outliers(df, sigma=3, limits=None):
    s = load_settings()
    s["sigma"] = sigma
    limits = fit_limits(df, s) if limits is None else limits
    flags, _ = outlier_flags(df, sensor_columns(df), limits, s)
    return df.loc[~flags.any(axis=1)]


def add_derived_metrics(df):
    return add_derived(df, load_settings())


if __name__ == "__main__":
    print(process_all().to_string())
