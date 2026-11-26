"""Tests for src/discretise.py on synthetic data with a known answer.
Run: python tests/test_discretise.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src import discretise as dz

rng = np.random.default_rng(0)
n = 6000
grp = rng.random(n) < 0.5
df = pd.DataFrame({
    "VAR_A": np.where(grp, rng.normal(5, 1, n), rng.normal(15, 1, n)),   # drives efficiency
    "NOISE": rng.normal(0, 1, n),                                         # unrelated
    "CHL_SW_TEMP_1": rng.normal(49, 4, n),
    "PLANT_COP": np.where(grp, rng.normal(3.5, .3, n), rng.normal(2.5, .3, n)),
    "CHL_STA_1": (rng.random(n) > .3).astype(float),
    "CONST": 3.0,
})
df.loc[:299, "CHL_SW_TEMP_1"] = np.nan

specs = dz.fit_breakpoints(df, s=dz.DEFAULTS)

# 1. the efficiency-relevant variable gets exactly one cut, between the two groups
assert specs["VAR_A"]["method"] == "entropy" and len(specs["VAR_A"]["breaks"]) == 1
assert 7 < specs["VAR_A"]["breaks"][0] < 13
# 2. an unrelated variable gets no entropy split (falls back to tertiles)
assert specs["NOISE"]["method"] == "tertiles"
# 3. design breakpoints are kept
assert {44.0, 54.0} <= set(specs["CHL_SW_TEMP_1"]["breaks"])
# 4. status columns are on/off, constant columns are dropped
assert specs["CHL_STA_1"]["type"] == "binary" and "CONST" not in specs
# 5. transactions use "VARIABLE=interval" and skip missing values
tx = dz.to_transactions(dz.discretise(df, specs))
assert all("=" in item for t in tx for item in t)
assert not any(i.startswith("CHL_SW_TEMP_1=") for i in tx[0])     # row 0 has no value
assert not any("nan" in i for t in tx for i in t)
# 6. at most max_bins intervals per variable
assert all(len(v["breaks"]) <= dz.DEFAULTS["max_bins"] - 1 for v in specs.values() if v["type"] == "interval")
print("All discretisation tests passed.")
