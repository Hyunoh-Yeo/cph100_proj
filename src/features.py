"""Step 3. Hourly features. Every feature at hour t uses only hours <= t of the same stay.

Feature groups
    value      last observed value of each variable (forward fill)
    trend      1h and 6h change in each vital sign
    static     Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS
    pattern    measurement-pattern features: whether each variable was measured this
               hour, and hours since it was last measured. These capture how the hospital
               orders tests rather than the patient's physiology, so we can train with
               and without them to see how much they hurt transfer from A to B.
"""
import numpy as np
import pandas as pd

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]


def build_features(data):
    """Returns (X, groups) where groups maps a group name to its column names."""
    pid = data["patient_id"]
    t = data["t"].astype(float)
    cols, groups = {}, {"value": [], "trend": [], "static": [], "pattern": []}

    for v in VITALS + LABS:
        measured = data[v].notna()
        cols[v] = data[v].groupby(pid, sort=False).ffill().astype(np.float32)
        groups["value"].append(v)

        cols[f"{v}_measured"] = measured.astype(np.float32)
        last_t = t.where(measured).groupby(pid, sort=False).ffill()
        cols[f"{v}_hrs_since"] = (t - last_t).astype(np.float32)
        groups["pattern"] += [f"{v}_measured", f"{v}_hrs_since"]

    filled = pd.DataFrame({v: cols[v] for v in VITALS})
    for k in (1, 6):
        diff = filled.groupby(pid, sort=False).diff(k)
        for v in VITALS:
            cols[f"{v}_diff{k}"] = diff[v].astype(np.float32)
            groups["trend"].append(f"{v}_diff{k}")

    for v in STATIC:
        cols[v] = data[v].astype(np.float32)
        groups["static"].append(v)

    X = pd.DataFrame(cols)
    del cols
    return X, groups
