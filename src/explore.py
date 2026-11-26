"""Step 2. Basic descriptive statistics by hospital (run after prepare_data.py)."""
import pandas as pd

data = pd.read_parquet("data/all.parquet")
VARS = [c for c in data.columns if c not in
        ("patient_id", "hospital", "t", "Age", "Gender", "Unit1", "Unit2",
         "HospAdmTime", "ICULOS", "SepsisLabel")]

stays = data.groupby("patient_id").agg(
    hospital=("hospital", "first"),
    hours=("t", "size"),
    septic=("SepsisLabel", "max"),
    age=("Age", "first"),
    male=("Gender", "first"),
)
first_pos = data[data.SepsisLabel == 1].groupby("patient_id")["t"].min()
stays["onset_t"] = first_pos + 6  # label turns on 6 hours before onset

print("=== stays ===")
summary = stays.groupby("hospital").agg(
    n_stays=("hours", "size"),
    median_hours=("hours", "median"),
    septic_pct=("septic", lambda s: 100 * s.mean()),
    median_age=("age", "median"),
    male_pct=("male", lambda s: 100 * s.mean()),
)
summary["positive_hour_pct"] = data.groupby("hospital")["SepsisLabel"].mean() * 100
print(summary.round(1).to_string())

print("\n=== onset timing among septic stays (hours since ICU admission) ===")
sep = stays[stays.septic == 1]
print(sep.groupby("hospital")["onset_t"].describe(percentiles=[.25, .5, .75]).round(1).to_string())
early = (sep.onset_t <= 6).groupby(sep.hospital).mean() * 100
print("\n% of septic stays whose label is already 1 in the first hour (onset <= 6h):")
print(early.round(1).to_string())

print("\n=== % of hours with a measurement, by variable ===")
miss = data.groupby("hospital")[VARS].apply(lambda d: d.notna().mean() * 100).T
miss["A_minus_B"] = miss["A"] - miss["B"]
print(miss.round(1).sort_values("A", ascending=False).to_string())
