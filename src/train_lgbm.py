"""Steps 4 to 7. Split, train LightGBM on hospital A, evaluate on A test and B, export.

Two variants are trained on the same split:
    full        all features
    no_pattern  drops the measurement-pattern features (measured / hrs_since)

Outputs in out/
    metrics.json, metrics.csv     AUROC, AUPRC, utility per variant and split
    predictions.parquet / .csv.gz one row per patient-hour for A_val, A_test, B
    importance_<variant>.csv      LightGBM gain importance
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features  # noqa: E402
from utility import best_threshold, normalized_utility, utility_components  # noqa: E402

SEED = 0
OUT = "out" if len(sys.argv) == 1 else "out_test"
SPLITS = ["A_val", "A_test", "B"]


def assign_splits(data, seed=SEED):
    """Hospital A split by patient 70/15/15. Hospital B is never trained or tuned on."""
    a_ids = np.random.default_rng(seed).permutation(
        data.loc[data.hospital == "A", "patient_id"].unique())
    n_tr, n_va = int(0.70 * len(a_ids)), int(0.15 * len(a_ids))
    split = {p: "A_train" for p in a_ids[:n_tr]}
    split.update({p: "A_val" for p in a_ids[n_tr:n_tr + n_va]})
    split.update({p: "A_test" for p in a_ids[n_tr + n_va:]})
    return data.patient_id.map(split).fillna("B")


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    data = pd.read_parquet("data/all.parquet")
    if len(sys.argv) > 1:  # quick test run: python src/train_lgbm.py 3000
        keep_ids = pd.Series(data.patient_id.unique()).sample(int(sys.argv[1]), random_state=SEED)
        data = data[data.patient_id.isin(keep_ids)].reset_index(drop=True)

    first_pos = data[data.SepsisLabel == 1].groupby("patient_id")["t"].min()
    data["onset_t"] = data.patient_id.map(first_pos + 6)  # label leads onset by 6h
    data["split"] = assign_splits(data)
    print(data.groupby("split").patient_id.nunique().to_string())

    X, groups = build_features(data)
    y = data.SepsisLabel.to_numpy()
    mask = {s: (data.split == s).to_numpy() for s in ["A_train"] + SPLITS}
    u_pos, u_neg, best = utility_components(data.t, data.onset_t)
    print(f"{X.shape[1]} features built ({time.time() - t0:.0f}s)")

    variants = {
        "full": list(X.columns),
        "no_pattern": [c for c in X.columns if c not in groups["pattern"]],
    }
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, metric="auc", verbose=-1, seed=SEED)

    probs, metrics = {}, {}
    for name, cols in variants.items():
        dtrain = lgb.Dataset(X.loc[mask["A_train"], cols], y[mask["A_train"]])
        dval = lgb.Dataset(X.loc[mask["A_val"], cols], y[mask["A_val"]], reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        p = model.predict(X[cols], num_iteration=model.best_iteration)
        probs[name] = p
        pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False) \
            .to_csv(f"{OUT}/importance_{name}.csv", header=["gain"])
        print(f"{name}: {len(cols)} features, best iteration {model.best_iteration} "
              f"({time.time() - t0:.0f}s)")

        k = mask["A_val"]
        thr, _ = best_threshold(p[k], u_pos[k], u_neg[k], best[k])
        metrics[name] = {"threshold_from_A_val": thr, "n_features": len(cols)}
        for s in SPLITS:
            k = mask[s]
            metrics[name][s] = {
                "auroc": roc_auc_score(y[k], p[k]),
                "auprc": average_precision_score(y[k], p[k]),
                "utility_at_A_val_threshold": normalized_utility(p[k] >= thr, u_pos[k], u_neg[k], best[k]),
                "utility_at_own_best_threshold": best_threshold(p[k], u_pos[k], u_neg[k], best[k])[1],
            }

    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    table = pd.DataFrame([{"variant": v, "split": s, **metrics[v][s]}
                          for v in metrics for s in SPLITS])
    table.to_csv(f"{OUT}/metrics.csv", index=False)
    print("\n" + table.to_string(index=False, float_format="%.3f"))

    keep = (data.split != "A_train").to_numpy()
    pred = data.loc[keep, ["patient_id", "hospital", "split", "t", "ICULOS",
                           "SepsisLabel", "onset_t"]].rename(
        columns={"ICULOS": "iculos", "SepsisLabel": "label"})
    for name, p in probs.items():
        pred[f"prob_{name}"] = p[keep].astype(np.float32)
    pred.to_parquet(f"{OUT}/predictions.parquet", index=False)
    pred.to_csv(f"{OUT}/predictions.csv.gz", index=False)
    print(f"\nwrote {len(pred)} prediction rows to {OUT}/ ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
