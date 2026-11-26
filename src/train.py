"""Train, evaluate and export every model in src/models on the same split and features.

    python src/train.py                    # all models (lgbm, gru)
    python src/train.py --models lgbm      # only some models; other models' results are kept
    python src/train.py --subset 3000      # quick test on 3,000 stays, writes to out_test/

Each model is trained twice:
    full        all features
    no_pattern  without the measurement-pattern features (measured / hrs_since)

Hospital A is split by patient 70/15/15 (seed 0). Hospital B is never used for training or
tuning. The decision threshold is chosen to maximize utility on A_val and applied
unchanged to A_test and B.

Outputs
    out/predictions.parquet, .csv.gz   one row per patient-hour (A_val, A_test, B),
                                       one prob_<model>_<variant> column per model
    out/metrics.csv, .json             AUROC, AUPRC, utility per model, variant and split
    out/importance_lgbm_<variant>.csv  LightGBM gain importance
    checkpoints/<model>_<variant>.*    saved models
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features  # noqa: E402
from models import AVAILABLE, get_model  # noqa: E402
from utility import best_threshold, normalized_utility, utility_components  # noqa: E402

SEED = 0
SPLITS = ["A_val", "A_test", "B"]
KEYS = ["patient_id", "t"]


def assign_splits(data, seed=SEED):
    a_ids = np.random.default_rng(seed).permutation(
        data.loc[data.hospital == "A", "patient_id"].unique())
    n_tr, n_va = int(0.70 * len(a_ids)), int(0.15 * len(a_ids))
    split = {p: "A_train" for p in a_ids[:n_tr]}
    split.update({p: "A_val" for p in a_ids[n_tr:n_tr + n_va]})
    split.update({p: "A_test" for p in a_ids[n_tr + n_va:]})
    return data.patient_id.map(split).fillna("B")


def load_data(subset):
    data = pd.read_parquet("data/all.parquet")
    if subset:
        ids = pd.Series(data.patient_id.unique()).sample(subset, random_state=SEED)
        data = data[data.patient_id.isin(ids)].reset_index(drop=True)
    first_pos = data[data.SepsisLabel == 1].groupby("patient_id")["t"].min()
    data["onset_t"] = data.patient_id.map(first_pos + 6)  # label leads onset by 6h
    data["split"] = assign_splits(data)
    return data


def evaluate(p_full, y, rows, u_pos, u_neg, best):
    """p_full has one probability per row of the data (NaN for A_train)."""
    k = rows["A_val"]
    thr, _ = best_threshold(p_full[k], u_pos[k], u_neg[k], best[k])
    out = []
    for s in SPLITS:
        k = rows[s]
        out.append({
            "split": s,
            "auroc": roc_auc_score(y[k], p_full[k]),
            "auprc": average_precision_score(y[k], p_full[k]),
            "threshold_from_A_val": thr,
            "utility_at_A_val_threshold": normalized_utility(p_full[k] >= thr, u_pos[k], u_neg[k], best[k]),
            "utility_at_own_best_threshold": best_threshold(p_full[k], u_pos[k], u_neg[k], best[k])[1],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=AVAILABLE, choices=AVAILABLE)
    ap.add_argument("--subset", type=int, default=None)
    args = ap.parse_args()
    out_dir = "out" if args.subset is None else "out_test"
    ckpt_dir = "checkpoints" if args.subset is None else os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    t0 = time.time()
    data = load_data(args.subset)
    print(data.groupby("split").patient_id.nunique().to_string())
    X_df, groups = build_features(data)
    y = data.SepsisLabel.to_numpy()
    stay = data.patient_id.factorize()[0]
    rows = {s: (data.split == s).to_numpy() for s in ["A_train"] + SPLITS}
    eval_rows = ~rows["A_train"]
    u_pos, u_neg, best = utility_components(data.t, data.onset_t)
    print(f"{X_df.shape[1]} features ({time.time() - t0:.0f}s)")

    variants = {
        "full": list(X_df.columns),
        "no_pattern": [c for c in X_df.columns if c not in groups["pattern"]],
    }

    probs, metrics = {}, []
    for model_name in args.models:
        Model = get_model(model_name)
        for variant, cols in variants.items():
            print(f"\n{model_name} / {variant} ({len(cols)} features)")
            X = X_df[cols].to_numpy()
            model = Model(seed=SEED).fit(X, y, stay, rows["A_train"], rows["A_val"])
            model.save(os.path.join(ckpt_dir, f"{model_name}_{variant}{Model.file_ext}"))
            if hasattr(model, "feature_importance"):
                pd.Series(model.feature_importance(), index=cols).sort_values(ascending=False) \
                    .to_csv(f"{out_dir}/importance_{model_name}_{variant}.csv", header=["gain"])

            p = np.full(len(data), np.nan)
            p[eval_rows] = model.predict(X, stay, eval_rows)
            probs[f"prob_{model_name}_{variant}"] = p
            for r in evaluate(p, y, rows, u_pos, u_neg, best):
                metrics.append({"model": model_name, "variant": variant, **r})
            print(f"  done ({time.time() - t0:.0f}s)")

    # merge with results of models that were not rerun this time
    pred = data.loc[eval_rows, ["patient_id", "hospital", "split", "t", "ICULOS",
                                "SepsisLabel", "onset_t"]].rename(
        columns={"ICULOS": "iculos", "SepsisLabel": "label"}).reset_index(drop=True)
    for col, p in probs.items():
        pred[col] = p[eval_rows].astype(np.float32)
    table = pd.DataFrame(metrics)
    old_pred_path, old_metrics_path = f"{out_dir}/predictions.parquet", f"{out_dir}/metrics.csv"
    if os.path.exists(old_pred_path) and os.path.exists(old_metrics_path):
        old = pd.read_parquet(old_pred_path)
        keep_cols = [c for c in old.columns if c.startswith("prob_") and c not in pred.columns]
        if keep_cols:
            pred = pred.merge(old[KEYS + keep_cols], on=KEYS, how="left", validate="one_to_one")
        old_m = pd.read_csv(old_metrics_path)
        if "model" in old_m.columns:
            table = pd.concat([old_m[~old_m.model.isin(args.models)], table], ignore_index=True)

    table.to_csv(old_metrics_path, index=False)
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(table.to_dict(orient="records"), f, indent=2)
    pred.to_parquet(old_pred_path, index=False)
    pred.to_csv(f"{out_dir}/predictions.csv.gz", index=False)

    show = table[["model", "variant", "split", "auroc", "auprc", "utility_at_A_val_threshold",
                  "utility_at_own_best_threshold"]]
    print("\n" + show.to_string(index=False, float_format="%.3f"))
    print(f"\nwrote {len(pred)} prediction rows to {out_dir}/ ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
