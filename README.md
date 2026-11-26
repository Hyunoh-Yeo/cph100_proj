# cph100_proj

CPH C100 final project (Rajan Tavathia and Hyunoh Yeo). We study alert burden and lead time of sepsis early warning models on the PhysioNet/Computing in Cardiology Challenge 2019 data, training on hospital A and testing transfer to hospital B.

## Reproducing Phase 1

### 1. Environment

```
conda env create -f environment.yml
conda activate cph100
```

Without conda, `pip install -r requirements.txt` also works. On macOS, LightGBM installed through pip may need `brew install libomp` first, which is why the conda route is recommended.

### 2. Data

The data is not in the repo (`data/` is gitignored). Download it from https://physionet.org/content/challenge-2019/1.0.0/ into `data/`. The fastest way is through the public S3 mirror.

```
pip install awscli
aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/ data/
```

You should end up with `data/training_setA/` (20,336 files) and `data/training_setB/` (20,000 files). Downloading the 42 MB ZIP from the PhysioNet page and unzipping it into `data/` works too.

### 3. Run

Run everything from the repo root, in this order.

```
python src/prepare_data.py   # reads all .psv files once, writes data/all.parquet (about 1 min)
python src/explore.py        # optional, prints stay counts, sepsis rates and measurement rates by hospital
python src/utility.py        # optional, checks our utility score against the official implementation
python src/train_lgbm.py     # trains both LightGBM variants and writes out/ (under 1 min)
```

`python src/train_lgbm.py 3000` runs on a random subset of 3,000 stays and writes to `out_test/` instead, which is handy for checking changes quickly.

### 4. Expected results

With the default seed (0) the full run should print numbers close to these.

| variant | split | AUROC | AUPRC | utility at A_val threshold |
|---|---|---|---|---|
| full | A_test | 0.807 | 0.107 | 0.413 |
| full | B | 0.760 | 0.062 | 0.260 |
| no_pattern | A_test | 0.786 | 0.099 | 0.371 |
| no_pattern | B | 0.792 | 0.066 | 0.293 |

Small differences can come from LightGBM version or thread count.

## What the code does

`src/prepare_data.py` merges every stay into one table and adds `patient_id`, `hospital` (A for `training_setA`, B for `training_setB`) and `t`, the 0-based hour index within a stay.

`src/features.py` builds hourly features that only use hours up to the current one. For each of the 34 vital and lab variables it uses the last observed value, and for each vital sign the 1h and 6h change. Age, sex, ICU unit, HospAdmTime and ICULOS are included as they are. It also builds measurement-pattern features (whether a variable was measured this hour and hours since it was last measured), which describe how each hospital orders tests rather than patient physiology.

`src/utility.py` implements the Challenge utility score following the official `evaluate_sepsis_score.py`.

`src/train_lgbm.py` splits hospital A by patient into 70% train, 15% validation and 15% test with seed 0. Hospital B is never used for training or tuning. It trains LightGBM twice, once with all 124 features (`full`) and once without the 68 measurement-pattern features (`no_pattern`), with early stopping on A validation AUROC. The decision threshold is chosen to maximize utility on A validation and applied unchanged to A test and B.

## Outputs (`out/`)

`predictions.csv.gz` and `predictions.parquet` have one row per patient-hour for the A_val, A_test and B splits. A_train is left out on purpose.

| column | meaning |
|---|---|
| patient_id | stay id, e.g. p000123 |
| hospital | A or B |
| split | A_val, A_test or B |
| t | 0-based hour index within the stay |
| iculos | hours since ICU admission, from the data |
| label | Challenge SepsisLabel, which turns 1 six hours before onset |
| onset_t | sepsis onset hour, equal to the first positive label hour + 6 (empty if never septic) |
| prob_full, prob_no_pattern | predicted probability from each variant |

Lead time should be measured against `onset_t`, not the first positive label. Thresholds and alert policies should be tuned on `A_val` and reported on `A_test` and `B`.

`metrics.json` and `metrics.csv` hold the numbers above, and `importance_full.csv` / `importance_no_pattern.csv` hold LightGBM gain importance.

## Notes

About 11% of septic stays in A and 20% in B already have a positive label at the first hour, so no early warning is possible for them. They are kept in the predictions, and the evaluation side decides how to treat them for lead time.

ICULOS is the most important feature in both variants. It reflects time in the ICU rather than physiology, which is worth keeping in mind when interpreting the models.
