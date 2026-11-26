"""Step 1. Read every .psv file once and save one table to data/all.parquet.

Adds three columns to the raw data:
    patient_id  file name without extension, e.g. p000001
    hospital    A for training_setA (p0xxxxx), B for training_setB (p1xxxxx)
    t           0-based hour index within the stay
"""
import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT_PATH = os.path.join(DATA_DIR, "all.parquet")


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "training_set*", "*.psv")))
    print(f"found {len(files)} files")

    frames = []
    for i, path in enumerate(files):
        df = pd.read_csv(path, sep="|")
        pid = os.path.splitext(os.path.basename(path))[0]
        df.insert(0, "patient_id", pid)
        df.insert(1, "hospital", "A" if "training_setA" in path else "B")
        df.insert(2, "t", np.arange(len(df)))
        frames.append(df)
        if (i + 1) % 10000 == 0:
            print(f"  read {i + 1} files")

    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(OUT_PATH, index=False)
    print(f"saved {len(data)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
