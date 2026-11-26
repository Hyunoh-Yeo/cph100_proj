"""Model classes. Each one has fit(X, y, stay, train_rows, val_rows), predict(X, stay, rows),
save(path) and load(path), so train.py can treat them the same way.

To add a model, write a class with that interface in a new file and register it below.
Imports are lazy so LightGBM runs work without PyTorch installed.
"""


def get_model(name):
    if name == "lgbm":
        from .lgbm import LGBMModel
        return LGBMModel
    if name == "gru":
        from .gru import GRUModel
        return GRUModel
    raise ValueError(f"unknown model {name!r}")


AVAILABLE = ["lgbm", "gru"]
