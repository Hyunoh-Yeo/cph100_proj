"""LightGBM baseline. Treats every patient-hour as an independent row."""
import lightgbm as lgb
import numpy as np

DEFAULT_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    metric="auc", verbose=-1,
)


class LGBMModel:
    """Same interface as the other models in this folder.

    fit(X, y, stay, train_rows, val_rows) trains on train_rows and early-stops on val_rows.
    predict(X, stay, rows) returns one probability per selected row, in row order.
    X is a float32 array (rows x features), y the hourly labels, stay the stay index of
    each row. LightGBM does not need stay, the GRU does.
    """

    name = "lgbm"
    file_ext = ".txt"

    def __init__(self, seed=0, num_boost_round=3000, early_stopping=100, **params):
        self.params = {**DEFAULT_PARAMS, **params, "seed": seed}
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.booster = None

    def fit(self, X, y, stay, train_rows, val_rows):
        dtrain = lgb.Dataset(X[train_rows], y[train_rows])
        dval = lgb.Dataset(X[val_rows], y[val_rows], reference=dtrain)
        self.booster = lgb.train(
            self.params, dtrain, num_boost_round=self.num_boost_round, valid_sets=[dval],
            callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
        print(f"  lgbm best iteration {self.booster.best_iteration}")
        return self

    def predict(self, X, stay, rows):
        return self.booster.predict(X[rows], num_iteration=self.booster.best_iteration)

    def feature_importance(self):
        return self.booster.feature_importance("gain")

    def save(self, path):
        self.booster.save_model(path, num_iteration=self.booster.best_iteration)

    @classmethod
    def load(cls, path):
        m = cls()
        m.booster = lgb.Booster(model_file=path)
        return m
