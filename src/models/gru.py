"""GRU sequence model. Reads each stay hour by hour and outputs a probability at every
hour. The GRU is unidirectional, so the prediction at hour t only depends on hours <= t."""
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class _GRUNet(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)  # one logit per hour


def _stay_bounds(stay, rows):
    """Start index and length of each stay among the selected rows (rows of a stay are contiguous)."""
    idx = np.flatnonzero(rows)
    s = stay[idx]
    first = np.r_[True, s[1:] != s[:-1]]
    starts = idx[first]
    lengths = np.diff(np.r_[np.flatnonzero(first), len(idx)])
    return starts, lengths


def _pad(seqs):
    return nn.utils.rnn.pad_sequence([torch.from_numpy(np.ascontiguousarray(a)) for a in seqs],
                                     batch_first=True)


class GRUModel:
    """Same interface as LGBMModel: fit(X, y, stay, train_rows, val_rows), predict(X, stay, rows)."""

    name = "gru"
    file_ext = ".pt"

    def __init__(self, seed=0, hidden=64, batch_size=64, lr=1e-3, max_epochs=30, patience=3):
        self.seed, self.hidden, self.batch_size = seed, hidden, batch_size
        self.lr, self.max_epochs, self.patience = lr, max_epochs, patience
        self.net, self.mu, self.sd = None, None, None

    # inputs are z-scored with train statistics, missing values become 0 (the train mean)
    def _transform(self, X):
        Z = (X - self.mu) / self.sd
        return np.clip(np.nan_to_num(Z, nan=0.0), -5, 5).astype(np.float32)

    def _batches(self, starts, lengths, shuffle, rng):
        order = np.argsort(lengths, kind="stable")  # similar lengths together, less padding
        batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if shuffle:
            rng.shuffle(batches)
        return batches

    def _predict_z(self, Z, starts, lengths):
        out = np.empty(lengths.sum(), dtype=np.float32)
        offsets = np.r_[0, np.cumsum(lengths)[:-1]]
        self.net.eval()
        with torch.no_grad():
            for b in self._batches(starts, lengths, False, None):
                x = _pad([Z[starts[i]:starts[i] + lengths[i]] for i in b])
                p = torch.sigmoid(self.net(x)).numpy()
                for j, i in enumerate(b):
                    out[offsets[i]:offsets[i] + lengths[i]] = p[j, :lengths[i]]
        return out

    def fit(self, X, y, stay, train_rows, val_rows):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        with warnings.catch_warnings():  # columns never measured in A (e.g. EtCO2) are all NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mu = np.nan_to_num(np.nanmean(X[train_rows], axis=0))
            sd = np.nanstd(X[train_rows], axis=0)
        self.sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        Z = self._transform(X)
        y = y.astype(np.float32)

        tr_starts, tr_len = _stay_bounds(stay, train_rows)
        va_starts, va_len = _stay_bounds(stay, val_rows)
        y_val = y[val_rows]

        self.net = _GRUNet(X.shape[1], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        best_auc, best_state, bad = -1.0, None, 0

        for epoch in range(self.max_epochs):
            self.net.train()
            for b in self._batches(tr_starts, tr_len, True, rng):
                x = _pad([Z[tr_starts[i]:tr_starts[i] + tr_len[i]] for i in b])
                yb = _pad([y[tr_starts[i]:tr_starts[i] + tr_len[i]] for i in b])
                mask = _pad([np.ones(tr_len[i], np.float32) for i in b])
                loss = (loss_fn(self.net(x), yb) * mask).sum() / mask.sum()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

            auc = roc_auc_score(y_val, self._predict_z(Z, va_starts, va_len))
            print(f"  gru epoch {epoch + 1}: A_val AUROC {auc:.3f}")
            if auc > best_auc:
                best_auc, bad = auc, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict(self, X, stay, rows):
        starts, lengths = _stay_bounds(stay, rows)
        return self._predict_z(self._transform(X), starts, lengths)

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "mu": self.mu, "sd": self.sd,
                    "n_in": len(self.mu), "hidden": self.hidden}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, weights_only=False)
        m = cls(hidden=ckpt["hidden"])
        m.mu, m.sd = ckpt["mu"], ckpt["sd"]
        m.net = _GRUNet(ckpt["n_in"], ckpt["hidden"])
        m.net.load_state_dict(ckpt["state"])
        return m
