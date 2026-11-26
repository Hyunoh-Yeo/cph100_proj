"""Challenge utility score, following the official evaluate_sepsis_score.py.

utility_components() turns the official per-hour rules into two arrays: the utility of
predicting 1 (u_pos) and of predicting 0 (u_neg) at every row. The score for any set of
binary predictions is then sum(where(pred, u_pos, u_neg)), normalized so that the best
possible predictions score 1 and predicting 0 everywhere scores 0.

Running this file checks it against a line-by-line port of the official function.
"""
import numpy as np

DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3.0
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1.0, -2.0, -0.05, 0.0


def utility_components(t, onset_t):
    t = np.asarray(t, dtype=float)
    onset = np.asarray(onset_t, dtype=float)
    septic = ~np.isnan(onset)
    rel = t - np.where(septic, onset, np.inf)

    m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY); b1 = -m1 * DT_EARLY
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL); b2 = -m2 * DT_LATE
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL); b3 = -m3 * DT_OPTIMAL

    with np.errstate(invalid="ignore"):
        before_opt = rel <= DT_OPTIMAL
        scored = rel <= DT_LATE
        pos = np.where(before_opt, np.maximum(m1 * rel + b1, U_FP), m2 * rel + b2)
        neg = np.where(before_opt, 0.0, m3 * rel + b3)
    u_pos = np.where(septic, np.where(scored, pos, 0.0), U_FP)
    u_neg = np.where(septic, np.where(scored, neg, 0.0), U_TN)
    best = septic & (rel >= DT_EARLY) & (rel <= DT_LATE)
    return u_pos, u_neg, best


def normalized_utility(pred, u_pos, u_neg, best):
    observed = np.where(pred, u_pos, u_neg).sum()
    optimal = np.where(best, u_pos, u_neg).sum()
    inaction = u_neg.sum()
    return float((observed - inaction) / (optimal - inaction))


def best_threshold(prob, u_pos, u_neg, best, n_grid=200):
    grid = np.unique(np.quantile(prob, np.linspace(0.5, 0.999, n_grid)))
    scores = [normalized_utility(prob >= g, u_pos, u_neg, best) for g in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def _official_utility(labels, predictions):
    """Line-by-line port of compute_prediction_utility from the official code."""
    if np.any(labels):
        is_septic, t_sepsis = True, np.argmax(labels) - DT_OPTIMAL
    else:
        is_septic, t_sepsis = False, float("inf")
    m_1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY); b_1 = -m_1 * DT_EARLY
    m_2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL); b_2 = -m_2 * DT_LATE
    m_3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL); b_3 = -m_3 * DT_OPTIMAL
    u = np.zeros(len(labels))
    for t in range(len(labels)):
        if t <= t_sepsis + DT_LATE:
            if is_septic and predictions[t]:
                u[t] = max(m_1 * (t - t_sepsis) + b_1, U_FP) if t <= t_sepsis + DT_OPTIMAL \
                    else m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u[t] = U_FP
            elif is_septic and not predictions[t]:
                u[t] = 0 if t <= t_sepsis + DT_OPTIMAL else m_3 * (t - t_sepsis) + b_3
            else:
                u[t] = U_TN
    return u.sum()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for trial in range(300):
        labs, preds, ts, ons = [], [], [], []
        obs = opt = ina = 0.0
        for _ in range(int(rng.integers(2, 20))):
            n = int(rng.integers(5, 60))
            lab = np.zeros(n, int)
            onset = np.nan
            if rng.random() < 0.4:
                first = int(rng.integers(0, n)); lab[first:] = 1; onset = first + 6
            pr = (rng.random(n) < rng.random()).astype(int)
            bp = np.zeros(n)
            if lab.any():
                s = np.argmax(lab) - DT_OPTIMAL
                bp[max(0, int(s + DT_EARLY)):min(int(s + DT_LATE + 1), n)] = 1
            obs += _official_utility(lab, pr); opt += _official_utility(lab, bp)
            ina += _official_utility(lab, np.zeros(n))
            preds.append(pr); ts.append(np.arange(n)); ons.append(np.full(n, onset))
        if opt == ina:
            continue
        ref = (obs - ina) / (opt - ina)
        mine = normalized_utility(np.concatenate(preds).astype(bool),
                                  *utility_components(np.concatenate(ts), np.concatenate(ons)))
        assert abs(ref - mine) < 1e-9, (trial, ref, mine)
    print("utility matches the official implementation on 300 random cases")
