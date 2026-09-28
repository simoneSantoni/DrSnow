"""Reference values for DrSnow's adaptively weighted AIPW estimators.

Generates one Thompson-sampling experiment with the simulation design of Hadad,
Hirshberg, Zhan, Wager & Athey (2021, PNAS) and computes arm values and contrasts
with the authors' reference code (gsbDBI/adaptive-confidence-intervals, commit on
GitHub master as of 2026-09; functions `run_mab_experiment`, `ts_mab_probs`,
`apply_floor`, `sample_mean`, `aw_scores`, `twopoint_stable_var_ratio`,
`stick_breaking`, `evaluate_aipw_stats`, `aw_contrast_stderr`, copied verbatim below
so that only numpy is needed).

Writes
  hadad_data.csv       t, arm (1-based), reward, p1..pK   (the logged experiment)
  hadad_reference.csv  weights, target, estimate, stderr

Run:  python3 hadad_reference.py
"""
import csv
import numpy as np


def repr_(x):
    return repr(float(x))

np.random.seed(20210415)


# ----------------------------------------------------------------- reference code
def collect(arr, idx):
    out = np.empty(len(idx), dtype=arr.dtype)
    for i, j in enumerate(idx):
        out[i] = arr[i, j]
    return out


def expand(values, idx, num_cols):
    out = np.zeros((len(idx), num_cols), dtype=values.dtype)
    for i, (j, v) in enumerate(zip(idx, values)):
        out[i, j] = v
    return out


def apply_floor(a, amin):
    new = np.maximum(a, amin)
    total_slack = np.sum(new) - 1
    individual_slack = new - amin
    c = total_slack / np.sum(individual_slack)
    return new - c * individual_slack


def stick_breaking(Z):
    T, K = Z.shape
    weights = np.zeros((T, K))
    weight_sum = np.zeros(K)
    for t in range(T):
        weights[t] = Z[t] * (1 - weight_sum)
        weight_sum += weights[t]
    return weights


def ts_mab_probs(sum, sum2, neff, prev_t, floor_start=0.005, floor_decay=0.0,
                 num_mc=20):
    K = len(sum)
    Z = np.random.normal(size=(num_mc, K))
    mu = sum / np.maximum(neff, 1)
    var = sum2 / np.maximum(neff, 1) - (sum / np.maximum(neff, 1)) ** 2
    posterior_var = 1 / (neff / var + 1 / 1.0)
    posterior_mean = neff / (var + neff) * mu
    idx = np.argmax(Z * np.sqrt(posterior_var) + posterior_mean, axis=1)
    w_mc = np.array([np.sum(idx == k) for k in range(K)])
    p_mc = w_mc / num_mc
    probs = apply_floor(p_mc, amin=floor_start / (prev_t + 1) ** floor_decay)
    return probs


def run_mab_experiment(ys, initial=0, floor_start=0.005, floor_decay=0.0):
    T, K = ys.shape
    T0 = initial * K
    arms = np.empty(T, dtype=np.int_)
    rewards = np.empty(T)
    probs = np.empty((T, K))
    sum = np.zeros(K)
    sum2 = np.zeros(K)
    neff = np.zeros(K)
    for t in range(T):
        if t < T0:
            p = np.full(K, 1 / K)
            w = t % K
        else:
            p = ts_mab_probs(sum, sum2, neff, t, floor_start=floor_start,
                             floor_decay=floor_decay)
            w = np.random.choice(K, p=p)
        sum[w] += ys[t, w]
        sum2[w] += ys[t, w] ** 2
        neff[w] += 1
        arms[t] = w
        rewards[t] = ys[t, w]
        probs[t] = p
    return arms, rewards, probs


def sample_mean(rewards, arms, K):
    T = len(arms)
    W = expand(np.ones(T), arms, K)
    Y = expand(rewards, arms, K)
    return np.cumsum(W * Y, 0) / np.maximum(np.cumsum(W, 0), 1)


def aw_scores(rewards, arms, assignment_probs, muhat=None):
    T, K = assignment_probs.shape
    balwts = 1 / collect(assignment_probs, arms)
    scores = expand(balwts * rewards, arms, K)
    if muhat is not None:
        scores += (1 - expand(balwts, arms, K)) * muhat
    return scores


def twopoint_stable_var_ratio(e, alpha):
    T, K = e.shape
    t = np.arange(1, T + 1)[:, np.newaxis]
    bad_lambda = (1 - alpha) / ((1 - alpha) + T * (t / T) ** alpha - t)
    good_lambda = 1 / (1 + T - t)
    lamb = (1 - e) * bad_lambda + e * good_lambda
    return np.clip(lamb, 0, 1)


def arm_stats(score, evalwts):
    estimate = np.sum(evalwts * score, 0) / np.sum(evalwts, 0)
    stderr = np.sqrt(np.sum(evalwts ** 2 * (score - estimate) ** 2, 0)) / \
        np.sum(evalwts, 0)
    return estimate, stderr


def aw_contrast_stderr(score, evalwts, estimate):
    h_sum = evalwts.sum(0)
    diff = score - estimate
    numerator = h_sum[:-1] * evalwts[:, -1:] * diff[:, -1:] - \
        h_sum[-1] * evalwts[:, :-1] * diff[:, :-1]
    numerator = np.sum(numerator ** 2, axis=0)
    denominator = h_sum[-1] ** 2 * h_sum[:-1] ** 2
    return np.sqrt(numerator / denominator)


# ----------------------------------------------------------------- experiment
truth = np.array([0.9, 1.0, 1.1])      # "lowSNR" design
K = len(truth)
T = 2000
floor_decay = 0.7
ys = truth + np.random.uniform(-1, 1, size=(T, K))
arms, rewards, probs = run_mab_experiment(ys, initial=5, floor_start=1 / K,
                                          floor_decay=floor_decay)

muhat = np.vstack([np.zeros(K), sample_mean(rewards, arms, K)[:-1]])
scores = aw_scores(rewards, arms, probs, muhat)
ipw = aw_scores(rewards, arms, probs, None)

twopoint = np.sqrt(np.maximum(0., stick_breaking(
    twopoint_stable_var_ratio(probs, floor_decay)) * probs))
weights = {"two_point": twopoint, "constant_allocation": np.sqrt(probs),
           "uniform": np.ones_like(probs)}

rows = []
for name, w in weights.items():
    for model, sc in (("running_mean", scores), ("none", ipw)):
        est, se = arm_stats(sc, w)
        for k in range(K):
            rows.append((name, model, f"value(arm {k + 1})", est[k], se[k]))
        cse = aw_contrast_stderr(sc, w, est)
        for k in range(K - 1):
            rows.append((name, model, f"value(arm {K}) - value(arm {k + 1})",
                         est[-1] - est[k], cse[k]))

with open("hadad_data.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["t", "arm", "reward"] + [f"p{k + 1}" for k in range(K)])
    for t in range(T):
        wr.writerow([t + 1, arms[t] + 1, repr_(rewards[t])] +
                    [repr_(v) for v in probs[t]])

with open("hadad_reference.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["weights", "outcome_model", "term", "estimate", "stderr"])
    for r in rows:
        wr.writerow([r[0], r[1], r[2], repr_(r[3]), repr_(r[4])])
print("wrote", len(rows), "reference rows")
