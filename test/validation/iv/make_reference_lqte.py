"""Reference values for DrSnow's DML local quantile treatment effects (dml_lqte).

Regenerate (from the repository root) with the Python package DoubleML (0.11.x):
    python test/validation/iv/make_reference_lqte.py

Writes lqte.csv (simulated data with fold ids), lqte_prelim.csv (the nested
sample splits and preliminary IPW quantile estimates DoubleML draws inside every
fold) and reference_lqte.csv (columns case, quantity, value), read by
test/iv/test_dml_lqte.jl.

DoubleMLLPQ estimates the nuisance g (probability of D = d and Y <= q given Z, X) at
a preliminary IPW quantile computed on half of every training fold, split with
sklearn's train_test_split(random_state=42) and cross-fitted with StratifiedKFold.
Those splits and the preliminary quantiles are recorded here (the root finder is
wrapped to log its results) so that DrSnow can be run with the same nested splits.
Learners: unpenalized logistic regression (deterministic), identical outer folds.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
import doubleml as dml
import doubleml.irm.lpq as lpq_mod

here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260928)
n = 1200
x1, x2 = rng.normal(size=n), rng.normal(size=n)
z = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.6 * x1))).astype(float)
u = rng.uniform(size=n)
at = u < 0.15
nt = u > 0.75
co = ~(at | nt)
d = (at | (co & (z == 1))).astype(float)
y0 = 0.5 * x1 - 0.3 * x2 + rng.normal(size=n)
y1 = y0 + 1.0 + 0.8 * rng.exponential(size=n)
y = np.where(d == 1, y1, y0)
K = 5
folds = np.zeros(n, dtype=int)
strata = d + 2 * z
for s in np.unique(strata):
    idx = np.where(strata == s)[0]
    rng.shuffle(idx)
    folds[idx] = np.arange(len(idx)) % K + 1
df = pd.DataFrame({"y": y, "d": d, "z": z, "x1": x1, "x2": x2, "fold": folds})
df.to_csv(os.path.join(here, "lqte.csv"), index=False)

smpls = [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(1, K + 1)]

# nested splits exactly as in DoubleMLLPQ._nuisance_est
prelim = []
for k, (train, _) in enumerate(smpls, start=1):
    t1, t2 = train_test_split(train, test_size=0.5, random_state=42,
                              stratify=strata[train])
    inner = np.zeros(len(t1), dtype=int)
    for j, (_, te) in enumerate(StratifiedKFold(n_splits=K).split(X=t1, y=strata[t1])):
        inner[te] = j + 1
    for pos, (i, f) in enumerate(zip(t1, inner)):
        prelim.append({"fold": k, "pos": pos + 1, "row": i + 1, "inner": f})
prelim = pd.DataFrame(prelim)

# log the preliminary IPW quantiles
log = []
orig = lpq_mod._solve_ipw_score


def logged(ipw_score, bracket_guess):
    est = orig(ipw_score=ipw_score, bracket_guess=bracket_guess)
    log.append(float(est))
    return est


lpq_mod._solve_ipw_score = logged


def learner():
    return LogisticRegression(C=np.inf, tol=1e-12, max_iter=10000)


data = dml.DoubleMLData(df, "y", "d", x_cols=["x1", "x2"], z_cols="z")
out = []
ipw_rows = []
quantiles = [0.25, 0.5, 0.75]
models = {}
for t in (0, 1):
    for iq, q in enumerate(quantiles):
        log.clear()
        m = dml.DoubleMLLPQ(data, learner(), learner(), treatment=t, quantile=q,
                            n_folds=K, normalize_ipw=True, draw_sample_splitting=False)
        m.set_sample_splitting([smpls])
        m.fit()
        models[(t, q)] = m
        assert len(log) == K
        for k in range(K):
            ipw_rows.append({"treatment": t, "quantile": q, "fold": k + 1,
                             "ipw": log[k]})
        out += [{"case": f"lpq_d{t}_q{q}", "quantity": "coef", "value": m.coef[0]},
                {"case": f"lpq_d{t}_q{q}", "quantity": "se", "value": m.se[0]}]
        # DoubleML's standardized score (psi / J) for the joint variance
        psi = m.psi[:, 0, 0]
        J = np.mean(m.psi_deriv[:, 0, 0])
        out += [{"case": f"lpq_d{t}_q{q}", "quantity": "J", "value": J},
                {"case": f"lpq_d{t}_q{q}", "quantity": "mean_psi2", "value": np.mean(psi ** 2)}]
# LQTE = LPQ(1) - LPQ(0), DoubleML framework difference (as in DoubleMLQTE)
for q in quantiles:
    fw = models[(1, q)].framework - models[(0, q)].framework
    out += [{"case": f"lqte_q{q}", "quantity": "coef", "value": fw.thetas[0]},
            {"case": f"lqte_q{q}", "quantity": "se", "value": fw.ses[0]}]
# nuisance predictions of the (d = 1, q = 0.5) model for a direct check
p = models[(1, 0.5)].predictions
preds = pd.DataFrame({k: p[k][:, 0, 0] for k in p})
preds.to_csv(os.path.join(here, "lqte_preds.csv"), index=False)
ipw = pd.DataFrame(ipw_rows)
prelim.to_csv(os.path.join(here, "lqte_prelim.csv"), index=False)
ipw.to_csv(os.path.join(here, "lqte_ipw.csv"), index=False)
pd.DataFrame(out).to_csv(os.path.join(here, "reference_lqte.csv"), index=False)
print("DoubleML", dml.__version__, "reference values written")
