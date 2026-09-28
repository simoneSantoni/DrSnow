"""Reference values for DrSnow's `rd_flex` from DoubleML's RDFlex.

Uses flex_data.csv (written by make_flex_data.jl) with deterministic learners
(sklearn LinearRegression; LogisticRegression without penalty) and the committed fold
assignments, so that the cross-fitted adjustments are identical. Writes
flex_reference.csv (estimates, standard errors and bandwidths) and
flex_adjustment.csv (the adjusted outcomes / treatments of every case).

    python doubleml_rdflex_reference.py   (doubleml 0.11.4, rdrobust 2.1.0,
                                           scikit-learn 1.9.1)
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.rdd import RDFlex

DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(DIR, "flex_data.csv"))
X = df[["z1", "z2", "z3", "z4"]].to_numpy()


def smpls(fold):
    return [[(np.where(fold != k)[0], np.where(fold == k)[0]) for k in np.unique(fold)]]


def run(case, fuzzy, spec="cutoff", n_iterations=2, kernel="triangular", **kw):
    y = df["y_fuzzy" if fuzzy else "y_sharp"].to_numpy()
    d = df["d"].to_numpy() if fuzzy else (df["x"].to_numpy() >= 0).astype(int)
    data = dml.DoubleMLRDDData.from_arrays(x=X, y=y, d=d, score=df["x"].to_numpy())
    obj = RDFlex(data, ml_g=LinearRegression(),
                 ml_m=LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000)
                 if fuzzy else None,
                 fuzzy=fuzzy, cutoff=0, n_folds=5, fs_specification=spec,
                 fs_kernel=kernel, **kw)
    obj._smpls = smpls(df["fold_fuzzy" if fuzzy else "fold_sharp"].to_numpy())
    obj.fit(n_iterations=n_iterations)
    res = obj._rdd_obj[0]
    row = dict(case=case, h_fs=obj.h_fs, h=float(res.bws.loc["h"].max()),
               b=float(res.bws.loc["b"].max()),
               coef_conv=obj.coef[0], coef_bc=obj.coef[1], coef_rb=obj.coef[2],
               se_conv=obj.se[0], se_rb=obj.se[2],
               n_h_left=int(res.N_h[0]), n_h_right=int(res.N_h[1]))
    adj = {f"{case}_my": obj._M_Y[:, 0]}
    if fuzzy:
        adj[f"{case}_md"] = obj._M_D[:, 0]
    return row, adj


rows, adjs = [], {}
for case, args in [("sharp", dict(fuzzy=False)),
                   ("sharp_score", dict(fuzzy=False, spec="cutoff and score")),
                   ("sharp_interacted",
                    dict(fuzzy=False, spec="interacted cutoff and score")),
                   ("sharp_iter1", dict(fuzzy=False, n_iterations=1)),
                   ("sharp_uniform", dict(fuzzy=False, kernel="uniform")),
                   ("fuzzy", dict(fuzzy=True))]:
    r, a = run(case, **args)
    rows.append(r)
    adjs.update(a)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(DIR, "flex_reference.csv"), index=False, float_format="%.15g")
pd.DataFrame(adjs).to_csv(os.path.join(DIR, "flex_adjustment.csv"), index=False,
                          float_format="%.15g")
print(out.to_string())
