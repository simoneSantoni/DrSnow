"""Reference values for DrSnow's `dml_did_multi` from DoubleML's DoubleMLDIDMulti.

Uses the datasets and unit-level fold assignments written by make_did_multi_data.jl,
deterministic learners (OLS: sklearn LinearRegression; logit: LogisticRegression
without penalty) and, for every group-time cell, the sample split obtained by
restricting the common unit-level folds to the units of that cell (DrSnow draws the
folds once over units). Writes did_multi_reference.csv.

For every cell the file stores DoubleML's estimate and standard error and, in
`se_hajek`, the standard error from the influence function of the normalized
(Hajek) doubly-robust estimator computed from DoubleML's own nuisance predictions.
DrSnow (like R's DRDID / did) uses the latter; DoubleML's linear score treats the
normalizing constant of the comparison-group weights as known, which changes the
standard error slightly but not the point estimate.

    python doubleml_did_multi_reference.py      (doubleml 0.11.4, scikit-learn 1.9.1)
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.data import DoubleMLPanelData

DIR = os.path.dirname(os.path.abspath(__file__))


def smpls_from_folds(ids, fold_of):
    f = np.array([fold_of[i] for i in ids])
    ks = np.unique(f)
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in ks]


def hajek_se_panel(model):
    sub = model.data_subset
    d = sub["G_indicator"].to_numpy().astype(float)
    dy = sub["y_diff"].to_numpy()
    pos = model.id_positions
    g0 = model.predictions["ml_g0"][pos, 0, 0]
    m = model.predictions["ml_m"][pos, 0, 0]
    e = dy - g0
    w1 = d / d.mean()
    pw = m * (1 - d) / (1 - m)
    w0 = pw / pw.mean()
    psi = w1 * (e - np.mean(w1 * e)) - w0 * (e - np.mean(w0 * e))
    return np.sqrt(np.mean(psi**2) / len(psi))


def hajek_se_rcs(model):
    sub = model.data_subset
    D = sub["G_indicator"].to_numpy().astype(float)
    T = sub["t_indicator"].to_numpy().astype(float)
    y = sub[model._dml_data.y_col].to_numpy()
    pos = model.id_positions
    p = {k: model.predictions[k][pos, 0, 0] for k in
         ["ml_g_d0_t0", "ml_g_d0_t1", "ml_g_d1_t0", "ml_g_d1_t1", "ml_m"]}
    m = p["ml_m"]
    g0y = T * p["ml_g_d0_t1"] + (1 - T) * p["ml_g_d0_t0"]
    e = y - g0y
    prop = m * (1 - D) / (1 - m)

    def nz(w):
        return w / w.mean()

    w_tpost, w_tpre = nz(D * T), nz(D * (1 - T))
    w_cpost, w_cpre = nz(prop * T), nz(prop * (1 - T))
    w_d = nz(D)
    dpost = p["ml_g_d1_t1"] - p["ml_g_d0_t1"]
    dpre = p["ml_g_d1_t0"] - p["ml_g_d0_t0"]
    comps = [(1, w_tpost, e), (-1, w_tpre, e), (-1, w_cpost, e), (1, w_cpre, e),
             (1, w_d, dpost), (-1, w_tpost, dpost), (-1, w_d, dpre), (1, w_tpre, dpre)]
    psi = np.zeros(len(y))
    for s, w, v in comps:
        psi += s * w * (v - np.mean(w * v))
    return np.sqrt(np.mean(psi**2) / len(psi))


def run(case, df, folds, x_cols, panel, **kw):
    data = DoubleMLPanelData(df, y_col="y", d_cols="g", t_col="t", id_col="id",
                             x_cols=x_cols)
    obj = dml.did.DoubleMLDIDMulti(data, ml_g=LinearRegression(),
                                   ml_m=LogisticRegression(C=np.inf, tol=1e-12,
                                                           max_iter=10_000),
                                   gt_combinations="standard", n_folds=5, panel=panel,
                                   **kw)
    fold_of = dict(zip(folds["id"], folds["fold"]))
    for model in obj.modellist:
        model.set_sample_splitting(
            smpls_from_folds(model.data_subset["id"].to_numpy(), fold_of))
    obj.fit()
    rows = []
    for i, model in enumerate(obj.modellist):
        g, tpre, teval = obj.gt_combinations[i]
        se_h = hajek_se_panel(model) if panel else hajek_se_rcs(model)
        rows.append(dict(case=case, kind="att_gt", g=g, t_pre=tpre, t_eval=teval,
                         name="", coef=obj.all_coef[i, 0], se=obj.all_se[i, 0],
                         se_hajek=se_h))
    for agg in ["group", "time", "eventstudy"]:
        a = obj.aggregate(agg)
        fw = a.aggregated_frameworks
        names = a.aggregation_names
        for j, nm in enumerate(names):
            rows.append(dict(case=case, kind=agg, g=0, t_pre=0, t_eval=0, name=nm,
                             coef=fw.thetas[j], se=fw.ses[j], se_hajek=np.nan))
        ov = a.overall_aggregated_framework
        rows.append(dict(case=case, kind=agg, g=0, t_pre=0, t_eval=0, name="overall",
                         coef=ov.thetas[0], se=ov.ses[0], se_hajek=np.nan))
    return rows


panel = pd.read_csv(os.path.join(DIR, "did_multi_panel.csv"))
pf = pd.read_csv(os.path.join(DIR, "did_multi_panel_folds.csv"))
rcs = pd.read_csv(os.path.join(DIR, "did_multi_rcs.csv"))
rf = pd.read_csv(os.path.join(DIR, "did_multi_rcs_folds.csv"))

rows = []
rows += run("panel_never", panel, pf, ["x1", "x2", "x3"], True)
rows += run("panel_notyet", panel, pf, ["x1", "x2", "x3"], True,
            control_group="not_yet_treated")
rows += run("panel_notyet_antic1", panel, pf, ["x1", "x2", "x3"], True,
            control_group="not_yet_treated", anticipation_periods=1)
rows += run("rcs_never", rcs, rf, ["x1", "x2"], False)
rows += run("rcs_notyet", rcs, rf, ["x1", "x2"], False,
            control_group="not_yet_treated")
out = pd.DataFrame(rows)
out.to_csv(os.path.join(DIR, "did_multi_reference.csv"), index=False,
           float_format="%.15g", na_rep="NaN")
print(out.to_string())
