"""Reference confidence sequences from the `confseq` Python package (v0.0.11).

Run:  python confseq_reference.py
Writes, next to this file,
  confseq_data.csv       the two fixed data streams (read by the Julia tests), and
  confseq_reference.csv  CS bounds of confseq for those streams.

confseq 0.0.11 was built from source against pybind11 >= 2.11 and Boost 1.84 headers
(its pinned pybind11 2.6 does not compile with current Python).
"""
import os
import numpy as np
from confseq.boundaries import normal_mixture_bound
from confseq.predmix import predmix_hoeffding_cs, predmix_empbern_twosided_cs
from confseq.betting import hedged_cs

here = os.path.dirname(os.path.abspath(__file__))

rng = np.random.default_rng(20260928)
n = 300
data = {
    "beta": rng.beta(2, 5, size=n),                          # bounded, skewed
    "bern": (rng.uniform(size=n) < 0.3).astype(float),       # Bernoulli(0.3)
}

with open(os.path.join(here, "confseq_data.csv"), "w") as f:
    f.write("t,beta,bern\n")
    for i in range(n):
        f.write(f"{i + 1},{data['beta'][i]:.17g},{data['bern'][i]:.17g}\n")

rows = []
# Two-sided normal-mixture boundary for a unit-variance sum.
for alpha in [0.05, 0.1]:
    for v_opt in [100.0, 1000.0]:
        for v in [1.0, 10.0, 100.0, 1000.0, 1e5]:
            b = float(normal_mixture_bound(v, alpha, v_opt, alpha, False))
            rows.append(("nm_bound", "none", alpha, v_opt, False, v, b, float("nan")))

for name, x in data.items():
    for alpha in [0.05, 0.1]:
        for fixed_n in [None, 200]:
            topt = -1 if fixed_n is None else fixed_n
            for ri in [False, True]:
                l, u = predmix_hoeffding_cs(x, alpha=alpha, running_intersection=ri,
                                            fixed_n=fixed_n)
                rows += [("hoeffding", name, alpha, topt, ri, t + 1, l[t], u[t])
                         for t in range(n)]
                l, u = predmix_empbern_twosided_cs(x, alpha=alpha,
                                                   running_intersection=ri,
                                                   fixed_n=fixed_n)
                rows += [("empirical_bernstein", name, alpha, topt, ri, t + 1, l[t],
                          u[t]) for t in range(n)]
        for ri in [False, True]:
            l, u = hedged_cs(x, alpha=alpha, running_intersection=ri, breaks=1000)
            rows += [("betting", name, alpha, -1, ri, t + 1, l[t], u[t])
                     for t in range(n)]


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.17g}"
    return str(v)


with open(os.path.join(here, "confseq_reference.csv"), "w") as f:
    f.write("method,data,alpha,t_opt,running_intersection,t,lower,upper\n")
    for r in rows:
        # keep the start of each path and every 10th step afterwards
        if r[0] != "nm_bound" and r[5] > 30 and r[5] % 10 != 0:
            continue
        f.write(",".join(fmt(v) for v in r) + "\n")
print("rows", len(rows))
