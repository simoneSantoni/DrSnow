# Positivity and weight diagnostics of the longitudinal estimators.

"""
    regime_weights(r::LongitudinalEstimate) -> DataFrame

Support of each regime in the data, period by period: how many units follow it, how
small the estimated probabilities of its actions are, and how the cumulative inverse
probability weights are distributed.

The longitudinal estimators rely on positivity: at every period, units with every
history must have a positive probability of receiving the regime's action. The
cumulative probability of following a regime through ``t`` is the product
``\\hat g_{1:t} = \\prod_{s ≤ t} \\hat g_s(d_s \\mid H_s)``, which shrinks with ``t``, so
the weights ``1 / \\hat g_{1:t}`` of the followers grow and the estimators lean on
fewer and fewer units (Cole and Hernán 2008). Practical violations show up as a
small share of followers in the last periods, estimated probabilities near zero
among followers, a heavy right tail of the weights, and many weights bounded at
`1 / g_bound`. In such cases the inverse-probability estimators are unstable and
every estimator, including g-computation, extrapolates from the followers to the
whole population; report the table and consider regimes with more support, fewer
periods or a larger `g_bound` (with its bias). The table describes the fitted
nuisances of the first repetition of the sample split; it tests nothing, and a
regime with good empirical support can still be confounded.

# Arguments
- `r::LongitudinalEstimate`: a fitted result.

# Returns
- `DataFrame` with one row per regime and period and the columns `regime`,
  `period`, `n_following` and `share_following` (followers through the period),
  `g_min` and `g_q05` (minimum and 5% quantile of ``\\hat g_t(d_t \\mid H_t)`` among the
  followers), `cumulative_min` (minimum unbounded ``\\hat g_{1:t}`` among followers),
  `weight_mean`, `weight_sd`, `weight_max` and `weight_q99` (the bounded cumulative
  weights of the followers) and `n_bounded` (followers whose cumulative probability
  was raised to `g_bound`).

# Examples
```julia
using DrSnow
# with r from longitudinal_regime(...):
# regime_weights(r)
```

# References
- Cole, S. R., & Hernán, M. A. (2008). Constructing inverse probability weights for
  marginal structural models. *American Journal of Epidemiology*, 168(6), 656–664.
- Petersen, M. L., Porter, K. E., Gruber, S., Wang, Y., & van der Laan, M. J. (2012).
  Diagnosing and responding to violations in the positivity assumption. *Statistical
  Methods in Medical Research*, 21(1), 31–54.
"""
function regime_weights(r::LongitudinalEstimate)
    rows = NamedTuple[]
    for (j, nm) in enumerate(r.names)
        follow = r.followers[j]
        cum = r.cumulative_probability[j]
        W = r.weights[j]
        for t in 1:r.K
            f = findall(follow[:, t])
            g = t == 1 ? cum[f, 1] : cum[f, t] ./ cum[f, t - 1]
            w = W[f, t]
            push!(rows, (regime=nm, period=r.times[t], n_following=length(f),
                         share_following=length(f) / r.n,
                         g_min=isempty(g) ? NaN : minimum(g),
                         g_q05=isempty(g) ? NaN : quantile(g, 0.05),
                         cumulative_min=isempty(f) ? NaN : minimum(cum[f, t]),
                         weight_mean=isempty(w) ? NaN : mean(w),
                         weight_sd=length(w) < 2 ? NaN : std(w),
                         weight_max=isempty(w) ? NaN : maximum(w),
                         weight_q99=isempty(w) ? NaN : quantile(w, 0.99),
                         n_bounded=r.n_bounded[j, t]))
        end
    end
    return DataFrame(rows)
end
