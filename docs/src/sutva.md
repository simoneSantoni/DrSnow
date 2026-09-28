# Interference and spillovers

```@meta
CurrentModule = DrSnow
```

The stable unit treatment value assumption (SUTVA) rules out *interference*: one
unit's treatment affecting another unit's outcome. Spatial policies, network
experiments and clustered programmes routinely violate it. When they do, the
familiar treated-versus-control contrast mixes the direct effect with spillovers onto
the "controls", and standard errors that ignore the dependence between nearby or
connected units are too small.

DrSnow's interference area provides:

| Task | Functions |
|---|---|
| Describe who can affect whom | [`SpatialStructure`](@ref), [`NetworkStructure`](@ref), [`PartitionStructure`](@ref) |
| Compute exposures | [`compute_exposure`](@ref) with [`NeighborExposure`](@ref), [`RingExposure`](@ref), [`HopExposure`](@ref), [`CustomExposure`](@ref) |
| Design-based effects (known assignment) | [`exposure_probabilities`](@ref), [`exposure_effects`](@ref) |
| Exact tests of no spillovers | [`spillover_fisher_test`](@ref) |
| Checks of the assignment mechanism | [`exposure_balance_test`](@ref), [`treatment_moran_test`](@ref) |
| Panel DiD with spillovers | [`spillover_did`](@ref), [`spillover_event_study`](@ref), [`spillover_pretrend_test`](@ref) |
| Cross-sectional exposure regressions | [`exposure_regression`](@ref) |
| Dependence-robust standard errors | [`ConleyVcov`](@ref), [`NetworkHACVcov`](@ref), [`conley_vcov`](@ref), [`network_hac_vcov`](@ref) |
| Two-stage (saturation) designs | [`TwoStageRandomization`](@ref), [`two_stage_effects`](@ref) |

## Concepts

**Exposure mappings.** Under general interference a unit has one potential outcome
per assignment vector, which is not estimable. An *exposure mapping* (Aronow & Samii
2017; Manski 2013) summarizes the assignment vector into a low-dimensional exposure
for each unit, e.g. "own treatment × whether any friend is treated", the share of
treated neighbours, or the distance band containing the nearest treated unit. The
key assumption is that potential outcomes depend on the assignment only through the
unit's own treatment and its exposure. Every estimate in this area is relative to a
stated exposure mapping; if the mapping is wrong (e.g. spillovers reach farther than
the outermost ring), estimates are biased.

**Estimands.** With a discrete mapping, the design-based estimand for conditions `a`
and `b` is `τ(a, b) = (1/N) Σᵢ [Yᵢ(a) − Yᵢ(b)]`, the average over the units for which
both conditions are possible. Regression-based estimators (`spillover_did`,
`exposure_regression`) target the coefficients of a linear exposure model:
the direct effect for units with no exposure, and spillover effects for untreated
(`spill_control`) and treated (`spill_treated`) units in each exposure column.
Two-stage designs (Hudgens & Halloran 2008) target group-averaged direct, indirect
(spillover on the untreated), total and overall effects of changing the treatment
saturation.

**Assumptions.** Design-based estimators and Fisher tests require the assignment
mechanism to be known (an experiment, or a credible "as-if random" natural
experiment) and use it through an [`AssignmentMechanism`](@ref). Positivity is
required: a unit that can never be in condition `a` contributes no information about
`Yᵢ(a)` (see [`exposure_positivity`](@ref)); by default such units raise an error,
and `positivity=:restrict` redefines the estimand on the remaining units. The panel
estimators require parallel trends for treated, exposed and clean-control units, that
spillovers vanish beyond the outermost exposure band (so clean controls exist), and
treatment-timing variation.

## Structures and alignment

Structures are keyed by unit identifiers. Data are always matched to them by id,
never by row order, and the set of units in the data must equal the set in the
structure (an exposure depends on every unit's treatment). All results are invariant
to shuffling data rows.

```julia
using DrSnow, DataFrames

# Spatial: coordinates by name (latitude/longitude cannot be swapped silently)
s = SpatialStructure(counties, :fips; lat=:lat, lon=:lon)        # haversine, km
s_xy = SpatialStructure(plots.id; x=plots.x, y=plots.y)           # Euclidean

# Network from an edge list: source → target means source's treatment can affect
# target. Units without edges (isolates) must be listed in `ids`.
g = NetworkStructure(people.id, friendships; source=:from, target=:to)

# Partial interference: households, villages, classrooms
p = PartitionStructure(people, :id; group=:village)
```

`shortest_path_hops` computes breadth-first-search distances; `HopExposure` uses them,
so a unit reachable in one hop is never also counted at two hops (powers of the
adjacency matrix count walks and do count it).

## Exposures

```julia
# one row per unit
e = compute_exposure(people, :treated, g, NeighborExposure(:share); unit=:id)

# panel: time-varying exposure from the period-t treatments, via sparse products
ex = compute_exposure(panel, :policy, s, RingExposure([20.0, 40.0]);
                      unit=:fips, time=:year)
```

Exposures are `missing` when undefined: isolates (unless `isolates=:zero`), empty
distance bands for `:share`, or a neighbour's treatment missing in that period.

## Design-based estimation

```julia
design = CompleteRandomization(n_units(g), 150)       # over structure_units(g)
P = exposure_probabilities(g, design; draws=10_000, rng=StableRNG(1))
exposure_positivity(P)
r = exposure_effects(people, :y, :treated, g, P; unit=:id)   # Hájek by default
confint(r)
```

Probabilities are computed exactly by enumerating the design's support when it is
small, and by Monte Carlo otherwise. The variance estimator is the conservative
Aronow–Samii estimator (linearized for Hájek). The Horvitz–Thompson estimator is
unbiased with exact probabilities but skewed when few units fall in a condition, so
its normal-approximation intervals can undercover in small samples; the Hájek
estimator is the default.

## Randomization tests

[`spillover_fisher_test`](@ref) tests the sharp null of no spillovers exactly: it
fixes a set of focal units, re-randomizes the other units' treatments *conditional on
the focal units' treatments*, and compares a statistic of the focal units' outcomes
and exposures with its conditional randomization distribution (Athey, Eckles &
Imbens 2018). [`exposure_balance_test`](@ref) and [`treatment_moran_test`](@ref)
check the assumed assignment mechanism itself: their reference distributions come
from the design, so structural differences between exposed and unexposed units
(well-connected units are exposed more often) do not produce false rejections.
Non-rejection is never evidence that SUTVA holds.

## Difference-in-differences with spillovers

```julia
r = spillover_did(panel, :y, :policy, s; unit=:fips, time=:year,
                  exposure=RingExposure([20.0, 40.0]),
                  vcov=ConleyVcov(s; unit=:fips, cutoff=60.0, time=:year))
es = spillover_event_study(panel, :y, :policy, s; unit=:fips, time=:year,
                           exposure=RingExposure([20.0, 40.0]), leads=3, lags=3)
spillover_pretrend_test(es)
```

Following Butts (2021), the regression includes unit and period fixed effects, the
own-treatment indicator, and exposure terms interacted with own treatment; units
beyond the outermost ring are the clean controls. Each coefficient is identified by
the units in its exposure cell; cells with few units (few clusters) give unreliable
cluster-robust intervals. With staggered adoption and heterogeneous effects, TWFE
coefficients inherit the known weighting problems of TWFE DiD.

## Standard errors under spatial and network dependence

[`ConleyVcov`](@ref) and [`NetworkHACVcov`](@ref) are covariance estimators for
`FixedEffectModels.reg` (and for the spillover regressions above): fixed effects are
partialled out, instrumental-variable models are supported, and panel options allow
within-period spatial correlation together with serial correlation within units
(Hsiang 2010). The uniform-kernel Conley estimator reproduces `fixest::vcov_conley`
exactly when the same Earth radius is used (fixest uses 6376 km; DrSnow's default is
the mean radius 6371 km, configurable with `earth_radius`). Like all HAC estimators
they are biased downward when the dependence range is large relative to the study
area; report results for several cutoffs or bandwidths.

## Two-stage randomized designs

```julia
design = TwoStageRandomization(df.village, [0.25, 0.75], [30, 30])
r = two_stage_effects(df, :y, :treated; group=:village, saturation=:saturation)
```

Signs are "treated minus untreated" and "higher minus reference saturation"
(Hudgens & Halloran define direct and indirect effects with the opposite sign). The
variance estimator is the between-group two-stage-sampling estimator, conservative in
expectation; intervals use `t(min groups per saturation − 1)`.

## Validation

- Aronow–Samii Horvitz–Thompson and Hájek estimates and variances match the R package
  `interference` (Zonszein, Samii & Aronow) to machine precision, given the same
  probability matrices; exposure conditions match its `make_exposure_map_AS`.
- Conley covariance matrices match `fixest::vcov_conley` (cross-section, with fixed
  effects, and pooled panel) to machine precision.
- Exact enumeration of small designs shows the Horvitz–Thompson estimator is exactly
  unbiased and the variance estimator conservative; the Cliff–Ord moments of Moran's
  I equal the exact permutation moments; the Fisher test p-value equals the exact
  conditional p-value.
- Monte Carlo checks of size and coverage for most inferential procedures are part of
  the test suite (`DRSNOW_SLOW_TESTS=true` for full replication counts).

## References

- Aronow, P. M., & Samii, C. (2017). Estimating average causal effects under general
  interference, with application to a social network experiment. *Annals of Applied
  Statistics*, 11(4), 1912–1947.
- Athey, S., Eckles, D., & Imbens, G. W. (2018). Exact p-values for network
  interference. *Journal of the American Statistical Association*, 113(521), 230–240.
- Baird, S., Bohren, J. A., McIntosh, C., & Özler, B. (2018). Optimal design of
  experiments in the presence of interference. *Review of Economics and Statistics*,
  100(5), 844–860.
- Butts, K. (2021). Difference-in-differences estimation with spatial spillovers.
  arXiv:2105.03737.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway
  clustering. *Journal of Business & Economic Statistics*, 29(2), 238–249.
- Cliff, A. D., & Ord, J. K. (1981). *Spatial Processes: Models and Applications*. Pion.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of
  Econometrics*, 92(1), 1–45.
- Hsiang, S. M. (2010). Temperatures and cyclones strongly associated with economic
  production in the Caribbean and Central America. *Proceedings of the National Academy
  of Sciences*, 107(35), 15367–15372.
- Hudgens, M. G., & Halloran, M. E. (2008). Toward causal inference with interference.
  *Journal of the American Statistical Association*, 103(482), 832–842.
- Kojevnikov, D., Marmer, V., & Song, K. (2021). Limit theorems for network dependent
  random variables. *Journal of Econometrics*, 222(2), 882–908.
- Leung, M. P. (2022). Causal inference under approximate neighborhood interference.
  *Econometrica*, 90(1), 267–293.
- Manski, C. F. (2013). Identification of treatment response with social interactions.
  *Econometrics Journal*, 16(1), S1–S23.

The functions and types described on this page are documented in the [API reference](reference/sutva.md).
