# Regression discontinuity with DrSnow: sharp and fuzzy designs.
#
# Run from the repository root:
#     julia --project=. examples/rd_sharp_fuzzy_demo.jl
#
# Part 1 uses the U.S. Senate data of Cattaneo, Frandsen & Titiunik (2015), shipped
# with the R package rdrobust (a copy is in test/validation/rd/senate.csv): the
# running variable is the Democratic margin of victory in a Senate election, the
# outcome the Democratic vote share in the next election for the same seat.
# Part 2 simulates a fuzzy design with a known effect. Part 3 shows honest
# (bias-aware) inference under a bound on the second derivative (RDHonest).

using DrSnow
using DataFrames
using Random
using Statistics
using Printf

# Minimal reader for the numeric/quoted CSV written by R (avoids a CSV dependency).
function read_simple_csv(path)
    lines = readlines(path)
    split_line(l) = [strip(f, '"') for f in split(l, ',')]
    header = split_line(lines[1])
    rows = split_line.(lines[2:end])
    df = DataFrame()
    for (j, name) in enumerate(header)
        raw = [r[j] for r in rows]
        vals = [isempty(v) || v == "NA" ? missing : tryparse(Float64, v) for v in raw]
        df[!, Symbol(name)] = any(v -> v === nothing, vals) ? raw :
                              Vector{Union{Missing,Float64}}(vals)
    end
    return df
end

senate = read_simple_csv(joinpath(@__DIR__, "..", "test", "validation", "rd", "senate.csv"))

println("="^78)
println("Part 1. Sharp RD: incumbency advantage in the U.S. Senate")
println("="^78)

# 1. Visualise the discontinuity (data for a plot: binned means + global polynomial).
pd = rd_plot_data(senate, :vote, :margin)
println("\nRD plot data: $(pd.nbins) bins (left, right); $(pd.binselect_description)")
println(first(pd.bins[:, [:side, :bin_mid, :mean_y, :n]], 5))

# 2. Estimation with MSE-optimal bandwidth and robust bias-corrected inference.
r = rd_estimate(senate, :vote, :margin)
show(stdout, MIME"text/plain"(), r)
println()
ci = confint(r)
@printf("\nRobust bias-corrected estimate: %.2f percentage points, 95%% CI [%.2f, %.2f]\n",
        coef(r)[1], ci[1, 1], ci[1, 2])
println("The conventional interval is shown for comparison only; at the MSE-optimal")
println("bandwidth its coverage is below the nominal level.")

# 3. All bandwidth selectors.
println("\nBandwidth selectors:")
println(rd_bandwidth(senate, :vote, :margin; bwselect=:all).table)

# 4. Falsification: density of the running variable, covariate balance, placebos.
println()
show(stdout, MIME"text/plain"(), rd_density_test(senate, :margin))
println("\nMcCrary (2008) binned test, for comparison with older studies:")
show(stdout, MIME"text/plain"(), rd_mccrary_test(senate, :margin))
println("\n\nCovariate balance at the cutoff (predetermined covariates as outcomes):")
cb = rd_covariate_balance(senate, [:presdemvoteshlag1, :demvoteshlag1, :demvoteshlag2,
                                   :dopen], :margin)
println(cb[:, [:covariate, :estimate, :pvalue, :pvalue_holm, :h_left]])
println(DataFrames.metadata(cb, "note"))

println("\nPlacebo cutoffs (control units below 0, treated units above 0):")
println(rd_placebo_cutoffs(senate, :vote, :margin;
                           placebo_cutoffs=[-20, -10, 10, 20])[:, [:cutoff, :estimate,
                                                                   :pvalue, :n_h_left,
                                                                   :n_h_right]])

println("\nDonut-hole estimates (observations with |margin| < r excluded):")
println(rd_donut(senate, :vote, :margin; radii=[0, 0.5, 1, 2])[:, [:radius, :estimate,
                                                                   :ci_lower, :ci_upper]])

println("\nBandwidth sensitivity (h and b scaled around the MSE-optimal choice):")
println(rd_bandwidth_sensitivity(senate, :vote, :margin)[:, [:multiplier, :h_left,
                                                             :estimate, :ci_lower,
                                                             :ci_upper]])

# 5. Local randomization inference in a small window (a different assumption).
println()
lr = rd_randomization_test(senate, :vote, :margin; window=0.75, rng=Xoshiro(2015))
show(stdout, MIME"text/plain"(), lr)
@printf("Difference in means in the window: %.2f (Neyman 95%% CI [%.2f, %.2f])\n",
        lr.details.estimate, lr.details.neyman_ci...)

println("\n", "="^78)
println("Part 2. Fuzzy RD with a known effect (simulated)")
println("="^78)

rng = Xoshiro(1)
n = 3000
x = 2 .* rand(rng, n) .- 1
u = randn(rng, n)
# take-up jumps by 0.4 at the cutoff; take-up is related to the unobservable u
d = Float64.((0.2 .+ 0.4 .* (x .>= 0) .+ 0.1 .* u) .> rand(rng, n))
y = 1 .+ 0.8 .* x .- 0.4 .* x .^ 2 .+ 2.0 .* d .+ 0.5 .* u .+ 0.3 .* randn(rng, n)
sim = DataFrame(y=y, x=x, d=d)

rf = rd_estimate(sim, :y, :x; treatment=:d)
show(stdout, MIME"text/plain"(), rf)
println()
cs = rd_weak_iv_confidence_set(rf)
println("True effect: 2.0")
println("Weak-identification-robust (Anderson–Rubin type) 95% confidence set: ",
        cs.kind, " ", cs.intervals)
@printf("First-stage robust z statistic: %.2f\n", cs.first_stage_z)

println("\n", "="^78)
println("Part 3. Honest (bias-aware) inference under a bound on the second derivative")
println("="^78)

# The honest interval is valid for every regression function with |f''| <= M on each
# side of the cutoff. M must be chosen a priori; the default is the rule of thumb of
# Armstrong & Kolesár (2020), so report a range of M values.
rh = rd_honest(senate, :vote, :margin)
show(stdout, MIME"text/plain"(), rh)
println("\nSensitivity of the honest 95% CI to M (FLCI-optimal bandwidth):")
for M in (0.02, 0.05, 0.1, 0.2)
    rh_m = rd_honest(senate, :vote, :margin; M=M, opt_criterion=:flci)
    @printf("  M = %4.2f: estimate %6.2f, h = %5.2f, max bias %5.2f, CI [%6.2f, %6.2f]\n",
            M, rh_m.estimate, rh_m.bandwidth, rh_m.max_bias, rh_m.conf_low,
            rh_m.conf_high)
end
# The data can only bound M from below: curvature estimates from blocks of 50 support
# points; a lower confidence bound of 0 means the data do not rule out any M >= 0.
sb = rd_smoothness_bound(senate, :vote, :margin; s=50, rng=Xoshiro(1))
println("\nLower bound on M implied by the data:")
println(sb)

# Fuzzy design: delta-method honest interval and the bias-aware Anderson–Rubin set,
# which stays valid when the first stage is weak (Noack & Rothe 2024).
rhf = rd_honest(sim, :y, :x; treatment=:d, M=(1.0, 0.5))
@printf("\nFuzzy honest estimate %.3f, 95%% CI [%.3f, %.3f] (true effect 2.0)\n",
        rhf.estimate, rhf.conf_low, rhf.conf_high)
ar = rd_honest_ar_confidence_set(rhf)
println("Bias-aware Anderson–Rubin 95% confidence set: ", ar.kind, " ", ar.intervals)
