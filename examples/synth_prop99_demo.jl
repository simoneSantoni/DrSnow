# Synthetic control methods on California's Proposition 99 (1989 tobacco tax).
#
# Data: annual cigarette packs sold per capita for 39 US states, 1970-2000
# (Abadie, Diamond & Hainmueller 2010; the version distributed with the R package
# synthdid). California is treated from 1989 onward.
#
# Run from the repository root:
#     julia --project=. examples/synth_prop99_demo.jl

using DrSnow
using DataFrames
using Printf
using Random

# Minimal reader for the numeric/quoted-string CSV shipped with the tests (avoids a
# CSV.jl dependency).
function read_prop99(path)
    lines = readlines(path)
    header = Symbol.(strip.(split(lines[1], ','), '"'))
    cols = [String[] for _ in header]
    for l in lines[2:end]
        for (j, f) in enumerate(split(l, ','))
            push!(cols[j], strip(f, '"'))
        end
    end
    return DataFrame(State=cols[1], Year=parse.(Int, cols[2]),
                     PacksPerCapita=parse.(Float64, cols[3]),
                     treated=parse.(Int, cols[4]))
end

df = read_prop99(joinpath(@__DIR__, "..", "test", "validation", "synth", "prop99.csv"))
panel = synth_panel(df, :PacksPerCapita, :treated, :State, :Year)
show(stdout, MIME"text/plain"(), panel)
println()

rng = Xoshiro(2021)

# Synthetic DiD and the SC / DiD estimators from the same machinery (synthdid).
# With a single treated unit, only the placebo variance is available.
results = Tuple{String,CausalEstimate}[
    ("Synthetic DiD", synthetic_did(panel; method=:sdid, rng=rng)),
    ("Synthetic control (synthdid)", synthetic_did(panel; method=:sc, rng=rng)),
    ("Difference-in-differences", synthetic_did(panel; method=:did, rng=rng)),
]

# Augmented synthetic control (ridge) with placebo standard error.
ascm = augmented_synthetic_control(panel; rng=rng)
push!(results, ("Augmented SC (ridge)", ascm))

println("\nEstimated effect on packs per capita, 1989-2000 average")
println(rpad("Method", 32), lpad("Estimate", 10), lpad("Std. err.", 11), "   95% CI")
for (name, r) in results
    lo, hi = confint(r)[1, :]
    @printf("%-32s %9.2f %10.2f   [%.2f, %.2f]\n", name, coef(r)[1], stderror(r)[1],
            lo, hi)
end
println("Standard errors: placebo (treatment reassigned to random control states).")

sdid = results[1][2]
println("\nLargest SDID unit weights:")
w = sort(synth_weights(sdid), :weight; rev=true)
show(stdout, MIME"text/plain"(), first(w, 5))
println("\n\nSDID time weights with positive weight:")
tw = synth_time_weights(sdid)
show(stdout, MIME"text/plain"(), tw[tw.weight .> 0, :])
println()

# Classic synthetic control (outcome-only) with in-space placebo inference.
sc = synthetic_control(panel)
println()
show(stdout, MIME"text/plain"(), sc)
t = synth_in_space_placebo(sc)
println()
show(stdout, MIME"text/plain"(), t)

# Conformal inference for the ASCM effects (moving-block permutations). With 19
# pre-treatment periods the smallest attainable p-value is 1/20, so a 90% level is
# used for the intervals.
ci = synth_conformal_inference(ascm; level=0.90)
println("\nConformal 90% intervals for the ASCM effect by year:")
show(stdout, MIME"text/plain"(), ci.per_period)
println()
