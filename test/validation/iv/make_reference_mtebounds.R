# Reference values for mte_bounds (Mogstad, Santos & Torgovitsky 2018) from the R
# package ivmte (Shea & Torgovitsky), version recorded in the output, solver
# lpSolveAPI.
#
# Regenerate (from the repository root) with
#   Rscript test/validation/iv/make_reference_mtebounds.R
# Writes
#   ae_collapsed.csv      the Angrist & Evans (1998) extract shipped with ivmte
#                         (`AE`, 209,133 women), collapsed to the distinct values of
#                         (worked, morekids, samesex, yob) with frequency `count`
#                         (DrSnow uses `weights = :count`, which reproduces the
#                         unweighted analysis of the full data exactly);
#   ivmte_sim.csv         ivmte's simulated example data (`ivmteSimData`) with
#                         instrument dummies z1, z2, z3;
#   reference_mtebounds.csv   (case, quantity, value): lower / upper bounds and the
#                         minimum criterion.
# ivmte imposes the shape restrictions on its audit grid (u = 0, 1 and the first 25
# points of the base-2 Halton sequence, and the covariate support); the Julia tests
# pass the same u-grid, so the linear programs coincide.

suppressPackageStartupMessages(library(ivmte))
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))

out <- list()
add <- function(case, r) {
  out[[length(out) + 1]] <<- data.frame(case = case,
                                        quantity = c("lower", "upper", "criterion"),
                                        value = c(r$bounds[1], r$bounds[2],
                                                  r$audit.criterion))
}
show <- function(case, r) {
  cat(case, ":", r$bounds, " criterion", r$audit.criterion, "\n")
  add(case, r)
}

ae <- AE[, c("worked", "morekids", "samesex", "yob")]
agg <- aggregate(list(count = rep(1L, nrow(ae))), ae, sum)
write.csv(agg, file.path(here, "ae_collapsed.csv"), row.names = FALSE)

sim <- ivmteSimData
sim$z1 <- as.numeric(sim$z == 1); sim$z2 <- as.numeric(sim$z == 2)
sim$z3 <- as.numeric(sim$z == 3)
write.csv(sim, file.path(here, "ivmte_sim.csv"), row.names = FALSE)

# A: AE, ATT and ATE, linear MTRs in u plus yob, OLS IV-like with interaction
for (tg in c("att", "ate")) {
  show(paste0("ae_linear_", tg), ivmte(data = AE, target = tg,
      m0 = ~ u + yob, m1 = ~ u + yob,
      ivlike = worked ~ morekids + samesex + morekids * samesex,
      propensity = morekids ~ samesex + yob,
      solver = "lpSolveAPI", noisy = FALSE))
}
# B: AE, quadratic splines with monotone MTRs and decreasing MTE, ATT
show("ae_spline_mono", ivmte(data = AE, target = "att",
    m0 = ~ 0 + uSplines(degree = 2, knots = c(1 / 3, 2 / 3)),
    m1 = ~ 0 + uSplines(degree = 2, knots = c(1 / 3, 2 / 3)),
    m1.inc = TRUE, m0.inc = TRUE, mte.dec = TRUE,
    ivlike = worked ~ morekids + samesex + morekids * samesex,
    propensity = morekids ~ samesex,
      solver = "lpSolveAPI", noisy = FALSE))
# C: simulated data, LATE for z: 1 -> 3, cubic MTRs plus x
show("sim_late", ivmte(data = ivmteSimData, target = "late",
    late.from = c(z = 1), late.to = c(z = 3),
    m0 = ~ u + I(u^2) + I(u^3) + x, m1 = ~ u + I(u^2) + I(u^3) + x,
    ivlike = y ~ d + z + d * z, propensity = d ~ z + x,
      solver = "lpSolveAPI", noisy = FALSE))
# D: simulated data, several IV-like specifications (OLS, OLS, IV), linear splines
show("sim_multi", ivmte(data = ivmteSimData, target = "ate",
    ivlike = c(y ~ (z == 1) + (z == 2) + (z == 3) + x, y ~ d + x, y ~ d | z),
    m0 = ~ uSplines(degree = 1, knots = c(.25, .5, .75)) + x,
    m1 = ~ uSplines(degree = 1, knots = c(.25, .5, .75)) + x,
    propensity = d ~ z + x,
      solver = "lpSolveAPI", noisy = FALSE))
# E: simulated data, generalized LATE on [0.2, 0.4]
show("sim_genlate", ivmte(data = ivmteSimData, target = "genlate",
    genlate.lb = .2, genlate.ub = .4,
    m0 = ~ u + I(u^2) + I(u^3) + x, m1 = ~ u + I(u^2) + I(u^3) + x,
    ivlike = y ~ d + z + d * z, propensity = d ~ z + x,
      solver = "lpSolveAPI", noisy = FALSE))
# F: simulated data, constant splines, decreasing MTRs, saturated IV-like, ATE
show("sim_const_dec", ivmte(data = ivmteSimData, target = "ate",
    m0 = ~ uSplines(degree = 0, knots = c(.2, .4, .6, .8)),
    m1 = ~ uSplines(degree = 0, knots = c(.2, .4, .6, .8)),
    m0.dec = TRUE, m1.dec = TRUE,
    ivlike = y ~ d + factor(z) + d * factor(z), propensity = d ~ factor(z),
      solver = "lpSolveAPI", noisy = FALSE))
# G: simulated data, ATU with the probit link and u-varying x (u * x)
show("sim_atu_probit", ivmte(data = ivmteSimData, target = "atu",
    m0 = ~ u + I(u^2) + x + u:x, m1 = ~ u + I(u^2) + x + u:x,
    ivlike = y ~ d + factor(z) + x, propensity = d ~ z + x, link = "probit",
      solver = "lpSolveAPI", noisy = FALSE))

res <- do.call(rbind, out)
res$value <- sprintf("%.15g", res$value)
write.csv(res, file.path(here, "reference_mtebounds.csv"), row.names = FALSE,
          quote = FALSE)
cat("ivmte", as.character(packageVersion("ivmte")), "\n")
