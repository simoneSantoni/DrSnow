# Reference values for DrSnow's experiment analysis, greedy matching and simulation
# diagnosands (src/design).
#
# Regenerate from the repository root with
#   R_LIBS_USER=<lib with estimatr, blockTools, DeclareDesign, randomizr> \
#     Rscript test/validation/design/make_reference_experiments.R
# Writes, in test/validation/design/:
#   experiment_data.csv          simulated experiment (complete, blocked, paired arms)
#   reference_estimatr.csv       estimatr estimates (case, estimate, std_error, df)
#   reference_blocktools.csv     blockTools optGreedy Mahalanobis pairs (id, pair)
#   reference_declaredesign.csv  DeclareDesign diagnosands with bootstrap SEs
#   reference_experiments_versions.txt

suppressPackageStartupMessages({
  library(estimatr); library(blockTools); library(DeclareDesign); library(randomizr)
})
dir <- "test/validation/design"

# --- data -------------------------------------------------------------------------
set.seed(20260928)
N <- 60
d <- data.frame(id = sprintf("u%02d", 1:N), x1 = rnorm(N), x2 = rnorm(N),
                block = rep(1:10, each = 6), pair = rep(1:30, each = 2))
d$z <- complete_ra(N, m = 30)
d$zb <- block_ra(blocks = d$block, block_m = rep(3, 10))
d$zp <- block_ra(blocks = d$pair, block_m = rep(1, 30))
base <- 1 + d$x1 + 0.5 * d$x2 + 0.3 * d$block / 10 + rnorm(N)
d$y <- base + 0.4 * d$z
d$yb <- base + 0.4 * d$zb
d$yp <- base + 0.4 * d$zp
write.csv(d, file.path(dir, "experiment_data.csv"), row.names = FALSE)

# --- estimatr -----------------------------------------------------------------------
rows <- list()
add <- function(case, fit, term) {
  tt <- tidy(fit)
  r <- tt[tt$term == term, ]
  rows[[length(rows) + 1]] <<- data.frame(case = case,
    estimate = sprintf("%.17g", r$estimate), std_error = sprintf("%.17g", r$std.error),
    df = sprintf("%.17g", r$df))
}
add("dim", difference_in_means(y ~ z, data = d), "z")
add("dim_blocked", difference_in_means(yb ~ zb, blocks = block, data = d), "zb")
add("dim_pairs", difference_in_means(yp ~ zp, blocks = pair, data = d), "zp")
add("lin", lm_lin(y ~ z, covariates = ~ x1 + x2, data = d, se_type = "HC1"), "z")
add("block_fe", lm_robust(yb ~ zb + x1, fixed_effects = ~ block, data = d,
                          se_type = "HC1"), "zb")
add("lin_hc1_noblock_zb", lm_lin(yb ~ zb, covariates = ~ x1, data = d,
                                 se_type = "HC1"), "zb")
write.csv(do.call(rbind, rows), file.path(dir, "reference_estimatr.csv"),
          row.names = FALSE, quote = FALSE)

# --- blockTools greedy pairs ------------------------------------------------------
b <- block(d[, c("id", "x1", "x2")], n.tr = 2, id.vars = "id",
           block.vars = c("x1", "x2"), algorithm = "optGreedy",
           distance = "mahalanobis")
bt <- b$blocks[[1]]
pairs <- data.frame(id = c(as.character(bt[, 1]), as.character(bt[, 2])),
                    pair = rep(seq_len(nrow(bt)), 2))
pairs <- pairs[!is.na(pairs$id), ]
write.csv(pairs, file.path(dir, "reference_blocktools.csv"), row.names = FALSE,
          quote = FALSE)

# --- DeclareDesign diagnosands --------------------------------------------------------
diags <- declare_diagnosands(
  mean_estimand = mean(estimand), mean_estimate = mean(estimate),
  bias = mean(estimate - estimand), sd_estimate = sd(estimate),
  rmse = sqrt(mean((estimate - estimand)^2)), power = mean(p.value <= 0.05),
  coverage = mean(estimand <= conf.high & estimand >= conf.low),
  mean_se = mean(std.error),
  type_s_rate = mean((sign(estimate) != sign(estimand))[p.value <= 0.05]),
  exaggeration_ratio = mean((estimate / estimand)[p.value <= 0.05]))
mk <- function(n, effect) {
  declare_model(N = n, U = rnorm(N), potential_outcomes(Y ~ effect * Z + U)) +
    declare_inquiry(ATE = mean(Y_Z_1 - Y_Z_0)) +
    declare_assignment(Z = complete_ra(N, m = N / 2)) +
    declare_measurement(Y = reveal_outcomes(Y ~ Z)) +
    declare_estimator(Y ~ Z, .method = difference_in_means, inquiry = "ATE")
}
out <- list()
set.seed(42)
for (cfg in list(c(100, 0.3), c(40, 0.2))) {
  dx <- diagnose_design(mk(cfg[1], cfg[2]), diagnosands = diags, sims = 20000,
                        bootstrap_sims = 200)
  dd <- dx$diagnosands_df
  for (nm in c("mean_estimand", "mean_estimate", "bias", "sd_estimate", "rmse", "power",
               "coverage", "mean_se", "type_s_rate", "exaggeration_ratio")) {
    out[[length(out) + 1]] <- data.frame(n = cfg[1], effect = cfg[2], diagnosand = nm,
      value = sprintf("%.17g", dd[[nm]]),
      mc_se = sprintf("%.17g", dd[[paste0("se(", nm, ")")]]), sims = 20000)
  }
}
write.csv(do.call(rbind, out), file.path(dir, "reference_declaredesign.csv"),
          row.names = FALSE, quote = FALSE)
writeLines(c(paste("R", getRversion()),
             paste("estimatr", packageVersion("estimatr")),
             paste("blockTools", packageVersion("blockTools")),
             paste("DeclareDesign", packageVersion("DeclareDesign")),
             paste("randomizr", packageVersion("randomizr"))),
           file.path(dir, "reference_experiments_versions.txt"))
