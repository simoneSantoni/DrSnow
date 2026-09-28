# Monte Carlo reference for full forest fits with the R package grf (2.6.1).
#
# Forests are random, so DrSnow and grf cannot agree fit by fit. Both implement the
# same algorithm, so every summary has the same distribution over forest seeds. For
# S seeds (spaced far apart: grf's trees use seeds seed + i, so nearby seeds give
# nearly identical forests) this script records, on grf_forest_data.csv, the
# average effects and their standard errors, calibration and best linear
# projection coefficients, variable importance, and CATE / regression predictions
# with variance estimates at the points of grf_forest_test.csv. The DrSnow tests
# compare their fits with the across-seed means and standard deviations stored in
# grf_forest_reference.csv.
#
#   R_LIBS_USER=<lib> Rscript test/validation/ml/grf_forest_reference.R

suppressPackageStartupMessages(library(grf))

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("--file=", "", args[grep("--file=", args)])
dir <- if (length(file_arg) == 1) dirname(normalizePath(file_arg)) else "."

d <- read.csv(file.path(dir, "grf_forest_data.csv"))
tst <- read.csv(file.path(dir, "grf_forest_test.csv"))
xs <- paste0("x", 1:6)
X <- as.matrix(d[, xs])
Xt <- as.matrix(tst[, xs])
S <- 20
rows <- list()
put <- function(quantity, s, value) {
  rows[[length(rows) + 1]] <<- data.frame(quantity = quantity, seed = s,
                                          index = seq_along(value),
                                          value = as.numeric(value))
}
for (s in 1:S) {
  seed <- s * 1000003
  cf <- causal_forest(X, d$y, d$w, num.trees = 2000, seed = seed)
  for (ts in c("all", "treated", "overlap")) {
    a <- average_treatment_effect(cf, target.sample = ts)
    put(paste0("ate_", ts, "_est"), s, a[["estimate"]])
    put(paste0("ate_", ts, "_se"), s, a[["std.err"]])
  }
  cal <- test_calibration(cf)
  put("cal_coef", s, cal[, 1])
  put("cal_se", s, cal[, 2])
  blp <- best_linear_projection(cf, X[, 1:2])
  put("blp_coef", s, blp[, 1])
  put("blp_se", s, blp[, 2])
  put("varimp", s, variable_importance(cf))
  pr <- predict(cf, Xt, estimate.variance = TRUE)
  put("cate_test", s, pr$predictions)
  put("cate_test_var", s, pr$variance.estimates)
  oob <- predict(cf, estimate.variance = TRUE)
  put("cate_oob_mean", s, mean(oob$predictions))
  put("cate_oob_var_mean", s, mean(oob$variance.estimates))
  put("Y_hat_mean_abs_resid", s, mean(abs(d$y - cf$Y.hat)))
  put("W_hat_mean_abs_resid", s, mean(abs(d$w - cf$W.hat)))
  rate <- rank_average_treatment_effect(cf, oob$predictions, R = 100)
  put("rate_autoc", s, rate$estimate)
  rf <- regression_forest(X, d$y, num.trees = 2000, seed = seed)
  rp <- predict(rf, Xt, estimate.variance = TRUE)
  put("reg_test", s, rp$predictions)
  put("reg_test_var", s, rp$variance.estimates)
  ivf <- instrumental_forest(X, d$yiv, d$d, d$z, num.trees = 2000, seed = seed)
  ip <- predict(ivf, Xt, estimate.variance = TRUE)
  put("iv_test", s, ip$predictions)
  put("iv_test_var", s, ip$variance.estimates)
  la <- average_treatment_effect(ivf)
  put("late_est", s, la[["estimate"]])
  put("late_se", s, la[["std.err"]])
  cat("seed", s, "done\n")
}
res <- do.call(rbind, rows)
agg <- do.call(rbind, lapply(split(res, list(res$quantity, res$index), drop = TRUE),
  function(g) data.frame(quantity = g$quantity[1], index = g$index[1],
                         mean = mean(g$value), sd = sd(g$value), S = nrow(g))))
agg <- agg[order(agg$quantity, agg$index), ]
write.csv(agg, file.path(dir, "grf_forest_reference.csv"), row.names = FALSE)
print(sessionInfo())
