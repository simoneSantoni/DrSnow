# Reference values for DrSnow's generalized random forest post-estimation from the
# R package grf (2.6.1).
#
# grf's average_treatment_effect, best_linear_projection, test_calibration,
# get_scores and rank_average_treatment_effect are deterministic functions of the
# data and of the forest's nuisance estimates (Y.hat, W.hat, Z.hat) and out-of-bag
# predictions. This script builds grf forest objects on grf_data.csv (written by
# make_grf_data.jl), replaces their nuisance estimates and predictions by the
# DrSnow values stored in the same file, and evaluates grf's functions, so DrSnow
# must reproduce the results to numerical precision (RATE standard errors come from
# a random bootstrap and are not compared).
#
#   R_LIBS_USER=<lib> Rscript test/validation/ml/grf_reference.R
#
# Writes grf_reference.csv (columns case, quantity, value) next to this script.

suppressPackageStartupMessages(library(grf))

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("--file=", "", args[grep("--file=", args)])
dir <- if (length(file_arg) == 1) dirname(normalizePath(file_arg)) else "."

d <- read.csv(file.path(dir, "grf_data.csv"))
X <- as.matrix(d[, paste0("x", 1:5)])
n <- nrow(d)
out <- list()
put <- function(case, quantity, value) {
  out[[length(out) + 1]] <<- data.frame(case = case, quantity = quantity,
                                        value = as.numeric(value))
}

set_preds <- function(f, tau) {
  f$predictions <- matrix(tau, ncol = 1)
  f
}

variants <- list(
  plain = list(clusters = NULL, sw = NULL, eq = FALSE),
  cluster = list(clusters = d$cl, sw = NULL, eq = FALSE),
  weights = list(clusters = NULL, sw = d$sw, eq = FALSE),
  equalize = list(clusters = d$cl, sw = NULL, eq = TRUE)
)

for (v in names(variants)) {
  a <- variants[[v]]
  cf <- causal_forest(X, d$y, d$w, Y.hat = d$y_hat, W.hat = d$w_hat, num.trees = 50,
                      clusters = a$clusters, sample.weights = a$sw,
                      equalize.cluster.weights = a$eq, seed = 1)
  cf <- set_preds(cf, d$tau)
  for (ts in c("all", "treated", "control", "overlap")) {
    ate <- suppressWarnings(average_treatment_effect(cf, target.sample = ts))
    put(v, paste0("ate_", ts, "_est"), ate[["estimate"]])
    put(v, paste0("ate_", ts, "_se"), ate[["std.err"]])
  }
  sub <- d$x2 > 0.3
  ate <- suppressWarnings(average_treatment_effect(cf, subset = sub))
  put(v, "ate_subset_est", ate[["estimate"]])
  put(v, "ate_subset_se", ate[["std.err"]])
  blp <- suppressWarnings(best_linear_projection(cf, X[, 1:2]))
  put(v, "blp_coef", blp[, 1])
  put(v, "blp_se", blp[, 2])
  blp1 <- suppressWarnings(best_linear_projection(cf, X[, 1:2], vcov.type = "HC1"))
  put(v, "blp_hc1_se", blp1[, 2])
  blpo <- best_linear_projection(cf, X[, 1], target.sample = "overlap")
  put(v, "blp_overlap_coef", blpo[, 1])
  put(v, "blp_overlap_se", blpo[, 2])
  cal <- test_calibration(cf)
  put(v, "cal_coef", cal[, 1])
  put(v, "cal_se", cal[, 2])
  put(v, "cal_t", cal[, 3])
  put(v, "cal_p", cal[, 4])
  put(v, "scores", get_scores(cf))
  for (tg in c("AUTOC", "QINI")) {
    rate <- suppressWarnings(rank_average_treatment_effect(
      cf, cbind(d$tau, d$prio2), target = tg, R = 2))
    put(v, paste0("rate_", tg), rate$estimate)
    put(v, paste0("toc_", tg), rate$TOC$estimate)
    rate2 <- suppressWarnings(rank_average_treatment_effect(
      cf, d$prio2, target = tg, q = c(0.05, 0.25, 0.33, 0.5, 0.9, 1), R = 2,
      subset = which(d$x4 > 0.2)))
    put(v, paste0("rate_ties_", tg), rate2$estimate)
    put(v, paste0("toc_ties_", tg), rate2$TOC$estimate)
  }
}

# continuous treatment, with supplied debiasing weights
cc <- causal_forest(X, d$yc, d$wc, Y.hat = d$yc_hat, W.hat = d$wc_hat, num.trees = 50,
                    seed = 1)
cc <- set_preds(cc, d$tauc)
ate <- average_treatment_effect(cc, debiasing.weights = d$gammac)
put("continuous", "ate_all_est", ate[["estimate"]])
put("continuous", "ate_all_se", ate[["std.err"]])
ato <- average_treatment_effect(cc, target.sample = "overlap")
put("continuous", "ate_overlap_est", ato[["estimate"]])
put("continuous", "ate_overlap_se", ato[["std.err"]])
blp <- best_linear_projection(cc, X[, 1], debiasing.weights = d$gammac)
put("continuous", "blp_coef", blp[, 1])
put("continuous", "blp_se", blp[, 2])

# instrumental forest with supplied compliance scores
ivf <- instrumental_forest(X, d$yiv, d$d, d$z, Y.hat = d$yiv_hat, W.hat = d$d_hat,
                           Z.hat = d$z_hat, num.trees = 50, seed = 1)
ivf <- set_preds(ivf, d$tauiv)
ate <- average_treatment_effect(ivf, compliance.score = d$compliance)
put("iv", "ate_all_est", ate[["estimate"]])
put("iv", "ate_all_se", ate[["std.err"]])
blp <- best_linear_projection(ivf, X[, 1:2], compliance.score = d$compliance)
put("iv", "blp_coef", blp[, 1])
put("iv", "blp_se", blp[, 2])
put("iv", "scores", get_scores(ivf, compliance.score = d$compliance))

# regression forest calibration
rf <- regression_forest(X, d$y, num.trees = 50, seed = 1)
rf <- set_preds(rf, d$y_hat)
cal <- test_calibration(rf)
put("regression", "cal_coef", cal[, 1])
put("regression", "cal_se", cal[, 2])
put("regression", "cal_p", cal[, 4])

res <- do.call(rbind, out)
write.csv(res, file.path(dir, "grf_reference.csv"), row.names = FALSE)
print(sessionInfo())
