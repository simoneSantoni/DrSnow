# Reference values for the residual-prediction specification tests
# (Scheidegger, Londschien & Bühlmann 2025), computed with the authors' R package
# RPIV (CRAN, version 1.1.1).
#
# Regenerate (from the repository root) with
#   Rscript test/validation/iv/make_reference_rpiv.R
# Writes rpiv.csv (simulated data, one column per case with the auxiliary-sample
# indicator and the learned predictions) and reference_rpiv.csv (columns case,
# quantity, value), read by test/iv/test_ml_spec_test.jl.
#
# RPIV's random forest (ranger, with OOB-error tuning) is not reimplemented in
# DrSnow. The script therefore replays each RPIV call step by step with the package
# internals under the same seed, checks that the replay reproduces the exported
# functions RPIV_test / weak_RPIV_test exactly, and exports the sample split and the
# forest predictions (OOB on the auxiliary sample, as RPIV uses them for clipping;
# out-of-sample on the main sample). The Julia test feeds those predictions to
# DrSnow through a fixed-prediction learner and compares the test statistics.

suppressPackageStartupMessages(library(RPIV))
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
stopifnot(packageVersion("RPIV") >= "1.1.1")

set.seed(20260928)
n <- 400
G <- 40
cl <- sample(1:G, n, replace = TRUE)
Z <- cbind(rnorm(n) + 0.5 * rnorm(G)[cl], rnorm(n))
C <- rnorm(n)
H <- rnorm(n)
X <- Z[, 1] + 0.5 * Z[, 2] + 0.3 * C + H + rnorm(n)
eps <- rnorm(n) * (0.5 + abs(Z[, 1]))
Y_null <- 1 + X - C - H + eps
Y_alt <- Y_null + 0.8 * Z[, 1]^2
dat <- data.frame(y_null = Y_null, y_alt = Y_alt, x = X, c = C, z1 = Z[, 1],
                  z2 = Z[, 2], cl = cl)

out <- list()
add <- function(case, quantity, value) {
  out[[length(out) + 1]] <<- data.frame(case = case, quantity = quantity,
                                        value = as.numeric(value))
}

# ---- strong-identification test (Procedure 2), replayed -------------------------
strong_case <- function(case, y, seed, clustering = NULL, ve) {
  N <- length(y)
  frac_A <- min(0.5, exp(1) / log(N))
  set.seed(seed)
  ref <- RPIV_test(y, X, C, Z, variance_estimator = ve, clustering = clustering)
  set.seed(seed)
  if (!is.null(clustering)) {
    clusters <- unique(clustering)
    clusters_train <- sample(clusters, round(length(clusters) * frac_A))
    train <- which(clustering %in% clusters_train)
  } else {
    train <- sample(1:N, round(N * frac_A))
  }
  Xbar <- cbind(1, X, C); Zbar <- cbind(1, Z, C)
  fs <- lm(Xbar[train, ] ~ -1 + Zbar[train, ])$fitted.values
  b <- lm(y[train] ~ -1 + fs)$coefficients
  res_train <- y[train] - Xbar[train, ] %*% b
  rf <- RPIV:::tune_rf(res_train, Zbar[train, -1], Zbar[-train, -1], list())
  w <- RPIV:::clip_w(rf$pred_train, rf$pred_test, 0.8)
  test <- setdiff(1:N, train)
  fst <- lm(Xbar[test, ] ~ -1 + Zbar[test, ])$fitted.values
  bt <- lm(y[test] ~ -1 + fst)$coefficients
  rt <- y[test] - Xbar[test, ] %*% bt
  ZAw <- -fst %*% solve(t(fst) %*% fst, t(Xbar[test, ]) %*% w)
  res <- list()
  for (v in ve) {
    s2 <- RPIV:::calc_sigmahatw2(v, rt, w, ZAw, clustering[test])
    frac <- s2 / mean(rt^2)
    if (frac < 0.05) s2 <- 0.05 * mean(rt^2)
    res[[v]] <- sum(w * rt) / sqrt(length(w) * s2)
    r_ref <- if (length(ve) == 1) ref else ref[[v]]
    stopifnot(abs(res[[v]] - r_ref$test_statistic) < 1e-10)
    add(case, paste0("T_", v), r_ref$test_statistic)
    add(case, paste0("p_", v), r_ref$p_value)
    add(case, paste0("varfrac_", v), r_ref$var_fraction)
  }
  aux <- rep(0, N); aux[train] <- 1
  pred <- rep(NA_real_, N); pred[train] <- rf$pred_train; pred[test] <- rf$pred_test
  dat[[paste0("aux_", case)]] <<- aux
  dat[[paste0("pred_", case)]] <<- pred
}

strong_case("strong_null", Y_null, 11, NULL, c("heteroskedastic", "homoskedastic"))
strong_case("strong_alt", Y_alt, 12, NULL, c("heteroskedastic", "homoskedastic"))
strong_case("strong_cluster", Y_null, 13, cl, c("cluster", "heteroskedastic"))

# ---- weak-IV-robust test at beta0 (Procedure 3, type = "fit"), replayed ---------
weak_case <- function(case, y, seed, beta0, clustering = NULL, ve) {
  N <- length(y)
  frac_A <- min(0.5, exp(1) / log(N))
  set.seed(seed)
  f <- weak_RPIV_test(y, X, C, Z, variance_estimator = ve, clustering = clustering)
  ref <- f(beta0, "fit")
  set.seed(seed)
  if (!is.null(clustering)) {
    clusters <- unique(clustering)
    clusters_train <- sample(clusters, round(length(clusters) * frac_A))
    train <- which(clustering %in% clusters_train)
  } else {
    train <- sample(1:N, round(N * frac_A))
  }
  test <- setdiff(1:N, train)
  Cb <- cbind(1, C)
  MY <- lm(y[train] ~ -1 + Cb[train, ])$residuals
  MX <- lm(X[train] ~ -1 + Cb[train, ])$residuals
  Zb_tr <- cbind(C[train], Z[train, ]); Zb_te <- cbind(C[test], Z[test, ])
  fs <- lm(MX ~ -1 + Z[train, ])$fitted.values
  bt <- lm(MY ~ -1 + fs)$coefficients
  tuned <- RPIV:::tune_rf(MY - MX * bt, Zb_tr, Ztest = NULL, list())
  resid <- MY - MX * beta0
  rf <- RPIV:::get_rf_predictions_from_tuned(resid, Zb_tr, Zb_te, tuned$par_opt)
  w <- RPIV:::clip_w(rf$pred_train, rf$pred_test, 0.8)
  R <- y[test] - X[test] * beta0
  MR <- lm(R ~ -1 + Cb[test, ])$residuals
  Mw <- lm(w ~ -1 + Cb[test, ])$residuals
  for (v in ve) {
    s2 <- RPIV:::calc_sigmahatw2_weak(v, MR, Mw, clustering[test])
    if (s2 / mean(MR^2) < 0.05) s2 <- 0.05 * mean(MR^2)
    Tm <- sum(MR * Mw) / sqrt(length(test)) / sqrt(s2)
    stopifnot(abs(Tm - ref[[v]]) < 1e-10)
    add(case, paste0("T_", v), ref[[v]])
  }
  aux <- rep(0, N); aux[train] <- 1
  pred <- rep(NA_real_, N); pred[train] <- rf$pred_train; pred[test] <- rf$pred_test
  dat[[paste0("aux_", case)]] <<- aux
  dat[[paste0("pred_", case)]] <<- pred
}

weak_case("weak_b1", Y_null, 21, 1.0, NULL, c("heteroskedastic", "homoskedastic"))
weak_case("weak_b0", Y_null, 22, 0.0, NULL, c("heteroskedastic", "homoskedastic"))
weak_case("weak_alt_b1", Y_alt, 23, 1.0, NULL, c("heteroskedastic"))
weak_case("weak_cluster_b1", Y_null, 24, 1.0, cl, c("cluster", "heteroskedastic"))

write.csv(dat, file.path(here, "rpiv.csv"), row.names = FALSE)
write.csv(do.call(rbind, out), file.path(here, "reference_rpiv.csv"), row.names = FALSE)
cat("RPIV", as.character(packageVersion("RPIV")), "reference values written\n")
