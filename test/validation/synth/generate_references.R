# Reference values for DrSnow's synthetic-control area.
#
# Regenerate (from the repository root) with
#     R_LIBS_USER=<lib> Rscript test/validation/synth/generate_references.R
# Packages used (versions recorded in reference_versions.csv):
#   synthdid (synth-inference/synthdid), Synth (CRAN), augsynth (ebenmichael/augsynth),
#   MCPanel (susanathey/MCPanel).
#
# Every file written here is read by test/synth/test_reference.jl.

suppressMessages({
    library(synthdid)
    library(Synth)
    library(augsynth)
    library(MCPanel)
})

args <- commandArgs(trailingOnly = FALSE)
script <- sub("--file=", "", args[grep("--file=", args)])
outdir <- getOption("drsnow.outdir",
                  if (length(script) == 1) dirname(normalizePath(script)) else ".")
out <- function(df, name) write.csv(df, file.path(outdir, name), row.names = FALSE)
# Sections can be run separately (e.g. in parallel): options(drsnow.sections = 2).
run <- function(k) k %in% getOption("drsnow.sections", 1:6)

versions <- data.frame(
    package = c("R", "synthdid", "Synth", "augsynth", "MCPanel"),
    version = c(R.version.string, as.character(packageVersion("synthdid")),
                as.character(packageVersion("Synth")),
                as.character(packageVersion("augsynth")),
                as.character(packageVersion("MCPanel"))))
out(versions, "reference_versions.csv")

# ---------------------------------------------------------------------------------------
# 1. California Proposition 99 (Abadie, Diamond & Hainmueller 2010; synthdid data set)
# ---------------------------------------------------------------------------------------
data("california_prop99")
prop99 <- california_prop99
names(prop99) <- c("State", "Year", "PacksPerCapita", "treated")
out(prop99, "prop99.csv")

setup <- panel.matrices(california_prop99)
Y <- setup$Y; N0 <- setup$N0; T0 <- setup$T0

weights_long <- function(ests, Y, N0, T0) {
    do.call(rbind, lapply(names(ests), function(nm) {
        w <- attr(ests[[nm]], "weights")
        rbind(data.frame(estimator = nm, type = "omega", name = rownames(Y)[1:N0],
                         weight = w$omega),
              data.frame(estimator = nm, type = "lambda", name = colnames(Y)[1:T0],
                         weight = w$lambda))
    }))
}

# Replicate estimates exactly as synthdid:::placebo_se / bootstrap_sample compute
# them (used for draw-by-draw checks and for parallel standard errors).
placebo_theta <- function(est, ind) {
    setup <- attr(est, "setup"); opts <- attr(est, "opts"); w <- attr(est, "weights")
    N1 <- nrow(setup$Y) - setup$N0
    n0 <- length(ind) - N1
    wb <- w
    wb$omega <- synthdid:::sum_normalize(w$omega[ind[1:n0]])
    c(do.call(synthdid_estimate, c(list(Y = setup$Y[ind, ], N0 = n0, T0 = setup$T0,
                                        X = setup$X[ind, , ], weights = wb), opts)))
}
boot_theta <- function(est, ind) {
    setup <- attr(est, "setup"); opts <- attr(est, "opts"); w <- attr(est, "weights")
    if (all(ind <= setup$N0) || all(ind > setup$N0)) return(NA)
    wb <- w
    wb$omega <- synthdid:::sum_normalize(w$omega[sort(ind[ind <= setup$N0])])
    c(do.call(synthdid_estimate, c(list(Y = setup$Y[sort(ind), ],
                                        N0 = sum(ind <= setup$N0), T0 = setup$T0,
                                        X = setup$X[sort(ind), , ], weights = wb), opts)))
}
rep_se <- function(draws) sqrt((length(draws) - 1) / length(draws)) * sd(draws)

if (run(1)) {
ests <- list(sdid = synthdid_estimate(Y, N0, T0),
             sc = sc_estimate(Y, N0, T0),
             did = did_estimate(Y, N0, T0))

# Point estimates, regularisation, and placebo SEs (package default 200 draws and a
# larger-B version whose Monte Carlo error is ~2%).
point <- do.call(rbind, lapply(names(ests), function(nm) {
    e <- ests[[nm]]
    set.seed(12345)
    se200 <- sqrt(vcov(e, method = "placebo", replications = 200))
    set.seed(54321)
    se1000 <- sqrt(vcov(e, method = "placebo", replications = 1000))
    opts <- attr(e, "opts")
    data.frame(estimator = nm, estimate = c(e), se_placebo_200 = c(se200),
               se_placebo_1000 = c(se1000), zeta_omega = opts$zeta.omega,
               zeta_lambda = opts$zeta.lambda,
               noise_level = sd(apply(Y[1:N0, 1:T0], 1, diff)))
}))
out(point, "sdid_prop99_estimates.csv")

out(weights_long(ests, Y, N0, T0), "sdid_prop99_weights.csv")

curves <- do.call(rbind, lapply(names(ests), function(nm) {
    data.frame(estimator = nm, time = colnames(Y)[(T0 + 1):ncol(Y)],
               effect = c(synthdid_effect_curve(ests[[nm]])))
}))
out(curves, "sdid_prop99_effect_curves.csv")

# Draw-by-draw placebo replications (mirrors synthdid:::placebo_se's theta) so the
# Julia implementation can be checked on identical permutations.
set.seed(777)
draws <- do.call(rbind, lapply(1:8, function(b) {
    ind <- sample(1:N0)
    do.call(rbind, lapply(names(ests), function(nm) {
        data.frame(estimator = nm, draw = b, index = paste(ind, collapse = ";"),
                   theta = placebo_theta(ests[[nm]], ind))
    }))
}))
out(draws, "sdid_prop99_placebo_draws.csv")

}

# ---------------------------------------------------------------------------------------
# 2. Simulated block design with several treated units and two covariates
#    (jackknife and bootstrap need N1 > 1).
# ---------------------------------------------------------------------------------------
set.seed(20260927)
N <- 40; TT <- 30; N1 <- 5; T0s <- 20
units <- sprintf("u%02d", 1:N)
alpha <- rnorm(N, sd = 2); beta <- cumsum(rnorm(TT, sd = 0.5))
f1 <- rnorm(N); g1 <- sin((1:TT) / 4)
f2 <- rnorm(N); g2 <- (1:TT) / TT
x1 <- matrix(rnorm(N * TT), N, TT); x2 <- matrix(runif(N * TT), N, TT)
treat <- matrix(0, N, TT); treat[(N - N1 + 1):N, (T0s + 1):TT] <- 1
f1[(N - N1 + 1):N] <- f1[(N - N1 + 1):N] + 1      # treated units differ in loadings
Ysim <- 10 + outer(alpha, rep(1, TT)) + outer(rep(1, N), beta) + 2 * outer(f1, g1) +
    outer(f2, g2) + 0.8 * x1 - 1.5 * x2 + 2 * treat + matrix(rnorm(N * TT, sd = 0.7), N, TT)
sim <- data.frame(unit = rep(units, each = TT), time = rep(1:TT, N),
                  y = c(t(Ysim)), treated = c(t(treat)), x1 = c(t(x1)), x2 = c(t(x2)))
out(sim, "sdid_sim_block.csv")

rownames(Ysim) <- units; colnames(Ysim) <- 1:TT
N0s <- N - N1
Xsim <- array(c(x1, x2), dim = c(N, TT, 2))

if (run(2)) {
sests <- list(sdid = synthdid_estimate(Ysim, N0s, T0s),
              sc = sc_estimate(Ysim, N0s, T0s),
              did = did_estimate(Ysim, N0s, T0s),
              sdid_cov = synthdid_estimate(Ysim, N0s, T0s, X = Xsim))
# Bootstrap and placebo SEs use synthdid's replicate functions (above) in parallel
# with L'Ecuyer streams: the covariate-adjusted fit takes seconds per replicate in R.
RNGkind("L'Ecuyer-CMRG")
cores <- getOption("drsnow.cores", 1)
boot_draw <- function(est) {
    repeat {
        th <- boot_theta(est, sample(1:N, replace = TRUE))
        if (!is.na(th)) return(th)
    }
}
spoint <- do.call(rbind, lapply(names(sests), function(nm) {
    e <- sests[[nm]]
    set.seed(99)
    seb <- rep_se(unlist(parallel::mclapply(1:1000, function(b) boot_draw(e),
                                            mc.cores = cores)))
    set.seed(98)
    sep <- rep_se(unlist(parallel::mclapply(1:1000,
                                            function(b) placebo_theta(e, sample(1:N0s)),
                                            mc.cores = cores)))
    sej <- sqrt(vcov(e, method = "jackknife"))
    beta <- attr(e, "weights")$beta
    data.frame(estimator = nm, estimate = c(e), se_jackknife = c(sej),
               se_bootstrap_1000 = c(seb), se_placebo_1000 = c(sep),
               beta1 = if (is.null(beta)) NA else beta[1],
               beta2 = if (is.null(beta)) NA else beta[2])
}))
out(spoint, "sdid_sim_estimates.csv")
out(weights_long(sests, Ysim, N0s, T0s), "sdid_sim_weights.csv")

RNGkind("default")
set.seed(4242)
bdraws <- do.call(rbind, lapply(1:6, function(b) {
    ind <- sample(1:N, replace = TRUE)
    do.call(rbind, lapply(names(sests), function(nm) {
        data.frame(estimator = nm, draw = b, index = paste(ind, collapse = ";"),
                   theta = boot_theta(sests[[nm]], ind))
    }))
}))
out(bdraws, "sdid_sim_bootstrap_draws.csv")

}

# ---------------------------------------------------------------------------------------
# 3. Simulated staggered design: cohort-wise synthdid on (never-treated + cohort) panels,
#    aggregated with weights proportional to treated unit-periods (Clarke et al. 2023).
# ---------------------------------------------------------------------------------------
if (run(3)) {
stag <- sim[, c("unit", "time", "y")]
adopt <- c(rep(Inf, 32), rep(26, 3), rep(21, 5))
names(adopt) <- units
stag$treated <- as.integer(stag$time >= adopt[stag$unit])
stag$y <- stag$y + 1.5 * stag$treated      # additional effect for the extra cohort
stag$y[stag$unit %in% units[36:40]] <- sim$y[sim$unit %in% units[36:40]]
out(stag, "sdid_sim_staggered.csv")
Ys <- matrix(stag$y, N, TT, byrow = TRUE, dimnames = list(units, 1:TT))
never <- which(!is.finite(adopt))
cohorts <- sort(unique(adopt[is.finite(adopt)]))
coh <- do.call(rbind, lapply(cohorts, function(a) {
    tr <- which(adopt == a)
    Ya <- Ys[c(never, tr), ]
    ea <- synthdid_estimate(Ya, length(never), a - 1)
    data.frame(adoption = a, n_treated = length(tr), n_post = TT - a + 1,
               estimate = c(ea))
}))
coh$weight <- coh$n_treated * coh$n_post / sum(coh$n_treated * coh$n_post)
coh$att <- sum(coh$weight * coh$estimate)
out(coh, "sdid_sim_staggered_cohorts.csv")

}

# ---------------------------------------------------------------------------------------
# 4. Augmented synthetic control (Ben-Michael, Feller & Rothstein 2021) on Prop 99
# ---------------------------------------------------------------------------------------
if (run(4)) {
aug_ridge <- augsynth(PacksPerCapita ~ treated, State, Year, prop99,
                      progfunc = "Ridge", scm = TRUE)
aug_scm <- augsynth(PacksPerCapita ~ treated, State, Year, prop99,
                    progfunc = "None", scm = TRUE)
aug_fe <- augsynth(PacksPerCapita ~ treated, State, Year, prop99,
                   progfunc = "Ridge", scm = TRUE, fixedeff = TRUE)
aug_fixed <- augsynth(PacksPerCapita ~ treated, State, Year, prop99,
                      progfunc = "Ridge", scm = TRUE, lambda = 1e4)
augs <- list(ridge = aug_ridge, scm = aug_scm, ridge_fe = aug_fe, ridge_fixed = aug_fixed)
aug_summ <- do.call(rbind, lapply(names(augs), function(nm) {
    a <- augs[[nm]]
    sj <- summary(a, inf_type = "jackknife")
    data.frame(model = nm, lambda = if (is.null(a$lambda)) NA else a$lambda,
               average_att = sj$average_att$Estimate,
               se_jackknife = sj$average_att$Std.Error)
}))
out(aug_summ, "ascm_prop99_summary.csv")
aug_w <- do.call(rbind, lapply(names(augs), function(nm) {
    data.frame(model = nm, unit = rownames(augs[[nm]]$weights),
               weight = c(augs[[nm]]$weights))
}))
out(aug_w, "ascm_prop99_weights.csv")
aug_att <- do.call(rbind, lapply(names(augs), function(nm) {
    data.frame(model = nm, time = augs[[nm]]$data$time,
               att = predict(augs[[nm]], att = TRUE))
}))
out(aug_att, "ascm_prop99_att.csv")
cv <- data.frame(lambda = aug_ridge$lambdas, error = aug_ridge$lambda_errors,
                 error_se = aug_ridge$lambda_errors_se)
out(cv, "ascm_prop99_cv.csv")

# Conformal inference (Chernozhukov, Wuthrich & Zhu 2021) with moving-block
# permutations, which are deterministic.
conf <- lapply(c("ridge", "scm"), function(nm) {
    s <- summary(augs[[nm]], inf_type = "conformal", type = "block")
    list(per = data.frame(model = nm, time = s$att$Time, estimate = s$att$Estimate,
                          lower = s$att$lower_bound, upper = s$att$upper_bound,
                          p_value = s$att$p_val),
         avg = data.frame(model = nm, estimate = s$average_att$Estimate,
                          p_value = s$average_att$p_val))
})
out(do.call(rbind, lapply(conf, `[[`, "per")), "ascm_prop99_conformal_block.csv")
out(do.call(rbind, lapply(conf, `[[`, "avg")), "ascm_prop99_conformal_block_joint.csv")

}

# ---------------------------------------------------------------------------------------
# 5. Classic ADH synthetic control: Basque Country (Abadie & Gardeazabal 2003)
# ---------------------------------------------------------------------------------------
if (run(5)) {
data("basque")
bq <- basque
out(bq, "basque.csv")
dp <- dataprep(foo = basque,
               predictors = c("school.illit", "school.prim", "school.med",
                              "school.high", "school.post.high", "invest"),
               predictors.op = "mean", time.predictors.prior = 1964:1969,
               special.predictors = list(
                   list("gdpcap", 1960:1969, "mean"),
                   list("sec.agriculture", seq(1961, 1969, 2), "mean"),
                   list("sec.energy", seq(1961, 1969, 2), "mean"),
                   list("sec.industry", seq(1961, 1969, 2), "mean"),
                   list("sec.construction", seq(1961, 1969, 2), "mean"),
                   list("sec.services.venta", seq(1961, 1969, 2), "mean"),
                   list("sec.services.nonventa", seq(1961, 1969, 2), "mean"),
                   list("popdens", 1969, "mean")),
               dependent = "gdpcap", unit.variable = "regionno",
               unit.names.variable = "regionname", time.variable = "year",
               treatment.identifier = 17, controls.identifier = c(2:16, 18),
               time.optimize.ssr = 1960:1969, time.plot = 1955:1997)
so <- synth(dp)
out(data.frame(predictor = rownames(dp$X1), treated = c(dp$X1),
               v = as.numeric(so$solution.v)), "adh_basque_v.csv")
out(data.frame(unit = colnames(dp$X0), w = as.numeric(so$solution.w)),
    "adh_basque_w.csv")
out(data.frame(loss_v = c(so$loss.v), loss_w = c(so$loss.w)), "adh_basque_loss.csv")
gaps <- dp$Y1plot - dp$Y0plot %*% so$solution.w
out(data.frame(year = as.numeric(rownames(dp$Y1plot)), gap = c(gaps)),
    "adh_basque_gaps.csv")
out(data.frame(predictor = rownames(dp$X0), dp$X0, check.names = FALSE),
    "adh_basque_X0.csv")

}

# ---------------------------------------------------------------------------------------
# 6. Matrix completion with nuclear-norm penalty (Athey et al. 2021) on Prop 99
# ---------------------------------------------------------------------------------------
if (run(6)) {
W <- setup$W
mask <- 1 - W
cvfit <- mcnnm_cv(Y, mask, to_estimate_u = 1, to_estimate_v = 1, num_lam_L = 30,
                  niter = 1000, rel_tol = 1e-5, cv_ratio = 0.8, num_folds = 5)
lam <- cvfit$best_lambda
fit <- mcnnm_fit(Y, mask, lambda_L = lam, to_estimate_u = 1, to_estimate_v = 1,
                 niter = 1000, rel_tol = 1e-5)
Yhat <- fit$L + outer(fit$u, rep(1, ncol(Y))) + outer(rep(1, nrow(Y)), fit$v)
att_fit <- mean((Y - Yhat)[W == 1])
fit2 <- mcnnm_fit(Y, mask, lambda_L = lam / 4, to_estimate_u = 1, to_estimate_v = 1,
                  niter = 1000, rel_tol = 1e-5)
Yhat2 <- fit2$L + outer(fit2$u, rep(1, ncol(Y))) + outer(rep(1, nrow(Y)), fit2$v)
out(data.frame(lambda = c(lam, lam / 4),
               att = c(att_fit, mean((Y - Yhat2)[W == 1])),
               rank = c(sum(svd(fit$L)$d > 1e-6), sum(svd(fit2$L)$d > 1e-6))),
    "mcnnm_prop99.csv")
out(data.frame(unit = rep(rownames(Y), ncol(Y)), time = rep(colnames(Y), each = nrow(Y)),
               yhat = c(Yhat)), "mcnnm_prop99_fitted.csv")
}
