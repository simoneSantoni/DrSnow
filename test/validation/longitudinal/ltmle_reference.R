# Reference values for DrSnow's longitudinal_regime from the R packages ltmle and
# lmtp, on the simulated data of make_data.jl (longitudinal_data.csv).
#
# Outputs:
#   ltmle_reference.csv  one row per (case, estimator): estimate and standard error
#                        (sqrt(var(IC)/n), ltmle's IC-based variance), plus effect
#                        measures (ATE, and RR / OR on the log scale) of the two-regime
#                        calls;
#   lmtp_reference.csv   lmtp_sdr / lmtp_tmle estimates and standard errors without
#                        cross-fitting (folds = 1) and with glm learners;
#   reference_versions.txt
#
# Run from this directory:  Rscript ltmle_reference.R

suppressPackageStartupMessages({ library(ltmle); library(lmtp) })
writeLines(c(paste("R", getRversion()), paste("ltmle", packageVersion("ltmle")),
             paste("lmtp", packageVersion("lmtp")),
             paste("SuperLearner", packageVersion("SuperLearner"))),
           "reference_versions.txt")

long <- read.csv("longitudinal_data.csv")
K <- max(long$t)
wide <- long[long$t == 1, c("id", "w1", "w2")]
for (s in 1:K) {
  sub <- long[long$t == s, c("id", "l1", "l2", "a")]
  names(sub) <- c("id", paste0(c("l1_", "l2_", "a_"), s))
  wide <- merge(wide, sub, by = "id")
}
wide <- merge(wide, long[long$t == 1, c("id", "y_cont", "y_bin")], by = "id")
wide <- wide[order(wide$id), ]
cols <- c("w1", "w2", unlist(lapply(1:K, function(s) paste0(c("l1_", "l2_", "a_"), s))))
Anodes <- paste0("a_", 1:K)
Lnodes <- unlist(lapply(1:K, function(s) paste0(c("l1_", "l2_"), s)))

out <- list()
add <- function(case, estimator, estimate, se) {
  out[[length(out) + 1]] <<- data.frame(case = case, estimator = estimator,
                                        estimate = unname(estimate),
                                        std_error = unname(se))
}
run_single <- function(case, outcome, abar = NULL, rule = NULL, gcomp = FALSE) {
  d <- wide[, c(cols, outcome)]
  fit <- suppressMessages(ltmle(d, Anodes = Anodes, Lnodes = Lnodes, Ynodes = outcome,
                                abar = abar, rule = rule, gcomp = gcomp,
                                variance.method = "ic", estimate.time = FALSE))
  est <- if (gcomp) "gcomp" else "tmle"
  s <- summary(fit, estimator = est)
  add(case, est, s$treatment$estimate, s$treatment$std.dev)
  s2 <- summary(fit, estimator = "iptw")
  add(case, "iptw", s2$treatment$estimate, s2$treatment$std.dev)
  invisible(fit)
}
run_pair <- function(case, outcome) {
  d <- wide[, c(cols, outcome)]
  fit <- suppressMessages(ltmle(d, Anodes = Anodes, Lnodes = Lnodes, Ynodes = outcome,
                                abar = list(rep(1, K), rep(0, K)),
                                variance.method = "ic", estimate.time = FALSE))
  for (est in c("tmle", "iptw")) {
    s <- summary(fit, estimator = est)
    em <- s$effect.measures
    add(case, est, em$treatment$estimate, em$treatment$std.dev)
    add(paste0(case, "_control"), est, em$control$estimate, em$control$std.dev)
    add(paste0(case, "_ATE"), est, em$ATE$estimate, em$ATE$std.dev)
    if (!is.null(em$RR)) {
      add(paste0(case, "_logRR"), est, log(em$RR$estimate), em$RR$std.dev)
      add(paste0(case, "_logOR"), est, log(em$OR$estimate), em$OR$std.dev)
    }
  }
}

run_single("cont_always", "y_cont", abar = rep(1, K))
run_single("cont_never", "y_cont", abar = rep(0, K))
run_single("bin_always", "y_bin", abar = rep(1, K))
run_single("bin_never", "y_bin", abar = rep(0, K))
run_single("cont_treat_once", "y_cont", abar = c(1, rep(0, K - 1)))
run_single("cont_always_gcomp", "y_cont", abar = rep(1, K), gcomp = TRUE)
run_single("bin_always_gcomp", "y_bin", abar = rep(1, K), gcomp = TRUE)
# dynamic regime: treat in period t when l1_t > 0
rule_l1 <- function(row) as.numeric(c(row["l1_1"] > 0, row["l1_2"] > 0, row["l1_3"] > 0))
run_single("cont_dynamic", "y_cont", rule = rule_l1)
run_single("bin_dynamic", "y_bin", rule = rule_l1)
run_pair("cont_pair", "y_cont")
run_pair("bin_pair", "y_bin")

res <- do.call(rbind, out)
write.csv(res, "ltmle_reference.csv", row.names = FALSE)
print(res)

# ---------------------------------------------------------------------------- lmtp
# lmtp estimates density ratios by the classification trick (observed versus
# regime-shifted treatment stacked), so its weights differ from the propensity-based
# ones; the comparison is a cross-check, not an exact-match reference.
lout <- list()
ladd <- function(case, estimator, fit) {
  lout[[length(lout) + 1]] <<- data.frame(case = case, estimator = estimator,
                                          estimate = fit$estimate@x,
                                          std_error = fit$estimate@std_error)
}
tv <- lapply(1:K, function(s) paste0(c("l1_", "l2_"), s))
ctrl <- lmtp_control(.trim = 1)
for (oc in c("y_cont", "y_bin")) {
  d <- wide[, c(cols, oc)]
  type <- if (oc == "y_bin") "binomial" else "continuous"
  for (sh in c("on", "off")) {
    shift <- if (sh == "on") static_binary_on else static_binary_off
    case <- paste0(ifelse(oc == "y_bin", "bin_", "cont_"),
                   ifelse(sh == "on", "always", "never"))
    f1 <- lmtp_sdr(d, trt = Anodes, outcome = oc, baseline = c("w1", "w2"),
                   time_vary = tv, shift = shift, outcome_type = type, folds = 1,
                   mtp = FALSE, learners_outcome = "SL.glm", learners_trt = "SL.glm",
                   control = ctrl)
    ladd(case, "sdr", f1)
    f2 <- lmtp_tmle(d, trt = Anodes, outcome = oc, baseline = c("w1", "w2"),
                    time_vary = tv, shift = shift, outcome_type = type, folds = 1,
                    mtp = FALSE, learners_outcome = "SL.glm", learners_trt = "SL.glm",
                    control = ctrl)
    ladd(case, "tmle", f2)
  }
}
lres <- do.call(rbind, lout)
write.csv(lres, "lmtp_reference.csv", row.names = FALSE)
print(lres)
