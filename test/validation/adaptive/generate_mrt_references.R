# Reference values for DrSnow's wcls / emee from the R package MRTAnalysis.
#
# Datasets:
#   mrt_heartsteps.csv  MRTAnalysis::data_mimicHeartSteps (37 participants, 7770 rows)
#   mrt_binary.csv      MRTAnalysis::data_binary (100 participants, 3000 rows)
#   mrt_sim.csv         simulated continuous MRT with 120 participants (> 50, so
#                       MRTAnalysis applies no small-sample correction in wcls)
# Output: mrt_reference.csv with one row per (case, term): estimate, std error, df.
#
# Run from this directory:  Rscript generate_mrt_references.R

suppressPackageStartupMessages(library(MRTAnalysis))
writeLines(c(paste("R", getRversion()),
             paste("MRTAnalysis", packageVersion("MRTAnalysis"))),
           "mrt_versions.txt")

hs <- data_mimicHeartSteps
write.csv(hs, "mrt_heartsteps.csv", row.names = FALSE)
bin <- data_binary
write.csv(bin, "mrt_binary.csv", row.names = FALSE)

set.seed(42)
n <- 120; Tn <- 30
sim <- expand.grid(dp = 1:Tn, id = 1:n)[, c("id", "dp")]
sim$x <- rnorm(nrow(sim))
sim$avail <- rbinom(nrow(sim), 1, 0.8)
sim$prob <- ifelse(sim$x > 0, 0.7, 0.4)
sim$a <- rbinom(nrow(sim), 1, sim$prob)
sim$y <- 1 + 0.5 * sim$x + sim$a * (0.3 - 0.2 * sim$dp / Tn) + rnorm(nrow(sim))
sim$ptilde <- 0.55
write.csv(sim, "mrt_sim.csv", row.names = FALSE)

out <- list()
add <- function(case, est, se, df) {
  out[[length(out) + 1]] <<- data.frame(case = case, term = names(est),
                                        estimate = unname(est),
                                        std_error = unname(se), df = df)
}
wcls_rows <- function(case, fit) {
  s <- summary(fit)$causal_excursion_effect
  add(case, setNames(s[, "Estimate"], rownames(s)),
      setNames(s[, "StdErr"], rownames(s)), s[1, "df2"])
}
emee_rows <- function(case, fit) {
  s <- summary(fit)$causal_excursion_effect
  add(case, setNames(s[, "Estimate"], rownames(s)),
      setNames(s[, "StdErr"], rownames(s)), s[1, "df"])
}

# WCLS, HeartSteps mimic (37 ids: small-sample correction applied)
wcls_rows("wcls_hs_marginal", wcls(hs, id = "userid", outcome = "logstep_30min",
  treatment = "intervention", rand_prob = 0.6, moderator_formula = ~1,
  control_formula = ~1, availability = "avail", verbose = FALSE))
wcls_rows("wcls_hs_moderated", wcls(hs, id = "userid", outcome = "logstep_30min",
  treatment = "intervention", rand_prob = 0.6, moderator_formula = ~logstep_pre30min,
  control_formula = ~logstep_pre30min + logstep_30min_lag1 + is_at_home_or_work,
  availability = "avail", verbose = FALSE))
wcls_rows("wcls_hs_numerator", wcls(hs, id = "userid", outcome = "logstep_30min",
  treatment = "intervention", rand_prob = "rand_prob",
  moderator_formula = ~is_at_home_or_work,
  control_formula = ~is_at_home_or_work + logstep_pre30min,
  availability = "avail", numerator_prob = 0.5, verbose = FALSE))
# WCLS, simulated (120 ids: no small-sample correction in MRTAnalysis)
wcls_rows("wcls_sim", wcls(sim, id = "id", outcome = "y", treatment = "a",
  rand_prob = "prob", moderator_formula = ~dp, control_formula = ~dp + x,
  availability = "avail", numerator_prob = "ptilde", verbose = FALSE))

# EMEE, binary data
emee_rows("emee_bin_marginal", emee(bin, id = "userid", outcome = "Y",
  treatment = "A", rand_prob = "rand_prob", moderator_formula = ~1,
  control_formula = ~1, availability = "avail", verbose = FALSE))
emee_rows("emee_bin_moderated", emee(bin, id = "userid", outcome = "Y",
  treatment = "A", rand_prob = "rand_prob", moderator_formula = ~time_var1,
  control_formula = ~time_var1 + time_var2, availability = "avail",
  verbose = FALSE))
emee_rows("emee_bin_numerator", emee(bin, id = "userid", outcome = "Y",
  treatment = "A", rand_prob = "rand_prob", moderator_formula = ~time_var2,
  control_formula = ~time_var1 + time_var2, availability = "avail",
  numerator_prob = 0.4, verbose = FALSE))

res <- do.call(rbind, out)
write.csv(res, "mrt_reference.csv", row.names = FALSE)   # 15 significant digits
print(res)
