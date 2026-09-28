# Reference values for group-sequential designs and analyses from gsDesign and rpact.
#
# Run (from this directory):  Rscript gs_reference.R
# Writes gs_designs_reference.csv (one row per design x look) and
# gs_analysis_reference.csv (rpact stagewise-ordering inference).
suppressPackageStartupMessages({
  library(gsDesign)
  library(rpact)
})
cat("gsDesign", as.character(packageVersion("gsDesign")),
    "rpact", as.character(packageVersion("rpact")), "\n")

rows <- list()
add <- function(id, source, k, sided, alpha, beta, timing, rule, par, futility,
                fpar, binding, upper, lower, alpha_spent, inflation, en0, en1) {
  for (i in seq_len(k)) {
    rows[[length(rows) + 1]] <<- data.frame(
      id = id, source = source, k = k, sided = sided, alpha = alpha, beta = beta,
      timing = paste(format(timing, digits = 17), collapse = ";"), look = i,
      rule = rule, par = par, futility = futility, fpar = fpar, binding = binding,
      upper = upper[i], lower = lower[i], alpha_spent = alpha_spent[i],
      inflation = inflation, en0 = en0, en1 = en1)
  }
}

sfmap <- list(obf = sfLDOF, pocock = sfLDPocock, power = sfPower, hsd = sfHSD)

id <- 0
# --- gsDesign: efficacy-only spending designs, one- and two-sided ---
for (k in c(3, 5)) for (rule in c("obf", "pocock", "power", "hsd")) {
  for (tim in list("equal", "unequal")) {
    timing <- if (tim == "equal") (1:k) / k else c(0.2, 0.45, 0.7, 0.9, 1)[(6 - k):5]
    if (tim == "unequal" && k == 3) timing <- c(0.3, 0.7, 1)
    par <- switch(rule, obf = 0, pocock = 0, power = 3, hsd = -4)
    for (tt in c(1, 2)) {
      id <- id + 1
      alpha <- 0.025; beta <- 0.1
      g <- tryCatch(gsDesign(k = k, test.type = tt, alpha = alpha, beta = beta,
                             timing = timing, sfu = sfmap[[rule]], sfupar = par,
                             n.fix = 1, tol = 1e-8),
                    error = function(e) tryCatch(
                      gsDesign(k = k, test.type = tt, alpha = alpha, beta = beta,
                               timing = timing, sfu = sfmap[[rule]], sfupar = par,
                               n.fix = 1), error = function(e) NULL))
      if (is.null(g)) {
        cat("gsDesign failed: k =", k, rule, tim, "test.type", tt, "\n")
        next
      }
      spent <- cumsum(g$upper$prob[, 1]) * (if (tt == 2) 2 else 1)
      add(id, "gsDesign", k, tt, if (tt == 2) 2 * alpha else alpha, beta, timing, rule,
          par, "none", 0, FALSE, g$upper$bound,
          if (tt == 2) g$lower$bound else rep(-Inf, k), spent,
          max(g$n.I), g$en[1], g$en[2])
    }
  }
}

# --- gsDesign: classical Wang-Tsiatis boundaries ---
for (k in c(3, 4)) for (rule in c("OF", "Pocock")) {
  id <- id + 1
  g <- gsDesign(k = k, test.type = 1, alpha = 0.025, beta = 0.1, sfu = rule,
                n.fix = 1, tol = 1e-8)
  add(id, "gsDesign", k, 1, 0.025, 0.1, (1:k) / k,
      if (rule == "OF") "obrien_fleming" else "pocock_classic", 0, "none", 0, FALSE,
      g$upper$bound, rep(-Inf, k), cumsum(g$upper$prob[, 1]), max(g$n.I), g$en[1],
      g$en[2])
}

# --- gsDesign: futility (beta spending), non-binding (test.type 4) and binding (3) ---
for (tt in c(4, 3)) for (k in c(3, 4)) for (fr in list(c("hsd", -2), c("power", 2))) {
  id <- id + 1
  timing <- (1:k) / k
  g <- gsDesign(k = k, test.type = tt, alpha = 0.025, beta = 0.1, sfu = sfLDOF,
                sfl = sfmap[[fr[1]]], sflpar = as.numeric(fr[2]), n.fix = 1,
                tol = 1e-8)
  # alpha spent: upper crossing probabilities under H0 ignoring the lower bound for
  # test.type 4 (gsDesign stores them in upper$spend), with it for test.type 3
  spent <- cumsum(g$upper$spend)
  add(id, "gsDesign", k, 1, 0.025, 0.1, timing, "obf", 0, fr[1], as.numeric(fr[2]),
      tt == 3, g$upper$bound, g$lower$bound, spent, max(g$n.I), g$en[1], g$en[2])
}

# --- rpact: spending, classical and Haybittle-Peto designs ---
rp <- list(c("asOF", "obf", NA), c("asP", "pocock", NA), c("asKD", "power", 3),
           c("asHSD", "hsd", -4), c("OF", "obrien_fleming", NA),
           c("P", "pocock_classic", NA), c("HP", "haybittle_peto", NA))
for (k in c(3, 4)) for (r in rp) for (sided in c(1, 2)) {
  id <- id + 1
  alpha <- if (sided == 1) 0.025 else 0.05
  args <- list(kMax = k, alpha = alpha, beta = 0.2, sided = sided,
               typeOfDesign = r[1])
  if (!is.na(r[3])) args$gammaA <- as.numeric(r[3])
  d <- do.call(getDesignGroupSequential, args)
  ch <- getDesignCharacteristics(d)
  en <- ch$averageSampleNumber0
  en1 <- ch$averageSampleNumber1
  add(id, "rpact", k, sided, alpha, 0.2, (1:k) / k, r[2],
      if (is.na(r[3])) 0 else as.numeric(r[3]), "none", 0, FALSE, d$criticalValues,
      if (sided == 2) -d$criticalValues else rep(-Inf, k), d$alphaSpent,
      ch$inflationFactor, en, en1)
}

out <- do.call(rbind, rows)
write.csv(out, "gs_designs_reference.csv", row.names = FALSE)
cat("designs:", length(unique(out$id)), "rows:", nrow(out), "\n")

# --- rpact analysis: stagewise ordering after stopping (normal approximation) ---
arows <- list()
cases <- list(
  list(des = "asOF", k = 3, n = c(40, 80), m1 = c(0.9, 0.85), m2 = c(0.2, 0.25),
       s1 = c(1.1, 1.05), s2 = c(1.0, 1.02)),
  list(des = "asOF", k = 3, n = c(40, 80, 120), m1 = c(0.4, 0.45, 0.42),
       m2 = c(0.1, 0.12, 0.11), s1 = c(1.1, 1.05, 1.07), s2 = c(1.0, 1.02, 1.01)),
  list(des = "asP", k = 3, n = c(40, 80), m1 = c(0.8, 0.7), m2 = c(0.2, 0.25),
       s1 = c(1.1, 1.05), s2 = c(1.0, 1.02)),
  list(des = "asKD", k = 4, n = c(30, 60, 90), m1 = c(0.5, 0.6, 0.62),
       m2 = c(0.1, 0.12, 0.1), s1 = c(1.1, 1.05, 1.02), s2 = c(1.0, 1.02, 1.0)))
for (ci in seq_along(cases)) {
  cs <- cases[[ci]]
  args <- list(kMax = cs$k, alpha = 0.025, beta = 0.2, sided = 1,
               typeOfDesign = cs$des)
  if (cs$des == "asKD") args$gammaA <- 3
  d <- do.call(getDesignGroupSequential, args)
  L <- length(cs$n)
  # stagewise (per-stage) summary data
  nst <- c(cs$n[1], diff(cs$n))
  ds <- getDataset(n1 = nst, n2 = nst, means1 = cs$m1, means2 = cs$m2,
                   stDevs1 = cs$s1, stDevs2 = cs$s2)
  res <- getAnalysisResults(design = d, dataInput = ds, normalApproximation = TRUE,
                            directionUpper = TRUE)
  st <- getStageResults(design = d, dataInput = ds, normalApproximation = TRUE)
  fs <- res$finalStage
  for (i in seq_len(L)) {
    arows[[length(arows) + 1]] <- data.frame(
      case = ci, design = cs$des, k = cs$k, look = i, info_rate = d$informationRates[i],
      effect = st$effectSizes[i], z = st$overallTestStatistics[i],
      critical = d$criticalValues[i],
      rci_lower = res$repeatedConfidenceIntervalLowerBounds[i],
      rci_upper = res$repeatedConfidenceIntervalUpperBounds[i],
      repeated_p = res$repeatedPValues[i],
      final_stage = fs, final_p = res$finalPValues[fs],
      final_lower = res$finalConfidenceIntervalLowerBounds[fs],
      final_upper = res$finalConfidenceIntervalUpperBounds[fs],
      median_unbiased = res$medianUnbiasedEstimates[fs])
  }
}
aout <- do.call(rbind, arows)
write.csv(aout, "gs_analysis_reference.csv", row.names = FALSE)
print(aout)
