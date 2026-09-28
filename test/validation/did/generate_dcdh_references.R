# Reference values for DrSnow's did_multiplegt_dyn (de Chaisemartin &
# D'Haultfoeuille 2024), generated with the DIDmultiplegtDYN R package (polars
# backend).
#
# Run from the repository root:
#   Rscript test/validation/did/generate_dcdh_references.R
#
# Outputs (test/validation/did):
#   favara_imbs.csv    Favara & Imbs (2015) data shipped with DIDmultiplegtDYN
#   dcdh_sim.csv       simulated panel with a non-binary, non-absorbing treatment,
#                      switchers in and out, and missing cells
#   r_dcdh.csv         effects, placebos and average total effect per case
#   r_dcdh_tests.csv   p-values of the joint tests that all effects / all placebos
#                      are zero (these use the full covariance of the effects and
#                      of the placebos)
#
# The covariance matrix returned in `coef$vcov` by DIDmultiplegtDYN 2.4.0 is not
# used: its off-diagonal elements are computed from columns built with
# `ifelse(is.null(x), NA, x)`, which keeps only the first element of x, so they are
# not the influence-function covariances. The joint tests are computed from the
# correct covariance (U_Gg_var_glob columns) and are compared instead.

suppressPackageStartupMessages({
  library(polars)
  library(DIDmultiplegtDYN)
})
outdir <- "test/validation/did"
fmt <- function(df) {
  for (nm in names(df)) if (is.numeric(df[[nm]])) {
    df[[nm]] <- ifelse(is.na(df[[nm]]), "NA", sprintf("%.17g", df[[nm]]))
  }
  df
}
save_csv <- function(df, name) {
  utils::write.csv(fmt(df), file.path(outdir, name), row.names = FALSE, quote = FALSE)
}
writeLines(c(paste("DIDmultiplegtDYN", packageVersion("DIDmultiplegtDYN")),
             paste("polars", packageVersion("polars")), R.version.string),
           file.path(outdir, "dcdh_versions.txt"))

data(favara_imbs)
fi <- as.data.frame(favara_imbs)
fi <- fi[, c("year", "county", "state_n", "Dl_vloans_b", "inter_bra", "w1", "Dl_hpi")]
for (nm in names(fi)) attributes(fi[[nm]]) <- NULL
save_csv(fi, "favara_imbs.csv")

# Simulated panel: groups start at treatment 0, 1 or 2; switchers move up or down
# once (some later move back towards the status quo), a few cells are missing.
set.seed(20240927)
G <- 150; TT <- 9
sim <- expand.grid(t = 1:TT, g = 1:G)
dsq <- sample(0:2, G, replace = TRUE)
Fg <- sample(c(3:8, rep(99, 3)), G, replace = TRUE)
dir <- ifelse(dsq == 0, 1, ifelse(dsq == 2, -1, sample(c(-1, 1), G, replace = TRUE)))
back <- runif(G) < 0.25
region <- sample(1:25, G, replace = TRUE)
a_g <- rnorm(G); l_t <- cumsum(rnorm(TT, sd = 0.3))
sim$d <- with(sim, dsq[g] + ifelse(t >= Fg[g], dir[g] * (1 + (t - Fg[g] >= 2)), 0) -
                ifelse(back[g] & t >= Fg[g] + 4, dir[g], 0))
sim$y <- with(sim, a_g[g] + l_t[t] + 0.5 * (d - dsq[g]) +
                0.2 * (t >= Fg[g]) * (t - Fg[g]) * dir[g] + rnorm(nrow(sim)))
sim$region <- region[sim$g]
sim$w <- round(runif(G, 1, 5))[sim$g]
drop <- sample(nrow(sim), 40)
sim$y[drop[1:20]] <- NA
sim <- sim[-drop[21:40], ]
# a time-varying control correlated with the outcome (drawn last, so the columns
# above do not depend on it)
sim$x <- 0.5 * sim$t / TT + rnorm(nrow(sim))
sim$y <- sim$y + 0.3 * sim$x
save_csv(sim[, c("g", "t", "y", "d", "region", "w", "x")], "dcdh_sim.csv")

res <- list(); vc <- list()
run <- function(case, df, ...) {
  r <- did_multiplegt_dyn(df = df, graph_off = TRUE, ...)
  R <- r$results
  e <- R$Effects
  out <- data.frame(case = case, kind = "effect", index = seq_len(nrow(e)),
                    estimate = e[, 1], se = e[, 2], n = e[, 5],
                    switchers = e[, 6])
  if (!is.null(R$Placebos)) {
    p <- R$Placebos
    out <- rbind(out, data.frame(case = case, kind = "placebo",
                                 index = seq_len(nrow(p)), estimate = p[, 1],
                                 se = p[, 2], n = p[, 5], switchers = p[, 6]))
  }
  a <- R$ATE
  out <- rbind(out, data.frame(case = case, kind = "ate", index = 0,
                               estimate = a[1], se = a[2], n = a[5],
                               switchers = a[6]))
  res[[length(res) + 1]] <<- out
  pe <- if (is.null(R$p_jointeffects)) NA else R$p_jointeffects
  pp <- if (is.null(R$p_jointplacebo)) NA else R$p_jointplacebo
  vc[[length(vc) + 1]] <<- data.frame(case = case, p_effects = pe, p_placebos = pp)
}
fa <- fi
run("fi_cl", fa, outcome = "Dl_vloans_b", group = "county", time = "year",
    treatment = "inter_bra", effects = 5, placebo = 3, cluster = "state_n")
run("fi_nocl", fa, outcome = "Dl_vloans_b", group = "county", time = "year",
    treatment = "inter_bra", effects = 4, placebo = 2)
run("fi_norm", fa, outcome = "Dl_vloans_b", group = "county", time = "year",
    treatment = "inter_bra", effects = 5, placebo = 2, normalized = TRUE,
    cluster = "state_n")
run("fi_w", fa, outcome = "Dl_vloans_b", group = "county", time = "year",
    treatment = "inter_bra", effects = 3, placebo = 2, weight = "w1",
    cluster = "state_n")
run("fi_never", fa, outcome = "Dl_vloans_b", group = "county", time = "year",
    treatment = "inter_bra", effects = 3, placebo = 1, only_never_switchers = TRUE)
run("fi_same", fa, outcome = "Dl_vloans_b", group = "county", time = "year",
    treatment = "inter_bra", effects = 3, placebo = 2, same_switchers = TRUE)
run("sim_both", sim, outcome = "y", group = "g", time = "t", treatment = "d",
    effects = 5, placebo = 3)
run("sim_cl", sim, outcome = "y", group = "g", time = "t", treatment = "d",
    effects = 5, placebo = 2, cluster = "region")
run("sim_in", sim, outcome = "y", group = "g", time = "t", treatment = "d",
    effects = 4, placebo = 2, switchers = "in")
run("sim_out", sim, outcome = "y", group = "g", time = "t", treatment = "d",
    effects = 4, placebo = 2, switchers = "out")
run("sim_norm", sim, outcome = "y", group = "g", time = "t", treatment = "d",
    effects = 5, placebo = 2, normalized = TRUE, weight = "w")
run("sim_never", sim, outcome = "y", group = "g", time = "t", treatment = "d",
    effects = 3, placebo = 2, only_never_switchers = TRUE)

# (trends_nonparam = "state_n" fails Design Restriction 1 in favara_imbs: all
# counties of a state are deregulated at the same date)
sim$big <- as.integer(sim$region > 12)
run("sim_tnp", sim, outcome = "y", group = "g", time = "t", treatment = "d",
    effects = 4, placebo = 2, trends_nonparam = "big", cluster = "region")
run("sim_tnp2", sim, outcome = "y", group = "g", time = "t", treatment = "d",
    effects = 3, placebo = 1, trends_nonparam = "big", normalized = TRUE,
    switchers = "in")

run("fi_ctrl", fa, outcome = "Dl_vloans_b", group = "county", time = "year",
    treatment = "inter_bra", effects = 3, placebo = 2, controls = "Dl_hpi",
    cluster = "state_n")
run("sim_ctrl", sim, outcome = "y", group = "g", time = "t", treatment = "d",
    effects = 4, placebo = 2, controls = "x")
run("sim_ctrl_n", sim, outcome = "y", group = "g", time = "t", treatment = "d",
    effects = 3, placebo = 1, controls = "x", normalized = TRUE, weight = "w",
    cluster = "region")

save_csv(do.call(rbind, res), "r_dcdh.csv")
save_csv(do.call(rbind, vc), "r_dcdh_tests.csv")
