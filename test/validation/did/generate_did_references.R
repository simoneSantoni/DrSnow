# Reference values for DrSnow's DiD estimators.
#
# Run from the repository root:
#   Rscript test/validation/did/generate_did_references.R
#
# Package versions used for the committed CSVs (R 4.6.1): did 2.5.1, DRDID 1.3.0,
# fixest 0.14.2, didimputation 0.5.1, bacondecomp 0.1.1, TwoWayFEWeights 2.1.0.
# All numbers are written with 17 significant digits so the Julia tests can use
# tight tolerances. Every dataset the Julia tests use is written here too, so both
# sides read exactly the same inputs.

suppressPackageStartupMessages({
  library(did); library(DRDID); library(fixest)
  library(didimputation); library(bacondecomp)
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

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
data(mpdta, package = "did")
mpdta <- mpdta[order(mpdta$countyreal, mpdta$year), ]
mpdta$first_treat <- mpdta$first.treat
mpdta$d <- as.numeric(mpdta$first_treat > 0 & mpdta$year >= mpdta$first_treat)
# Deterministic, time-invariant sampling weight (for weighted estimators).
mpdta$w <- 1 + (mpdta$countyreal %% 7) / 3
mp <- mpdta[, c("countyreal", "year", "lpop", "lemp", "first_treat", "d", "w")]
save_csv(mp, "mpdta.csv")

# LaLonde (NSW experimental treated vs a deterministic subsample of CPS controls),
# two periods (1975, 1978), panel. Same construction as DRDID's examples, with the
# CPS comparison group subsampled to keep the file small.
data(nsw_long, package = "DRDID")
lal <- subset(nsw_long, nsw_long$treated == 0 | nsw_long$sample == 2)
set.seed(20200101)
cps_ids <- unique(lal$id[lal$experimental == 0])
keep_ids <- c(unique(lal$id[lal$experimental == 1]), sample(cps_ids, 1500))
lal <- lal[lal$id %in% keep_ids, c("id", "year", "re", "experimental", "age", "educ",
                                  "black", "married", "nodegree", "hisp", "re74")]
lal <- lal[order(lal$id, lal$year), ]
lal$w <- 1 + (lal$id %% 5) / 4
save_csv(lal, "lalonde_panel.csv")

data(sim_rc, package = "DRDID")
sim_rc$w <- 1 + (sim_rc$id %% 3) / 2
save_csv(sim_rc, "sim_rc.csv")

# ---------------------------------------------------------------------------
# Sant'Anna & Zhao (2020) doubly robust DiD (DRDID)
# ---------------------------------------------------------------------------
xs <- c("age", "educ", "black", "married", "nodegree", "hisp", "re74")
l0 <- lal[lal$year == 1975, ]; l1 <- lal[lal$year == 1978, ]
stopifnot(all(l0$id == l1$id))
X <- cbind(1, as.matrix(l0[, xs]))
Xr <- cbind(1, as.matrix(sim_rc[, c("x1", "x2", "x3", "x4")]))
res <- list()
add <- function(data, method, weighted, r) {
  res[[length(res) + 1]] <<- data.frame(data = data, method = method,
                                         weighted = as.numeric(weighted),
                                         att = r$ATT, se = r$se)
}
for (wt in c(FALSE, TRUE)) {
  iw <- if (wt) l0$w else NULL
  add("panel", "dr_imp", wt, drdid_imp_panel(l1$re, l0$re, l0$experimental, X, i.weights = iw))
  add("panel", "dr", wt, drdid_panel(l1$re, l0$re, l0$experimental, X, i.weights = iw))
  add("panel", "reg", wt, reg_did_panel(l1$re, l0$re, l0$experimental, X, i.weights = iw))
  add("panel", "std_ipw", wt, std_ipw_did_panel(l1$re, l0$re, l0$experimental, X, i.weights = iw))
  add("panel", "ipw", wt, ipw_did_panel(l1$re, l0$re, l0$experimental, X, i.weights = iw))
  add("panel_nox", "dr_imp", wt, drdid_imp_panel(l1$re, l0$re, l0$experimental, NULL, i.weights = iw))
  add("panel_nox", "dr", wt, drdid_panel(l1$re, l0$re, l0$experimental, NULL, i.weights = iw))
  iwr <- if (wt) sim_rc$w else NULL
  add("rc", "dr_imp", wt, drdid_imp_rc(sim_rc$y, sim_rc$post, sim_rc$d, Xr, i.weights = iwr))
  add("rc", "dr", wt, drdid_rc(sim_rc$y, sim_rc$post, sim_rc$d, Xr, i.weights = iwr))
  add("rc", "reg", wt, reg_did_rc(sim_rc$y, sim_rc$post, sim_rc$d, Xr, i.weights = iwr))
  add("rc", "std_ipw", wt, std_ipw_did_rc(sim_rc$y, sim_rc$post, sim_rc$d, Xr, i.weights = iwr))
  add("rc", "ipw", wt, ipw_did_rc(sim_rc$y, sim_rc$post, sim_rc$d, Xr, i.weights = iwr))
}
save_csv(do.call(rbind, res), "r_drdid.csv")

# ---------------------------------------------------------------------------
# Callaway & Sant'Anna (2021): att_gt + aggte on mpdta (analytic SEs)
# ---------------------------------------------------------------------------
scen <- list(
  S1 = list(xformla = ~1, control_group = "nevertreated", est_method = "dr",
            base_period = "varying", anticipation = 0, weightsname = NULL),
  S2 = list(xformla = ~lpop, control_group = "nevertreated", est_method = "dr",
            base_period = "varying", anticipation = 0, weightsname = NULL),
  S3 = list(xformla = ~lpop, control_group = "notyettreated", est_method = "dr",
            base_period = "varying", anticipation = 0, weightsname = NULL),
  S4 = list(xformla = ~lpop, control_group = "nevertreated", est_method = "ipw",
            base_period = "varying", anticipation = 0, weightsname = NULL),
  S5 = list(xformla = ~lpop, control_group = "nevertreated", est_method = "reg",
            base_period = "varying", anticipation = 0, weightsname = NULL),
  S6 = list(xformla = ~1, control_group = "nevertreated", est_method = "dr",
            base_period = "universal", anticipation = 0, weightsname = NULL),
  S7 = list(xformla = ~lpop, control_group = "notyettreated", est_method = "dr",
            base_period = "varying", anticipation = 1, weightsname = NULL),
  S8 = list(xformla = ~lpop, control_group = "nevertreated", est_method = "dr",
            base_period = "varying", anticipation = 0, weightsname = "w"),
  S9 = list(xformla = ~lpop, control_group = "notyettreated", est_method = "reg",
            base_period = "universal", anticipation = 0, weightsname = NULL)
)
gt_rows <- list(); ag_rows <- list()
for (nm in names(scen)) {
  s <- scen[[nm]]
  m <- att_gt(yname = "lemp", tname = "year", idname = "countyreal", gname = "first_treat",
              xformla = s$xformla, data = mp, control_group = s$control_group,
              est_method = s$est_method, base_period = s$base_period,
              anticipation = s$anticipation, weightsname = s$weightsname,
              bstrap = FALSE, cband = FALSE)
  gt_rows[[nm]] <- data.frame(scenario = nm, group = m$group, t = m$t, att = m$att,
                              se = m$se)
  aggs <- list(simple = list(type = "simple"), group = list(type = "group"),
               dynamic = list(type = "dynamic"), calendar = list(type = "calendar"),
               dynamic_bal1 = list(type = "dynamic", balance_e = 1),
               dynamic_win = list(type = "dynamic", min_e = -2, max_e = 2))
  for (an in names(aggs)) {
    a <- aggs[[an]]
    ag <- aggte(m, type = a$type, balance_e = a$balance_e,
                min_e = if (is.null(a$min_e)) -Inf else a$min_e,
                max_e = if (is.null(a$max_e)) Inf else a$max_e,
                bstrap = FALSE, cband = FALSE, na.rm = TRUE)
    ag_rows[[length(ag_rows) + 1]] <- data.frame(scenario = nm, agg = an,
      label = "overall", att = ag$overall.att, se = ag$overall.se)
    if (!is.null(ag$egt)) {
      ag_rows[[length(ag_rows) + 1]] <- data.frame(scenario = nm, agg = an,
        label = as.character(ag$egt), att = ag$att.egt, se = ag$se.egt)
    }
  }
}
save_csv(do.call(rbind, gt_rows), "r_att_gt.csv")
save_csv(do.call(rbind, ag_rows), "r_aggte.csv")

# Repeated cross-sections: mpdta rows treated as independent observations.
rc_scen <- list(
  R1 = list(xformla = ~1, control_group = "nevertreated", est_method = "dr",
            base_period = "varying"),
  R2 = list(xformla = ~lpop, control_group = "notyettreated", est_method = "dr",
            base_period = "varying"),
  R3 = list(xformla = ~lpop, control_group = "nevertreated", est_method = "reg",
            base_period = "universal"),
  R4 = list(xformla = ~lpop, control_group = "nevertreated", est_method = "ipw",
            base_period = "varying"))
rc_gt <- list(); rc_ag <- list()
for (nm in names(rc_scen)) {
  s <- rc_scen[[nm]]
  m <- att_gt(yname = "lemp", tname = "year", gname = "first_treat",
              xformla = s$xformla, data = mp, panel = FALSE,
              control_group = s$control_group, est_method = s$est_method,
              base_period = s$base_period, bstrap = FALSE, cband = FALSE)
  rc_gt[[nm]] <- data.frame(scenario = nm, group = m$group, t = m$t, att = m$att,
                            se = m$se)
  for (tp in c("simple", "group", "dynamic", "calendar")) {
    ag <- aggte(m, type = tp, bstrap = FALSE, cband = FALSE, na.rm = TRUE)
    rc_ag[[length(rc_ag) + 1]] <- data.frame(scenario = nm, agg = tp,
      label = "overall", att = ag$overall.att, se = ag$overall.se)
    if (!is.null(ag$egt)) {
      rc_ag[[length(rc_ag) + 1]] <- data.frame(scenario = nm, agg = tp,
        label = as.character(ag$egt), att = ag$att.egt, se = ag$se.egt)
    }
  }
}
save_csv(do.call(rbind, rc_gt), "r_att_gt_rc.csv")
save_csv(do.call(rbind, rc_ag), "r_aggte_rc.csv")

# ---------------------------------------------------------------------------
# TWFE and binned event study (fixest), for SE conventions
# ---------------------------------------------------------------------------
tw <- feols(lemp ~ d | countyreal + year, data = mp, cluster = ~countyreal)
mp$rel <- ifelse(mp$first_treat > 0, mp$year - mp$first_treat, -1000)
mp$relb <- ifelse(mp$rel == -1000, -1000, pmin(pmax(mp$rel, -3), 2))
es <- feols(lemp ~ i(relb, ref = c(-1, -1000)) | countyreal + year, data = mp,
            cluster = ~countyreal)
tww <- feols(lemp ~ d | countyreal + year, data = mp, cluster = ~countyreal,
             weights = ~w)
twfe_out <- rbind(
  data.frame(model = "twfe", term = "d", estimate = coef(tw)[["d"]], se = se(tw)[["d"]]),
  data.frame(model = "twfe_weighted", term = "d", estimate = coef(tww)[["d"]],
             se = se(tww)[["d"]]),
  data.frame(model = "es_bin_m3_p2", term = sub("relb::", "", names(coef(es))),
             estimate = unname(coef(es)), se = unname(se(es))))
save_csv(twfe_out, "r_twfe.csv")

# ---------------------------------------------------------------------------
# Sun & Abraham (2021) via fixest::sunab
# ---------------------------------------------------------------------------
sa <- feols(lemp ~ sunab(first_treat, year) | countyreal + year, data = mp,
            cluster = ~countyreal)
sa_att <- summary(sa, agg = "ATT")
sa_out <- rbind(
  data.frame(term = sub("year::", "", names(coef(sa))), estimate = unname(coef(sa)),
             se = unname(se(sa))),
  data.frame(term = "ATT", estimate = unname(coef(sa_att)[["ATT"]]),
             se = unname(se(sa_att)[["ATT"]])))
save_csv(sa_out, "r_sunab.csv")
Vsa <- vcov(sa)
Vdf <- as.data.frame(Vsa)
names(Vdf) <- paste0("e", sub("year::", "", colnames(Vsa)))
save_csv(cbind(data.frame(term = sub("year::", "", rownames(Vsa))), Vdf),
         "r_sunab_vcov.csv")
sa_c <- summary(sa, agg = FALSE)
save_csv(data.frame(term = gsub("[:]", "_", names(coef(sa_c))),
                    estimate = unname(coef(sa_c)), se = unname(se(sa_c))),
         "r_sunab_cohort.csv")

# ---------------------------------------------------------------------------
# Borusyak, Jaravel & Spiess (2024) imputation (didimputation)
# ---------------------------------------------------------------------------
bj <- did_imputation(data = mp, yname = "lemp", gname = "first_treat", tname = "year",
                     idname = "countyreal")
bjh <- did_imputation(data = mp, yname = "lemp", gname = "first_treat", tname = "year",
                      idname = "countyreal", horizon = TRUE, pretrends = -3:-1)
bj_out <- rbind(data.frame(term = "ATT", estimate = bj$estimate, se = bj$std.error),
                data.frame(term = bjh$term, estimate = bjh$estimate,
                           se = bjh$std.error))
save_csv(bj_out, "r_imputation.csv")

# ---------------------------------------------------------------------------
# Goodman-Bacon (2021) decomposition (bacondecomp)
# ---------------------------------------------------------------------------
bd <- bacon(lemp ~ d, data = mp, id_var = "countyreal", time_var = "year",
            quietly = TRUE)
save_csv(data.frame(treated = bd$treated, untreated = bd$untreated,
                    estimate = bd$estimate, weight = bd$weight,
                    type = gsub(" ", "_", bd$type)),
         "r_bacon.csv")
# ---------------------------------------------------------------------------
# de Chaisemartin & D'Haultfoeuille (2020) TWFE weights (TwoWayFEWeights 2.x).
# Note: `sensibility` divides by the sample SD of the weights (n - 1 denominator);
# DrSnow's `sigma_fe` uses the paper's population formula (Corollary 1).
# ---------------------------------------------------------------------------
if (requireNamespace("TwoWayFEWeights", quietly = TRUE)) {
  tw <- TwoWayFEWeights::twowayfeweights(mp, "lemp", "countyreal", "year", "d",
                                         type = "feTR", summary_measures = TRUE)
  save_csv(data.frame(beta = tw$beta, sensibility = tw$sensibility,
                      sum_minus = tw$sum_minus, sum_plus = tw$sum_plus,
                      nr_minus = tw$nr_minus, nr_plus = tw$nr_plus),
           "r_twfe_weights_summary.csv")
  wr <- as.data.frame(tw$dat_result)
  wr <- wr[wr$weight != 0, ]
  save_csv(data.frame(unit = wr$G, time = wr$T, weight = wr$weight),
           "r_twfe_weights.csv")
}
cat("done\n")
