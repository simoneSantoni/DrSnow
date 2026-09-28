# Reference values for DrSnow's honest RD inference, McCrary test and stdvars option.
#
# Writes the datasets (subsets of data shipped with RDHonest) and reference outputs of
#   RDHonest::RDHonest / RDHonestBME / RDSmoothnessBound / CVb   -> rd_honest & co.
#   rdd::DCdensity                                               -> rd_mccrary_test
#   rdrobust::rdrobust / rdbwselect with stdvars = TRUE          -> stdvars option
# used by test/rd/test_validation_honest.jl.
#
# Usage (from the repository root):
#   Rscript test/validation/rd/generate_reference_honest.R
#
# Versions used for the committed files are written to reference_honest_versions.txt
# (RDHonest from GitHub kolesarm/RDHonest, rdd 0.57 from the CRAN archive,
# rdrobust 4.0.0).

suppressPackageStartupMessages({
  library(RDHonest)
  library(rdd)
  library(rdrobust)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
outdir <- if (length(file_arg) == 1) dirname(normalizePath(file_arg)) else "."

write_full <- function(df, path) {
  out <- df
  for (nm in names(out)) {
    if (is.double(out[[nm]])) {
      out[[nm]] <- ifelse(is.na(out[[nm]]), NA, sprintf("%.17g", out[[nm]]))
    }
  }
  write.csv(out, path, row.names = FALSE, na = "")
}
read_back <- function(path) read.csv(path, stringsAsFactors = FALSE)

# ------------------------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------------------------
lee <- data.frame(voteshare = lee08$voteshare, margin = lee08$margin)
write_full(lee, file.path(outdir, "lee08.csv"))
lee <- read_back(file.path(outdir, "lee08.csv"))

hs <- headst[, c("mortHS", "povrate", "urban", "black", "sch1417", "pop", "statefp")]
hs <- hs[complete.cases(hs), ]
hs$pop_k <- hs$pop / 1000
hs$pop <- NULL
write_full(hs, file.path(outdir, "headst.csv"))
hs <- read_back(file.path(outdir, "headst.csv"))

set.seed(20260927)
rr <- rcp[sort(sample(nrow(rcp), 3000)), ]
rcps <- data.frame(elig_year = rr$elig_year, retired = as.numeric(rr$retired),
                   log_cn = log(rr$cn), log_c = log(rr$c), survey_year = rr$survey_year)
write_full(rcps, file.path(outdir, "rcp_sample.csv"))
rcps <- read_back(file.path(outdir, "rcp_sample.csv"))

cg <- cghs[sort(sample(nrow(cghs), 6000)), ]
cgs <- data.frame(log_earn = log(cg$earnings), yearat14 = cg$yearat14)
write_full(cgs, file.path(outdir, "cghs_sample.csv"))
cgs <- read_back(file.path(outdir, "cghs_sample.csv"))

sim <- read_back(file.path(outdir, "simulated.csv"))
senate <- read_back(file.path(outdir, "senate.csv"))

# ------------------------------------------------------------------------------------
# Collection helpers
# ------------------------------------------------------------------------------------
ref <- list()
add <- function(case, key, value) {
  value <- as.numeric(value)
  keys <- if (length(value) == 1) key else paste0(key, "[", seq_along(value), "]")
  ref[[length(ref) + 1]] <<- data.frame(case = case, key = keys, value = value)
}

rec_honest <- function(id, r) {
  co <- r$coefficients
  add(id, "estimate", co$estimate)
  add(id, "se", co$std.error)
  add(id, "max_bias", co$maximum.bias)
  add(id, "conf_low", co$conf.low)
  add(id, "conf_high", co$conf.high)
  add(id, "conf_low_onesided", co$conf.low.onesided)
  add(id, "conf_high_onesided", co$conf.high.onesided)
  add(id, "bandwidth", co$bandwidth)
  add(id, "eff_obs", co$eff.obs)
  add(id, "leverage", co$leverage)
  add(id, "cv", co$cv)
  add(id, "M", co$M)
  add(id, "pvalue", co$p.value)
  if (!is.null(co$first.stage) && !is.na(co$first.stage)) {
    add(id, "first_stage", co$first.stage)
    add(id, "M_rf", co$M.rf)
    add(id, "M_fs", co$M.fs)
  }
}

q <- function(expr) suppressMessages(suppressWarnings(expr))

# ------------------------------------------------------------------------------------
# RDHonest: sharp designs (Lee 2008)
# ------------------------------------------------------------------------------------
rec_honest("lee_uni_h10", q(RDHonest(voteshare ~ margin, data = lee, kern = "uniform",
                                     M = 0.1, h = 10)))
rec_honest("lee_default", q(RDHonest(voteshare ~ margin, data = lee)))
rec_honest("lee_flci", q(RDHonest(voteshare ~ margin, data = lee, M = 0.04,
                                  opt.criterion = "FLCI")))
rec_honest("lee_oci_epa", q(RDHonest(voteshare ~ margin, data = lee, M = 0.04,
                                     kern = "epanechnikov", opt.criterion = "OCI")))
rec_honest("lee_uni_mse", q(RDHonest(voteshare ~ margin, data = lee, M = 0.1,
                                     kern = "uniform")))
rec_honest("lee_ehw_h10", q(RDHonest(voteshare ~ margin, data = lee, M = 0.1, h = 10,
                                     se.method = "EHW")))
rec_honest("lee_taylor_flci", q(RDHonest(voteshare ~ margin, data = lee, M = 0.1,
                                         sclass = "T", opt.criterion = "FLCI")))
rec_honest("lee_alpha10", q(RDHonest(voteshare ~ margin, data = lee, M = 0.1,
                                     alpha = 0.1, opt.criterion = "FLCI")))
rec_honest("lee_J5_h8", q(RDHonest(voteshare ~ margin, data = lee, M = 0.1, h = 8,
                                   J = 5)))
rec_honest("lee_cutoff5", q(RDHonest(voteshare ~ margin, data = lee, M = 0.1,
                                     cutoff = 5)))
# Inference at a point: boundary (one-sided data) and interior
rec_honest("lee_ip_boundary", q(RDHonest(voteshare ~ margin, data = lee,
                                         subset = margin > 0, M = 0.1, h = 10,
                                         point.inference = TRUE)))
rec_honest("lee_ip_interior_h10", q(RDHonest(voteshare ~ margin, data = lee, M = 0.1,
                                             h = 10, cutoff = 20,
                                             point.inference = TRUE)))
rec_honest("lee_ip_interior_opt", q(RDHonest(voteshare ~ margin, data = lee, M = 0.1,
                                             cutoff = 20, point.inference = TRUE)))
rec_honest("lee_ip_rot", q(RDHonest(voteshare ~ margin, data = lee, subset = margin > 0,
                                    point.inference = TRUE)))

# Senate data (clusters by state)
senc <- senate[complete.cases(senate[, c("vote", "margin", "state")]), ]
rec_honest("senate_cluster", q(RDHonest(vote ~ margin, data = senc, se.method = "EHW",
                                        clusterid = state, M = 0.1)))
rec_honest("senate_cluster_h", q(RDHonest(vote ~ margin, data = senc, se.method = "EHW",
                                          clusterid = state, M = 0.1, h = 15)))

# Covariates (Head Start)
rec_honest("hs_covs_rot", q(RDHonest(mortHS ~ povrate | urban + black + sch1417,
                                     data = hs)))
rec_honest("hs_covs_M2", q(RDHonest(mortHS ~ povrate | urban + black + sch1417,
                                    data = hs, M = 2)))
rec_honest("hs_covs_M2_h8", q(RDHonest(mortHS ~ povrate | urban + black + sch1417,
                                       data = hs, M = 2, h = 8)))
rec_honest("hs_nocov", q(RDHonest(mortHS ~ povrate, data = hs, M = 2)))
rec_honest("hs_weights", q(RDHonest(mortHS ~ povrate, data = hs, M = 2,
                                    weights = pop_k)))
rec_honest("hs_cluster_covs", q(RDHonest(mortHS ~ povrate | urban + black, data = hs,
                                         M = 2, se.method = "EHW",
                                         clusterid = statefp)))

# Fuzzy designs (retirement consumption)
rec_honest("rcp_fuzzy_h3", q(RDHonest(log_cn | retired ~ elig_year, data = rcps,
                                      M = c(0.001, 0.002), h = 3)))
rec_honest("rcp_fuzzy_mse", q(RDHonest(log_cn | retired ~ elig_year, data = rcps,
                                       M = c(0.001, 0.002), T0 = 0)))
r0 <- q(RDHonest(log_cn | retired ~ elig_year, data = rcps, M = c(0.001, 0.002), T0 = 0))
rec_honest("rcp_fuzzy_T0", q(RDHonest(log_cn | retired ~ elig_year, data = rcps,
                                      M = c(0.001, 0.002),
                                      T0 = r0$coefficients$estimate)))
rec_honest("rcp_fuzzy_rot", q(RDHonest(log_cn | retired ~ elig_year, data = rcps)))
rec_honest("rcp_fuzzy_ehw_flci", q(RDHonest(log_cn | retired ~ elig_year, data = rcps,
                                            M = c(0.002, 0.004), se.method = "EHW",
                                            opt.criterion = "FLCI")))
rec_honest("rcp_fuzzy_covs", q(RDHonest(log_cn | retired ~ elig_year | survey_year,
                                        data = rcps, M = c(0.001, 0.002))))
rec_honest("rcp_fuzzy_cluster", q(RDHonest(log_cn | retired ~ elig_year, data = rcps,
                                           M = c(0.001, 0.002), se.method = "EHW",
                                           clusterid = survey_year)))
rec_honest("sim_fuzzy_uni", q(RDHonest(y_fuzzy | d ~ x, data = sim, M = c(1, 1),
                                       kern = "uniform")))

# Discrete running variable
rec_honest("sim_disc", q(RDHonest(y_disc ~ x_disc, data = sim, M = 1)))
rec_honest("sim_disc_uni", q(RDHonest(y_disc ~ x_disc, data = sim, M = 1,
                                      kern = "uniform", opt.criterion = "FLCI")))
rec_honest("cghs_h3", q(RDHonest(log_earn ~ yearat14, data = cgs, cutoff = 1947,
                                 M = 0.04, h = 3)))
rec_honest("cghs_opt", q(RDHonest(log_earn ~ yearat14, data = cgs, cutoff = 1947,
                                  M = 0.04)))

# BME (Kolesár & Rothe 2018)
rec_bme <- function(id, r) {
  co <- r$coefficients
  add(id, "estimate", co$estimate)
  add(id, "se", co$std.error)
  add(id, "max_bias", co$maximum.bias)
  add(id, "conf_low", co$conf.low)
  add(id, "conf_high", co$conf.high)
  add(id, "conf_low_onesided", co$conf.low.onesided)
  add(id, "conf_high_onesided", co$conf.high.onesided)
  add(id, "eff_obs", co$eff.obs)
  add(id, "leverage", co$leverage)
}
rec_bme("cghs_bme_o0_h3", q(RDHonestBME(log_earn ~ yearat14, data = cgs, cutoff = 1947,
                                        h = 3, order = 0)))
rec_bme("cghs_bme_o1_h5", q(RDHonestBME(log_earn ~ yearat14, data = cgs, cutoff = 1947,
                                        h = 5, order = 1)))
rec_bme("cghs_bme_o2_all", q(RDHonestBME(log_earn ~ yearat14, data = cgs, cutoff = 1947,
                                         order = 2, alpha = 0.1)))
rec_bme("sim_disc_bme_o1", q(RDHonestBME(y_disc ~ x_disc, data = sim, h = 0.2,
                                         order = 1)))

# Smoothness bound (deterministic variants: multiple = FALSE)
rs <- q(RDHonest(log_earn ~ yearat14, data = cgs, cutoff = 1947, M = 0.04, h = 3))
sb <- RDSmoothnessBound(rs, s = 2, separate = TRUE, multiple = FALSE)
add("cghs_sb_s2_sep", "estimate", sb$estimate)
add("cghs_sb_s2_sep", "conf_low", sb$conf.low)
sb <- RDSmoothnessBound(rs, s = 1, separate = TRUE, multiple = FALSE, sclass = "T")
add("cghs_sb_s1_T", "estimate", sb$estimate)
add("cghs_sb_s1_T", "conf_low", sb$conf.low)
rl <- q(RDHonest(voteshare ~ margin, data = lee, M = 0.1, h = 10))
sb <- RDSmoothnessBound(rl, s = 100, separate = TRUE, multiple = FALSE)
add("lee_sb_s100_sep", "estimate", sb$estimate)
add("lee_sb_s100_sep", "conf_low", sb$conf.low)
sb <- RDSmoothnessBound(rl, s = 50, separate = TRUE, multiple = FALSE, sclass = "T")
add("lee_sb_s50_sep_T", "estimate", sb$estimate)
add("lee_sb_s50_sep_T", "conf_low", sb$conf.low)
sb <- RDSmoothnessBound(rl, s = 100, separate = FALSE, multiple = TRUE)
add("lee_sb_s100_multi", "estimate", sb$estimate)
add("lee_sb_s100_multi", "conf_low", sb$conf.low)
sb <- RDSmoothnessBound(rs, s = 2, separate = FALSE, multiple = TRUE)
add("cghs_sb_s2_multi", "estimate", sb$estimate)
add("cghs_sb_s2_multi", "conf_low", sb$conf.low)

# Critical values
Bs <- c(0, 0.1, 0.5, 1, 2, 5, 9.5, 12)
add("cvb_a05", "cv", CVb(Bs, alpha = 0.05))
add("cvb_a10", "cv", CVb(Bs, alpha = 0.1))

# ------------------------------------------------------------------------------------
# McCrary (2008) test: rdd::DCdensity
# ------------------------------------------------------------------------------------
rec_dc <- function(id, r) {
  add(id, "theta", r$theta)
  add(id, "se", r$se)
  add(id, "z", r$z)
  add(id, "p", r$p)
  add(id, "bin", r$binsize)
  add(id, "bw", r$bw)
}
rec_dc("mcc_senate", DCdensity(senate$margin, 0, ext.out = TRUE, plot = FALSE))
rec_dc("mcc_lee", DCdensity(lee$margin, 0, ext.out = TRUE, plot = FALSE))
rec_dc("mcc_lee_bin_bw", DCdensity(lee$margin, 0, bin = 1, bw = 15, ext.out = TRUE,
                                   plot = FALSE))
rec_dc("mcc_sim_c02", DCdensity(sim$x, 0.2, ext.out = TRUE, plot = FALSE))
rec_dc("mcc_hs", DCdensity(hs$povrate, 0, ext.out = TRUE, plot = FALSE))

# ------------------------------------------------------------------------------------
# stdvars = TRUE (rdrobust 4.0.0)
# ------------------------------------------------------------------------------------
rec_rob <- function(id, r) {
  add(id, "tau_cl", r$Estimate[1, "tau.us"])
  add(id, "tau_bc", r$Estimate[1, "tau.bc"])
  add(id, "se_cl", r$Estimate[1, "se.us"])
  add(id, "se_rb", r$Estimate[1, "se.rb"])
  add(id, "h", r$bws[1, ])
  add(id, "b", r$bws[2, ])
}
rec_rob("std_senate", rdrobust(senate$vote, senate$margin, stdvars = TRUE))
rec_rob("std_senate_msetwo", rdrobust(senate$vote, senate$margin, stdvars = TRUE,
                                      bwselect = "msetwo"))
rec_rob("std_sim_fuzzy_covs", rdrobust(sim$y_fuzzy, sim$x, fuzzy = sim$d,
                                       covs = sim$z1, stdvars = TRUE))
rec_rob("std_sim_disc", rdrobust(sim$y_disc, sim$x_disc, stdvars = TRUE))
rec_rob("std_sim_cluster_cer", rdrobust(sim$y_sharp, sim$x, cluster = sim$g,
                                        bwselect = "cerrd", stdvars = TRUE))
bw <- rdbwselect(senate$vote, senate$margin, stdvars = TRUE, all = TRUE)
add("stdbw_senate_all", "bws", as.vector(t(bw$bws)))
bw <- rdbwselect(sim$y_kink, sim$x, deriv = 1, stdvars = TRUE, all = TRUE)
add("stdbw_sim_kink_all", "bws", as.vector(t(bw$bws)))

out <- do.call(rbind, ref)
out$value <- sprintf("%.17g", out$value)
write.csv(out, file.path(outdir, "reference_honest.csv"), row.names = FALSE)

writeLines(c(R.version.string,
             paste("RDHonest", as.character(packageVersion("RDHonest"))),
             paste("rdd", as.character(packageVersion("rdd"))),
             paste("rdrobust", as.character(packageVersion("rdrobust")))),
           file.path(outdir, "reference_honest_versions.txt"))
