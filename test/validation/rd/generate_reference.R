# Reference values for DrSnow's regression discontinuity area.
#
# Generates the validation datasets (Senate data shipped with rdrobust, plus simulated
# designs) and the reference outputs of rdrobust / rdbwselect / rdplot / rddensity /
# rdbwdensity used by test/rd/test_validation.jl.
#
# Usage (from the repository root):
#   Rscript test/validation/rd/generate_reference.R
#
# Versions used to produce the committed files: R 4.x, rdrobust 4.0.0,
# rddensity 3.0, lpdensity 3.0.1 (printed into reference_versions.txt).
#
# Datasets are written with 17 significant digits and read back before any estimation,
# so R and Julia work on bit-identical inputs.

suppressPackageStartupMessages({
  library(rdrobust)
  library(rddensity)
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
data(rdrobust_RDsenate)
senate <- rdrobust_RDsenate
senate$pop_m <- senate$population / 1e6
write_full(senate, file.path(outdir, "senate.csv"))
senate <- read_back(file.path(outdir, "senate.csv"))

set.seed(20260927)
n <- 1000
x <- 2 * rbeta(n, 2, 4) - 1
z1 <- rnorm(n)
z2 <- 0.5 * x + rnorm(n)
z3 <- z1 + z2                                   # exactly collinear with z1 and z2
g <- sample(1:60, n, replace = TRUE)
w <- runif(n, 0.5, 2)
e <- rnorm(n, sd = 0.3)
d <- rbinom(n, 1, 0.2 + 0.5 * (x >= 0) + 0.1 * x)
d_one <- (x >= 0) * rbinom(n, 1, 0.7)         # one-sided non-compliance
y_sharp <- 0.5 + 0.8 * x - 0.5 * x^2 + 0.3 * z1 + 0.6 * (x >= 0) + e
y_fuzzy <- 0.5 + 0.8 * x - 0.5 * x^2 + 0.3 * z1 + 1.0 * d + e
y_one <- 0.5 + 0.8 * x - 0.5 * x^2 + 1.0 * d_one + e
y_kink <- 0.5 + 0.4 * x + 0.8 * x * (x >= 0) + 0.3 * x^2 + e
d_kink <- 1 + 0.5 * x + 1.0 * x * (x >= 0) + 0.2 * rnorm(n)
y_fkink <- 0.3 + 0.5 * d_kink + e
x_disc <- round(x * 25) / 25
y_disc <- 0.5 + 0.8 * x_disc + 0.4 * (x_disc >= 0) + e
sim <- data.frame(x, z1, z2, z3, g, w, d, d_one, y_sharp, y_fuzzy, y_one, y_kink,
                  d_kink, y_fkink, x_disc, y_disc)
write_full(sim, file.path(outdir, "simulated.csv"))
sim <- read_back(file.path(outdir, "simulated.csv"))

# ------------------------------------------------------------------------------------
# Collection helpers: long format (case, key, value)
# ------------------------------------------------------------------------------------
ref <- list()
add <- function(case, key, value) {
  value <- as.numeric(value)
  keys <- if (length(value) == 1) key else paste0(key, "[", seq_along(value), "]")
  ref[[length(ref) + 1]] <<- data.frame(case = case, key = keys, value = value)
}

record_rdrobust <- function(id, r) {
  add(id, "tau_cl", r$Estimate[1, "tau.us"])
  add(id, "tau_bc", r$Estimate[1, "tau.bc"])
  add(id, "se_cl", r$Estimate[1, "se.us"])
  add(id, "se_rb", r$Estimate[1, "se.rb"])
  add(id, "pv", r$pv[, 1])
  add(id, "ci_lower", r$ci[, 1])
  add(id, "ci_upper", r$ci[, 2])
  add(id, "h", r$bws["h", ])
  add(id, "b", r$bws["b", ])
  add(id, "N", r$N)
  add(id, "N_h", r$N_h)
  add(id, "N_b", r$N_b)
  add(id, "M", r$M)
  add(id, "bias", r$bias)
  add(id, "beta_Y_p_l", r$beta_Y_p_l)
  add(id, "beta_Y_p_r", r$beta_Y_p_r)
  if (!is.null(r$tau_T)) {
    add(id, "tau_T", r$tau_T)
    add(id, "se_T", r$se_T)
  }
  if (!is.null(r$coef_covs)) add(id, "coef_covs", r$coef_covs)
}

run <- function(id, ...) {
  if (nzchar(Sys.getenv("RD_REF_VERBOSE"))) message(id)
  r <- suppressWarnings(rdrobust(...))
  record_rdrobust(id, r)
  invisible(r)
}

# ------------------------------------------------------------------------------------
# rdrobust: Senate data
# ------------------------------------------------------------------------------------
Y <- senate$vote; X <- senate$margin
run("senate_default", Y, X)
for (bs in c("mserd", "msetwo", "msesum", "msecomb1", "msecomb2",
             "cerrd", "certwo", "cersum", "cercomb1", "cercomb2")) {
  run(paste0("senate_bw_", bs), Y, X, bwselect = bs)
}
run("senate_epa", Y, X, kernel = "epa")
run("senate_uni", Y, X, kernel = "uni")
run("senate_p2", Y, X, p = 2)
run("senate_p0", Y, X, p = 0)
run("senate_p3q5", Y, X, p = 3, q = 5)
for (v in c("hc0", "hc1", "hc2", "hc3")) run(paste0("senate_", v), Y, X, vce = v)
run("senate_nn5", Y, X, nnmatch = 5)
run("senate_cluster", Y, X, cluster = senate$state)
run("senate_covs", Y, X, covs = cbind(senate$presdemvoteshlag1, senate$demvoteshlag1,
                                       senate$demvoteshlag2))
run("senate_covs_cluster", Y, X, covs = cbind(senate$presdemvoteshlag1,
                                               senate$demvoteshlag1),
    cluster = senate$state)
run("senate_covs_hc2", Y, X, covs = cbind(senate$presdemvoteshlag1), vce = "hc2")
run("senate_h10", Y, X, h = 10)
run("senate_h_b_twosided", Y, X, h = c(10, 15), b = c(20, 25))
run("senate_rho", Y, X, h = 12, rho = 0.5)
run("senate_rho_est", Y, X, rho = 0.8)
run("senate_mass_off", Y, X, masspoints = "off")
run("senate_mass_check", Y, X, masspoints = "check")
run("senate_regul0", Y, X, scaleregul = 0)
run("senate_kink", Y, X, deriv = 1)
run("senate_weights", Y, X, weights = senate$pop_m)
run("senate_cutoff5", Y, X, c = 5)
run("senate_bwcheck", Y, X, bwcheck = 50)
run("senate_bwrestrict_off", Y, X, bwrestrict = FALSE)
run("senate_scalepar", Y, X, scalepar = 2)

# ------------------------------------------------------------------------------------
# rdrobust: simulated designs
# ------------------------------------------------------------------------------------
s <- sim
run("sim_sharp", s$y_sharp, s$x)
run("sim_sharp_covs", s$y_sharp, s$x, covs = cbind(s$z1, s$z2))
run("sim_sharp_covs_collinear", s$y_sharp, s$x, covs = cbind(s$z1, s$z2, s$z3))
run("sim_sharp_cluster", s$y_sharp, s$x, cluster = s$g)
run("sim_sharp_cluster_msetwo", s$y_sharp, s$x, cluster = s$g, bwselect = "msetwo")
run("sim_sharp_cluster_cerrd", s$y_sharp, s$x, cluster = s$g, bwselect = "cerrd")
run("sim_sharp_cluster_hb", s$y_sharp, s$x, cluster = s$g, h = 0.3)
run("sim_sharp_weights", s$y_sharp, s$x, weights = s$w)
run("sim_sharp_hc3", s$y_sharp, s$x, vce = "hc3")
run("sim_sharp_hc1_weights_covs", s$y_sharp, s$x, vce = "hc1", weights = s$w,
    covs = cbind(s$z1))
run("sim_fuzzy", s$y_fuzzy, s$x, fuzzy = s$d)
run("sim_fuzzy_covs", s$y_fuzzy, s$x, fuzzy = s$d, covs = cbind(s$z1, s$z2))
run("sim_fuzzy_cluster", s$y_fuzzy, s$x, fuzzy = s$d, cluster = s$g)
run("sim_fuzzy_covs_cluster", s$y_fuzzy, s$x, fuzzy = s$d, covs = cbind(s$z1),
    cluster = s$g)
run("sim_fuzzy_hc1", s$y_fuzzy, s$x, fuzzy = s$d, vce = "hc1")
run("sim_fuzzy_hc2_covs", s$y_fuzzy, s$x, fuzzy = s$d, vce = "hc2", covs = cbind(s$z1))
run("sim_fuzzy_sharpbw", s$y_fuzzy, s$x, fuzzy = s$d, sharpbw = TRUE)
run("sim_fuzzy_msesum", s$y_fuzzy, s$x, fuzzy = s$d, bwselect = "msesum")
run("sim_fuzzy_certwo", s$y_fuzzy, s$x, fuzzy = s$d, bwselect = "certwo")
run("sim_fuzzy_onesided", s$y_one, s$x, fuzzy = s$d_one)
run("sim_kink", s$y_kink, s$x, deriv = 1)
run("sim_kink_uni", s$y_kink, s$x, deriv = 1, kernel = "uni")
run("sim_fuzzy_kink", s$y_fkink, s$x, fuzzy = s$d_kink, deriv = 1)
run("sim_fuzzy_kink_covs", s$y_fkink, s$x, fuzzy = s$d_kink, deriv = 1,
    covs = cbind(s$z2))
run("sim_disc", s$y_disc, s$x_disc)
run("sim_disc_off", s$y_disc, s$x_disc, masspoints = "off")
run("sim_disc_hc2", s$y_disc, s$x_disc, vce = "hc2")
run("sim_disc_msecomb2", s$y_disc, s$x_disc, bwselect = "msecomb2")
run("sim_p2_epa", s$y_sharp, s$x, p = 2, kernel = "epa")
run("senate_cr2", Y, X, cluster = senate$state, vce = "cr2")
run("senate_cr3", Y, X, cluster = senate$state, vce = "cr3")
run("sim_sharp_cr2", s$y_sharp, s$x, cluster = s$g, vce = "cr2")
run("sim_sharp_cr3", s$y_sharp, s$x, cluster = s$g, vce = "cr3")
run("sim_sharp_cr2_hb", s$y_sharp, s$x, cluster = s$g, vce = "cr2", h = 0.3)
run("sim_sharp_cr3_hb", s$y_sharp, s$x, cluster = s$g, vce = "cr3", h = 0.3)
run("sim_fuzzy_covs_cr3", s$y_fuzzy, s$x, fuzzy = s$d, covs = cbind(s$z1),
    cluster = s$g, vce = "cr3")
run("sim_fuzzy_cr2", s$y_fuzzy, s$x, fuzzy = s$d, cluster = s$g, vce = "cr2")

# ------------------------------------------------------------------------------------
# rdbwselect (all selectors)
# ------------------------------------------------------------------------------------
bw_all <- function(id, ...) {
  b <- suppressWarnings(rdbwselect(..., all = TRUE))
  for (i in seq_len(nrow(b$bws))) {
    add(id, paste0(rownames(b$bws)[i], "_h"), b$bws[i, 1:2])
    add(id, paste0(rownames(b$bws)[i], "_b"), b$bws[i, 3:4])
  }
}
bw_all("bw_senate", Y, X)
bw_all("bw_senate_cluster", Y, X, cluster = senate$state)
bw_all("bw_senate_p2_uni", Y, X, p = 2, kernel = "uni")
bw_all("bw_sim_fuzzy_covs", s$y_fuzzy, s$x, fuzzy = s$d, covs = cbind(s$z1, s$z2))
bw_all("bw_sim_fuzzy_covs_cluster", s$y_fuzzy, s$x, fuzzy = s$d,
       covs = cbind(s$z1, s$z2), cluster = s$g)
bw_all("bw_sim_kink_hc3", s$y_kink, s$x, deriv = 1, vce = "hc3")
bw_all("bw_sim_disc", s$y_disc, s$x_disc)
bw_all("bw_sim_cr2", s$y_sharp, s$x, cluster = s$g, vce = "cr2")
bw_all("bw_senate_cr3", Y, X, cluster = senate$state, vce = "cr3")

# ------------------------------------------------------------------------------------
# rdplot
# ------------------------------------------------------------------------------------
plot_case <- function(id, ...) {
  pdf(NULL)
  r <- suppressWarnings(suppressMessages(rdplot(..., hide = TRUE)))
  dev.off()
  add(id, "J", r$J)
  add(id, "J_IMSE", r$J_IMSE)
  add(id, "J_MV", r$J_MV)
  add(id, "coef_l", r$coef[, 1])
  add(id, "coef_r", r$coef[, 2])
  vb <- r$vars_bins
  add(id, "mean_bin", vb$rdplot_mean_bin)
  add(id, "mean_x", vb$rdplot_mean_x)
  add(id, "mean_y", vb$rdplot_mean_y)
  add(id, "min_bin", vb$rdplot_min_bin)
  add(id, "max_bin", vb$rdplot_max_bin)
  add(id, "N_bin", vb$rdplot_N)
  add(id, "se_y", vb$rdplot_se_y)
  add(id, "ci_l", vb$rdplot_ci_l)
  add(id, "ci_r", vb$rdplot_ci_r)
  add(id, "poly_x", r$vars_poly$rdplot_x[c(1, 250, 500, 501, 750, 1000)])
  add(id, "poly_y", r$vars_poly$rdplot_y[c(1, 250, 500, 501, 750, 1000)])
  add(id, "bin_avg", r$bin_avg)
  add(id, "bin_med", r$bin_med)
  add(id, "rscale", r$rscale)
}
Yc <- senate$vote[!is.na(senate$vote)]; Xc <- senate$margin[!is.na(senate$vote)]
for (bs in c("es", "espr", "esmv", "esmvpr", "qs", "qspr", "qsmv", "qsmvpr")) {
  plot_case(paste0("plot_senate_", bs), Yc, Xc, binselect = bs)
}
plot_case("plot_senate_nbins", Yc, Xc, nbins = c(15, 25))
plot_case("plot_senate_p1_scale", Yc, Xc, p = 1, scale = 2)
plot_case("plot_senate_tri_h", Yc, Xc, kernel = "tri", h = 40, p = 2)
plot_case("plot_senate_covs", senate$vote, senate$margin,
          covs = senate$presdemvoteshlag1)
plot_case("plot_sim_disc", s$y_disc, s$x_disc)
plot_case("plot_sim_support", s$y_sharp, s$x, support = c(-1.2, 1.2), binselect = "es")
plot_case("plot_sim_weights", s$y_sharp, s$x, weights = s$w)

# ------------------------------------------------------------------------------------
# rddensity / rdbwdensity
# ------------------------------------------------------------------------------------
dens_case <- function(id, X, ...) {
  r <- suppressWarnings(rddensity(X, ...))
  add(id, "hat", unlist(r$hat))
  add(id, "sd_jk", unlist(r$sd_jk))
  add(id, "sd_asy", unlist(r$sd_asy))
  add(id, "test", c(r$test$t_asy, r$test$t_jk, r$test$p_asy, r$test$p_jk))
  add(id, "h", c(r$h$left, r$h$right))
  add(id, "N", c(r$N$full, r$N$left, r$N$right, r$N$eff_left, r$N$eff_right))
  add(id, "bino_LN", r$bino$LeftN)
  add(id, "bino_RN", r$bino$RightN)
  add(id, "bino_LW", r$bino$LeftWindow)
  add(id, "bino_RW", r$bino$RightWindow)
  add(id, "bino_pval", r$bino$pval)
  if (isTRUE(r$opt$all)) {
    add(id, "hat_p", unlist(r$hat_p))
    add(id, "test_p", c(r$test_p$t_asy, r$test_p$t_jk, r$test_p$p_asy, r$test_p$p_jk))
  }
}
Xs <- senate$margin
dens_case("dens_senate", Xs)
dens_case("dens_senate_all", Xs, all = TRUE)
dens_case("dens_senate_p1", Xs, p = 1)
dens_case("dens_senate_uni", Xs, kernel = "uniform")
dens_case("dens_senate_epa", Xs, kernel = "epanechnikov")
dens_case("dens_senate_plugin", Xs, vce = "plugin")
dens_case("dens_senate_restricted", Xs, fitselect = "restricted")
dens_case("dens_senate_restricted_plugin", Xs, fitselect = "restricted", vce = "plugin")
dens_case("dens_senate_each", Xs, bwselect = "each")
dens_case("dens_senate_diff", Xs, bwselect = "diff")
dens_case("dens_senate_sum", Xs, bwselect = "sum")
dens_case("dens_senate_h", Xs, h = c(20, 30))
dens_case("dens_senate_nomass", Xs, massPoints = FALSE)
dens_case("dens_senate_c10", Xs, c = 10)
dens_case("dens_sim", sim$x)
dens_case("dens_sim_disc", sim$x_disc)
dens_case("dens_sim_binoW", sim$x, binoW = 0.05, binoNW = 5)

bwd_case <- function(id, X, ...) {
  r <- suppressWarnings(rdbwdensity(X, ...))
  add(id, "bw", r$h[, 1])
  add(id, "variance", r$h[, 2])
  add(id, "biassq", r$h[, 3])
}
bwd_case("bwd_senate", Xs)
bwd_case("bwd_senate_restricted", Xs, fitselect = "restricted")
bwd_case("bwd_senate_plugin_uni", Xs, vce = "plugin", kernel = "uniform")
bwd_case("bwd_sim", sim$x, p = 3)

out <- do.call(rbind, ref)
out$value <- sprintf("%.17g", out$value)
write.csv(out, file.path(outdir, "reference.csv"), row.names = FALSE)
writeLines(c(R.version.string,
             paste("rdrobust", packageVersion("rdrobust")),
             paste("rddensity", packageVersion("rddensity")),
             paste("lpdensity", packageVersion("lpdensity"))),
           file.path(outdir, "reference_versions.txt"))
cat("wrote", nrow(out), "reference values\n")
