# Reference values for DrSnow's did_etwfe (Wooldridge extended TWFE), generated
# with the R package etwfe (etwfe() + emfx(), which use fixest and marginaleffects).
#
# Run from the repository root:
#   Rscript test/validation/did/generate_etwfe_references.R
#
# Uses test/validation/did/mpdta.csv (written by generate_did_references.R).
# Outputs (test/validation/did):
#   r_etwfe.csv        emfx aggregations (simple / group / calendar / event)
#   r_etwfe_coef.csv   cohort x period coefficients of the linear models
# Standard errors are clustered by county unless the case name ends in "_iid"
# (fixest's model-based covariance; for Poisson/logit fixest multiplies it by
# (n - 1) / (n - K), which DrSnow does not).

suppressPackageStartupMessages({
  library(etwfe); library(fixest); library(marginaleffects)
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
writeLines(c(paste("etwfe", packageVersion("etwfe")),
             paste("fixest", packageVersion("fixest")),
             paste("marginaleffects", packageVersion("marginaleffects")),
             R.version.string), file.path(outdir, "etwfe_versions.txt"))

mp <- read.csv(file.path(outdir, "mpdta.csv"))
mp$emp <- round(exp(mp$lemp))
mp$high <- as.integer(mp$lemp > 5.5)
res <- list(); cf <- list()
agg <- function(case, m, types = c("simple", "group", "calendar", "event"), ...) {
  for (ty in types) {
    e <- as.data.frame(emfx(m, type = ty, ...))
    lab <- switch(ty, simple = rep(NA, nrow(e)), group = e$first_treat,
                  calendar = e$year, event = e$event)
    res[[length(res) + 1]] <<- data.frame(case = case, type = ty, label = lab,
                                          estimate = e$estimate, se = e$std.error)
  }
  ct <- coeftable(m)
  keep <- grepl("^\\.Dtreat", rownames(ct))
  cf[[length(cf) + 1]] <<- data.frame(case = case, term = rownames(ct)[keep],
                                      estimate = ct[keep, 1], se = ct[keep, 2])
}
agg("lin", etwfe(lemp ~ 0, tvar = year, gvar = first_treat, data = mp,
                 vcov = ~countyreal))
agg("lin_cov", etwfe(lemp ~ lpop, tvar = year, gvar = first_treat, data = mp,
                     vcov = ~countyreal))
agg("lin_never", etwfe(lemp ~ 0, tvar = year, gvar = first_treat, data = mp,
                       cgroup = "never", vcov = ~countyreal))
agg("lin_unit", etwfe(lemp ~ 0, tvar = year, gvar = first_treat, data = mp,
                      ivar = countyreal, vcov = ~countyreal))
agg("pois_iid", etwfe(emp ~ 0, tvar = year, gvar = first_treat, data = mp,
                      family = "poisson"))
agg("pois", etwfe(emp ~ 0, tvar = year, gvar = first_treat, data = mp,
                  family = "poisson", vcov = ~countyreal))
agg("pois_cov", etwfe(emp ~ lpop, tvar = year, gvar = first_treat, data = mp,
                      family = "poisson", vcov = ~countyreal), c("simple", "event"))
agg("logit_iid", etwfe(high ~ 0, tvar = year, gvar = first_treat, data = mp,
                       family = "logit"), c("simple", "event"))
save_csv(do.call(rbind, res), "r_etwfe.csv")
save_csv(do.call(rbind, cf), "r_etwfe_coef.csv")
