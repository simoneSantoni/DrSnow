# Reference stratified sandwich covariances from the R package `survey`: svyglm on a
# design with strata (and optionally clusters as primary sampling units), equal
# weights and no finite-population correction, i.e. the with-replacement
# linearization variance sum_h n_h/(n_h-1) sum_j (u_hj - ubar_h)(u_hj - ubar_h)'.
#
# Usage: Rscript stratified_vcov_reference.R [R library path]
# Reads stratified_vcov_data.csv, writes stratified_vcov_reference.csv with one row
# per (specification, coefficient pair): spec, row, col, coef_row, vcov, degf.
#   strata        y ~ z + x1 + x2, strata = stratum, units = observations
#   strata_fe     y ~ z + x1 + x2 + factor(stratum), strata = stratum
#   cluster       y ~ z + x1 + x2, strata = stratum, ids = cluster
#   cluster_fe    y ~ z + x1 + x2 + factor(stratum), strata = stratum, ids = cluster

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) .libPaths(c(args[1], .libPaths()))
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu = "fail")
file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- if (length(file_arg) == 1) dirname(sub("^--file=", "", file_arg)) else "."
f <- function(x) file.path(here, x)

d <- read.csv(f("stratified_vcov_data.csv"))
des_obs <- svydesign(ids = ~1, strata = ~stratum, data = d)
des_cl <- svydesign(ids = ~cluster, strata = ~stratum, data = d)
specs <- list(
  strata = svyglm(y ~ z + x1 + x2, design = des_obs),
  strata_fe = svyglm(y ~ z + x1 + x2 + factor(stratum), design = des_obs),
  cluster = svyglm(y ~ z + x1 + x2, design = des_cl),
  cluster_fe = svyglm(y ~ z + x1 + x2 + factor(stratum), design = des_cl)
)
rows <- list()
for (nm in names(specs)) {
  m <- specs[[nm]]
  V <- vcov(m)
  keep <- c("z", "x1", "x2")
  for (a in keep) for (b in keep) {
    rows[[length(rows) + 1]] <- data.frame(spec = nm, row = a, col = b,
                                           coef_row = unname(coef(m)[a]),
                                           vcov = V[a, b],
                                           degf = degf(m$survey.design))
  }
}
out <- do.call(rbind, rows)
write.csv(out, f("stratified_vcov_reference.csv"), row.names = FALSE)
cat("survey", as.character(packageVersion("survey")), "\n")
