# Reference Conley (1999) standard errors from fixest (vcov_conley: uniform kernel,
# great-circle distance with distance = "spherical"), without small-sample correction.
#
# Usage: Rscript conley_reference.R [R library path]
# Reads conley_data.csv, writes conley_reference.csv with one row per
# (specification, coefficient pair): spec, row, col, coef_row, vcov.
#   cs_100    period-1 cross-section, y ~ x1 + x2, cutoff 100 km
#   cs_fe_250 period-1 cross-section, y ~ x1 + x2 | g, cutoff 250 km
#   pool_200  pooled panel, y ~ x1 + x2 | g, cutoff 200 km (fixest ignores time:
#             repeated observations of a location are fully correlated)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) .libPaths(c(args[1], .libPaths()))
suppressPackageStartupMessages(library(fixest))
file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- if (length(file_arg) == 1) dirname(sub("^--file=", "", file_arg)) else "."
f <- function(x) file.path(here, x)

d <- read.csv(f("conley_data.csv"))
d1 <- d[d$period == 1, ]
nossc <- ssc(K.adj = FALSE, G.adj = FALSE)
specs <- list(
  cs_100 = list(m = feols(y ~ x1 + x2, d1), cutoff = 100),
  cs_fe_250 = list(m = feols(y ~ x1 + x2 | g, d1), cutoff = 250),
  pool_200 = list(m = feols(y ~ x1 + x2 | g, d), cutoff = 200)
)
rows <- list()
for (nm in names(specs)) {
  sp <- specs[[nm]]
  V <- vcov(sp$m, vcov = conley(cutoff = sp$cutoff, distance = "spherical"),
            lat = ~lat, lon = ~lon, ssc = nossc, vcov_fix = FALSE)
  cn <- colnames(V)
  for (a in cn) for (b in cn) {
    rows[[length(rows) + 1]] <- data.frame(spec = nm, row = a, col = b,
                                           coef_row = unname(coef(sp$m)[a]),
                                           vcov = V[a, b])
  }
}
out <- do.call(rbind, rows)
write.csv(out, f("conley_reference.csv"), row.names = FALSE)
print(out)
