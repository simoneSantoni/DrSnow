# Reference values for DrSnow's DR-DiD scores from the R package DRDID
# (Sant'Anna & Zhao 2020). DRDID fits the nuisances in-sample (OLS outcome
# regressions, logit propensity score), so the comparison evaluates DrSnow's score
# functions at in-sample OLS / logistic predictions: point estimates must agree to
# numerical precision. (Standard errors differ by design: DRDID's influence function
# adds first-stage estimation effects, whereas the cross-fitted DML score does not
# need them.)
#
#   Rscript test/validation/ml/drdid_reference.R

suppressPackageStartupMessages(library(DRDID))

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("--file=", "", args[grep("--file=", args)])
dir <- if (length(file_arg) == 1) dirname(normalizePath(file_arg)) else "."

p <- read.csv(file.path(dir, "did_panel.csv"))
cp <- cbind(1, as.matrix(p[, c("x1", "x2", "x3")]))  # DRDID >= 1.2 needs the intercept
rp <- drdid_panel(y1 = p$y1, y0 = p$y0, D = p$d, covariates = cp)
c <- read.csv(file.path(dir, "did_rcs.csv"))
cc <- cbind(1, as.matrix(c[, c("x1", "x2", "x3")]))
rc <- drdid_rc(y = c$y, post = c$post, D = c$d, covariates = cc)

out <- data.frame(case = c("drdid_panel", "drdid_rc"),
                  att = sprintf("%.17g", c(rp$ATT, rc$ATT)),
                  se = sprintf("%.17g", c(rp$se, rc$se)))
write.csv(out, file.path(dir, "drdid_reference.csv"), row.names = FALSE)
print(out)
cat("DRDID version:", as.character(packageVersion("DRDID")), "\n")
