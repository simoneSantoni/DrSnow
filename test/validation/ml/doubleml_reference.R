# Reference values for DrSnow's DML estimators from the R package DoubleML.
#
# Uses the datasets and fold assignments written by make_data.jl, deterministic
# learners (mlr3 regr.lm = OLS, classif.log_reg = logistic MLE) and externally set
# sample splits, so DoubleML and DrSnow must agree to numerical precision.
#
#   Rscript test/validation/ml/doubleml_reference.R
#
# Produced with R 4.x, DoubleML 1.0.2, mlr3 / mlr3learners (see sessionInfo()
# printed at the end). Writes dml_reference.csv next to this script.

suppressPackageStartupMessages({
  library(DoubleML)
  library(mlr3)
  library(mlr3learners)
  library(data.table)
})
lgr::get_logger("mlr3")$set_threshold("warn")

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("--file=", "", args[grep("--file=", args)])
dir <- if (length(file_arg) == 1) dirname(normalizePath(file_arg)) else "."

dat <- fread(file.path(dir, "dml_data.csv"))
folds <- fread(file.path(dir, "dml_folds.csv"))
n <- nrow(dat)
K <- 5
smpls <- lapply(seq_len(ncol(folds)), function(r) {
  f <- as.integer(folds[[r]])
  list(train_ids = lapply(1:K, function(k) which(f != k)),
       test_ids = lapply(1:K, function(k) which(f == k)))
})
xcols <- paste0("x", 1:5)

reg <- function() lrn("regr.lm")
cls <- function() lrn("classif.log_reg")

results <- list()
record <- function(case, obj) {
  for (j in seq_along(obj$coef)) {
    for (r in seq_len(ncol(obj$all_coef))) {
      results[[length(results) + 1]] <<- data.table(
        case = case, treatment = names(obj$coef)[j], rep = r,
        coef = obj$all_coef[j, r], se = obj$all_se[j, r])
    }
    results[[length(results) + 1]] <<- data.table(
      case = case, treatment = names(obj$coef)[j], rep = 0L,
      coef = obj$coef[j], se = obj$se[j])
  }
}
fit_with <- function(obj) {
  obj$set_sample_splitting(smpls)
  obj$fit()
  obj
}

# PLR, partialling out, one treatment
d <- DoubleMLData$new(dat, y_col = "y_plr", d_cols = "d1", x_cols = xcols)
record("plr_po", fit_with(DoubleMLPLR$new(d, ml_l = reg(), ml_m = reg(), n_folds = K,
                                          score = "partialling out")))
# PLR, IV-type score
record("plr_ivtype", fit_with(DoubleMLPLR$new(d, ml_l = reg(), ml_m = reg(),
                                              ml_g = reg(), n_folds = K,
                                              score = "IV-type")))
# PLR, two treatments
d2 <- DoubleMLData$new(dat, y_col = "y_plr", d_cols = c("d1", "d2"), x_cols = xcols)
record("plr_multi", fit_with(DoubleMLPLR$new(d2, ml_l = reg(), ml_m = reg(),
                                             n_folds = K)))
# IRM, ATE and ATTE (DoubleML 1.0.2 default trimming threshold 1e-12)
di <- DoubleMLData$new(dat, y_col = "y_irm", d_cols = "d_irm", x_cols = xcols)
record("irm_ate", fit_with(DoubleMLIRM$new(di, ml_g = reg(), ml_m = cls(),
                                           n_folds = K, score = "ATE")))
record("irm_atte", fit_with(DoubleMLIRM$new(di, ml_g = reg(), ml_m = cls(),
                                            n_folds = K, score = "ATTE")))
# PLIV, one instrument (partialling out and IV-type), two instruments
dp <- DoubleMLData$new(dat, y_col = "y_pliv", d_cols = "d_pliv", x_cols = xcols,
                       z_cols = "z1")
record("pliv_po", fit_with(DoubleMLPLIV$new(dp, ml_l = reg(), ml_m = reg(),
                                            ml_r = reg(), n_folds = K)))
record("pliv_ivtype", fit_with(DoubleMLPLIV$new(dp, ml_l = reg(), ml_m = reg(),
                                                ml_r = reg(), ml_g = reg(),
                                                n_folds = K, score = "IV-type")))
dp2 <- DoubleMLData$new(dat, y_col = "y_pliv", d_cols = "d_pliv", x_cols = xcols,
                        z_cols = c("z1", "z2"))
record("pliv_2z", fit_with(DoubleMLPLIV$new(dp2, ml_l = reg(), ml_m = reg(),
                                            ml_r = reg(), n_folds = K)))
# IIVM, two-sided and one-sided non-compliance
dv <- DoubleMLData$new(dat, y_col = "y_iv", d_cols = "d_iv", x_cols = xcols,
                       z_cols = "z_iv")
record("iivm", fit_with(DoubleMLIIVM$new(dv, ml_g = reg(), ml_m = cls(), ml_r = cls(),
                                         n_folds = K)))
do <- DoubleMLData$new(dat, y_col = "y_os", d_cols = "d_os", x_cols = xcols,
                       z_cols = "z_iv")
record("iivm_onesided", fit_with(DoubleMLIIVM$new(
  do, ml_g = reg(), ml_m = cls(), ml_r = cls(), n_folds = K,
  subgroups = list(always_takers = FALSE, never_takers = TRUE))))

out <- rbindlist(results)
out$coef <- sprintf("%.17g", out$coef)
out$se <- sprintf("%.17g", out$se)
options(digits = 17)
write.csv(out, file.path(dir, "dml_reference.csv"), row.names = FALSE)
print(out)
cat("DoubleML version:", as.character(packageVersion("DoubleML")), "\n")
cat("mlr3 version:", as.character(packageVersion("mlr3")), "\n")
cat(R.version.string, "\n")
