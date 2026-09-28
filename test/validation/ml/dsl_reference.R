# Reference values for DrSnow's design-based supervised learning (dsl_regression)
# from the R package dsl (naoki-egami/dsl, version 0.1.0, GitHub commit noted in
# dsl_reference_info.csv), on the package's example datasets.
#
# To make the comparison exact, dsl is run with a deterministic internal prediction
# model (sl_method = "lm": SuperLearner's SL.lm, i.e. OLS of each ML-measured
# variable on the prediction columns) and a single sample split (sample_split = 1).
# The fold of every row (or cluster) is reconstructed exactly as dsl draws it
# (set.seed(seed); sample(rep(1:K, ...))) and exported, so DrSnow can reuse it with
# OLSLearner(). dsl solves the moment equations numerically (optim, L-BFGS-B), so
# agreement is to optimizer tolerance, not machine precision.
#
# Usage: Rscript dsl_reference.R [output directory, default "."]

suppressMessages(library(dsl))
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) > 0) args[1] else "."
seed <- 1234
K <- 5

dsl_folds <- function(G, K, seed) {
  set.seed(seed)
  id_base0 <- rep(1:K, each = floor(G / K))
  extra <- G - length(id_base0)
  if (extra > 0) id_base0 <- c(id_base0, 1:extra)
  sample(id_base0, size = length(id_base0), replace = FALSE)
}

results <- list()
vcovs <- list()
run_case <- function(case, data, cols, cluster = NULL, file = case, ...) {
  fit <- NULL
  invisible(capture.output(fit <- dsl(data = data, cluster = cluster,
                                      sl_method = "lm", sample_split = 1,
                                      cross_fit = K, seed = seed, ...)))
  # fold of each row, as drawn inside dsl
  if (is.null(cluster)) {
    fold <- dsl_folds(nrow(data), K, seed)
  } else {
    cid <- as.numeric(as.factor(data[, cluster]))
    fb <- dsl_folds(length(unique(cid)), K, seed)
    fold <- fb[match(cid, sort(unique(cid)))]
  }
  d <- data[, cols, drop = FALSE]
  d$fold <- fold
  write.csv(d, file.path(out_dir, paste0("dsl_data_", file, ".csv")), row.names = FALSE,
            na = "")
  results[[case]] <<- data.frame(case = case, term = names(fit$coefficients),
                                 estimate = unname(fit$coefficients),
                                 std_error = unname(fit$standard_errors))
  V <- as.matrix(fit$vcov)
  vcovs[[case]] <<- data.frame(case = case, row = rep(seq_len(nrow(V)), ncol(V)),
                               col = rep(seq_len(ncol(V)), each = nrow(V)),
                               value = as.vector(V))
}

data("data_lm")
run_case("lm", data_lm, c("Y", "pred_Y", "X1", "X2", "X3", "X4", "X5"),
         model = "lm", formula = Y ~ X1 + X2 + X3 + X4 + X5,
         predicted_var = "Y", prediction = "pred_Y")

cl_lm <- data_lm
cl_lm$cl <- rep(1:500, each = 10)
run_case("lm_cluster", cl_lm, c("Y", "pred_Y", "X1", "X2", "X3", "cl"),
         cluster = "cl", model = "lm", formula = Y ~ X1 + X2 + X3,
         predicted_var = "Y", prediction = "pred_Y")

data("data_logit")
run_case("logit_xpred", data_logit, c("Y", "X1", "pred_Y", "pred_X1", "X2", "X4"),
         model = "logit", formula = Y ~ X1 + X2 + X4,
         predicted_var = c("Y", "X1"), prediction = c("pred_Y", "pred_X1"))
run_case("lm_xpred", data_logit, c("Y", "X1", "pred_Y", "pred_X1", "X2", "X4"),
         file = "logit_xpred",
         model = "lm", formula = Y ~ X1 + X2 + X4,
         predicted_var = c("Y", "X1"), prediction = c("pred_Y", "pred_X1"))

data("data_unequal")
run_case("logit_unequal", data_unequal,
         c("Y", "pred_Y", "sample_prob", "X1", "X2", "X3"),
         model = "logit", formula = Y ~ X1 + X2 + X3, predicted_var = "Y",
         prediction = "pred_Y", sample_prob = "sample_prob")
run_case("lm_unequal", data_unequal, file = "logit_unequal",
         c("Y", "pred_Y", "sample_prob", "X1", "X2", "X3"),
         model = "lm", formula = Y ~ X1 + X2 + X3, predicted_var = "Y",
         prediction = "pred_Y", sample_prob = "sample_prob")

data("data_felm")
fdat <- data_felm
fdat$state <- as.character(fdat$state)
run_case("felm_twoways", fdat,
         c("log_gsp", "pred_log_gsp", "log_pcap", "log_pc", "unemp", "state", "year"),
         cluster = "state", model = "felm",
         formula = log_gsp ~ log_pcap + log_pc + unemp, predicted_var = "log_gsp",
         prediction = "pred_log_gsp", index = c("state", "year"),
         fixed_effect = "twoways")
# dsl minimizes the squared moments with L-BFGS-B; in the two-way model the
# regressors are nearly collinear within state and year, and the optimizer stops
# about 1e-3 (relative; 0.002 standard errors) from the root. Cross-check with the
# exact within estimator: lm() on the same cross-fitted pseudo-outcome.
fold_tw <- read.csv(file.path(out_dir, "dsl_data_felm_twoways.csv"))$fold
lab <- !is.na(fdat$log_gsp)
g <- numeric(nrow(fdat))
for (k in 1:K) {
  tr <- fold_tw != k & lab
  cf <- coef(lm(log_gsp ~ pred_log_gsp, data = fdat[tr, ]))
  g[fold_tw == k] <- cf[1] + cf[2] * fdat$pred_log_gsp[fold_tw == k]
}
fdat$pseudo <- g + (lab / mean(lab)) * (ifelse(lab, fdat$log_gsp, 0) - g)
within <- lm(pseudo ~ log_pcap + log_pc + unemp + factor(state) + factor(year),
             data = fdat)
write.csv(data.frame(case = "felm_twoways", term = c("log_pcap", "log_pc", "unemp"),
                     estimate = unname(coef(within)[c("log_pcap", "log_pc", "unemp")])),
          file.path(out_dir, "dsl_reference_within.csv"), row.names = FALSE)
run_case("felm_oneway", fdat,
         c("log_gsp", "pred_log_gsp", "log_pcap", "log_pc", "unemp", "state", "year"),
         model = "felm", formula = log_gsp ~ log_pcap + log_pc + unemp,
         predicted_var = "log_gsp", prediction = "pred_log_gsp", index = c("state"),
         fixed_effect = "oneway")

write.csv(do.call(rbind, results), file.path(out_dir, "dsl_reference.csv"),
          row.names = FALSE)
write.csv(do.call(rbind, vcovs), file.path(out_dir, "dsl_reference_vcov.csv"),
          row.names = FALSE)
write.csv(data.frame(package = c("dsl", "SuperLearner", "R"),
                     version = c(as.character(packageVersion("dsl")),
                                 as.character(packageVersion("SuperLearner")),
                                 paste(R.version$major, R.version$minor, sep = "."))),
          file.path(out_dir, "dsl_reference_info.csv"), row.names = FALSE)
