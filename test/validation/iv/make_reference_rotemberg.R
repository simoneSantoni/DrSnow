# Reference values for panel Rotemberg weights (Goldsmith-Pinkham, Sorkin & Swift
# 2020) on the Autor, Dorn & Hanson (2013) data shipped with GPSS's R package
# `bartik.weight` (github.com/paulgp/bartik-weight, R-code/pkg, version 0.1.0).
#
# Regenerate (from the repository root) with
#   Rscript test/validation/iv/make_reference_rotemberg.R
# Writes
#   adh_panel.csv         czone × year rows: outcome, treatment, controls, weight and
#                         the shares of the 25 industries with the largest total
#                         employment share (sh_<ind>, the share in that row's year);
#   adh_shocks.csv        ind, year, trade_ (industry-period import growth);
#   reference_rotemberg.csv   (case, quantity, value).
# References:
#   - bartik.weight::bw on the 50 sector-period instruments t<year> × sh_<ind>
#     (alpha, beta per industry-year);
#   - the GPSS aggregation across periods coded as in their
#     make_rotemberg_summary_ADH.do (alpha_k = Σ_t alpha_kt,
#     beta_k = Σ_t alpha_kt beta_kt / alpha_k, g_k = Σ_t alpha_kt g_kt / alpha_k);
#   - the overidentified 2SLS with the 50 instruments (AER::ivreg, weighted) and its
#     Rotemberg weights a_k s_k'Mx / Σ a s'Mx with a = first-stage coefficients,
#     written out with explicit matrices.

suppressPackageStartupMessages({
  library(bartik.weight)
  library(AER)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))

controls <- c("reg_midatl", "reg_encen", "reg_wncen", "reg_satl", "reg_escen",
              "reg_wscen", "reg_mount", "reg_pacif", "l_sh_popedu_c", "l_sh_popfborn",
              "l_sh_empl_f", "l_sh_routine33", "l_task_outsource", "t2",
              "l_shind_manuf_cbp")
master <- as.data.frame(ADH_master)
local <- as.data.frame(ADH_local)
global <- as.data.frame(ADH_global)

# 25 industries with the largest total share mass
mass <- tapply(local$sh_ind_, local$ind, sum)
inds <- sort(names(sort(mass, decreasing = TRUE))[1:25])
local <- local[local$ind %in% inds, ]
global <- global[global$ind %in% inds, ]
global <- global[order(global$year, global$ind), ]
years <- c(1990, 2000)

# wide shares: one column per industry (share in the row's year)
key <- paste(master$czone, master$year)
for (k in inds) {
  sub <- local[local$ind == k, ]
  v <- sub$sh_ind_[match(key, paste(sub$czone, sub$year))]
  v[is.na(v)] <- 0
  master[[paste0("sh_", k)]] <- v
}
# sector-period instruments in (year, ind) order, matching `global`
Zn <- character(0)
for (t in years) for (k in inds) {
  nm <- paste0("t", t, "_", k)
  master[[nm]] <- master[[paste0("sh_", k)]] * (master$year == t)
  Zn <- c(Zn, nm)
}
y <- "d_sh_empl_mfg"; x <- "d_tradeusch_pw"; wt <- "timepwt48"
b <- bw(master, y, x, controls, wt, master, Zn, global, "trade_")

out <- list()
add <- function(case, quantity, value) {
  out[[length(out) + 1]] <<- data.frame(case = case, quantity = quantity,
                                        value = as.numeric(value))
}
for (j in seq_len(nrow(b))) {
  add("adh_bartik", paste0("alpha_", b$ind[j], "_", b$year[j]), b$alpha[j])
  add("adh_bartik", paste0("beta_", b$ind[j], "_", b$year[j]), b$beta[j])
}
add("adh_bartik", "estimate", sum(b$alpha * b$beta))
for (k in inds) {
  s <- b[b$ind == k, ]
  ak <- sum(s$alpha)
  add("adh_bartik", paste0("agg_alpha_", k), ak)
  add("adh_bartik", paste0("agg_beta_", k), sum(s$alpha * s$beta) / ak)
  add("adh_bartik", paste0("agg_g_", k), sum(s$alpha * s$trade_) / ak)
}
# the Bartik 2SLS itself
master$bartik <- 0
for (t in years) for (k in inds) {
  g <- global$trade_[global$year == t & global$ind == k]
  master$bartik <- master$bartik + master[[paste0("t", t, "_", k)]] * g
}
fb <- as.formula(paste(y, "~", x, "+", paste(controls, collapse = "+"), "|",
                       "bartik +", paste(controls, collapse = "+")))
add("adh_bartik", "ivreg", coef(ivreg(fb, data = master, weights = master[[wt]]))[x])

# overidentified 2SLS with the 50 share instruments
ft <- as.formula(paste(y, "~", x, "+", paste(controls, collapse = "+"), "|",
                       paste(c(Zn, controls), collapse = "+")))
add("adh_tsls", "ivreg", coef(ivreg(ft, data = master, weights = master[[wt]]))[x])
W <- cbind(as.matrix(master[controls]), 1)
w <- master[[wt]]
Mw <- function(v) v - W %*% solve(crossprod(W, w * W), crossprod(W, w * v))
Z <- as.matrix(master[Zn]); Zp <- Mw(Z); xp <- Mw(master[[x]]); yp <- Mw(master[[y]])
a <- solve(crossprod(Zp, w * Zp), crossprod(Zp, w * xp))
sx <- crossprod(Zp, w * xp); sy <- crossprod(Zp, w * yp)
al <- as.vector(a * sx / sum(a * sx)); be <- as.vector(sy / sx)
add("adh_tsls", "estimate", sum(al * be))
for (j in seq_along(Zn)) {
  add("adh_tsls", paste0("alpha_", global$ind[j], "_", global$year[j]), al[j])
}
for (k in inds) {
  idx <- which(global$ind == k)
  add("adh_tsls", paste0("agg_alpha_", k), sum(al[idx]))
  add("adh_tsls", paste0("agg_beta_", k), sum(al[idx] * be[idx]) / sum(al[idx]))
}

keep <- c("czone", "year", y, x, wt, controls, paste0("sh_", inds))
write.csv(master[keep], file.path(here, "adh_panel.csv"), row.names = FALSE)
write.csv(global, file.path(here, "adh_shocks.csv"), row.names = FALSE)
res <- do.call(rbind, out)
res$value <- sprintf("%.15g", res$value)
write.csv(res, file.path(here, "reference_rotemberg.csv"), row.names = FALSE,
          quote = FALSE)
cat("wrote", nrow(res), "reference values\n")
