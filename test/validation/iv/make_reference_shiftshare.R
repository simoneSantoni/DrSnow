# Reference values for the shift-share functions.
#
# Regenerate (from the repository root) with
#   Rscript test/validation/iv/make_reference_shiftshare.R
# Writes shiftshare.csv (simulated regions with share columns s1..s40),
# shiftshare_shocks.csv (sector, shock, cluster, q, and 199 permuted shock draws)
# and reference_shiftshare.csv (case, quantity, value); read by
# test/iv/test_shift_share.jl.
#
# Independent implementations used as references
#   - EHW and AKM standard errors, AKM0 confidence interval: ShiftShareSE::ivreg_ss
#     (Kolesar), with and without sector clusters, weighted;
#   - BHJ shock-level regression: aggregation coded from Borusyak, Hull & Jaravel
#     (2022) and lm + sandwich (HC0 / cluster HC0) at the shock level;
#   - Rotemberg weights: coded from Goldsmith-Pinkham, Sorkin & Swift (2020);
#   - randomization inference with counterfactual shocks (Borusyak & Hull 2023):
#     recentered instrument, p-value for beta = 0 by direct enumeration.

suppressPackageStartupMessages({
  library(ShiftShareSE); library(sandwich)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
set.seed(20260928)

n <- 300; K <- 40; C <- 8
cl <- rep(1:C, length.out = K)
g <- rnorm(K) + 0.5 * rnorm(C)[cl]
q <- rnorm(K)
raw <- matrix(rgamma(n * K, shape = 0.3), n, K)
tot <- runif(n, 0.4, 0.9)
S <- raw / rowSums(raw) * tot
colnames(S) <- paste0("s", 1:K)
x1 <- rnorm(n)
pop <- exp(rnorm(n, 0, 0.5))
eta <- rnorm(K)
B <- as.numeric(S %*% g)
v <- rnorm(n)
u <- as.numeric(S %*% eta) * 2 + 0.5 * v + rnorm(n)
d <- 0.8 * B + 0.3 * x1 + v
y <- 1.5 * d + x1 + u
dat <- data.frame(y = y, d = d, x1 = x1, pop = pop, S)
write.csv(dat, file.path(here, "shiftshare.csv"), row.names = FALSE)
R <- 199
draws <- sapply(1:R, function(r) sample(g))
sh <- data.frame(sector = paste0("s", 1:K), shock = g, cluster = cl, q = q, draws)
write.csv(sh, file.path(here, "shiftshare_shocks.csv"), row.names = FALSE)

out <- list()
add <- function(case, quantity, value) {
  out[[length(out) + 1]] <<- data.frame(case = case, quantity = quantity,
                                        value = as.numeric(value))
}
dat$B <- B
dat$ssum <- rowSums(S)

for (spec in list(list("unw", NULL, NULL), list("w", dat$pop, NULL),
                  list("wcl", dat$pop, cl))) {
  case <- spec[[1]]; w <- spec[[2]]; sc <- spec[[3]]
  fit <- if (is.null(w)) {
    ivreg_ss(y ~ x1 + ssum | d, X = B, data = dat, W = S,
             method = c("ehw", "akm", "akm0"), sector_cvar = sc)
  } else {
    ivreg_ss(y ~ x1 + ssum | d, X = B, data = dat, W = S, weights = w,
             method = c("ehw", "akm", "akm0"), sector_cvar = sc)
  }
  add(case, "beta", fit$beta)
  add(case, "se_ehw", fit$se["EHW"])
  add(case, "se_akm", fit$se["AKM"])
  add(case, "akm0_lower", fit$ci.l["AKM0"])
  add(case, "akm0_upper", fit$ci.r["AKM0"])
  add(case, "akm0_p_b0", fit$p["AKM0"])
  # BHJ shock-level regression
  ww <- if (is.null(w)) rep(1, n) else w
  Z <- cbind(1, dat$x1, dat$ssum)
  res <- function(v) as.numeric(lm.wfit(Z, v, ww)$residuals)
  yp <- res(dat$y); dp <- res(dat$d)
  sk <- colSums(ww * S)
  yb <- colSums(ww * S * yp) / sk; db <- colSums(ww * S * dp) / sk
  gi <- lm.wfit(cbind(1, g), db, sk)
  # IV at the shock level with intercept: instrument g
  gt <- g - sum(sk * g) / sum(sk)
  bb <- sum(sk * gt * yb) / sum(sk * gt * db)
  a0 <- sum(sk * (yb - bb * db)) / sum(sk)
  e <- yb - bb * db - a0
  sc_ <- sk * gt * e
  if (!is.null(sc)) sc_ <- tapply(sc_, factor(sc), sum)
  add(case, "bhj_beta", bb)
  add(case, "bhj_se", sqrt(sum(sc_^2)) / abs(sum(sk * gt * db)))
  # Rotemberg weights (GPSS 2020)
  Sp <- apply(S, 2, res)
  sx <- colSums(ww * Sp * dp); sy <- colSums(ww * Sp * yp)
  alpha <- g * sx / sum(g * sx)
  add(case, "rot_alpha_s1", alpha[1])
  add(case, "rot_alpha_s7", alpha[7])
  add(case, "rot_beta_s7", sy[7] / sx[7])
  add(case, "rot_negsum", sum(alpha[alpha < 0]))
  add(case, "rot_est", sum(alpha * sy / sx))
  # randomization inference with the permuted shocks (recentered instrument)
  mu <- as.numeric(S %*% rowMeans(draws))
  Bc <- res(as.numeric(S %*% g) - mu)
  T0 <- sum(ww * Bc * yp)
  Tr <- sapply(1:R, function(r) sum(ww * res(as.numeric(S %*% draws[, r]) - mu) * yp))
  add(case, "ri_p_b0", (1 + sum(abs(Tr) >= abs(T0) - 1e-9 * max(1, abs(T0)))) / (1 + R))
  add(case, "rc_beta", sum(ww * Bc * yp) / sum(ww * Bc * dp))
}

res <- do.call(rbind, out)
write.csv(res, file.path(here, "reference_shiftshare.csv"), row.names = FALSE)
cat("wrote", nrow(res), "reference values\n")
