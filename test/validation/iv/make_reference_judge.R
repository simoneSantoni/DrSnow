# Reference values for the judge-design functions.
#
# Regenerate (from the repository root) with
#   Rscript test/validation/iv/make_reference_judge.R
# Writes judge.csv (a simulated judge design) and reference_judge.csv (columns case,
# quantity, value), read by test/iv/test_judge.jl.
#
# Independent implementations used as references
#   - residualized leave-one-out leniency: lm residuals on court dummies, judge sums;
#   - 2SLS with court fixed effects and judge-clustered SEs: fixest::feols;
#   - UJIVE with judge indicators as instruments and court dummies as controls:
#     explicit hat matrices (Kolesar 2013), CR1 cluster variance by judge;
#   - balance test: fixest::feols of leniency on case characteristics with court
#     fixed effects, judge-clustered Wald statistic.

suppressPackageStartupMessages(library(fixest))
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
set.seed(20260927)

n_courts <- 8; judges <- 6; cases <- 40
J <- n_courts * judges
court_of <- rep(1:n_courts, each = judges)
lam <- 0.2 + 0.5 * runif(J) + 0.1 * rnorm(n_courts)[court_of]
n <- J * cases
court <- sample(1:n_courts, n, replace = TRUE)
judge <- (court - 1) * judges + sample(1:judges, n, replace = TRUE)
x <- rnorm(n); g <- as.numeric(runif(n) < 0.5)
U <- pnorm(0.7 * x + rnorm(n))
d <- as.numeric(U < lam[judge])
y <- 0.5 * x + d * (1 + 0.8 * (U - 0.5)) + rnorm(n)
# a judge with a single case (dropped by DrSnow with a warning)
dat <- data.frame(y = y, d = d, judge = judge, court = court, x = x, g = g)
dat <- rbind(dat, data.frame(y = 0.3, d = 1, judge = 999, court = 1, x = 0, g = 0))
write.csv(dat, file.path(here, "judge.csv"), row.names = FALSE)
dat <- dat[dat$judge != 999, ]
n <- nrow(dat)

out <- list()
add <- function(case, quantity, value) {
  out[[length(out) + 1]] <<- data.frame(case = case, quantity = quantity,
                                        value = as.numeric(value))
}

loo <- function(v, j) {
  s <- ave(v, j, FUN = sum); c <- ave(v, j, FUN = length)
  (s - v) / (c - 1)
}
dstar <- resid(lm(d ~ factor(court), data = dat))
dat$z <- loo(dstar, dat$judge)
dat$zraw <- loo(dat$d, dat$judge)
add("judge", "leniency_1", dat$z[1])
add("judge", "leniency_2", dat$z[2])
add("judge", "leniency_sd", sd(dat$z))

m <- feols(y ~ 1 | court | d ~ z, data = dat, cluster = ~judge)
add("judge", "tsls_coef", coef(m)["fit_d"])
add("judge", "tsls_se", se(m)["fit_d"])
fs <- feols(d ~ z | court, data = dat, cluster = ~judge)
add("judge", "fs_coef", coef(fs)["z"])
add("judge", "fs_se", se(fs)["z"])
m0 <- feols(y ~ 1 | court | d ~ zraw, data = dat, cluster = ~judge)
add("judge", "tsls_raw_coef", coef(m0)["fit_d"])

# UJIVE with judge indicators
Zd <- model.matrix(~ factor(judge) - 1, data = dat)
W <- model.matrix(~ factor(court) - 1, data = dat)
hat <- function(A) { Q <- qr.Q(qr(A)); Q[, 1:qr(A)$rank] %*% t(Q[, 1:qr(A)$rank]) }
PZ <- hat(cbind(Zd, W)); PW <- hat(W)
hZ <- diag(PZ); hW <- diag(PW)
xu <- as.numeric((PZ %*% dat$d - hZ * dat$d) / (1 - hZ) - (PW %*% dat$d - hW * dat$d) / (1 - hW))
bu <- sum(xu * dat$y) / sum(xu * dat$d)
e <- as.numeric((diag(n) - PW) %*% (dat$y - bu * dat$d))
sc <- rowsum(xu * e, dat$judge)
G <- length(unique(dat$judge)); K <- 1 + n_courts
V <- sum(sc^2) / sum(xu * dat$d)^2 * (n - 1) / (n - K) * G / (G - 1)
add("judge", "ujive_coef", bu)
add("judge", "ujive_se", sqrt(V))

# balance
b <- feols(z ~ x + g | court, data = dat, cluster = ~judge)
cf <- coef(b)[c("x", "g")]; Vb <- vcov(b)[c("x", "g"), c("x", "g")]
add("judge", "balance_F", as.numeric(t(cf) %*% solve(Vb, cf)) / 2)

res <- do.call(rbind, out)
write.csv(res, file.path(here, "reference_judge.csv"), row.names = FALSE)
cat("wrote", nrow(res), "reference values\n")
