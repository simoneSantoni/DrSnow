# Reference values for the marginal-treatment-effect functions.
#
# Regenerate (from the repository root) with
#   Rscript test/validation/iv/make_reference_mte.R
# Writes mte.csv (simulated normal-selection data) and reference_mte.csv (case,
# quantity, value), read by test/iv/test_mte.jl.
#
# Independent implementations used as references (coded from the papers; no R
# package implements exactly these estimators):
#   - probit propensity score: glm(binomial("probit"));
#   - polynomial MTE (Brinch, Mogstad & Wiswall 2017) and parametric normal local IV:
#     lm of y on x, p, p:x and p^2, p^3 (resp. dnorm(qnorm(p))), treatment-effect
#     parameters by numerical integration of the fitted MTE (stats::integrate);
#   - semiparametric local IV (Carneiro, Heckman & Vytlacil 2011): Gaussian-kernel
#     local linear residualization and local quadratic regression written out with
#     explicit weighted least squares; cross-checked against KernSmooth::locpoly.
# localIV::mte is not used as a reference because it residualizes with loess.

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
set.seed(20260929)

n <- 2000
z <- rnorm(n); x <- rnorm(n); V <- rnorm(n)
U0 <- 0.3 * V + rnorm(n); U1 <- -0.5 * V + rnorm(n)
D <- as.numeric(V <= 0.8 * z + 0.3 * x)
y <- ifelse(D == 1, 1 + 1.5 * x + U1, x + U0)
dat <- data.frame(y = y, d = D, z = z, x = x)
write.csv(dat, file.path(here, "mte.csv"), row.names = FALSE)

out <- list()
add <- function(case, quantity, value) {
  out[[length(out) + 1]] <<- data.frame(case = case, quantity = quantity,
                                        value = as.numeric(value))
}

ps_fit <- glm(d ~ x + z, family = binomial("probit"), data = dat)
p <- fitted(ps_fit)
lo <- max(min(p[D == 1]), min(p[D == 0])); hi <- min(max(p[D == 1]), max(p[D == 0]))
add("mte", "p1", p[1]); add("mte", "lower", lo); add("mte", "upper", hi)
keep <- p >= lo & p <= hi
add("mte", "n_keep", sum(keep))
yk <- y[keep]; xk <- x[keep]; pk <- p[keep]
xbar <- mean(xk)
ppol <- pmin(pk + 0.1, 1)

params <- function(case, kfun, delta) {
  K <- function(u) sapply(u, function(v) if (v == 0) 0 else
    integrate(kfun, 0, v, rel.tol = 1e-12, abs.tol = 1e-14)$value)
  mx <- xk * delta
  K1 <- K(1); Kp <- K(pk)
  add(case, "ATE", mean(mx) + K1)
  add(case, "ATT", sum(pk * mx + Kp) / sum(pk))
  add(case, "ATU", sum((1 - pk) * mx + K1 - Kp) / sum(1 - pk))
  add(case, "LATE", mean(mx) + (K(hi) - K(lo)) / (hi - lo))
  add(case, "PRTE", sum((ppol - pk) * mx + K(ppol) - Kp) / sum(ppol - pk))
  add(case, "mte_050", xbar * delta + kfun(0.5))
  add(case, "mte_020", xbar * delta + kfun(0.2))
}

# polynomial of degree 2 in u (K cubic)
m <- lm(yk ~ xk + pk + I(pk * xk) + I(pk^2) + I(pk^3))
b <- coef(m)
kpol <- function(u) b["pk"] + 2 * b["I(pk^2)"] * u + 3 * b["I(pk^3)"] * u^2
params("poly", kpol, b["I(pk * xk)"])
# normal
phiq <- dnorm(qnorm(pk))
m <- lm(yk ~ xk + pk + I(pk * xk) + phiq)
b <- coef(m)
knorm <- function(u) b["pk"] - b["phiq"] * qnorm(u)
params("normal", knorm, b["I(pk * xk)"])

# semiparametric local IV with explicit Gaussian-kernel local polynomials
h <- 0.15; hr <- 0.05
loclin_fit <- function(xv, yv, h) {
  sapply(xv, function(x0) {
    w <- dnorm((xv - x0) / h)
    coef(lm.wfit(cbind(1, xv - x0), yv, w))[1]
  })
}
locquad <- function(xv, yv, x0, h) {
  w <- dnorm((xv - x0) / h)
  coef(lm.wfit(cbind(1, xv - x0, (xv - x0)^2), yv, w))[1:2]
}
M <- cbind(yk, xk, pk * xk)
R <- M - apply(M, 2, function(col) loclin_fit(pk, col, hr))
bs <- coef(lm.fit(R[, 2:3], R[, 1]))
yt <- yk - xk * bs[1] - pk * xk * bs[2]
grid <- lo + (hi - lo) * (0:20) / 20
lq <- sapply(grid, function(u) locquad(pk, yt, u, h))
add("semi", "delta", bs[2])
add("semi", "beta0", bs[1])
add("semi", "late", xbar * bs[2] + (lq[1, 21] - lq[1, 1]) / (hi - lo))
for (j in c(5, 10, 15)) add("semi", paste0("mte_grid", j), xbar * bs[2] + lq[2, j + 1])
# no covariates: exact local quadratic derivative and KernSmooth cross-check
lq0 <- sapply(grid, function(u) locquad(pk, yk, u, h))
for (j in c(5, 10, 15)) add("semi0", paste0("mte_grid", j), lq0[2, j + 1])
lp <- KernSmooth::locpoly(pk, yk, drv = 1L, degree = 2, bandwidth = h,
                          gridsize = 401L, range.x = c(lo, hi))
for (j in c(5, 10, 15)) add("semi0", paste0("locpoly_grid", j), approx(lp$x, lp$y, grid[j + 1])$y)

res <- do.call(rbind, out)
write.csv(res, file.path(here, "reference_mte.csv"), row.names = FALSE)
cat("wrote", nrow(res), "reference values\n")
