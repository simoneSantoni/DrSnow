# Reference values for (1) the heteroskedasticity- / cluster-robust CLR and K tests
# (Kleibergen 2005, as in Stata's weakiv) and (2) the parametric LATE extrapolation
# (Angrist & Fernández-Val 2013) on the Card (1995) data (card.csv).
#
# Regenerate (from the repository root) with
#   Rscript test/validation/iv/make_reference_robust_extrap.R
# Writes reference_robust_extrap.csv (case, quantity, value).
#
# No R package implements the robust CLR, so (1) codes the statistics from
# Kleibergen (2005) / Finlay & Magnusson (weakiv) independently of DrSnow: controls
# are partialled out with lm(), the reduced-form covariance is built from explicit
# score matrices with the Stata / fixest small-sample factors, and the conditional
# p-value of the LR statistic is computed with stats::integrate of the Andrews,
# Moreira & Stock (2007) formula.
# (2) sets up the extrapolation as a just-identified GMM problem (first stage,
# interacted 2SLS, target means) and computes the sandwich covariance with a
# numerical Jacobian (numDeriv) — a different route from DrSnow's analytic influence
# functions.

suppressPackageStartupMessages({
  library(numDeriv)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
card <- read.csv(file.path(here, "card.csv"))
out <- list()
add <- function(case, quantity, value) {
  out[[length(out) + 1]] <<- data.frame(case = case, quantity = quantity,
                                        value = as.numeric(value))
}

# ---------------------------------------------------------------------------
# (1) robust CLR / K
# ---------------------------------------------------------------------------
ctrl <- c("exper", "expersq", "black", "smsa", "south")
W <- cbind(1, as.matrix(card[ctrl]))
res <- function(v) as.vector(lm.fit(W, v)$residuals)
y <- res(card$lwage); d <- res(card$educ)
Z <- cbind(res(card$nearc4), res(card$nearc2))
n <- nrow(card); k <- 2; kW <- ncol(W)
ZZi <- solve(crossprod(Z))
gam <- ZZi %*% crossprod(Z, y); pii <- ZZi %*% crossprod(Z, d)
ey <- as.vector(y - Z %*% gam); ed <- as.vector(d - Z %*% pii)
vc <- function(type) {
  S <- cbind(Z * ey, Z * ed)
  dof <- n - k - kW
  if (type == "hc1") {
    M <- crossprod(S) * n / dof
  } else {
    g <- card$region
    Sg <- rowsum(S, g)
    G <- nrow(Sg)
    M <- crossprod(Sg) * (n - 1) / dof * G / (G - 1)
  }
  B <- kronecker(diag(2), ZZi)
  B %*% M %*% B
}
lrp <- function(m, qt, k) {
  if (m <= 0) return(1)
  f <- function(s) pchisq((qt + m) / (1 + qt * s^2 / m), k) * (1 - s^2)^((k - 3) / 2)
  K <- gamma(k / 2) / (sqrt(pi) * gamma((k - 1) / 2))
  1 - 2 * K * integrate(f, 0, 1, rel.tol = 1e-12)$value
}
stats_at <- function(b0, V) {
  V11 <- V[1:k, 1:k]; V12 <- V[1:k, k + 1:k]; V22 <- V[k + 1:k, k + 1:k]
  g <- gam - b0 * pii
  Om <- V11 - b0 * (V12 + t(V12)) + b0^2 * V22
  Del <- t(V12) - b0 * V22                      # Cov(pi, g)
  Dt <- pii - Del %*% solve(Om, g)
  Psi <- V22 - Del %*% solve(Om, t(Del))
  AR <- as.numeric(t(g) %*% solve(Om, g))
  K <- as.numeric((t(Dt) %*% solve(Om, g))^2 / (t(Dt) %*% solve(Om, Dt)))
  J <- AR - K
  rk <- as.numeric(t(Dt) %*% solve(Psi, Dt))
  LR <- 0.5 * (AR - rk + sqrt((AR + rk)^2 - 4 * J * rk))
  c(AR = AR, K = K, rk = rk, LR = LR, p_clr = lrp(LR, rk, k),
    p_k = pchisq(K, 1, lower.tail = FALSE))
}
for (type in c("hc1", "cluster")) {
  V <- vc(type)
  for (b0 in c(0, 0.1, 0.3)) {
    s <- stats_at(b0, V)
    for (nm in names(s)) add(paste0("rclr_", type), paste0(nm, "_", b0), s[nm])
  }
}

# ---------------------------------------------------------------------------
# (2) parametric LATE extrapolation: treatment somecol, instrument nearc4,
#     covariates exper and black
# ---------------------------------------------------------------------------
yv <- card$lwage; dv <- card$somecol; zv <- card$nearc4
if (mean(dv[zv == 1]) < mean(dv[zv == 0])) zv <- 1 - zv
X <- cbind(1, card$exper, card$black)
q <- ncol(X)
W1 <- cbind(X, zv * X); R <- cbind(X, dv * X)
pihat <- solve(crossprod(W1), crossprod(W1, dv))
th <- solve(crossprod(W1, R), crossprod(W1, yv))
delta <- th[(q + 1):(2 * q)]
targets <- c("compliers", "population", "treated", "untreated", "always_takers",
             "never_takers")
sfun <- function(t, pi) {
  switch(t, compliers = as.vector(X %*% pi[(q + 1):(2 * q)]),
         population = rep(1, n), treated = dv, untreated = 1 - dv,
         always_takers = as.vector(X %*% pi[1:q]),
         never_takers = as.vector(1 - X %*% (pi[1:q] + pi[(q + 1):(2 * q)])))
}
est <- sapply(targets, function(t) {
  s <- sfun(t, pihat); sum(s * (X %*% delta)) / sum(s)
})
par <- c(pihat, th, est)
mom <- function(p) {
  pi <- p[1:(2 * q)]; thp <- p[2 * q + 1:(2 * q)]; tg <- p[4 * q + seq_along(targets)]
  dl <- thp[(q + 1):(2 * q)]
  m1 <- W1 * as.vector(dv - W1 %*% pi)
  m2 <- W1 * as.vector(yv - R %*% thp)
  m3 <- sapply(seq_along(targets), function(j) {
    s <- sfun(targets[j], pi); s * (as.vector(X %*% dl) - tg[j])
  })
  cbind(m1, m2, m3)
}
M0 <- mom(par)
G <- jacobian(function(p) colMeans(mom(p)), par)
S <- crossprod(M0) / n
Vp <- solve(G) %*% S %*% t(solve(G)) / n
idx <- 4 * q + seq_along(targets)
for (j in seq_along(targets)) {
  add("extrap_linear", paste0(targets[j], "_coef"), est[j])
  add("extrap_linear", paste0(targets[j], "_se_hc0"), sqrt(Vp[idx[j], idx[j]]))
}
for (j in 1:q) add("extrap_linear", paste0("delta_", j), delta[j])

res_ <- do.call(rbind, out)
res_$value <- sprintf("%.15g", res_$value)
write.csv(res_, file.path(here, "reference_robust_extrap.csv"), row.names = FALSE,
          quote = FALSE)
print(res_)
