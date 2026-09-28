# Reference values for the many-instrument IV estimators (k-class, JIVE, jackknife AR).
#
# Regenerate (from the repository root) with
#   Rscript test/validation/iv/make_reference_manyiv.R
# Reads card.csv (written by make_reference.R); writes reference_manyiv.csv with
# columns case, quantity, value (read by test/iv/test_validation_manyiv.jl).
#
# Independent implementations used as references
#   - LIML / Fuller point estimates, kappa, homoskedastic, HC0 and CR0 standard errors:
#     ivmodel (Kang, Jiang, Zhao & Small). ivmodel's heteroskedastic / cluster SEs
#     have no small-sample factor; the Julia test rescales DrSnow's HC1 / CR1.
#   - Bekker (1994) SEs: Hansen, Hausman & Newey (2008) Sigma_B on the full
#     (unpartialled) regressor matrix [D, W], coded from the paper.
#   - HLIM / HFUL and the Hausman et al. (2012) variance: coded from the paper with
#     explicit n x n projection matrices after partialling out W.
#   - JIVE1 / JIVE2: leave-one-out fitted values from explicit hat matrices, then
#     AER::ivreg + sandwich HC1. UJIVE (Kolesar 2013): explicit G matrix; EHW-type and
#     many-instrument robust variances from explicit n x n sums.
#   - Mikusheva & Sun (2022) jackknife AR statistic from explicit n x n matrices.
#
# Many-instrument specification: lwage on educ, instruments nearc4 x region (9) and
# nearc2 x region (9), controls exper, expersq, black, smsa, south and region
# dummies (absorbed as a fixed effect in DrSnow).

suppressPackageStartupMessages({
  library(AER); library(sandwich); library(ivmodel)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
cd <- read.csv(file.path(here, "card.csv"))
out <- list()
add <- function(case, quantity, value) {
  out[[length(out) + 1]] <<- data.frame(case = case, quantity = quantity,
                                        value = as.numeric(value))
}
ctrl <- c("exper", "expersq", "black", "smsa", "south")
n <- nrow(cd)
y <- cd$lwage; d <- cd$educ
Xc <- as.matrix(cd[, ctrl])

# ----------------------------------------------------------------------------
# Case card_kc: two instruments, controls; ivmodel LIML / Fuller
# ----------------------------------------------------------------------------
Z2 <- as.matrix(cd[, c("nearc4", "nearc2")])
iv2 <- ivmodel(Y = y, D = d, Z = Z2, X = Xc)
for (spec in list(list("liml", NULL), list("fuller1", 1), list("fuller4", 4))) {
  nm <- spec[[1]]
  fit <- function(...) if (is.null(spec[[2]])) LIML(iv2, ...) else Fuller(iv2, b = spec[[2]], ...)
  f0 <- fit()
  add("card_kc", paste0(nm, "_coef"), f0$point.est)
  add("card_kc", paste0(nm, "_kappa"), f0$k)
  add("card_kc", paste0(nm, "_se_iid"), f0$std.err)
  add("card_kc", paste0(nm, "_se_hc0"), fit(heteroSE = TRUE)$std.err)
  add("card_kc", paste0(nm, "_se_cr0"), fit(clusterID = cd$region)$std.err)
}

# ----------------------------------------------------------------------------
# Many-instrument case
# ----------------------------------------------------------------------------
R <- sapply(1:9, function(r) as.numeric(cd$region == r))
Zm <- cbind(cd$nearc4 * R, cd$nearc2 * R)
colnames(Zm) <- c(paste0("z4r", 1:9), paste0("z2r", 1:9))
W <- cbind(Xc, R)                      # region dummies span the intercept
K <- ncol(Zm)

run_many <- function(case, w) {
  sw <- sqrt(w)
  yt <- y * sw; dt <- d * sw; Zt <- Zm * sw; Wt <- W * sw
  hat <- function(A) { Q <- qr.Q(qr(A)); Q %*% t(Q) }
  PW <- hat(Wt); PZ <- hat(cbind(Zt, Wt))
  hW <- diag(PW); hZ <- diag(PZ)
  MW <- diag(n) - PW
  yp <- as.numeric(MW %*% yt); dp <- as.numeric(MW %*% dt)
  L <- K + ncol(W); Kx <- 1 + ncol(W)
  # --- LIML / Fuller (unweighted: ivmodel; any weights: explicit) --------------
  Yb <- cbind(yt, dt)
  MZ <- diag(n) - PZ
  kl <- min(Re(eigen(solve(t(Yb) %*% MZ %*% Yb) %*% (t(Yb) %*% MW %*% Yb))$values))
  kc <- function(kap) {
    Dk <- dp - kap * as.numeric(MZ %*% dt)
    b <- sum(Dk * yp) / sum(Dk * dp)
    e <- yp - dp * b
    list(b = b, e = e, se = sqrt(sum(e^2) / (n - Kx) / sum(Dk * dp)))
  }
  for (spec in list(list("liml", 0), list("fuller1", 1))) {
    kap <- kl - spec[[2]] / (n - L)
    f <- kc(kap)
    add(case, paste0(spec[[1]], "_coef"), f$b)
    add(case, paste0(spec[[1]], "_kappa"), kap)
    add(case, paste0(spec[[1]], "_se_iid"), f$se)
    # Bekker: HHN (2008) Sigma_B with full regressors X = [D, W] and P = PZ
    X <- cbind(dt, Wt)
    bfull <- solve(t(X) %*% (diag(n) - kap * MZ) %*% X, t(X) %*% (diag(n) - kap * MZ) %*% yt)
    u <- as.numeric(yt - X %*% bfull)
    s2 <- sum(u^2) / (n - ncol(X))
    at <- sum(u * (PZ %*% u)) / sum(u^2)
    Xt <- X - u %*% (t(u) %*% X) / sum(u^2)
    H <- t(X) %*% PZ %*% X - at * t(X) %*% X
    SB <- s2 * ((1 - at)^2 * t(Xt) %*% PZ %*% Xt + at^2 * t(Xt) %*% MZ %*% Xt)
    Vb <- solve(H) %*% SB %*% solve(H)
    add(case, paste0(spec[[1]], "_se_bekker"), sqrt(Vb[1, 1]))
  }
  if (all(w == 1)) {
    ivm <- ivmodel(Y = y, D = d, Z = Zm, X = W, intercept = FALSE)
    add(case, "ivmodel_liml_coef", LIML(ivm)$point.est)
    add(case, "ivmodel_liml_se_iid", LIML(ivm)$std.err)
    add(case, "ivmodel_fuller1_coef", Fuller(ivm, b = 1)$point.est)
  }
  # --- HLIM / HFUL on the partialled data (Hausman et al. 2012) ---------------
  Zp <- MW %*% Zt
  P <- hat(Zp); hp <- diag(P); Pt <- P - diag(hp)
  Xb <- cbind(yp, dp)
  at <- min(Re(eigen(solve(t(Xb) %*% Xb) %*% (t(Xb) %*% Pt %*% Xb))$values))
  for (spec in list(list("hlim", at), list("hful", (at - (1 - at) / n) / (1 - (1 - at) / n)))) {
    a <- spec[[2]]
    H <- sum(dp * (Pt %*% dp)) - a * sum(dp^2)
    b <- (sum(dp * (Pt %*% yp)) - a * sum(dp * yp)) / H
    e <- yp - dp * b
    xh <- dp - e * sum(e * dp) / sum(e^2)
    xd <- as.numeric(Pt %*% xh)
    S1 <- sum(e^2 * xd^2)
    A <- xh * e
    S2 <- sum((Pt^2) * outer(A, A))
    add(case, paste0(spec[[1]], "_coef"), b)
    add(case, paste0(spec[[1]], "_alpha"), a)
    add(case, paste0(spec[[1]], "_se"), sqrt((S1 + S2) / H^2))
  }
  # --- JIVE1 / JIVE2 / UJIVE ----------------------------------------------------
  PZd <- as.numeric(PZ %*% dt); PWd <- as.numeric(PW %*% dt)
  xj1 <- (PZd - hZ * dt) / (1 - hZ)
  xj2 <- PZd - hZ * dt
  for (spec in list(list("jive1", xj1), list("jive2", xj2))) {
    xh <- spec[[2]]
    m <- ivreg(yt ~ dt + Wt - 1 | xh + Wt - 1)
    add(case, paste0(spec[[1]], "_coef"), coef(m)["dt"])
    add(case, paste0(spec[[1]], "_se_hc1"), sqrt(vcovHC(m, type = "HC1")["dt", "dt"]))
    add(case, paste0(spec[[1]], "_se_iid"), sqrt(vcov(m)["dt", "dt"]))
  }
  G <- diag(1 / (1 - hZ)) %*% (PZ - diag(hZ)) - diag(1 / (1 - hW)) %*% (PW - diag(hW))
  xu <- as.numeric(G %*% dt)
  bu <- sum(xu * yt) / sum(xu * dt)
  e <- yp - dp * bu
  A <- sum(xu * dt)
  add(case, "ujive_coef", bu)
  add(case, "ujive_se_hc1", sqrt(sum(xu^2 * e^2) / A^2 * n / (n - Kx)))
  a <- dt * e
  Tm <- sum(G * t(G) * outer(a, a)) - sum(diag(G)^2 * a^2)
  add(case, "ujive_se_many", sqrt((sum(xu^2 * e^2) + Tm) / A^2))
  # --- Mikusheva & Sun (2022) jackknife AR on the partialled data -------------
  M <- diag(n) - P
  Pt2 <- Pt^2 / (outer(diag(M), diag(M)) + M^2)
  diag(Pt2) <- 0
  for (b0 in c(0, 0.1)) {
    e0 <- yp - b0 * dp
    num <- sum(e0 * (Pt %*% e0))
    f <- e0 * as.numeric(M %*% e0)
    Phi <- 2 / K * sum(Pt2 * outer(f, f))
    add(case, paste0("jar_stat_b", b0), num / sqrt(K * Phi))
  }
}

run_many("card_many", rep(1, n))
run_many("card_many_w", cd$weight / mean(cd$weight))

res <- do.call(rbind, out)
write.csv(res, file.path(here, "reference_manyiv.csv"), row.names = FALSE)
cat("wrote", nrow(res), "reference values\n")
