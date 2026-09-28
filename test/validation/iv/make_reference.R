# Reference values for the DrSnow IV area.
#
# Regenerate (from the repository root) with
#   Rscript test/validation/iv/make_reference.R
# Packages: wooldridge (data), AER, sandwich, ivmodel, fixest.
# Writes card.csv, mroz.csv and reference.csv next to this script. reference.csv has
# columns case, quantity, value; test/iv/test_validation.jl reads it.
#
# Datasets
#   card.csv  Card (1995) NLS Young Men, 3010 obs (wooldridge::card). Added columns:
#             region (1-9 from reg661-reg669), cl50 (id mod 50, an artificial
#             clustering variable used only to exercise cluster formulas), somecol =
#             1{educ >= 13} (binary treatment for the LATE / compliance checks).
#   mroz.csv  Mroz (1987) married women in the labour force, 428 obs.
#
# Independent implementations used as references
#   - 2SLS coefficients/SEs: AER::ivreg + sandwich (iid, HC1, CR1), fixest (FE,
#     weights, clusters).
#   - AER diagnostics: weak-instrument F, Wu-Hausman, Sargan.
#   - ivmodel: homoskedastic Anderson-Rubin test and set, CLR test and set.
#   - robust AR: brute-force regression of y - b d on Z and controls with HC1 /
#     cluster covariance, endpoints by uniroot (not DrSnow's polynomial method).
#   - Hansen J: two-step efficient GMM on the full (unpartialled) moment set.
#   - effective F, Olea-Pflueger critical values, tF (ivDiag's table / formula),
#     LTZ and UCI (Conley et al. 2012) coded directly from the papers' formulas.
#   - compliance / complier means: textbook arm-mean formulas and ivreg of X*D on D.
#   - IPW LATE with covariates: glm logit + Hajek means; SE by nonparametric bootstrap.

suppressPackageStartupMessages({
  library(wooldridge); library(AER); library(sandwich); library(ivmodel)
  library(fixest)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
out <- list()
add <- function(case, quantity, value) {
  out[[length(out) + 1]] <<- data.frame(case = case, quantity = quantity,
                                        value = as.numeric(value))
}

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
data(card)
card$region <- apply(card[, paste0("reg66", 1:9)], 1, function(r) which(r == 1))
card$cl50 <- card$id %% 50 + 1
card$somecol <- as.integer(card$educ >= 13)
card$agesq <- card$age^2
cd <- card[, c("id", "lwage", "educ", "nearc4", "nearc2", "exper", "expersq", "age",
               "agesq",
               "black", "smsa", "south", "smsa66", "region", "cl50", "weight",
               "somecol")]
write.csv(cd, file.path(here, "card.csv"), row.names = FALSE)

data(mroz)
mz <- subset(mroz, inlf == 1)[, c("lwage", "educ", "exper", "expersq", "fatheduc",
                                  "motheduc", "huseduc")]
write.csv(mz, file.path(here, "mroz.csv"), row.names = FALSE)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
ctrl <- c("exper", "expersq", "black", "smsa", "south")
X <- cbind(1, as.matrix(cd[, ctrl]))
resid_on <- function(v, W) as.numeric(v - W %*% qr.solve(W, v))

vc_of <- function(fit, type, cl = NULL) {
  if (type == "iid") return(vcov(fit))
  if (type == "hc1") return(vcovHC(fit, type = "HC1"))
  vcovCL(fit, cluster = cl, type = "HC1")
}

# AR statistic (F form) with a given covariance type: regress y - b d on Z + X
ar_stat <- function(b, y, d, Z, X, type, cl = NULL) {
  u <- y - b * d
  fit <- lm(u ~ Z + X - 1)
  k <- ncol(Z)
  V <- vc_of(fit, type, cl)[1:k, 1:k, drop = FALSE]
  g <- coef(fit)[1:k]
  as.numeric(t(g) %*% solve(V, g)) / k
}

# endpoints of {b : AR(b) <= c} by scanning then uniroot
ar_set <- function(y, d, Z, X, type, cl, c, center, halfwidth) {
  f <- function(b) ar_stat(b, y, d, Z, X, type, cl) - c
  grid <- center + halfwidth * sinh(seq(-6, 6, length.out = 4001)) / sinh(6)
  v <- sapply(grid, f)
  roots <- c()
  for (i in seq_len(length(grid) - 1)) {
    if (sign(v[i]) != sign(v[i + 1])) {
      roots <- c(roots, uniroot(f, c(grid[i], grid[i + 1]), tol = 1e-13)$root)
    }
  }
  list(roots = roots, left = v[1] <= 0, right = v[length(v)] <= 0)
}

# ----------------------------------------------------------------------------
# Case card_ji: just identified, lwage ~ ctrl | educ ~ nearc4
# ----------------------------------------------------------------------------
f_ji <- lwage ~ educ + exper + expersq + black + smsa + south |
  nearc4 + exper + expersq + black + smsa + south
m_ji <- ivreg(f_ji, data = cd)
for (tp in c("iid", "hc1", "cl")) {
  V <- vc_of(m_ji, tp, ~region)
  add("card_ji", paste0("coef_educ"), coef(m_ji)["educ"])
  add("card_ji", paste0("se_educ_", tp), sqrt(V["educ", "educ"]))
  add("card_ji", paste0("se_exper_", tp), sqrt(V["exper", "exper"]))
}
dg <- summary(m_ji, diagnostics = TRUE)$diagnostics
add("card_ji", "weak_F_iid", dg["Weak instruments", "statistic"])
add("card_ji", "wu_hausman_F_iid", dg["Wu-Hausman", "statistic"])

y <- cd$lwage; d <- cd$educ; Z1 <- as.matrix(cd[, "nearc4", drop = FALSE])
iv1 <- ivmodel(Y = y, D = d, Z = Z1, X = as.matrix(cd[, ctrl]))
ar1 <- AR.test(iv1, beta0 = 0)
add("card_ji", "ar_F_iid_b0", ar1$Fstat)
add("card_ji", "ar_p_iid_b0", ar1$p.value)
add("card_ji", "ar_lower_iid", ar1$ci[1, 1])
add("card_ji", "ar_upper_iid", ar1$ci[1, 2])
# robust AR (HC1) and cluster AR by brute force
for (tp in c("hc1", "cl")) {
  cl <- if (tp == "cl") cd$region else NULL
  add("card_ji", paste0("ar_F_", tp, "_b0"), ar_stat(0, y, d, Z1, X, tp, cl))
  dof <- if (tp == "cl") length(unique(cd$region)) - 1 else nrow(cd) - 1 - ncol(X)
  cc <- qf(0.95, 1, dof)
  s <- ar_set(y, d, Z1, X, tp, cl, cc, coef(m_ji)["educ"], 50)
  add("card_ji", paste0("ar_nroots_", tp), length(s$roots))
  for (i in seq_along(s$roots)) add("card_ji", paste0("ar_root", i, "_", tp), s$roots[i])
  add("card_ji", paste0("ar_left_accepted_", tp), s$left)
}
# first stage with HC1 and effective F (= robust F for k = 1)
fs <- lm(educ ~ nearc4 + exper + expersq + black + smsa + south, data = cd)
Vfs <- vcovHC(fs, type = "HC1")
Frob <- coef(fs)["nearc4"]^2 / Vfs["nearc4", "nearc4"]
add("card_ji", "fs_F_hc1", Frob)
# tF (ivDiag's implementation of LMMP Table 3)
source(file.path(here, "tF_ivDiag.R"))
Vh <- vcovHC(m_ji, type = "HC1")
tfo <- tF(coef(m_ji)["educ"], sqrt(Vh["educ", "educ"]), Frob, prec = 12)
add("card_ji", "tF_cF", tfo["cF"])
add("card_ji", "tF_lower", tfo["CI2.5%"])
add("card_ji", "tF_upper", tfo["CI97.5%"])
# LTZ (Conley et al. 2012; ivDiag formula), prior gamma ~ N(0.01, 0.01^2), HC1
Dt <- resid_on(d, X); Zt <- resid_on(cd$nearc4, X)
A <- solve(t(Dt) %*% Zt %*% solve(t(Zt) %*% Zt) %*% t(Zt) %*% Dt) %*% (t(Dt) %*% Zt)
add("card_ji", "ltz_A", A)
b_ltz <- coef(m_ji)["educ"] - A * 0.01
se_ltz <- sqrt(Vh["educ", "educ"] + A^2 * 0.01^2)
add("card_ji", "ltz_est", b_ltz)
add("card_ji", "ltz_se", se_ltz)
# UCI over gamma in [0, 0.02]: union of HC1 intervals at every vertex
tcrit <- qt(0.975, nrow(cd) - 7)
lo <- Inf; hi <- -Inf
for (g in c(0, 0.02)) {
  cdg <- cd; cdg$ly <- cdg$lwage - g * cdg$nearc4
  mg <- ivreg(ly ~ educ + exper + expersq + black + smsa + south |
                nearc4 + exper + expersq + black + smsa + south, data = cdg)
  seg <- sqrt(vcovHC(mg, type = "HC1")["educ", "educ"])
  lo <- min(lo, coef(mg)["educ"] - tcrit * seg)
  hi <- max(hi, coef(mg)["educ"] + tcrit * seg)
}
add("card_ji", "uci_lower", lo)
add("card_ji", "uci_upper", hi)

# ----------------------------------------------------------------------------
# Case card_oi: overidentified, educ ~ nearc4 + nearc2
# ----------------------------------------------------------------------------
f_oi <- lwage ~ educ + exper + expersq + black + smsa + south |
  nearc4 + nearc2 + exper + expersq + black + smsa + south
m_oi <- ivreg(f_oi, data = cd)
add("card_oi", "coef_educ", coef(m_oi)["educ"])
for (tp in c("iid", "hc1", "cl")) {
  V <- vc_of(m_oi, tp, ~region)
  add("card_oi", paste0("se_educ_", tp), sqrt(V["educ", "educ"]))
}
dg <- summary(m_oi, diagnostics = TRUE)$diagnostics
add("card_oi", "weak_F_iid", dg["Weak instruments", "statistic"])
add("card_oi", "wu_hausman_F_iid", dg["Wu-Hausman", "statistic"])
add("card_oi", "sargan", dg["Sargan", "statistic"])
Z2 <- as.matrix(cd[, c("nearc4", "nearc2")])
iv2 <- ivmodel(Y = y, D = d, Z = Z2, X = as.matrix(cd[, ctrl]))
ar2 <- AR.test(iv2, beta0 = 0)
add("card_oi", "ar_F_iid_b0", ar2$Fstat)
add("card_oi", "ar_lower_iid", ar2$ci[1, 1])
add("card_oi", "ar_upper_iid", ar2$ci[1, 2])
clr2 <- CLR(iv2, beta0 = 0)
add("card_oi", "clr_stat_b0", clr2$test.stat)
add("card_oi", "clr_p_b0", clr2$p.value)
add("card_oi", "clr_lower", clr2$ci[1, 1])
add("card_oi", "clr_upper", clr2$ci[1, 2])
clr2b <- CLR(iv2, beta0 = 0.2)
add("card_oi", "clr_p_b02", clr2b$p.value)
for (tp in c("hc1", "cl")) {
  cl <- if (tp == "cl") cd$region else NULL
  add("card_oi", paste0("ar_F_", tp, "_b0"), ar_stat(0, y, d, Z2, X, tp, cl))
  dof <- if (tp == "cl") length(unique(cd$region)) - 1 else nrow(cd) - 2 - ncol(X)
  cc <- qf(0.95, 2, dof)
  s <- ar_set(y, d, Z2, X, tp, cl, cc, coef(m_oi)["educ"], 50)
  add("card_oi", paste0("ar_nroots_", tp), length(s$roots))
  for (i in seq_along(s$roots)) add("card_oi", paste0("ar_root", i, "_", tp), s$roots[i])
  add("card_oi", paste0("ar_left_accepted_", tp), s$left)
}
# effective F with HC1 and Olea-Pflueger simplified critical values
fs2 <- lm(educ ~ nearc4 + nearc2 + exper + expersq + black + smsa + south, data = cd)
pi2 <- coef(fs2)[c("nearc4", "nearc2")]
S2 <- vcovHC(fs2, type = "HC1")[c("nearc4", "nearc2"), c("nearc4", "nearc2")]
Zt2 <- apply(Z2, 2, resid_on, W = X)
Q <- t(Zt2) %*% Zt2
add("card_oi", "eff_F_hc1", t(pi2) %*% Q %*% pi2 / sum(diag(S2 %*% Q)))
ev <- eigen(Q, symmetric = TRUE)
Qh <- ev$vectors %*% diag(sqrt(ev$values)) %*% t(ev$vectors)
Sm <- Qh %*% S2 %*% Qh
for (tau in c(0.05, 0.1, 0.2, 0.3)) {
  x <- 1 / tau
  Keff <- sum(diag(Sm))^2 * (1 + 2 * x) /
    (sum(Sm * Sm) + 2 * x * sum(diag(Sm)) * max(eigen(Sm)$values))
  add("card_oi", paste0("op_cv_", tau), qchisq(0.95, Keff, ncp = x * Keff) / Keff)
}
add("card_oi", "fs_wald_F_hc1", t(pi2) %*% solve(S2, pi2) / 2)
# Hansen J: two-step efficient GMM on the full moment set [Z, X] (no partialling),
# weight matrix S = sum z_i z_i' e_i^2 from 2SLS residuals (Hansen 1982; ivreg2).
# (momentfit's gmmFit/specTest uses a different weighting convention.)
Zf <- cbind(Z2, X); Xf <- cbind(d, X)
PZf <- Zf %*% solve(crossprod(Zf), t(Zf))
b2s <- solve(t(Xf) %*% PZf %*% Xf, t(Xf) %*% PZf %*% y)
e2s <- as.numeric(y - Xf %*% b2s)
Wj <- solve(crossprod(Zf * e2s))
bgm <- solve(t(Xf) %*% Zf %*% Wj %*% t(Zf) %*% Xf, t(Xf) %*% Zf %*% Wj %*% t(Zf) %*% y)
gJ <- t(Zf) %*% (y - Xf %*% bgm)
add("card_oi", "hansen_J", t(gJ) %*% Wj %*% gJ)
add("card_oi", "gmm_coef_educ", bgm[1])

# ----------------------------------------------------------------------------
# Case card_fe: region fixed effects, sampling weights, cluster by cl50;
# and two-way clustering (region, cl50) without fixed effects
# ----------------------------------------------------------------------------
m_fe <- feols(lwage ~ exper + expersq + black + smsa + south | region |
                educ ~ nearc4 + nearc2, data = cd, weights = ~weight, cluster = ~cl50)
add("card_fe", "coef_educ", coef(m_fe)["fit_educ"])
add("card_fe", "se_educ", se(m_fe)["fit_educ"])
add("card_fe", "se_exper", se(m_fe)["exper"])
m_2w <- feols(lwage ~ exper + expersq + black + smsa + south |
                educ ~ nearc4 + nearc2, data = cd, cluster = ~region + cl50)
add("card_2w", "coef_educ", coef(m_2w)["fit_educ"])
add("card_2w", "se_educ", se(m_2w)["fit_educ"])

# ----------------------------------------------------------------------------
# Case card_p2: two endogenous regressors (educ, expersq), exper exogenous,
# instruments nearc4, nearc2, age^2 (numerical check only)
# ----------------------------------------------------------------------------
m_p2 <- ivreg(lwage ~ educ + expersq + exper + black + smsa + south |
                nearc4 + nearc2 + agesq + exper + black + smsa + south, data = cd)
add("card_p2", "coef_educ", coef(m_p2)["educ"])
add("card_p2", "coef_expersq", coef(m_p2)["expersq"])
add("card_p2", "se_educ_iid", sqrt(vcov(m_p2)["educ", "educ"]))
add("card_p2", "se_expersq_hc1", sqrt(vcovHC(m_p2, type = "HC1")["expersq", "expersq"]))
Xp <- cbind(1, as.matrix(cd[, c("exper", "black", "smsa", "south")]))
Dm <- apply(as.matrix(cd[, c("educ", "expersq")]), 2, resid_on, W = Xp)
Zm <- apply(as.matrix(cd[, c("nearc4", "nearc2", "agesq")]), 2, resid_on, W = Xp)
n <- nrow(cd); k <- 3; kx <- ncol(Xp)
PZ <- function(v) Zm %*% qr.solve(Zm, v)
Evv <- Dm - PZ(Dm)
Svv <- t(Evv) %*% Evv / (n - k - kx)
L <- t(chol(Svv))
Mcd <- solve(L) %*% t(Dm) %*% PZ(Dm) %*% t(solve(L))
add("card_p2", "cragg_donald_F", min(eigen(Mcd, symmetric = TRUE)$values) / k)
for (j in 1:2) {
  o <- setdiff(1:2, j)
  Do <- Dm[, o, drop = FALSE]
  Dh <- PZ(Do)
  delta <- solve(t(Dh) %*% Do, t(Dh) %*% Dm[, j])
  e <- Dm[, j] - Do %*% delta
  Pe <- PZ(e)
  swF <- (sum(e * Pe) / (k - 2 + 1)) / ((sum(e^2) - sum(e * Pe)) / (n - k - kx))
  add("card_p2", paste0("sw_F_", c("educ", "expersq")[j]), swF)
}

# ----------------------------------------------------------------------------
# Case mroz: textbook 2SLS (Wooldridge Example 15.5)
# ----------------------------------------------------------------------------
m_mz <- ivreg(lwage ~ educ + exper + expersq | fatheduc + motheduc + exper + expersq,
              data = mz)
add("mroz", "coef_educ", coef(m_mz)["educ"])
add("mroz", "se_educ_iid", sqrt(vcov(m_mz)["educ", "educ"]))
dg <- summary(m_mz, diagnostics = TRUE)$diagnostics
add("mroz", "sargan", dg["Sargan", "statistic"])
add("mroz", "wu_hausman_F_iid", dg["Wu-Hausman", "statistic"])

# ----------------------------------------------------------------------------
# Case card_bin: binary treatment somecol, binary instrument nearc4
# ----------------------------------------------------------------------------
Dz <- cd$somecol; Zb <- cd$nearc4; n <- nrow(cd)
n1 <- sum(Zb); n0 <- n - n1
p1 <- mean(Dz[Zb == 1]); p0 <- mean(Dz[Zb == 0])
add("card_bin", "share_compliers", p1 - p0)
add("card_bin", "share_always", p0)
add("card_bin", "share_never", 1 - p1)
# HC0-type (independent arms) SEs, scaled by n/(n-1) as in DrSnow
fac <- n / (n - 1)
add("card_bin", "se_compliers", sqrt(fac * (p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)))
add("card_bin", "se_always", sqrt(fac * p0 * (1 - p0) / n0))
# complier means: 2SLS of X*D on D instrumented by Z; HC0 * n/(n-1)
for (v in c("black", "exper", "south")) {
  cdv <- cd; cdv$xd <- cdv[[v]] * cdv$somecol
  mv <- ivreg(xd ~ somecol | nearc4, data = cdv)
  add("card_bin", paste0("complier_mean_", v), coef(mv)["somecol"])
  add("card_bin", paste0("complier_se_", v),
      sqrt(fac * vcovHC(mv, type = "HC0")["somecol", "somecol"]))
  add("card_bin", paste0("always_mean_", v), mean(cd[[v]][Dz == 1 & Zb == 0]))
  add("card_bin", paste0("never_mean_", v), mean(cd[[v]][Dz == 0 & Zb == 1]))
  add("card_bin", paste0("pop_mean_", v), mean(cd[[v]]))
}
# LATE without covariates = Wald = 2SLS
mw <- ivreg(lwage ~ somecol | nearc4, data = cd)
add("card_bin", "late_wald", coef(mw)["somecol"])
add("card_bin", "late_wald_se", sqrt(fac * vcovHC(mw, type = "HC0")["somecol", "somecol"]))
# IPW (kappa) LATE with covariates: logit propensity, Hajek means
ipw_late <- function(dd) {
  ps <- fitted(glm(nearc4 ~ black + smsa + south + exper, family = binomial, data = dd))
  w1 <- dd$nearc4 / ps; w0 <- (1 - dd$nearc4) / (1 - ps)
  m1 <- function(f) sum(w1 * f) / sum(w1); m0 <- function(f) sum(w0 * f) / sum(w0)
  pc <- m1(dd$somecol) - m0(dd$somecol)
  c(late = (m1(dd$lwage) - m0(dd$lwage)) / pc,
    y1 = (m1(dd$lwage * dd$somecol) - m0(dd$lwage * dd$somecol)) / pc,
    y0 = (m0(dd$lwage * (1 - dd$somecol)) - m1(dd$lwage * (1 - dd$somecol))) / pc,
    pc = pc)
}
est <- ipw_late(cd)
add("card_bin", "ipw_late", est["late"])
add("card_bin", "ipw_y1c", est["y1"])
add("card_bin", "ipw_y0c", est["y0"])
add("card_bin", "ipw_share", est["pc"])
set.seed(20260927)
B <- 2000
bs <- t(replicate(B, ipw_late(cd[sample.int(n, n, replace = TRUE), ])))
# robust bootstrap spread (IQR / 1.349): with a small complier share the bootstrap
# distribution of the ratio has heavy tails that inflate its standard deviation
rsd <- function(v) IQR(v) / 1.349
add("card_bin", "ipw_late_boot_se", rsd(bs[, "late"]))
add("card_bin", "ipw_y1c_boot_se", rsd(bs[, "y1"]))
add("card_bin", "ipw_y0c_boot_se", rsd(bs[, "y0"]))
add("card_bin", "ipw_share_boot_se", rsd(bs[, "pc"]))

# ----------------------------------------------------------------------------
# Case card_ext: reweighted cell LATEs (Angrist & Fernandez-Val 2013), cells = black
# (both cells have a clear first stage); D = somecol, Z = nearc4. Point estimates by
# direct cell arithmetic, SEs by
# nonparametric bootstrap (IQR / 1.349).
# ----------------------------------------------------------------------------
extrap <- function(dd) {
  cellid <- dd$black
  res <- c()
  cs <- sort(unique(cellid))
  pc <- sapply(cs, function(k) mean(cellid == k))
  late <- sapply(cs, function(k) {
    s <- dd[cellid == k, ]
    (mean(s$lwage[s$nearc4 == 1]) - mean(s$lwage[s$nearc4 == 0])) /
      (mean(s$somecol[s$nearc4 == 1]) - mean(s$somecol[s$nearc4 == 0]))
  })
  fs <- sapply(cs, function(k) {
    s <- dd[cellid == k, ]
    mean(s$somecol[s$nearc4 == 1]) - mean(s$somecol[s$nearc4 == 0])
  })
  ptr <- sapply(cs, function(k) mean(dd$somecol[cellid == k]))
  c(compliers = sum(pc * fs * late) / sum(pc * fs),
    population = sum(pc * late),
    treated = sum(pc * ptr * late) / sum(pc * ptr))
}
ex <- extrap(cd)
for (nm in names(ex)) add("card_ext", nm, ex[nm])
# delta method, coded independently: basic moments (cell-arm shares, cell-arm means
# of Y and D), their influence functions, numDeriv Jacobian, variance * n/(n-1)
suppressPackageStartupMessages(library(numDeriv))
cid <- cd$black; cells <- sort(unique(cid)); n <- nrow(cd)
basic <- c(); IFm <- NULL
for (v in c("p", "y", "d")) for (k in cells) for (zz in 0:1) {
  sel <- as.numeric(cid == k & cd$nearc4 == zz)
  if (v == "p") { m <- mean(sel); f <- (sel - m) / n }
  if (v == "y") { m <- sum(sel * cd$lwage) / sum(sel); f <- sel * (cd$lwage - m) / sum(sel) }
  if (v == "d") { m <- sum(sel * cd$somecol) / sum(sel); f <- sel * (cd$somecol - m) / sum(sel) }
  basic <- c(basic, m); IFm <- cbind(IFm, f)
}
hfun <- function(t) {
  C <- length(cells); P <- matrix(t[1:(2 * C)], 2); Y <- matrix(t[2 * C + 1:(2 * C)], 2)
  Dm <- matrix(t[4 * C + 1:(2 * C)], 2)          # rows: z = 0, 1; columns: cells
  pc <- colSums(P); fs <- Dm[2, ] - Dm[1, ]; late <- (Y[2, ] - Y[1, ]) / fs
  ptr <- (P[2, ] * Dm[2, ] + P[1, ] * Dm[1, ]) / pc
  c(sum(pc * fs * late) / sum(pc * fs), sum(pc * late) / sum(pc),
    sum(pc * ptr * late) / sum(pc * ptr))
}
Jx <- jacobian(hfun, basic)
Vx <- crossprod(IFm %*% t(Jx)) * n / (n - 1)
for (j in 1:3) add("card_ext", paste0(c("compliers", "population", "treated")[j],
                                      "_delta_se"), sqrt(Vx[j, j]))
set.seed(20260928)
bx <- t(replicate(2000, extrap(cd[sample.int(nrow(cd), nrow(cd), replace = TRUE), ])))
for (nm in colnames(bx)) add("card_ext", paste0(nm, "_boot_se"), IQR(bx[, nm]) / 1.349)

ref <- do.call(rbind, out)
write.csv(ref, file.path(here, "reference.csv"), row.names = FALSE)
cat(sprintf("wrote %d reference values\n", nrow(ref)))
