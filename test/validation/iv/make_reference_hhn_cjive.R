# Reference values for the Hansen–Hausman–Newey (2008) many-instrument variance of
# LIML / Fuller and for the cluster jackknife IV estimator (CJIVE, Frandsen, Leslie &
# McIntyre 2025).
#
# Regenerate (from the repository root) with
#   Rscript test/validation/iv/make_reference_hhn_cjive.R
# Writes manyiv_ext.csv (simulated data: skewed errors, 150 group-indicator
# instruments, a covariate, 80 clusters) and reference_hhn_cjive.csv.
# References:
#   - HHN variance: the formula of HHN (2008, Section 2) written out with explicit
#     n × n projection matrices (X = [d, 1, w], instruments [Z, 1, w]);
#   - ManyIV (Kolesár, github.com/kolesarm/ManyIV): LIML estimate and its
#     minimum-distance many-instrument standard error allowing non-normal errors
#     (Kolesár 2018), which is asymptotically equivalent to the HHN variance (not
#     numerically identical) — used as a closeness check;
#   - clusterIV::cjive (CRAN, version recorded below): CJIVE estimate and
#     cluster-robust standard error, with and without the covariate.

suppressPackageStartupMessages({
  library(ManyIV)
  library(clusterIV)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
set.seed(20260930)

n <- 1200; K <- 150; G <- 80
grp <- c(rep(1:K, each = 3),
         sample(1:K, n - 3 * K, replace = TRUE, prob = exp(1.5 * rnorm(K))))
cl <- sample(1:G, n, replace = TRUE)
w <- rnorm(n)
str <- rnorm(K, sd = 0.35)
a <- rnorm(G, sd = 0.3)
e1 <- (rchisq(n, 2) - 2) / 2
e2 <- 0.6 * e1 + 0.8 * (rexp(n) - 1)
d <- str[grp] + 0.5 * w + a[cl] + e2
y <- 1 + 0.5 * d + 0.3 * w + 0.5 * a[cl] + e1
dat <- data.frame(y = y, d = d, w = w, grp = grp, cl = cl)
Z <- model.matrix(~ factor(grp))[, -1]
colnames(Z) <- paste0("z", 2:K)
dat <- cbind(dat, Z)
write.csv(dat, file.path(here, "manyiv_ext.csv"), row.names = FALSE)

out <- list()
add <- function(case, quantity, value) {
  out[[length(out) + 1]] <<- data.frame(case = case, quantity = quantity,
                                        value = as.numeric(value))
}

# --- LIML / Fuller with the HHN variance, explicit matrices ---------------------
X <- cbind(d, 1, w)
Zf <- cbind(Z, 1, w)
P <- Zf %*% solve(crossprod(Zf), t(Zf))
Kf <- ncol(Zf)
Yb <- cbind(y, d)
Mz <- diag(n) - P
Wm <- cbind(1, w)
Mw <- diag(n) - Wm %*% solve(crossprod(Wm), t(Wm))
kl <- min(eigen(solve(t(Yb) %*% Mz %*% Yb, t(Yb) %*% Mw %*% Yb))$values)
hhn <- function(kappa) {
  # k-class with W included: delta = [X'(I - kappa Mz) X]^{-1} X'(I - kappa Mz) y
  A <- diag(n) - kappa * Mz
  delta <- solve(t(X) %*% A %*% X, t(X) %*% A %*% y)
  u <- as.vector(y - X %*% delta)
  Gx <- ncol(X)
  s2 <- sum(u^2) / (n - Gx)
  al <- as.numeric(t(u) %*% P %*% u / sum(u^2))
  Xt <- X - u %*% (t(u) %*% X) / sum(u^2)
  Vh <- (diag(n) - P) %*% Xt
  Ups <- P %*% X
  H <- t(X) %*% P %*% X - al * t(X) %*% X
  SB <- s2 * ((1 - al)^2 * t(Xt) %*% P %*% Xt + al^2 * t(Xt) %*% (diag(n) - P) %*% Xt)
  ptt <- diag(P); tau <- Kf / n; kap <- sum(ptt^2) / Kf
  Am <- (t(Ups) %*% (ptt - tau)) %*% t(colSums(u^2 * Vh) / n)
  Bm <- Kf * (kap - tau) * t(Vh) %*% ((u^2 - s2) * Vh) / (n * (1 - 2 * tau + kap * tau))
  Hi <- solve(H)
  L <- Hi %*% (SB + Am + t(Am) + Bm) %*% Hi
  LB <- Hi %*% SB %*% Hi
  c(delta[1], sqrt(L[1, 1]), sqrt(LB[1, 1]))
}
r <- hhn(kl)
add("hhn", "liml_coef", r[1]); add("hhn", "liml_se_hhn", r[2])
add("hhn", "liml_se_bekker", r[3])
r4 <- hhn(kl - 1 / (n - Kf))
add("hhn", "fuller1_coef", r4[1]); add("hhn", "fuller1_se_hhn", r4[2])

mi <- IVreg(y ~ d + w | factor(grp) + w, data = dat, inference = c("standard", "md"))
add("hhn", "manyiv_liml_coef", mi$estimate["liml", "estimate"])
add("hhn", "manyiv_liml_se_md", mi$estimate["liml", "md"])

# --- CJIVE (clusterIV) -----------------------------------------------------------
cj <- cjive(y ~ d | factor(grp), data = dat, cluster = ~cl)
add("cjive", "coef", cj$coefficient); add("cjive", "se", cj$se)
cjw <- cjive(y ~ d | factor(grp), data = dat, cluster = ~cl, controls = ~w)
add("cjive", "coef_w", cjw$coefficient); add("cjive", "se_w", cjw$se)
wts <- 0.5 + (dat$cl %% 3) / 2
cjww <- cjive(y ~ d | factor(grp), data = dat, cluster = ~cl, controls = ~w,
              weights = wts)
add("cjive", "coef_wt", cjww$coefficient); add("cjive", "se_wt", cjww$se)

res <- do.call(rbind, out)
res$value <- sprintf("%.15g", res$value)
write.csv(res, file.path(here, "reference_hhn_cjive.csv"), row.names = FALSE,
          quote = FALSE)
cat("ManyIV", as.character(packageVersion("ManyIV")), "clusterIV",
    as.character(packageVersion("clusterIV")), "\n")
print(res)
