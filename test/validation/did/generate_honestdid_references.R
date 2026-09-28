# Reference values for DrSnow's honest_did (Rambachan & Roth 2023 sensitivity
# analysis), generated with the HonestDiD R package.
#
# Run from the repository root:
#   Rscript test/validation/did/generate_honestdid_references.R
#
# Package versions: see honestdid_versions.txt. Numbers are written with 17
# significant digits. Outputs (CSV, in test/validation/did):
#   hd_bc_beta.csv, hd_bc_sigma.csv   Benzarti & Carloni (2019) event study shipped
#                                     with HonestDiD (4 pre, 4 post periods)
#   hd_mpdta_es.csv, hd_mpdta_sigma.csv  Callaway–Sant'Anna event study on mpdta
#                                     (universal base period), as used by
#                                     HonestDiD's honest_did.AGGTEobj
#   r_honestdid.csv                   robust confidence sets (one row per case)
#
# Conditional / hybrid confidence sets are computed on an explicit theta grid
# (grid_lb, grid_ub, grid_points) so the Julia tests can use the same grid; the
# reported bounds are the smallest and largest accepted grid points, as in
# HonestDiD. FLCI half-lengths use HonestDiD's simulated folded-normal quantiles
# (10^6 draws) and least-favorable critical values use 1000 draws, so the C-LF and
# FLCI rows are compared with tolerances reflecting that Monte Carlo error.

suppressPackageStartupMessages({
  library(HonestDiD)
  library(did)
})
outdir <- "test/validation/did"
fmt <- function(df) {
  for (nm in names(df)) if (is.numeric(df[[nm]])) {
    df[[nm]] <- ifelse(is.na(df[[nm]]), "NA", sprintf("%.17g", df[[nm]]))
  }
  df
}
save_csv <- function(df, name) {
  utils::write.csv(fmt(df), file.path(outdir, name), row.names = FALSE, quote = FALSE)
}
sessioninfo <- c(paste("HonestDiD", packageVersion("HonestDiD")),
                 paste("did", packageVersion("did")), R.version.string)
writeLines(sessioninfo, file.path(outdir, "honestdid_versions.txt"))

data(BCdata_EventStudy)
bc <- BCdata_EventStudy
b <- bc$betahat
S <- bc$sigma
npre <- length(bc$prePeriodIndices)
npost <- length(bc$postPeriodIndices)
save_csv(data.frame(beta = b), "hd_bc_beta.csv")
save_csv(as.data.frame(unname(S)), "hd_bc_sigma.csv")

rows <- list()
add <- function(case, delta, method, M, lb, ub, glb = NA, gub = NA, gp = NA,
                l = NA, post = NA, hl = NA, vec = NA) {
  rows[[length(rows) + 1]] <<- data.frame(case = case, delta = delta,
    method = method, M = M, lb = lb, ub = ub, grid_lb = glb, grid_ub = gub,
    grid_points = gp, l = l, npost = post, flci_halflength = hl, flci_vec = vec)
}
bounds <- function(ci) c(min(ci$grid[ci$accept == 1]), max(ci$grid[ci$accept == 1]))

lfirst <- basisVector(1, npost)
lavg <- rep(1 / npost, npost)
lstr <- function(l) paste(sprintf("%.17g", l), collapse = ";")

# FLCI for Delta^SD
for (l in list(lfirst, lavg)) for (M in c(0, 0.01, 0.02, 0.04)) {
  f <- findOptimalFLCI(betahat = b, sigma = S, numPrePeriods = npre,
                       numPostPeriods = npost, l_vec = l, M = M)
  add("bc", "DeltaSD", "FLCI", M, f$FLCI[1], f$FLCI[2], l = lstr(l), post = npost,
      hl = f$optimalHalfLength, vec = lstr(f$optimalVec))
}

gp <- 400
# Conditional (ARP) and hybrids on explicit grids
for (l in list(lfirst, lavg)) {
  th <- sum(l * b[(npre + 1):(npre + npost)])
  glb <- th - 0.4; gub <- th + 0.4
  for (M in c(0, 0.02)) {
    ci <- computeConditionalCS_DeltaSD(b, S, npre, npost, l_vec = l, M = M,
            hybrid_flag = "ARP", gridPoints = gp, grid.lb = glb, grid.ub = gub)
    bb <- bounds(ci); add("bc", "DeltaSD", "Conditional", M, bb[1], bb[2], glb, gub, gp, lstr(l), npost)
    ci <- computeConditionalCS_DeltaSD(b, S, npre, npost, l_vec = l, M = M,
            hybrid_flag = "FLCI", gridPoints = gp, grid.lb = glb, grid.ub = gub)
    bb <- bounds(ci); add("bc", "DeltaSD", "C-F", M, bb[1], bb[2], glb, gub, gp, lstr(l), npost)
    ci <- computeConditionalCS_DeltaSDB(b, S, npre, npost, l_vec = l, M = M,
            biasDirection = "positive", hybrid_flag = "ARP", gridPoints = gp,
            grid.lb = glb, grid.ub = gub)
    bb <- bounds(ci); add("bc", "DeltaSDPB", "Conditional", M, bb[1], bb[2], glb, gub, gp, lstr(l), npost)
    ci <- computeConditionalCS_DeltaSDM(b, S, npre, npost, l_vec = l, M = M,
            monotonicityDirection = "increasing", hybrid_flag = "ARP", gridPoints = gp,
            grid.lb = glb, grid.ub = gub)
    bb <- bounds(ci); add("bc", "DeltaSDI", "Conditional", M, bb[1], bb[2], glb, gub, gp, lstr(l), npost)
  }
  for (Mbar in c(0, 0.5, 1, 2)) {
    ci <- computeConditionalCS_DeltaRM(b, S, npre, npost, l_vec = l, Mbar = Mbar,
            hybrid_flag = "ARP", gridPoints = gp, grid.lb = glb, grid.ub = gub)
    bb <- bounds(ci); add("bc", "DeltaRM", "Conditional", Mbar, bb[1], bb[2], glb, gub, gp, lstr(l), npost)
  }
  for (Mbar in c(0.5, 1)) {
    ci <- computeConditionalCS_DeltaRM(b, S, npre, npost, l_vec = l, Mbar = Mbar,
            hybrid_flag = "LF", gridPoints = gp, grid.lb = glb, grid.ub = gub)
    bb <- bounds(ci); add("bc", "DeltaRM", "C-LF", Mbar, bb[1], bb[2], glb, gub, gp, lstr(l), npost)
    ci <- computeConditionalCS_DeltaRMB(b, S, npre, npost, l_vec = l, Mbar = Mbar,
            biasDirection = "positive", hybrid_flag = "ARP", gridPoints = gp,
            grid.lb = glb, grid.ub = gub)
    bb <- bounds(ci); add("bc", "DeltaRMPB", "Conditional", Mbar, bb[1], bb[2], glb, gub, gp, lstr(l), npost)
    ci <- computeConditionalCS_DeltaRMM(b, S, npre, npost, l_vec = l, Mbar = Mbar,
            monotonicityDirection = "decreasing", hybrid_flag = "ARP", gridPoints = gp,
            grid.lb = glb, grid.ub = gub)
    bb <- bounds(ci); add("bc", "DeltaRMD", "Conditional", Mbar, bb[1], bb[2], glb, gub, gp, lstr(l), npost)
    ci <- computeConditionalCS_DeltaSDRM(b, S, npre, npost, l_vec = l, Mbar = Mbar,
            hybrid_flag = "ARP", gridPoints = gp, grid.lb = glb, grid.ub = gub)
    bb <- bounds(ci); add("bc", "DeltaSDRM", "Conditional", Mbar, bb[1], bb[2], glb, gub, gp, lstr(l), npost)
  }
}

# One post-period (no nuisance parameters): first 4 pre + first post coefficient
b1 <- b[1:(npre + 1)]; S1 <- S[1:(npre + 1), 1:(npre + 1)]
glb <- b1[npre + 1] - 0.4; gub <- b1[npre + 1] + 0.4
for (M in c(0, 0.02)) {
  f <- findOptimalFLCI(betahat = b1, sigma = S1, numPrePeriods = npre,
                       numPostPeriods = 1, M = M)
  add("bc1", "DeltaSD", "FLCI", M, f$FLCI[1], f$FLCI[2], l = "1", post = 1,
      hl = f$optimalHalfLength, vec = lstr(f$optimalVec))
  ci <- computeConditionalCS_DeltaSD(b1, S1, npre, 1, M = M, hybrid_flag = "ARP",
          gridPoints = gp, grid.lb = glb, grid.ub = gub)
  bb <- bounds(ci); add("bc1", "DeltaSD", "Conditional", M, bb[1], bb[2], glb, gub, gp, "1", 1)
  ci <- computeConditionalCS_DeltaSD(b1, S1, npre, 1, M = M, hybrid_flag = "FLCI",
          gridPoints = gp, grid.lb = glb, grid.ub = gub)
  bb <- bounds(ci); add("bc1", "DeltaSD", "C-F", M, bb[1], bb[2], glb, gub, gp, "1", 1)
  ci <- computeConditionalCS_DeltaSDB(b1, S1, npre, 1, M = M, biasDirection = "negative",
          hybrid_flag = "ARP", gridPoints = gp, grid.lb = glb, grid.ub = gub)
  bb <- bounds(ci); add("bc1", "DeltaSDNB", "Conditional", M, bb[1], bb[2], glb, gub, gp, "1", 1)
}
for (Mbar in c(0, 0.5, 1, 2)) {
  ci <- computeConditionalCS_DeltaRM(b1, S1, npre, 1, Mbar = Mbar, hybrid_flag = "ARP",
          gridPoints = gp, grid.lb = glb, grid.ub = gub)
  bb <- bounds(ci); add("bc1", "DeltaRM", "Conditional", Mbar, bb[1], bb[2], glb, gub, gp, "1", 1)
  ci <- computeConditionalCS_DeltaRM(b1, S1, npre, 1, Mbar = Mbar, hybrid_flag = "LF",
          gridPoints = gp, grid.lb = glb, grid.ub = gub)
  bb <- bounds(ci); add("bc1", "DeltaRM", "C-LF", Mbar, bb[1], bb[2], glb, gub, gp, "1", 1)
}

# Callaway-Sant'Anna event study on mpdta (universal base period)
data(mpdta)
gt <- att_gt(yname = "lemp", tname = "year", idname = "countyreal",
             gname = "first.treat", data = mpdta, base_period = "universal",
             bstrap = FALSE, cband = FALSE)
es <- aggte(gt, type = "dynamic", bstrap = FALSE, cband = FALSE)
inf <- es$inf.function$dynamic.inf.func.e
V <- t(inf) %*% inf / nrow(inf) / nrow(inf)
ref <- which(es$egt == -1)
save_csv(data.frame(e = es$egt, att = es$att.egt), "hd_mpdta_es.csv")
save_csv(as.data.frame(unname(V)), "hd_mpdta_sigma.csv")
beta <- es$att.egt[-ref]; Vm <- V[-ref, -ref]
mpre <- sum(es$egt < -1); mpost <- length(beta) - mpre
l0 <- basisVector(1, mpost)
glb <- beta[mpre + 1] - 0.3; gub <- beta[mpre + 1] + 0.3
for (Mbar in c(0, 0.5, 1)) {
  ci <- computeConditionalCS_DeltaRM(beta, Vm, mpre, mpost, l_vec = l0, Mbar = Mbar,
          hybrid_flag = "ARP", gridPoints = gp, grid.lb = glb, grid.ub = gub)
  bb <- bounds(ci); add("mpdta", "DeltaRM", "Conditional", Mbar, bb[1], bb[2], glb, gub, gp, lstr(l0), mpost)
}
for (M in c(0, 0.01, 0.02)) {
  f <- findOptimalFLCI(betahat = beta, sigma = Vm, numPrePeriods = mpre,
                       numPostPeriods = mpost, l_vec = l0, M = M)
  add("mpdta", "DeltaSD", "FLCI", M, f$FLCI[1], f$FLCI[2], l = lstr(l0), post = mpost,
      hl = f$optimalHalfLength, vec = lstr(f$optimalVec))
}
orig <- constructOriginalCS(beta, Vm, mpre, mpost, l_vec = l0)
add("mpdta", "Original", "Original", NA, orig$lb, orig$ub, l = lstr(l0), post = mpost)

out <- do.call(rbind, rows)
save_csv(out, "r_honestdid.csv")
