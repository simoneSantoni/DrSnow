# Reference values for exposure_effects from the R package `interference`
# (Zonszein, Samii & Aronow; https://github.com/szonszein/interference, v0.1.0).
#
# Usage: Rscript aronow_samii_reference.R [R library path]
# Reads as_edges.csv, as_units.csv, as_draws.txt (see generate_aronow_samii.jl) and
# writes as_reference.csv.
#
# The package's own probability helper adds 1 to marginal counts; to compare the
# estimators and variance formulas themselves we pass it probability matrices built
# from the stored draws with plain frequencies, which is DrSnow's Monte Carlo
# convention: pi_ij(k,l) = mean_r 1{D_i = k} 1{D_j = l}.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) .libPaths(c(args[1], .libPaths()))
library(interference)

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- if (length(file_arg) == 1) dirname(sub("^--file=", "", file_arg)) else "."
f <- function(x) file.path(here, x)

edges <- read.csv(f("as_edges.csv"))
units <- read.csv(f("as_units.csv"))
draws_txt <- readLines(f("as_draws.txt"))
N <- nrow(units)
R <- length(draws_txt)
draws <- t(sapply(draws_txt, function(s) as.integer(strsplit(s, "")[[1]]),
                  USE.NAMES = FALSE))                       # R x N

adj <- matrix(0, N, N)
adj[cbind(edges$source, edges$target)] <- 1
adj[cbind(edges$target, edges$source)] <- 1

# exposure conditions with the package's own mapping (1-hop, 4 conditions)
obs_exposure <- make_exposure_map_AS(adj, units$z, hop = 1)
conds <- colnames(obs_exposure)
I_exposure <- lapply(conds, function(k) matrix(0, N, R))
names(I_exposure) <- conds
for (r in 1:R) {
  e <- make_exposure_map_AS(adj, draws[r, ], hop = 1)
  for (k in conds) I_exposure[[k]][, r] <- e[, k]
}
kk <- list(); kl <- list()
for (k in conds) for (l in conds) {
  m <- I_exposure[[k]] %*% t(I_exposure[[l]]) / R
  if (k == l) kk[[paste(k, k, sep = ",")]] <- m else kl[[paste(k, l, sep = ",")]] <- m
}
prob <- list(I_exposure = I_exposure, prob_exposure_k_k = kk, prob_exposure_k_l = kl)

est <- estimates(obs_exposure, units$y, prob, control_condition = "no",
                 effect_estimators = c("hajek", "horvitz-thompson"),
                 variance_estimators = c("hajek", "horvitz-thompson"))

map <- c(dir_ind1 = "treated_exposed", isol_dir = "treated_unexposed",
         ind1 = "control_exposed", no = "control_unexposed")
out <- data.frame(contrast = paste(map[names(est$tau_ht)], "-", map[["no"]]),
                  tau_ht = as.numeric(est$tau_ht),
                  var_tau_ht = as.numeric(est$var_tau_ht[names(est$tau_ht)]),
                  tau_h = as.numeric(est$tau_h[names(est$tau_ht)]),
                  var_tau_h = as.numeric(est$var_tau_h[names(est$tau_ht)]))
cond_obs <- apply(obs_exposure, 1, function(r) conds[which(r == 1)])
write.csv(out, f("as_reference.csv"), row.names = FALSE)
write.csv(data.frame(unit = 1:N, condition = map[cond_obs]), f("as_conditions.csv"),
          row.names = FALSE)
print(out)
