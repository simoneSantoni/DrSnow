# Reference values for DrSnow's analytic power calculators (src/design).
#
# Regenerate from the repository root with
#   R_LIBS_USER=<lib with pwr, PowerUpR, rdpower, rdrobust> \
#     Rscript test/validation/design/make_reference_power.R
# Writes test/validation/design/reference_power.csv (case, quantity, value) and
# test/validation/design/reference_power_versions.txt.
# PowerUpR 1.1.0 is archived on CRAN: install it from
#   https://cran.r-project.org/src/contrib/Archive/PowerUpR/PowerUpR_1.1.0.tar.gz

suppressPackageStartupMessages({
  library(pwr); library(PowerUpR); library(rdpower); library(rdrobust)
})
out <- list()
add <- function(case, quantity, value) {
  out[[length(out) + 1]] <<- data.frame(case = case, quantity = quantity,
                                        value = sprintf("%.17g", value))
}
quiet <- function(expr) {
  res <- NULL
  invisible(capture.output(res <- suppressMessages(suppressWarnings(expr))))
  res
}

# --- two-sample t (pwr) ------------------------------------------------------
add("t_power", "power", pwr.t.test(n = 64, d = 0.5)$power)
add("t_power_a01_greater", "power",
    pwr.t.test(n = 50, d = 0.3, sig.level = 0.01, alternative = "greater")$power)
add("t_power_less", "power",
    pwr.t.test(n = 40, d = -0.4, alternative = "less")$power)
add("t_mde", "d", pwr.t.test(n = 64, power = 0.8)$d)
add("t_n", "n", pwr.t.test(d = 0.5, power = 0.8)$n)
add("t2n_power", "power", pwr.t2n.test(n1 = 30, n2 = 70, d = 0.6)$power)
add("t2n_mde", "d", pwr.t2n.test(n1 = 30, n2 = 70, power = 0.9)$d)

# --- two proportions ------------------------------------------------------------
h <- ES.h(0.35, 0.25)
add("p2_power", "power", pwr.2p.test(h = h, n = 300)$power)
add("p2_n", "n", pwr.2p.test(h = h, power = 0.8)$n)
add("p2n_power", "power", pwr.2p2n.test(h = h, n1 = 200, n2 = 400)$power)
add("p2_greater", "power", pwr.2p.test(h = h, n = 200, alternative = "greater")$power)
add("prop_test_power", "power",
    power.prop.test(n = 300, p1 = 0.35, p2 = 0.25, strict = TRUE)$power)
add("prop_test_n", "n",
    power.prop.test(p1 = 0.35, p2 = 0.25, power = 0.8, strict = TRUE,
                    tol = 1e-12)$n)

# --- cluster RCT (PowerUpR, cra2r2) ------------------------------------------------
r <- quiet(power.cra2r2(es = 0.25, rho2 = 0.1, n = 20, J = 40, r21 = 0.3, r22 = 0.5,
                        g2 = 1, p = 0.5))
add("cra2_power", "power", r$power)
r <- quiet(power.cra2r2(es = 0.3, rho2 = 0.2, n = 10, J = 30, p = 0.4,
                        two.tailed = FALSE))
add("cra2_power_onesided", "power", r$power)
r <- quiet(mdes.cra2r2(power = 0.8, rho2 = 0.1, n = 20, J = 40, r21 = 0.3, r22 = 0.5,
                       g2 = 1, p = 0.5))
add("cra2_mdes", "mdes", r$mdes[1])
r <- quiet(mrss.cra2r2(es = 0.25, power = 0.8, rho2 = 0.1, n = 20, r21 = 0.3,
                       r22 = 0.5, g2 = 1))
add("cra2_mrss", "J", r$J)

# --- blocked individual RCT (PowerUpR, bira2c1) --------------------------------------
r <- quiet(power.bira2c1(es = 0.2, n = 2, J = 100, r21 = 0.5))
add("bira_pairs_power", "power", r$power)
r <- quiet(power.bira2c1(es = 0.2, n = 8, J = 50, r21 = 0.3, g1 = 2, p = 0.5))
add("bira_power", "power", r$power)
r <- quiet(mdes.bira2c1(power = 0.8, n = 8, J = 50, r21 = 0.3, g1 = 2))
add("bira_mdes", "mdes", r$mdes[1])

# --- regression discontinuity (rdpower) --------------------------------------------
sen <- read.csv("test/validation/rd/senate.csv")
Y <- sen$vote; X <- sen$margin
r <- quiet(rdpower(data = cbind(Y, X), tau = 5))
for (nm in c("power.rbc", "se.rbc", "power.conv", "se.conv", "bias.l", "bias.r",
             "Vl.rb", "Vr.rb", "samph.l", "N.l", "N.r", "Nh.l", "Nh.r"))
  add("rd_senate", nm, r[[nm]])
r <- quiet(rdpower(data = cbind(Y, X), tau = 4, sampsi = c(300, 350)))
add("rd_senate_sampsi", "power.rbc", r$power.rbc)
add("rd_senate_sampsi", "se.rbc", r$se.rbc)
add("rd_senate_sampsi", "power.conv", r$power.conv)
r <- quiet(rdpower(data = cbind(Y, X), tau = 4, samph = 12))
add("rd_senate_samph", "power.rbc", r$power.rbc)
add("rd_senate_samph", "se.rbc", r$se.rbc)
# rdpower's parameter interface (nsamples/variance without data) fails in rdpower
# 3.0 ("object 'Vl.cl' not found"); DrSnow's parameter method is checked against the
# data method and the closed form in the tests instead.

res <- do.call(rbind, out)
write.csv(res, "test/validation/design/reference_power.csv", row.names = FALSE,
          quote = FALSE)
writeLines(c(paste("R", getRversion()),
             paste("pwr", packageVersion("pwr")),
             paste("PowerUpR", packageVersion("PowerUpR")),
             paste("rdpower", packageVersion("rdpower")),
             paste("rdrobust", packageVersion("rdrobust"))),
           "test/validation/design/reference_power_versions.txt")
