# ============================================================
# marx_estimation.R
# MAR model estimation for Nickel prices (1980-2019)
# Output: estimated parameters + filtered series -> data/processed/
# ============================================================

library(MARX)
library(mFilter)

# ── 1. Load data ─────────────────────────────────────────────
nickel <- read.csv("../data/raw/nickel_prices_1980_2019.csv")
price  <- nickel$Nickel_Price_USD_per_MT

# ── 2. HP filter (lambda = 129600 for monthly data) ──────────
hp    <- hpfilter(price, freq = 129600, type = "lambda")
cycle <- hp$cycle

# ── 3. Lag order selection via pseudo-causal OLS ─────────────
lag_sel <- selection.lag(cycle, NULL, p_max = 5)
best_p  <- 2  # BIC and HQ both select p=2, consistent with the paper
cat("Selected lag order p:", best_p, "\n")

# ── 4. MAR model selection: find best (r,s) with r+s = best_p ─
# Compute t log-likelihood from residuals for a given fit
t_loglik <- function(residuals, scale, df) {
  sum(dt(residuals / scale, df = df, log = TRUE) - log(scale))
}

best_ll  <- -Inf
best_r   <- NA
best_s   <- NA
best_fit <- NULL

for (r in 0:best_p) {
  s <- best_p - r
  fit <- tryCatch(
    marx.t(cycle, NULL, p_C = r, p_NC = s),
    error = function(e) NULL
  )
  if (!is.null(fit)) {
    ll <- t_loglik(fit$residuals, fit$scale, fit$df)
    cat(sprintf("MAR(%d,%d) log-likelihood: %.4f\n", r, s, ll))
    if (!is.nan(ll) && !is.na(ll) && ll > best_ll) {
      best_ll  <- ll
      best_r   <- r
      best_s   <- s
      best_fit <- fit
    }
  }
}

cat(sprintf("\nSelected model: MAR(%d,%d)\n", best_r, best_s))

# ── 5. Extract estimated parameters ──────────────────────────
phi   <- best_fit$coef.c
psi   <- best_fit$coef.nc
df_t  <- best_fit$df
scale <- best_fit$scale

cat("phi (causal):      ", phi,   "\n")
cat("psi (noncausal):   ", psi,   "\n")
cat("degrees of freedom:", df_t,  "\n")
cat("scale:             ", scale, "\n")

# ── 6. Extract noncausal component u_t ───────────────────────
T       <- length(cycle)
n_res   <- length(best_fit$residuals)  # 475 for MAR(1,1)
u_t     <- cycle[(best_r+1):T] -
  as.numeric(filter(cycle, phi, sides = 1))[(best_r+1):T]
u_t     <- u_t[1:n_res]  # trim to match residuals length

# ── 7. Export to data/processed/ ─────────────────────────────
series_out <- data.frame(
  Date      = nickel$Date[(best_r+1):(best_r+n_res)],
  cycle     = cycle[(best_r+1):(best_r+n_res)],
  u_t       = u_t,
  residuals = best_fit$residuals
)

write.csv(series_out,
          "/Users/derek/mar-bubble-forecasting/data/processed/nickel_filtered.csv",
          row.names = FALSE)

# Get standard errors
inf <- inference(
  y         = cycle,
  x         = NULL,
  B_C       = best_fit$coef.c,
  B_NC      = best_fit$coef.nc,
  B_x       = best_fit$coef.exo,
  IC        = best_fit$coef.int,
  sig       = best_fit$scale,
  df        = best_fit$df,
  sig_level = 0.05
)

params_out <- data.frame(
  r     = best_r,
  s     = best_s,
  phi   = phi,
  psi   = psi,
  df    = df_t,
  scale = scale,
  phi_se   = inf$se.c,
  psi_se   = inf$se.nc
)
write.csv(params_out,
          "/Users/derek/mar-bubble-forecasting/data/processed/mar_parameters.csv",
          row.names = FALSE)

cat("\nDone. Files saved to data/processed/\n")
