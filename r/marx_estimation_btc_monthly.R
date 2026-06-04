# ============================================================
# marx_estimation_btc_monthly.R
# MAR model estimation for Bitcoin monthly prices
# HP filter lambda = 129,600 (Ravn & Uhlig 2002, monthly standard)
# Model selection based on AIC
# Training sample: up to end of 2024
# Output: parameters + filtered series -> data/processed/
# ============================================================

library(MARX)
library(mFilter)

# ── 1. Load data ─────────────────────────────────────────────
btc      <- read.csv("data/raw/btc_prices_monthly_2014m09_2026m05.csv")
btc$Date <- as.character(btc$Date)

# Training sample: up to end of 2024
train <- btc[btc$Date <= "2024-09", ]
price <- train$BTC_Price_USD
dates <- train$Date

cat("Training sample:", min(dates), "to", max(dates), "\n")
cat("Observations:", length(price), "\n")

# ── 2. HP filter (lambda = 129600) ───────────────────────────
hp    <- hpfilter(price, freq = 129600, type = "lambda")
cycle <- hp$cycle
trend <- hp$trend
cat("HP filter applied with lambda = 129600\n")

# ── 3. Lag order selection ────────────────────────────────────
lag_sel <- selection.lag(cycle, NULL, p_max = 5)
cat("\nLag selection results:\n")
print(lag_sel)

# Use AIC for lag selection
best_p <- as.integer(which.min(lag_sel$aic)) - 1
cat("AIC selected lag order p:", best_p, "\n")

# ── 4. MAR model selection ────────────────────────────────────
t_loglik <- function(residuals, scale, df) {
  sum(dt(residuals / scale, df = df, log = TRUE) - log(scale))
}

cat("\nModel comparison (p=2):\n")
cat(sprintf("%-12s %-10s %-10s %-10s %-10s\n",
            "Model", "LL", "AIC", "phi", "psi"))

for (r in 0:best_p) {
  s   <- best_p - r
  fit <- tryCatch(
    marx.t(cycle, NULL, p_C = r, p_NC = s),
    error = function(e) NULL
  )
  if (!is.null(fit)) {
    ll  <- t_loglik(fit$residuals, fit$scale, fit$df)
    k   <- r + s + 2
    aic <- -2 * ll + 2 * k
    cat(sprintf("MAR(%d,%d)     %-10.2f %-10.2f %-10s %-10s\n",
      r, s, ll, aic,
      paste(round(fit$coef.c,  4), collapse=","),
      paste(round(fit$coef.nc, 4), collapse=",")))
  }
}

# MAR(0,2) selected by AIC but psi1=1.123 > 1 is economically
# uninterpretable within the MAR framework. MAR(1,1) selected
# as best economically interpretable specification at p=2.
best_r   <- 1
best_s   <- 1
best_fit <- marx.t(cycle, NULL, p_C = 1, p_NC = 1)
cat("\nSelected: MAR(1,1) — best economically interpretable model at p=2\n")

# ── 5. Extract parameters ─────────────────────────────────────
phi   <- best_fit$coef.c
psi   <- best_fit$coef.nc
df_t  <- best_fit$df
scale <- best_fit$scale

cat("phi:", phi, "\n")
cat("psi:", psi, "\n")
cat("df:", df_t, "\n")
cat("scale:", scale, "\n")

# ── 6. Standard errors ────────────────────────────────────────
inf <- inference(
  y = cycle, x = NULL,
  B_C  = best_fit$coef.c,
  B_NC = best_fit$coef.nc,
  B_x  = best_fit$coef.exo,
  IC   = best_fit$coef.int,
  sig  = best_fit$scale,
  df   = best_fit$df,
  sig_level = 0.05
)

# ── 7. Extract noncausal component u_t ───────────────────────
T     <- length(cycle)
n_res <- length(best_fit$residuals)
u_t   <- cycle[(best_r+1):T] -
  as.numeric(filter(cycle, phi, sides = 1))[(best_r+1):T]
u_t   <- u_t[1:n_res]

# ── 8. Export ─────────────────────────────────────────────────
series_out <- data.frame(
  Date      = dates[(best_r+1):(best_r+n_res)],
  cycle     = cycle[(best_r+1):(best_r+n_res)],
  u_t       = u_t,
  residuals = best_fit$residuals
)
write.csv(series_out,
          "data/processed/btc_filtered_monthly.csv",
          row.names = FALSE)

params_out <- data.frame(
  r      = best_r,
  s      = best_s,
  phi    = phi,
  psi    = psi,
  df     = df_t,
  scale  = scale,
  phi_se = inf$se.c,
  psi_se = inf$se.nc
)
write.csv(params_out,
          "data/processed/btc_mar_parameters_monthly.csv",
          row.names = FALSE)

cat("\nDone. Files saved to data/processed/\n")
