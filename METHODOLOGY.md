This serves as extra technical details to models and code structure.

**This methodology** validates two numerical forecasting methods for MAR processes when closed-form predictive densities do not exist (Student-t errors with df ≠ 1), and applies them to the 2007 Nickel price bubble.

# 1. Model Specification

Financial and commodity markets show **locally explosive episodes**: rapid price increases followed by sudden crashes. Traditional autoregressive models fail to capture this pattern: they assume either explosive roots that violate stationarity, or Gaussian errors that produce symmetric, unimodal forecasts.

**The challenge:** During bubble episodes, the future is typically bi-modal: the price either crashes back to fundamentals OR continues rising. Standard point forecasts (conditional means) hide this split entirely. Moreover, existing bubble models (e.g., regime-switching autoregressive processes with time-varying roots) are parameter-heavy and difficult to forecast with.

**Mixed Causal-Noncausal Autoregressive (MAR) models** offer a simpler alternative. By combining lags (past prices) and leads (forward-looking) with fat-tailed errors, MAR models:
- Generate explosive episodes while staying stationary overall
- Produce bi-modal forecasts during bubbles (showing both crash and continuation) 
- Require only 4-5 parameters (φ, ψ, ν, σ) versus 10+ in regime-switching models

The forward-looking (noncausal) component $u_t = \sum_{i=0}^{\infty} \psi^i \varepsilon_{t+i}$ aggregates future shocks, creating the price build-up before a crash. This mirrors how markets anticipate future events.

## 1.1 Definition of the MAR Model 

The MAR(r,s) process is defined as:

$$\Phi(L)\Psi(L^{-1})y_t = \varepsilon_t$$

where:
- $L$ = lag operator, $L^{-1}$ = lead operator
- $\Phi(L) = 1 - \phi_1 L - \cdots - \phi_r L^r$ (causal polynomial, degree r)
- $\Psi(L^{-1}) = 1 - \psi_1 L^{-1} - \cdots - \psi_s L^{-s}$ (noncausal polynomial, degree s)
- $\varepsilon_t$ ~ i.i.d. non-Gaussian (Student-t or Cauchy)
- All roots of $\Phi$ and $\Psi$ lie outside the unit circle (stationarity)

This analysis focuses on **MAR(r,1)** processes with a single positive lead:

$$\Phi(L)(1 - \psi L^{-1})y_t = \varepsilon_t, \quad \psi > 0$$

**Why single lead?** Multiple leads create complex dynamics. A single positive lead generates the asymmetric bubble pattern we observe: gradual rise followed by sudden crash.

The noncausal component simplifies to:
$$u_t = \sum_{i=0}^{\infty} \psi^i \varepsilon_{t+i}$$

This aggregates all future shocks with exponentially decreasing weights. During bubbles, $u_t$ rises at rate $\psi^{-1}$ until a large negative shock arrives and triggers the crash.

AR(p) and MAR(r,s) models with $r + s = p$ have identical autocovariance functions. Standard methods (OLS, Gaussian MLE) cannot distinguish between them.

**Solution:** Fat-tailed errors distributions break this symmetry. Maximum likelihood with non-Gaussian errors can separately identify the number of lags ($r$) and leads ($s$).

This analysis uses three error distributions:
- **Cauchy** (Student-t with df = 1): admits closed-form predictive densities
- **Student-t with df = 2 and df = 3**: no closed forms, require numerical methods

All have significantly fatter tails than Gaussian, enabling MAR identification.

## 1.2 Estimation of the MAR Model

MAR models are estimated in two steps. First, fit a pseudo-causal AR(p) model via OLS and select $p$ using information criteria (BIC, AIC, HQ). Second, for all combinations where $r + s = p$, estimate MAR(r,s) via maximum likelihood with Student-t errors. Select the model with highest likelihood.

Standard point forecasts ($E[y_{T+1} | y_T]$) work during normal periods but fail during bubbles. A single average hides the crash-or-continue split. For example, 60% crash to 0 and 40% rise to 15,000 gives mean = 6,000, yet neither outcome is near 6,000.

MAR models require the **full predictive density** $f(y_{T+h} | y_T)$ to capture both modes. From this density, we compute crash probabilities and visualize bi-modality. For Student-t errors with df $\neq$ 1, no closed-form exists. Two numerical approximation method are used: simulation-based (Lanne et al. 2012) and sample-based (Gouriéroux & Jasiak 2016).

# 2. Data Pipeline

The data flows from raw commodity prices through R-based MAR estimation to Python analysis: Raw CSV → HP filtering → MAR estimation → Processed outputs → Python forecasting modules. This chapter describes each step.

## 2.1 Data Source

**Dataset:** Monthly global Nickel prices, January 1980 to September 2019 (476 observations).

**Source:** World Bank Global Economic Monitor commodity price database. The original paper (Hecq & Voisin 2021) uses IMF Primary Commodity Prices. Minor parameter differences between this replication and the paper are attributed to this data source difference.

**Why Nickel:** Commodity markets exhibit clear speculative bubble episodes. The 2007 Nickel price spike provides a well-documented bubble-crash pattern ideal for testing MAR forecasting methods.

**Format:** CSV with two columns:
- `Date`: Monthly period in format `1980M01`, `1980M02`, etc.
- `Nickel_Price_USD_per_MT`: Price in USD per metric ton

**File location:** `data/raw/nickel_prices_1980_2019.csv`

## 2.2 `r/marx_estimation.R`

The R script performs the model estimation using the MARX package. It runs in seven sequential steps, exporting processed data and estimated parameters for use in Python.

1. **Load raw data** — reads `nickel_prices_1980_2019.csv` and extracts the price column

2. **HP filtering** — applies Hodrick-Prescott filter with λ = 129,600 (Ravn & Uhlig 2002 monthly standard) via `mFilter::hpfilter()`. MAR models require stationarity, but Nickel prices exhibit an upward trend over 1980-2019. HP filtering removes this trend while preserving locally explosive episodes. Decomposes price into trend + cycle, where the cycle component is used for estimation.

3. **Lag order selection** — fits pseudo-causal AR(p) via OLS for p = 0,...,5 using `MARX::selection.lag()`. Compares information criteria:
   - BIC selects p = 2
   - AIC selects p = 5  
   - HQ selects p = 2
   - **Selected: p = 2** (BIC/HQ consensus)

4. **MAR(r,s) selection** — for r + s = 2, estimates three candidates via Student-t MLE using `MARX::marx.t()`:
   - MAR(0,2): purely noncausal
   - MAR(1,1): mixed  
   - MAR(2,0): purely causal
   - **Selected: MAR(1,1)** (highest log-likelihood)

5. **Parameter extraction** — from `marx.t()` output:
   - φ = 0.617, ψ = 0.777, df = 1.49, scale = 402.9
   - Standard errors via `MARX::inference()`: SE(φ) = 0.017, SE(ψ) = 0.013

6. **Compute u_t** — applies causal filter: $u_t = y_t^{\text{cycle}} - \phi \cdot y_{t-1}^{\text{cycle}}$

7. **Export outputs** — saves to `data/processed/`:
   - `mar_parameters.csv`: r, s, φ, ψ, df, scale, standard errors
   - `nickel_filtered.csv`: Date, cycle, u_t, residuals (475 observations)

These files serve as inputs for all Python modules.

## 2.3 `src/preprocessing.py`

This module loads the R outputs and makes them available to all Python forecasting modules. It provides a single entry point for data access.

**Design decisions:**

**MARParams dataclass** — stores parameters as a typed object rather than a dictionary:
```python
@dataclass
class MARParams:
    phi: float
    psi: float
    df: float
    scale: float
    r: int
    s: int
    phi_se: float
    psi_se: float
```

This enables IDE autocomplete (`params.phi` instead of `params['phi']`) and runtime type validation.

**Date parsing** — converts `'1980M02'` string format to `pd.Timestamp`:
```python
def _parse_mar_date(date_str: str) -> pd.Timestamp:
    year, month = int(date_str[:4]), int(date_str[5:])
    return pd.Timestamp(year=year, month=month, day=1)
```

Sets `freq='MS'` (month start) on the DatetimeIndex for alignment in forecasting operations.

**Load function** — returns a `(params, series)` tuple:
```python
params, series = load_data()
```

where `params` is a `MARParams` object and `series` is a DataFrame with columns `[cycle, u_t, residuals]` and a DatetimeIndex.

**Output:** All forecasting modules import from `preprocessing.py` to access data consistently. No module loads CSV files directly.

# 3. Forecasting Methods

This chapter describes three approaches to computing the predictive density $f(y_{T+h} | y_T)$ for MAR processes. The **closed-form method** provides exact predictive densities for Cauchy errors (df = 1). It serves as a **benchmark** for validation but applies only to this special case. The estimated MAR model yielded df = 1.49. For Student-t errors with df ≠ 1, no closed form exists, hence two numerical approximation methods are used. All three methods are implemented as pure functions in `src/`.

## 3.1 `src/closed_form.py`

For Cauchy-distributed errors (Student-t with df = 1), the predictive density admits a closed-form expression derived by Gouriéroux & Zakoïan (2017). This provides an exact benchmark for validating the numerical methods.

**Equation 5 — One-step ahead density for MAR(0,1):**

For a purely noncausal process $u_t = \varepsilon_t + \psi \varepsilon_{t+1} + \psi^2 \varepsilon_{t+2} + \cdots$ with Cauchy errors, the predictive density of $u^*_{T+1}$ given $u_T$ is:

$$l(u^* \mid u_T) = \frac{1}{\pi\gamma} \frac{1}{1 + \frac{(u_T - \psi u^*)^2}{\gamma^2}} \frac{\gamma^2 + (1-\psi)^2 u_T^2}{\gamma^2 + (1-\psi)^2 (u^*)^2}$$

where $\gamma$ is the scale parameter of $\varepsilon_t$.

**Three factors:**
1. Normalizing constant: $1/(\pi\gamma)$
2. Transition kernel: $g(u_T - \psi u^*)$ — Cauchy density centered at $\psi u^*$
3. Stationary density ratio: $l(u_T) / l(u^*)$ — adjusts for marginal density

The ratio term makes this a **conditional** density rather than just the transition density.

**MAR(1,1) extension:**

For the full MAR(1,1) model $y_t = \phi y_{t-1} + u_t$, substitute:
- $u_T = y_T - \phi y_{T-1}$
- $u^* = y^* - \phi y_T$

The transformation is linear with Jacobian = 1, so:

$$l(y^* | y_T, y_{T-1}) = l(u^* | u_T)$$

evaluated at $u_T = y_T - \phi y_{T-1}$ and $u^* = y^* - \phi y_T$.

**Implementation:**

`cauchy_predictive_density(u_T, psi, gamma, grid)` — evaluates Equation 5 over a grid of candidate values

`mar11_predictive_density(y_T, y_T_lag, phi, psi, gamma, grid)` — applies the MAR(1,1) substitution wrapper

`make_grid(center, gamma, psi, n_points)` — constructs a grid wide enough to capture both modes (crash near 0, continuation near $\psi^{-1} u_T$)

**Grid construction:** Uses ±20 × marginal scale of $u_t$ to ensure tails are well-covered. The marginal scale is $\gamma / (1 - \psi)$.

**Validation use:** This closed form is used to validate Method 1 (simulation-based) for Cauchy errors before applying Method 1 to Student-t errors where no closed form exists.
