This serves as extra technical details to models and code structure. (concisely mention structure of methodology)

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

**One-step ahead density for MAR(0,1):**

For a purely noncausal process $u_t = \varepsilon_t + \psi \varepsilon_{t+1} + \psi^2 \varepsilon_{t+2} + \cdots$ with Cauchy errors, the predictive density of $u^*_{T+1}$ given $u_T$ is:

$$l(u^{\ast}_{T+1} | u_T) = \frac{1}{\pi\gamma} \cdot \frac{1}{1 + \frac{(u_T - \psi u^{\ast}_{T+1})^2}{\gamma^2}} \cdot \frac{\gamma^2 + (1-\psi)^2 u_T^2}{\gamma^2 + (1-\psi)^2 (u^{\ast}_{T+1})^2}$$

where $\gamma$ is the scale parameter of $\varepsilon_t$.

**MAR(1,1) extension:**

For the full MAR(1,1) model $y_t = \phi y_{t-1} + u_t$, we transform from the noncausal component $u_t$ back to the observed series $y_t$.

Substitute:
- $u_T = y_T - \phi y_{T-1}$ (current noncausal component)
- $u^{\ast}_{T+1} = y^{\ast}_{T+1} - \phi y_T$ (future noncausal component)

where $y^{\ast}_{T+1}$ is the candidate future value of the observed series.

The transformation is linear with Jacobian = 1, so:

$$l(y^{\ast}_{T+1} | y_T, y_{T-1}) = l(u^{\ast}_{T+1} | u_T)$$

evaluated at the substituted values above.

**Implementation:**

`cauchy_predictive_density(u_T, psi, gamma, grid)` — evaluates the closed-form density over a grid of candidate values

`mar11_predictive_density(y_T, y_T_lag, phi, psi, gamma, grid)` — applies the MAR(1,1) transformation

`make_grid(center, gamma, psi, n_points)` — constructs a grid capturing both crash mode (near 0) and continuation mode (near $\psi^{-1} u_T$). Uses ±20 × marginal scale ($\gamma / (1 - \psi)$) to cover tails.

**Validation use:** The closed form serves as a benchmark for validating Method 1 (simulation-based) on Cauchy errors before applying it to Student-t errors where no closed form exists.

## 3.2 `src/forecasting_sim.py`

When no closed form exists (Student-t with df ≠ 1), the predictive density is approximated by simulating many possible future paths and weighting them by consistency with the current observation.

This implements the simulation-based method of Lanne et al. (2012a). The algorithm has four steps:

**Step 1: Simulate future errors**

Draw $N$ independent sequences of $M$ future errors from the Student-t distribution:

$$\varepsilon^*_{T+1}, \varepsilon^*_{T+2}, \ldots, \varepsilon^*_{T+M}$$

Each sequence represents one possible future scenario. Implemented via `simulate_future_errors(N, M, df, scale)` which returns an $(N \times M)$ matrix.

**Step 2: Compute implied future values**

For each simulated sequence, compute the implied value of $u_{T+1}$ using the truncated noncausal representation:

$$u^*_{T+1} \approx \varepsilon^*_{T+1} + \psi \varepsilon^*_{T+2} + \psi^2 \varepsilon^*_{T+3} + \cdots + \psi^{M-1} \varepsilon^*_{T+M}$$

This gives $N$ candidate values for $u_{T+1}$. Implemented via `compute_future_u(errors, psi, h=1)`.

**Step 3: Weight by consistency**

Not all simulated paths are equally plausible given the current observation $u_T$. Each path is weighted by how well it explains $u_T$.

The weight for path $j$ is:

$$w_j = g\left(u_T - \sum_{i=1}^{M} \psi^i \varepsilon^*_{j,T+i}\right)$$

where $g$ is the Student-t pdf. Weights are normalized to sum to 1. Implemented via `compute_weights(u_T, errors, psi, df, scale)`.

**Intuition:** If the simulated future errors cannot produce the current high bubble value $u_T$, that path gets near-zero weight. Only paths consistent with the bubble get significant weight.

**Step 4: Construct weighted empirical CDF**

For each point $x$ on a grid, the CDF is:

$$F(x | u_T) = \sum_{j: u^*_{j,T+1} \leq x} w_j$$

This gives the full predictive distribution. Crash probability = CDF evaluated at the threshold.

**Implementation details:**

**Log-weights for numerical stability:** Weights are computed in log-space to prevent underflow when $N$ is large (100,000+):
```python
log_w = student_t.logpdf(residuals, df=df, scale=scale)
log_w -= log_w.max()  # normalize
w = np.exp(log_w)
w /= w.sum()
```

**Parameters:**
- $N$ = number of simulated paths (typically 100,000 for regular periods, up to 1,000,000 for extreme bubbles)
- $M$ = truncation parameter (typically 100, controls accuracy of the infinite sum approximation)

**Functions:**

`crash_probability(u_T, psi, df, scale, threshold, N, M)` — returns $P(u_{T+1} \leq \text{threshold} | u_T)$

`simulate_predictive_cdf(u_T, psi, df, scale, grid, N, M)` — returns full CDF over grid

## 3.3 `src/forecasting_sample.py`

Method 2 approximates the predictive density using historical data instead of simulating random futures. This implements the sample-based approach of Gouriéroux & Jasiak (2016).

**Key difference from Method 1:** Instead of drawing future errors randomly, Method 2 uses the **empirical distribution** of past $u_t$ values to approximate the density. This introduces a **learning mechanism** — predictions depend on what the series has done historically.

**Algorithm:**

For each candidate value $u^*_{T+1}$ on a grid, the predictive density is:

$$l(u^*_{T+1} | u_T) \propto g(u_T - \psi u^*_{T+1}) \times \frac{\sum_{i=1}^{T} g(u^*_{T+1} - \psi u_i)}{\sum_{i=1}^{T} g(u_T - \psi u_i)}$$

where the sum is over all historical observations $u_1, \ldots, u_T$.

**Three components:**
1. **Transition kernel:** $g(u_T - \psi u^*_{T+1})$ — same as Method 1
2. **Sample-averaged numerator:** weights $u^*_{T+1}$ by how often similar values appeared historically
3. **Sample-averaged denominator:** normalizes by the current value's historical frequency

**Learning mechanism:**

If the current level $u_T$ has never been reached before (new all-time high), the denominator is small, inflating the crash probability. The method "learns" that uncharted territory is riskier.

Conversely, if the series previously reached $u_T$ and continued rising, Method 2 assigns lower crash probability than Method 1.

**Comparison to Method 1:**

| Aspect | Method 1 (Simulation) | Method 2 (Sample) |
|---|---|---|
| Data source | Random draws from t(df) | Historical $u_1, \ldots, u_T$ |
| Dependence | Only on model parameters | Parameters + past realizations |
| Interpretation | Model-implied probability | Learned probability |
| New highs | Constant probability | Higher crash probability |

The difference between the two methods captures the **learning component** — how much sample-based probabilities deviate from theoretical due to historical patterns.

**Implementation:**

`sample_crash_probability(u_T, u_sample, psi, df, scale, threshold)` — returns crash probability using all historical $u_t$ values

`sample_predictive_density(u_T, u_sample, psi, df, scale, grid)` — returns full density over grid

**Note:** The density is normalized via numerical integration (`np.trapz`) to ensure it integrates to 1.

# 4. Validation (`src/monte_carlo.py`)

The forecasting methods must be validated before applying them to real data. This chapter describes the Monte Carlo simulation framework used to reproduce Tables 1 and 2 from the paper.

## 4.1 Why Validation is Needed

For Student-t errors with df ≠ 1, no theoretical benchmark exists. We validate by:

1. **Comparing Method 1 to closed form** for Cauchy errors (df = 1) — if Method 1 matches the exact solution, we trust it for other df values
2. **Comparing Method 2 to Method 1** for all df values — Method 1 serves as a proxy for the theoretical probability

## 4.2 MAR(0,1) Simulation

Monte Carlo requires simulating MAR(0,1) processes with known parameters. The purely noncausal process is simulated **backwards** from a terminal condition.

**Algorithm:**

Starting from $u_T = 0$, iterate backwards for $t = T-1, T-2, \ldots, 1$:

$$u_t = \psi u_{t+1} + \varepsilon_t$$

where $\varepsilon_t \sim t(\text{df}, \sigma)$ is drawn randomly at each step.

This produces a stationary series with known parameters $\psi$, df, $\sigma$ that can be used to test forecasting accuracy.

**Bubble detection:** Find the index where $u_t$ reaches a specified quantile (e.g., Q0.995) — this represents the bubble peak. Forecast from that point and measure crash probability.

## 4.3 Tables 1 & 2 Generation

**Table 1 — Method 1 validation:**

For each combination of:
- $\psi \in \{0.2, 0.5, 0.8\}$
- df $\in \{1, 2, 3\}$

Run 200 replications:
1. Simulate MAR(0,1) series (T = 500)
2. Find bubble point (Q0.995)
3. Compute crash probability using Method 1 (N = 100,000)
4. For df = 1, compare against closed form

Report mean and standard deviation of crash probabilities across replications.

**Table 2 — Method 2 validation:**

Same parameter grid, but also vary sample size $T \in \{100, 200, 500, 1000\}$ to test how Method 2 converges with more data.

**Implementation decisions:**
- N = 100,000 (reduced from paper's 1,000,000 for computational efficiency)
- 200 replications (reduced from paper's 1,000)
- For bachelor thesis, these settings provide sufficient validation

**Output:** CSV files in `outputs/tables/` containing crash probabilities for all parameter combinations.

# 5. Empirical Application (Notebooks)

The four Jupyter notebooks in `notebooks/` apply the MAR model and forecasting methods to Nickel prices. They are designed to run sequentially and produce all figures and tables for the thesis.

## 5.1 Notebook Execution Flow

**`01_data_exploration.ipynb`**

Visualizes raw and filtered data:
- Plot 1: Raw monthly Nickel prices (1980-2019)
- Plot 2: HP-filtered cycle component  
- Plot 3: Noncausal component $u_t$

Outputs: `outputs/figures/01_raw_nickel_prices.png`, `02_hp_filtered_cycle.png`, `03_noncausal_component.png`

**`02_estimation_results.ipynb`**

Presents MAR(1,1) estimation results:
- Model selection table (BIC/AIC/HQ for p = 0,...,5)
- Parameter estimates with standard errors
- Residual diagnostics (time series plot, histogram vs Student-t fit, Shapiro-Wilk test)

Outputs: `outputs/figures/04_residuals.png`, `05_residual_distribution.png`

**`03_forecasting_methods.ipynb`**

Validates forecasting methods:
- Bi-modal density plot at different bubble levels (reproduces Figure 2)
- Method 1 vs closed form comparison (Cauchy validation)
- Tables 1 & 2 from Monte Carlo results

Outputs: Table 1 and Table 2 formatted for thesis, density visualization

**`04_empirical_application.ipynb`**

Core empirical results:
- Figure 9: HP-filtered cycle with six forecast points marked
- Table 3: Crash probabilities at all six points (Methods 1 & 2)
- Predictive density plots for all six points showing bi-modality

Outputs: `outputs/figures/09_forecast_points.png`, `10_predictive_densities.png`, `outputs/tables/table3_crash_probabilities.csv`

## 5.2 Key Results

**Forecast points:** Six dates along the 2007 Nickel bubble trajectory, identified by quantile positions (Q0.48, Q0.96, Q0.99, Q0.99, Q1.00, Q0.97).

**Bi-modal densities:** Points 2-5 exhibit clear bi-modality (crash vs continuation modes). Point 1 (normal period) shows uni-modal density. Point 6 (post-crash) shows reduced bi-modality.

**Method comparison:** Sample-based consistently assigns higher crash probability than simulation-based during bubble episodes. The difference (0.20-0.28) represents the learning component — Method 2 recognizes that extreme values have historically preceded crashes.

## 5.3 Reproducibility

All notebooks import from `src/` modules — no data loading or computation is duplicated. Parameters are loaded once via `preprocessing.load_data()` and passed to forecasting functions.

Computational requirements:
- Notebooks 01, 02, 04: ~5 minutes total
- Notebook 03: depends on Monte Carlo completion (~6-8 hours on VU server)

Figures and tables are automatically saved to `outputs/` for direct inclusion in thesis.
