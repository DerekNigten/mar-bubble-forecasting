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
