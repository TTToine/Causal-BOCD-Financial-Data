# Causal-BOCD: Analytical Online Changepoint Detection for High-Frequency Financial Data

![Status](https://img.shields.io/badge/Status-Exam%20Project-blue)
![Language](https://img.shields.io/badge/Language-Python%20%7C%20R-yellowgreen)
![Complexity](https://img.shields.io/badge/Update%20Complexity-O(1)-brightgreen)

## 📖 Overview
This project implements an advanced **Bayesian Online Changepoint Detection (BOCD)** architecture tailored for non-stationary, high-frequency financial time series. 

Traditional batch models or non-parametric architectures relying on Markov Chain Monte Carlo (MCMC) introduce deterministic computational latencies incompatible with real-time streaming, leading to rapid **alpha decay** and severe risk underestimation. Furthermore, standard financial time series sampled in chronological time violate the fundamental i.i.d. assumptions required for exact Bayesian inference.

To bridge this gap, this framework shifts the paradigm from retrospective segmentation to **causal predictive filtering**, calculating the exact marginal predictive distribution $P(x_{t+1} | x_{1:t})$ in $O(1)$ time complexity per update.

## 🚀 Key Contributions & Features

### 1. Information-Theoretic Sampling (Dollar Bars)
Replaces chronological time-based sampling with **Dollar Bars** (sampling when a fixed monetary volume $V$ is traded). This regularizes the microstructure noise, compresses low-activity periods, and restores the conditional i.i.d. properties required for exact exponential-conjugate inference.

### 2. Analytical Robustness via Student-t Predictive
By leveraging the exact conjugacy of the **Normal-Gamma** prior, the posterior predictive distribution is analytically integrated into a non-standardized **Student-t distribution**. The heavy tails natively absorb transitory anomalies and fat-tailed market shocks without triggering false changepoint alerts.

### 3. Power-Law Hazard Function (Volatility Clustering)
Replaces the standard memoryless geometric prior with a **Power-Law Hazard Function** $H(r) \propto r^{-\gamma}$. This introduces "regime memory," mathematically encoding the empirical phenomenon of volatility clustering: the longer a regime persists, the lower the a priori probability of an immediate structural break.

### 4. Intra-Bar Updating & Bayesian Associativity
Exploits the associative property of Bayes' Theorem to perform sequential updates on micro-observations *within* a forming Dollar Bar. This eliminates "information blindness" and allows the model to detect structural breaks before the bar officially closes, gaining a critical latency advantage.

### 5. Ultra-Low Latency Architecture ($O(1)$)
Unlike MCMC methods, the BOCD engine updates only 4 sufficient statistics $(\mu, \kappa, \alpha, \beta)$ using pure algebraic operations (sums and multiplications). The system is designed with a decoupled, asynchronous 3-tier architecture (Stateful Accumulator $\rightarrow$ Async Message Queue $\rightarrow$ Analytical BOCD Engine) to guarantee microsecond-level inference.

## 🧮 Mathematical Core

The core message-passing algorithm updates the joint distribution of the run length $r_t$ and the data:
$$P(r_t, x_{1:t}) = \sum_{r_{t-1}} P(r_t | r_{t-1}) P(x_t | r_{t-1}, x^{(r)}_t) P(r_{t-1}, x_{1:t-1})$$

**Posterior Predictive Distribution (Student-t):**
$$p(x_t | x_{1:t-1}) = \text{Student-t}\left(x_t \mid \mu_{t-1}, \frac{\beta_{t-1}(\kappa_{t-1}+1)}{\alpha_{t-1}\kappa_{t-1}}, 2\alpha_{t-1}\right)$$

## 📊 Evaluation Framework

The model is evaluated not just on point-prediction error, but on probabilistic calibration and financial impact:
*   **Negative Log-Predictive Density (NLPD):** Measures the "algorithmic surprise". (Target: NLPD $\approx 1.38$, proving the Student-t correctly calibrates extreme events).
*   **Information Coefficient (IC):** Rank correlation between changepoint probability and subsequent volatility/returns.
*   **Financial Metrics:** Maximum Drawdown (MDD) reduction, Post-Slippage Sharpe/Sortino Ratios.
*   **Execution Metrics:** Tick-to-Trade latency (constrained to the microsecond domain).

## 🏗️ System Architecture

1.  **Level 1: Stateful Accumulator:** Intercepts raw ticks, calculates accumulated dollar volume, and emits bar-close events.
2.  **Level 2: Async Message Queue:** Decouples high-frequency ingestion from statistical inference (e.g., using ZeroMQ/Redis).
3.  **Level 3: Analytical BOCD Engine:** Consumes bars, updates Normal-Gamma hyperparameters in $O(1)$, and outputs the marginal predictive distribution.
