import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from bocd import BOCD, ConstantHazard, DecreasingHazard, NormalGammaPrior, dollar_bar_indices


def reference_bocd(returns, mu_0=0.0, kappa_0=1.0, alpha_0=2.0, beta_0=1.0, gamma=0.5, lambda_h=50.0, R_max=250):
    """Core loop of the original exam notebook (bocd_ap.py), kept verbatim as a regression oracle."""
    T = len(returns)
    log_P_r = np.full((T, R_max + 1), -np.inf)
    log_P_r[0, 0] = 0.0
    mu, kappa = np.full(R_max + 1, mu_0), np.full(R_max + 1, kappa_0)
    alpha, beta = np.full(R_max + 1, alpha_0), np.full(R_max + 1, beta_0)
    nlpd_series = []
    for t_idx in range(T):
        xt = returns[t_idx]
        r_limit = min(t_idx, R_max)
        var_p = (beta[:r_limit+1] * (kappa[:r_limit+1] + 1)) / (alpha[:r_limit+1] * kappa[:r_limit+1])
        log_lik = student_t.logpdf(xt, df=2*alpha[:r_limit+1], loc=mu[:r_limit+1], scale=np.sqrt(var_p))
        r_range = np.arange(r_limit + 1)
        h_val = 1.0 / (lambda_h + r_range**gamma)
        log_H, log_1minusH = np.log(h_val), np.log(1 - h_val)
        log_new_P_joint = np.full(r_limit + 2, -np.inf)
        log_new_P_joint[1:r_limit+2] = log_P_r[t_idx, :r_limit+1] + log_1minusH + log_lik
        log_new_P_joint[0] = logsumexp(log_P_r[t_idx, :r_limit+1] + log_H + log_lik)
        log_evidence = logsumexp(log_new_P_joint)
        if t_idx + 1 < T:
            log_P_r[t_idx + 1, :min(t_idx+2, R_max+1)] = log_new_P_joint[:min(t_idx+2, R_max+1)] - log_evidence
            log_P_r[t_idx + 1, :R_max+1] -= logsumexp(log_P_r[t_idx + 1, :R_max+1])
        nlpd_series.append(-log_evidence)
        old_mu, old_kappa = mu[:r_limit+1].copy(), kappa[:r_limit+1].copy()
        old_alpha, old_beta = alpha[:r_limit+1].copy(), beta[:r_limit+1].copy()
        upd_len = min(r_limit + 1, R_max)
        mu[1:upd_len+1] = (old_kappa[:upd_len] * old_mu[:upd_len] + xt) / (old_kappa[:upd_len] + 1)
        kappa[1:upd_len+1] = old_kappa[:upd_len] + 1
        alpha[1:upd_len+1] = old_alpha[:upd_len] + 0.5
        beta[1:upd_len+1] = old_beta[:upd_len] + (old_kappa[:upd_len] * (xt - old_mu[:upd_len])**2) / (2 * (old_kappa[:upd_len] + 1))
        mu[0], kappa[0], alpha[0], beta[0] = mu_0, kappa_0, alpha_0, beta_0
    return log_P_r, np.array(nlpd_series)


def piecewise_series(seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, 1.0, 300), rng.normal(0.0, 3.0, 300)])


def test_matches_original_implementation_including_truncation():
    xs = piecewise_series()
    ref_log_r, ref_nlpd = reference_bocd(xs, R_max=100)
    res = BOCD(hazard=DecreasingHazard(50.0, 0.5), r_max=100, truncation="drop").run(xs)
    np.testing.assert_allclose(res.nlpd, ref_nlpd, rtol=1e-10)
    # original row t+1 is the posterior after observation t
    np.testing.assert_allclose(np.exp(res.log_run_length[:-1]), np.exp(ref_log_r[1:]), atol=1e-12)


def test_posterior_is_normalised():
    res = BOCD(r_max=50).run(piecewise_series())
    np.testing.assert_allclose(res.run_length_posterior.sum(axis=1), 1.0, atol=1e-10)


def test_first_prediction_is_prior_predictive():
    p = NormalGammaPrior(0.5, 2.0, 3.0, 1.5)
    model = BOCD(prior=p)
    x = 1.3
    expected = student_t.logpdf(x, df=2 * p.alpha0, loc=p.mu0, scale=np.sqrt(p.beta0 * (p.kappa0 + 1) / (p.alpha0 * p.kappa0)))
    assert np.isclose(model.update(x), expected)


def test_sequential_updates_equal_batch_conjugate_update():
    """Bayesian associativity: K one-at-a-time updates give the batch Normal-Gamma posterior."""
    rng = np.random.default_rng(1)
    xs = rng.normal(0.3, 1.2, 40)
    p = NormalGammaPrior()
    model = BOCD(prior=p, r_max=100)
    for x in xs:
        model.update(x)
    n, xbar = len(xs), xs.mean()
    kappa_n = p.kappa0 + n
    mu_n = (p.kappa0 * p.mu0 + n * xbar) / kappa_n
    alpha_n = p.alpha0 + n / 2
    beta_n = p.beta0 + 0.5 * np.sum((xs - xbar) ** 2) + p.kappa0 * n * (xbar - p.mu0) ** 2 / (2 * kappa_n)
    assert np.allclose([model.mu[n], model.kappa[n], model.alpha[n], model.beta[n]], [mu_n, kappa_n, alpha_n, beta_n])


def test_detects_variance_shift():
    xs = piecewise_series()
    res = BOCD(r_max=250).run(xs)
    recent = res.prob_recent_change(window=10)
    assert recent[300:320].max() > 0.9
    assert np.median(recent[100:290]) < 0.2  # background level ~ 10 x hazard
    assert res.map_run_length[400] < 120  # the new regime is recognised


def test_p_r0_is_only_a_weighted_hazard():
    """Without truncation, P(r_t = 0 | x_1..x_t) cannot leave the range of H, whatever the data."""
    xs = piecewise_series()
    hz = DecreasingHazard(50.0, 0.5)
    res = BOCD(hazard=hz, r_max=len(xs)).run(xs)
    p0 = res.run_length_posterior[:, 0]
    h = hz(np.arange(len(xs) + 1))
    assert p0.min() >= h.min() - 1e-12 and p0.max() <= h.max() + 1e-12
    res_c = BOCD(hazard=ConstantHazard(50.0), r_max=len(xs)).run(xs)
    np.testing.assert_allclose(res_c.run_length_posterior[:, 0], 1 / 50.0)


def test_drop_truncation_inflates_short_run_lengths_merge_does_not():
    xs = piecewise_series()
    hz = DecreasingHazard(50.0, 0.5)
    p0_drop = BOCD(hazard=hz, r_max=250, truncation="drop").run(xs).run_length_posterior[:, 0]
    p0_merge = BOCD(hazard=hz, r_max=250, truncation="merge").run(xs).run_length_posterior[:, 0]
    assert p0_drop.max() > 2 * hz(0)  # spurious mass once the regime outlives r_max
    assert p0_merge.max() <= hz(0) + 1e-12


def test_dollar_bar_indices():
    dv = np.array([1.0, 1.0, 1.0, 5.0, 0.5, 0.5, 1.0])
    # cumulative volume 1,2,3,8,8.5,9,10 -> bars close at 1, 3 (8-2), 6 (10-8)
    assert dollar_bar_indices(dv, 2.0).tolist() == [0, 1, 3, 6]
