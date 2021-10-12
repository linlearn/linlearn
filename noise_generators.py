import numpy as np

def gaussian(rng, n_samples, sigma=20):
    noise = sigma * rng.normal(size=n_samples)
    expect_noise = 0
    noise_2nd_moment = sigma ** 2

    return noise, expect_noise, noise_2nd_moment


def lognormal(rng, n_samples, sigma=1.75):
    noise = rng.lognormal(0, sigma, n_samples)
    expect_noise = np.exp(0.5 * sigma ** 2)
    noise_2nd_moment = np.exp(2 * sigma ** 2)

    return noise, expect_noise, noise_2nd_moment


def pareto(rng, n_samples, sigma=10, pareto=2.05):
    noise = sigma * rng.pareto(pareto, n_samples)
    expect_noise = (sigma) / (pareto - 1)
    noise_2nd_moment = expect_noise ** 2 + (sigma ** 2) * pareto / (
        ((pareto - 1) ** 2) * (pareto - 2)
    )

    return noise, expect_noise, noise_2nd_moment


def student(rng, n_samples, sigma=10, df=2.1):
    noise = sigma * rng.standard_t(df, n_samples)
    expect_noise = 0
    noise_2nd_moment = expect_noise ** 2 + (sigma ** 2) * df / (df - 2)

    return noise, expect_noise, noise_2nd_moment


def weibull(rng, n_samples, sigma=10, a=0.65):
    from scipy.special import gamma

    noise = sigma * rng.weibull(a, n_samples)
    expect_noise = sigma * gamma(1 + 1 / a)
    noise_2nd_moment = (sigma ** 2) * gamma(1 + 2 / a)

    return noise, expect_noise, noise_2nd_moment


def frechet(rng, n_samples, sigma=10, alpha=2.2):
    from scipy.special import gamma

    noise = sigma * (1 / rng.weibull(alpha, n_samples))
    expect_noise = sigma * gamma(1 - 1 / alpha)
    noise_2nd_moment = (sigma ** 2) * gamma(1 - 2 / alpha)

    return noise, expect_noise, noise_2nd_moment


def loglogistic(rng, n_samples, sigma=10, c=2.2):
    from scipy.stats import fisk

    noise = sigma * fisk.rvs(c, size=n_samples)
    expect_noise = sigma * (np.pi / c) / np.sin(np.pi / c)
    noise_2nd_moment = (sigma ** 2) * (2 * np.pi / c) / np.sin(2 * np.pi / c)

    return noise, expect_noise, noise_2nd_moment
