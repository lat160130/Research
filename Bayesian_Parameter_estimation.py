import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the model logic in a function (Best Practice)
def run_bayesian_regression():
    # Generate Synthetic Data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    true_slope = 2.5
    true_intercept = 5.0
    y_obs = true_slope * x + true_intercept + np.random.normal(0, 1.0, size=x.shape)

    print("Building Model...")
    with pm.Model() as model:
        # Priors
        slope = pm.Normal('slope', mu=0, sigma=10)
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=2)

        # Likelihood
        mu = slope * x + intercept
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_obs)

        # Sampling
        print("Sampling...")
        # NUTS is the default sampler. It uses multiprocessing.
        trace = pm.sample(1000, tune=1000, chains=2, progressbar=True)
    
    return trace, x, y_obs

# 2. THE CRITICAL FIX FOR WINDOWS
if __name__ == "__main__":
    trace, x, y_obs = run_bayesian_regression()
    
    # Analyze Results
    print("\nEstimated Slope:", trace.posterior['slope'].mean().item())
    print("Estimated Intercept:", trace.posterior['intercept'].mean().item())

    pm.plot_trace(trace)
    plt.tight_layout()
    plt.show()