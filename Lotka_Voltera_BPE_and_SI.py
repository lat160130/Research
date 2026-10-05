import numpy as np
import sympy as sp
import pymc as pm
import pytensor.tensor as pt
from scipy.integrate import odeint
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
import matplotlib.pyplot as plt
import pandas as pd

# Import SALib for Sensitivity Analysis
from SALib.sample import saltelli
from SALib.analyze import sobol

# ==========================================
# 1. SYMBOLIC DEFINITION (The "Symbol Code")
# ==========================================
print("--- 1. Defining ODEs Symbolically ---")

# A. Define independent symbols
t = sp.symbols('t')
x, y = sp.symbols('x y')           # State variables (Prey, Predator)
alpha, beta, gamma, delta = sp.symbols('alpha beta gamma delta') # Parameters

# B. Define the Equations (RHS)
# dx/dt = alpha*x - beta*x*y
# dy/dt = delta*x*y - gamma*y
dxdt = alpha * x - beta * x * y
dydt = delta * x * y - gamma * y

# Display the equations
print(f"dx/dt = {dxdt}")
print(f"dy/dt = {dydt}")

# C. Compile to a Python Function
ode_params = [alpha, beta, gamma, delta]
ode_states = [x, y]
ode_system_func = sp.lambdify((ode_states, t) + tuple(ode_params), [dxdt, dydt])

# Wrapper for odeint that unpacks the arguments correctly
def system_wrapper(y_vec, t, *p):
    return ode_system_func(y_vec, t, *p)

# ==========================================
# 2. GENERATE SYNTHETIC DATA
# ==========================================
print("\n--- 2. Generating Synthetic Data ---")

# True parameters
true_params = [1.1, 0.1, 1.0, 0.1] # alpha, beta, gamma, delta
t_obs = np.linspace(1, 15, 15)     # Observation times
y0 = [10, 5]                       # Initial conditions

# Solve using our generated symbolic function
true_sol = odeint(system_wrapper, y0, t_obs, args=tuple(true_params))

# Add noise
np.random.seed(42)
y_obs = true_sol + np.random.normal(0, 0.5, size=true_sol.shape)

# ==========================================
# 3. DEFINE PYTENSOR OP (The Bridge)
# ==========================================
class ODEop(Op):
    def make_node(self, *inputs):
        pt_inputs = [pt.as_tensor_variable(i) for i in inputs]
        outputs = [pt.matrix()]
        return Apply(self, pt_inputs, outputs)

    def perform(self, node, inputs, output_storage):
        params = [float(x) for x in inputs]
        sol = odeint(system_wrapper, y0, t_obs, args=tuple(params))
        output_storage[0][0] = sol

ode_op = ODEop()

# ==========================================
# 4. BAYESIAN ESTIMATION (PyMC)
# ==========================================
print("\n--- 3. Running Bayesian Estimation ---")

with pm.Model() as model:
    # A. Priors
    p_alpha = pm.TruncatedNormal('alpha', mu=1.0, sigma=0.5, lower=0.01)
    p_beta  = pm.TruncatedNormal('beta',  mu=0.1, sigma=0.05, lower=0.001)
    p_gamma = pm.TruncatedNormal('gamma', mu=1.0, sigma=0.5, lower=0.01)
    p_delta = pm.TruncatedNormal('delta', mu=0.1, sigma=0.05, lower=0.001)
    
    sigma = pm.HalfNormal('sigma', sigma=1.0)

    # B. Model Prediction
    mu_pred = ode_op(p_alpha, p_beta, p_gamma, p_delta)

    # C. Likelihood
    likelihood = pm.Normal('likelihood', mu=mu_pred, sigma=sigma, observed=y_obs)

    # D. Sampling
    # Reduced draws/chains for demonstration speed; increase for production
    trace = pm.sample(draws=500, tune=250, step=pm.Metropolis(), chains=2, cores=1, progressbar=True)

# ==========================================
# 5. VISUALIZATION
# ==========================================
print("\n--- 4. Bayesian Results ---")
print(f"True Alpha: {true_params[0]}, Estimated: {trace.posterior['alpha'].mean():.3f}")
print(f"True Beta:  {true_params[1]}, Estimated: {trace.posterior['beta'].mean():.3f}")

# Optional: Plot trace if running interactively
# pm.plot_trace(trace)
# plt.show()

# ==========================================
# 6. SENSITIVITY ANALYSIS (SOBOL)
# ==========================================
print("\n--- 5. Running Sobol Sensitivity Analysis ---")

# A. Define the Problem for SALib
# We analyze how parameters impact the Average Prey Population over time
problem = {
    'num_vars': 4,
    'names': ['alpha', 'beta', 'gamma', 'delta'],
    'bounds': [
        [0.5, 1.5],   # Bounds for alpha
        [0.05, 0.2],  # Bounds for beta
        [0.5, 1.5],   # Bounds for gamma
        [0.05, 0.2]   # Bounds for delta
    ]
}

# B. Generate Samples
# N is the number of base samples. Total runs = N * (2D + 2)
# For D=4 and N=1024, we run the model ~10,000 times.
N = 1024
param_values = saltelli.sample(problem, N)

print(f"Generated {param_values.shape[0]} parameter sets for analysis.")

# C. Run the Model
Y_prey_mean = np.zeros([param_values.shape[0]])

# We use a slightly finer time grid for sensitivity to capture dynamics better
t_sens = np.linspace(0, 15, 100)

for i, X in enumerate(param_values):
    # X contains [alpha, beta, gamma, delta]
    try:
        # Solve ODE
        sol = odeint(system_wrapper, y0, t_sens, args=tuple(X))
        
        # Define Quantity of Interest (QoI): Average Prey Population
        # You could also choose Max Predator, Final Prey, etc.
        Y_prey_mean[i] = np.mean(sol[:, 0]) 
    except:
        # Handle unstable parameters if necessary
        Y_prey_mean[i] = 0

# D. Analyze
Si = sobol.analyze(problem, Y_prey_mean, print_to_console=False)

# E. Formatting and Interpretation
print("\n--- 6. Sensitivity Interpretation ---")

# Convert to DataFrame for easy viewing
total_si = Si['ST']
first_si = Si['S1']
params = problem['names']

df_sens = pd.DataFrame({
    'Parameter': params,
    'First Order (S1)': first_si,
    'Total Order (ST)': total_si,
    'Interaction (ST - S1)': total_si - first_si
}).set_index('Parameter')

print("Sensitivity Indices (Quantity of Interest: Mean Prey Population):")
print(df_sens.round(4))

print("\n--- AUTOMATED INTERPRETATION ---")

# Identify most influential parameter (Total Order)
max_idx = np.argmax(total_si)
dominant_param = params[max_idx]

print(f"1. DOMINANT FACTOR: The parameter '{dominant_param}' has the highest Total Effect (ST={total_si[max_idx]:.3f}).")
print(f"   -> Changes in '{dominant_param}' cause the largest variance in the average prey population.")

# Check for interactions
print("\n2. INTERACTIONS:")
for i, param in enumerate(params):
    interaction = total_si[i] - first_si[i]
    if interaction > 0.1: # Threshold for "significant" interaction
        print(f"   -> '{param}' has high interactions (difference={interaction:.3f}).")
        print(f"      Its effect depends heavily on the values of other parameters.")
    else:
        print(f"   -> '{param}' acts mostly independently (low interaction).")

print("\n3. SUMMARY:")
print("   - First Order (S1): The effect of the parameter varying alone.")
print("   - Total Order (ST): The effect of the parameter + its interactions with others.")