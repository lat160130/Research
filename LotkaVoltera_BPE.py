import numpy as np
import sympy as sp
import pymc as pm
import pytensor.tensor as pt
from scipy.integrate import odeint
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
import matplotlib.pyplot as plt

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

# Display the equations (optional, prints nicely)
print(f"dx/dt = {dxdt}")
print(f"dy/dt = {dydt}")

# C. Compile to a Python Function
# 'lambdify' turns symbolic math into a standard Python function we can use with scipy
# We create a function f(y_vec, t, alpha, beta, gamma, delta)
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
# This connects the numerical solver to the Bayesian engine

class ODEop(Op):
    def make_node(self, *inputs):
        # Convert all inputs to PyTensor variables
        pt_inputs = [pt.as_tensor_variable(i) for i in inputs]
        # Output is a matrix (Time x States)
        outputs = [pt.matrix()]
        return Apply(self, pt_inputs, outputs)

    def perform(self, node, inputs, output_storage):
        # Unpack parameters (they come as 0-D arrays, need conversion to float)
        params = [float(x) for x in inputs]
        
        # Run the solver using the compiled symbolic function
        # Note: 'system_wrapper' is defined in Part 1
        sol = odeint(system_wrapper, y0, t_obs, args=tuple(params))
        
        output_storage[0][0] = sol

# Initialize the operator
ode_op = ODEop()

# ==========================================
# 4. BAYESIAN ESTIMATION (PyMC)
# ==========================================
print("\n--- 3. Running Bayesian Estimation ---")

with pm.Model() as model:
    # A. Priors
    # We define stochastic variables for the parameters we want to find.
    # We rename them p_* to avoid any python variable conflicts.
    p_alpha = pm.TruncatedNormal('alpha', mu=1.0, sigma=0.5, lower=0.01)
    p_beta  = pm.TruncatedNormal('beta',  mu=0.1, sigma=0.05, lower=0.001)
    p_gamma = pm.TruncatedNormal('gamma', mu=1.0, sigma=0.5, lower=0.01)
    p_delta = pm.TruncatedNormal('delta', mu=0.1, sigma=0.05, lower=0.001)
    
    # Noise prior
    sigma = pm.HalfNormal('sigma', sigma=1.0)

    # B. Model Prediction
    # Pass the PyMC variables into our ODE operator
    mu_pred = ode_op(p_alpha, p_beta, p_gamma, p_delta)

    # C. Likelihood
    # Compare the prediction (mu_pred) to the observed data (y_obs)
    likelihood = pm.Normal('likelihood', mu=mu_pred, sigma=sigma, observed=y_obs)

    # D. Sampling
    # Using Metropolis because we don't have gradients for odeint
    trace = pm.sample(draws=1000, tune=500, step=pm.Metropolis(), chains=2, cores=1, progressbar=True)

# ==========================================
# 5. VISUALIZATION
# ==========================================
print("\n--- 4. Results ---")
print(f"True Alpha: {true_params[0]}, Estimated: {trace.posterior['alpha'].mean():.3f}")
print(f"True Beta:  {true_params[1]}, Estimated: {trace.posterior['beta'].mean():.3f}")

pm.plot_trace(trace)
plt.tight_layout()
plt.show()