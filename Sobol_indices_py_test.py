import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- LIBRARY FIXES ---
# 1. Fix SALib import for newer versions
try:
    from SALib.sample import saltelli
except ImportError:
    # In SALib >= 1.5.0, saltelli is inside 'sobol'
    from SALib.sample import sobol as saltelli
from SALib.analyze import sobol

# 2. Fix PyMC/PyTensor imports
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply  # <--- Correct location for Apply

# ==========================================
# PART 1: MODEL DEFINITION
# ==========================================

def ode_model(y, t, alpha, beta, gamma, delta):
    """Lotka-Volterra equations"""
    x, predator = y
    dxdt = alpha * x - beta * x * predator
    dydt = delta * x * predator - gamma * predator
    return [dxdt, dydt]

def run_model_for_sensitivity(params):
    """Wrapper for SALib: Returns max predator population"""
    y0 = [10, 5]
    t = np.linspace(0, 15, 100)
    solution = odeint(ode_model, y0, t, args=(params[0], params[1], params[2], params[3]))
    return np.max(solution[:, 1])

# ==========================================
# PART 2: SENSITIVITY ANALYSIS (Sobol)
# ==========================================
print("--- Starting Sensitivity Analysis ---")

problem = {
    'num_vars': 4,
    'names': ['alpha', 'beta', 'gamma', 'delta'],
    'bounds': [
        [0.5, 1.5],
        [0.05, 0.2],
        [0.5, 1.5],
        [0.05, 0.2]
    ]
}

# Generate samples (N=512 is usually enough for a quick test)
param_values = saltelli.sample(problem, 512)
Y = np.zeros([param_values.shape[0]])

for i, X in enumerate(param_values):
    Y[i] = run_model_for_sensitivity(X)

Si = sobol.analyze(problem, Y)

# Visualize Sensitivity
print("\nSobol Total Effect Indices:")
for name, val in zip(problem['names'], Si['ST']):
    print(f"{name}: {val:.4f}")

plt.figure(figsize=(8, 4))
plt.bar(problem['names'], Si['ST'])
plt.title('Sensitivity: Total Effect ($S_T$)')
plt.show()




