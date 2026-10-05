"""
# -- Import Bloc ------------------------------------------ Import Bloc
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
 
# --------------------------------------------------------- Import Bloc
"""


"""Second-order Sobol indices for the GMT model with SALib.

    dx/dt = k1 * (-x + b12*y + alpha1)
    dy/dt = k2 * ((1 - alpha2)*x - y + alpha2)

pip install SALib scipy numpy matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol

# ---- 1. Problem: parameters (the thetas) and their prior ranges ----------
problem = {
    "num_vars": 5,
    "names": ["k1", "k2", "b12", "alpha1", "alpha2"],
    "bounds": [[0, 3], [0, 3], [0, 1], [0, 1], [0, 1]],
}
d = problem["num_vars"]
N = 2**12
tout = np.linspace(0, 80, 81)

# ---- 2. Sample: N*(2d+2) rows when calc_second_order=True ----------------
X = sobol_sample.sample(problem, N, calc_second_order=True, seed=1)
print(f"{len(X)} model runs")

# ---- 3. Model: solve ALL parameter sets at once as one big ODE system ----
def simulate(X, x0=0.0, y0=0.0):
    k1, k2, b12, a1, a2 = X.T
    n = len(X)

    def rhs(t, z):
        x, y = z[:n], z[n:]
        return np.concatenate([k1 * (-x + b12 * y + a1),
                               k2 * ((1 - a2) * x - y + a2)])

    z0 = np.concatenate([np.full(n, x0), np.full(n, y0)])
    sol = solve_ivp(rhs, (tout[0], tout[-1]), z0, t_eval=tout,
                    rtol=1e-6, atol=1e-9)
    return sol.y[:n], sol.y[n:]              # each: runs x times

Yx, Yy = simulate(X)

# ---- 4. Analyze at each output time (skip t=0: zero variance) ------------
def indices_over_time(Y):
    S1, ST, S2 = [], [], []
    for k in range(1, len(tout)):
        r = sobol.analyze(problem, Y[:, k], calc_second_order=True,
                          print_to_console=False)
        S1.append(r["S1"]); ST.append(r["ST"]); S2.append(r["S2"])
    return np.array(S1), np.array(ST), np.array(S2)   # S2: time x d x d (upper triangle)

S1, ST, S2 = indices_over_time(Yy)   # use Yx for the x observable
t = tout[1:]
names = problem["names"]

# ---- 5. Plot ------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
ax[0].plot(t, S1); ax[0].set_title("First order $S_i$")
ax[1].plot(t, ST); ax[1].set_title("Total order $S_{T_i}$")
for i in range(d):
    for j in range(i + 1, d):
        ax[2].plot(t, S2[:, i, j], label=f"{names[i]} x {names[j]}")
ax[2].set_title("Second order $S_{ij}$")
ax[0].legend(names); ax[2].legend(fontsize=7)
for a in ax:
    a.set_xlabel("t"); a.axhline(0, color="gray", lw=0.5)
plt.tight_layout()
plt.savefig("sobol_GMT_y.png", dpi=150)

# Heatmap at one time: diagonal = S_i, off-diagonal = S_ij
k = np.argmin(np.abs(t - 36))
M = np.nan_to_num(S2[k]); M = M + M.T + np.diag(S1[k])
fig, a = plt.subplots(figsize=(5, 4))
im = a.imshow(M, vmin=0, vmax=1, cmap="viridis")
a.set_xticks(range(d), names); a.set_yticks(range(d), names)
for i in range(d):
    for j in range(d):
        a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
fig.colorbar(im); a.set_title(f"y at t = {t[k]:g}")
plt.tight_layout()
plt.savefig("sobol_GMT_y_heatmap.png", dpi=150)

print(f"t = {t[k]:g}:  sum S1 + sum S2 = {S1[k].sum() + np.nansum(S2[k]):.3f}")
plt.show()






