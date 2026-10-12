# ==============================================================================
#  Sobol sensitivity analysis of the GMT labeled/unlabeled flux model
#  Equation (16): dU/dt = K*A*U + alpha
#     https://link.springer.com/article/10.1007/s11538-024-01386-x
#
#  What this script does:
#    1. Defines the ODE system (Eq. 16) for 2 metabolites.
#    2. Solves it once at nominal parameter values and plots U1(t), U2(t).
#    3. For each output of interest (mean U1, mean U2) computes
#         - first-order Sobol indices  S1  (effect of each parameter alone)
#         - second-order Sobol indices S2  (effect of each PAIR of parameters)
#         - total-order Sobol indices  ST  (alone + all interactions)
#    4. Plots S1 and ST as bar charts, and S2 as a heatmap, and saves PNGs.
#
#  To reuse for a different model, change sections 2 (model), 3 (parameters),
#  and 5 (outputs). Everything else works unchanged.
# ==============================================================================


# -- 1. Packages ---------------------------------------------------------------
# Activate the project whose Project.toml is one folder above this file.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Pkg; Pkg.activate(joinpath(@__DIR__))
using OrdinaryDiffEq      # ODE solvers (Tsit5)
using GlobalSensitivity   # gsa(), Sobol()
using QuasiMonteCarlo     # low-discrepancy (Sobol) sample matrices
using Statistics          # mean()
using Plots, StatsPlots   # plotting (StatsPlots gives groupedbar)


# -- 2. The model --------------------------------------------------------------
# In-place ODE function: writes the derivative into du.
#   u = [U1, U2]                          current concentrations
#   p = [alpha1, alpha2, beta12, k1, k2]  parameters (ORDER MATTERS: everything
#                                         below indexes p in this order)
#   t = time (unused here: the system is autonomous)
#
# Eq. (16) written out component-wise, with
#   K = diag(k1, k2),  A = [ -1          beta12 ;
#                            1 - alpha2  -1     ]
function gmt_model!(du, u, p, t)
    alpha1, alpha2, beta12, k1, k2 = p
    du[1] = k1 * ((-u[1] + beta12 * u[2])       + alpha1)
    du[2] = k2 * (((1 - alpha2) * u[1] - u[2])  + alpha2)
    return nothing
end


# -- 3. Nominal parameters, initial condition, time span -----------------------
param_names = ["alpha1", "alpha2", "beta12", "k1", "k2"]   # labels for plots
p_nominal   = [3.0, 3.0, 1.0, 1.0, 1.0]                    # same order as above
u0          = [1.0, 1.0]                                    # U1(0), U2(0)
tspan       = (0.0, 10.0)                                   # start, end time
t_save      = range(tspan[1], tspan[2]; length = 200)       # times to record

# Range [lower, upper] each parameter is varied over in the sensitivity
# analysis. Sobol indices are only meaningful relative to these ranges -
# use the same ranges as your priors / physically plausible values.
# NOTE: check alpha2 against the paper; (1 - alpha2) multiplies a flux, so
# alpha2 may need to lie in [0, 1].
lower = [1.0, 1.0, 1.0, 1.0, 1.0]
upper = [5.0, 5.0, 5.0, 5.0, 5.0]


# -- 4. Solve once and plot the ODE --------------------------------------------
prob = ODEProblem(gmt_model!, u0, tspan, p_nominal)
sol  = solve(prob, Tsit5(); saveat = t_save)

fig_ode = plot(sol; lw = 2, label = ["U1(t)" "U2(t)"],
               xlabel = "t", ylabel = "concentration",
               title = "GMT flux model, nominal parameters")
display(fig_ode)
savefig(fig_ode, joinpath(@__DIR__, "ode_solution.png"))


# -- 5. Outputs (quantities of interest) ---------------------------------------
# Sobol analysis needs a function: parameter vector -> ONE number.
# solve_model(p) solves the ODE for a given parameter vector p;
# each output function then reduces the solution to a single number.
function solve_model(p)
    prob_p = remake(prob; p = collect(p))   # same ODE, new parameters
    return solve(prob_p, Tsit5(); saveat = t_save)
end

mean_U1(p) = mean(solve_model(p)[1, :])    # time-averaged U1 over tspan
mean_U2(p) = mean(solve_model(p)[2, :])    # time-averaged U2 over tspan

# Put every output to analyze in this list as (label, function).
outputs = [("mean U1", mean_U1),
           ("mean U2", mean_U2)]


# -- 6. Sobol sample matrices --------------------------------------------------
# Sobol's method needs two independent N x d sample matrices A and B
# (d = number of parameters). Cost per output is about
#     N * (d + 2) + N * d*(d-1)/2     ODE solves
# = N * 17 for d = 5. Increase N until the indices stop changing.
N = 2^11
A, B = QuasiMonteCarlo.generate_design_matrices(N, lower, upper, SobolSample(), 2)


# -- 7. Helper: turn S2 into a symmetric matrix for plotting -------------------
# S2[i, j] is the pure interaction effect of parameters i and j.
# GlobalSensitivity fills only one triangle, so mirror it; the diagonal
# (a parameter with itself) is not defined and is left as NaN (blank).
function symmetric_S2(S2)
    d = size(S2, 1)
    M = fill(NaN, d, d)
    for i in 1:d, j in (i + 1):d
        v = (isfinite(S2[i, j]) && S2[i, j] != 0) ? S2[i, j] : S2[j, i]
        M[i, j] = v
        M[j, i] = v
    end
    return M
end


# -- 8. Run Sobol analysis and plot, one output at a time ----------------------
for (name, output_function) in outputs
    println("Computing Sobol indices for: ", name)

    # order = [0, 1, 2] -> compute total (0), first (1) and second (2) order
    res = gsa(output_function, Sobol(order = [0, 1, 2]), A, B)

    # Monte Carlo noise can push estimates slightly below 0 or above 1; clamp.
    S1 = clamp.(res.S1, 0, 1)          # vector, one entry per parameter
    ST = clamp.(res.ST, 0, 1)          # vector, one entry per parameter
    S2 = symmetric_S2(res.S2)          # d x d matrix, one entry per pair

    # Print the numbers too, for tables in the paper.
    println("  S1 = ", round.(S1; digits = 3))
    println("  ST = ", round.(ST; digits = 3))

    # (a) First- and total-order indices side by side.
    #     ST - S1 = how much of a parameter's effect comes from interactions.
    fig_bar = groupedbar(hcat(S1, ST);
                         label = ["first order S1" "total order ST"],
                         xticks = (1:length(param_names), param_names),
                         ylims = (0, 1.05), ylabel = "Sobol index (dimensionless)",
                         title = "First and total order: $name")

    # (b) Second-order indices: interaction strength of each parameter pair.
    fig_S2 = heatmap(param_names, param_names, S2;
                     c = :viridis, clims = (0, max(1e-6, maximum(filter(isfinite, S2)))),
                     yflip = true, aspect_ratio = 1,
                     title = "Second order S2: $name")

    # Combine both panels into one figure, show it, and save it.
    fig = plot(fig_bar, fig_S2; layout = (1, 2), size = (1100, 450),
               left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
    display(fig)
    savefig(fig, joinpath(@__DIR__, "sobol_" * replace(name, " " => "_") * ".png"))
end