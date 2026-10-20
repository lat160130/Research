# ==============================================================================
#  Time-resolved Sobol sensitivity analysis of the GMT flux model
#  Equation (16): dU/dt = K*(A*U + alpha)
#     https://link.springer.com/article/10.1007/s11538-024-01386-x
#
#  Output of interest: U_j(t_k) at EVERY saved time t_k, so for each state
#  U1, U2 we get Sobol indices as functions of time:
#    S1(t)  first order   - effect of each parameter alone, at time t
#    S2(t)  second order  - extra effect of each PAIR of parameters, at time t
#    ST(t)  total order   - alone + all interactions, at time t
#
#  To reuse for another model, change sections 2 (model) and 3 (parameters).
# ==============================================================================


# -- 1. Packages ---------------------------------------------------------------
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))   # project one folder up

using OrdinaryDiffEq      # ODE solver (Tsit5)
using GlobalSensitivity   # gsa(), Sobol()
using QuasiMonteCarlo     # Sobol sample matrices
using Plots


# -- 2. The model --------------------------------------------------------------
#   u = [U1, U2],  p = [alpha1, alpha2, beta12, k1, k2]  (order matters)
function gmt_model!(du, u, p, t)
    alpha1, alpha2, beta12, k1, k2 = p
    du[1] = k1 * ((-u[1] + beta12 * u[2]) + alpha1)
    du[2] = k2 * (((1 - alpha2) * u[1] - u[2]) + alpha2)
    return nothing
end


# -- 3. Parameters, initial condition, time points -----------------------------
param_names = ["alpha1", "alpha2", "beta12", "k1", "k2"]
p_nominal   = [1/4, 3/10, 3/20, 7/20, 3/10] # Check fig 7
u0          = [1.0, 1.0]
tspan       = (0.0, 10.0)

# Time points where Sobol indices are computed. t = 0 is EXCLUDED: there
# U(0) = u0 for every parameter set, so Var = 0 and the indices are 0/0.
t_save = range(tspan[1], tspan[2]; length = 101)[2:end]

# Range each parameter is varied over (use your priors / physical ranges).
# Check figs 4-13
lower = [0.0, 0.0, 0.0, 0.0, 0.0]
upper = [1.0, 1.0, 1.0, 3.0, 3.0]

d = length(param_names)   # number of parameters


# -- 4. Solve once and plot the ODE --------------------------------------------
prob = ODEProblem(gmt_model!, u0, tspan, p_nominal)
sol  = solve(prob, Tsit5(); saveat = t_save)

fig_ode = plot(sol; lw = 2, label = ["U1(t)" "U2(t)"],
               xlabel = "t", ylabel = "proportion unlabled",
               title = "GMT flux model, nominal parameters")
display(fig_ode)
savefig(fig_ode, joinpath(@__DIR__, "ode_solution.png"))


# -- 5. Output functions: parameters -> vector of U_j(t_k) ---------------------
# Returning a VECTOR (one entry per time point) makes gsa compute indices for
# every entry at once:
#   res.S1, res.ST : (number of time points) x d
#   res.S2         : d x d x (number of time points), upper triangle filled
function solve_model(p)
    return solve(remake(prob; p = collect(p)), Tsit5(); saveat = t_save)
end

U1_over_time(p) = Vector(solve_model(p)[1, :])
U2_over_time(p) = Vector(solve_model(p)[2, :])

outputs = [("U1", U1_over_time),
           ("U2", U2_over_time)]


# -- 6. Sobol sample matrices --------------------------------------------------
# Cost per output ~ N * (2d + 2) = 12N ODE solves for d = 5.
# Increase N until the curves stop changing (they get noisy where Var is small).
N = 2^11
A, B = QuasiMonteCarlo.generate_design_matrices(N, lower, upper, SobolSample(), 2)


# -- 7. List of parameter pairs (i < j) for the second-order plot --------------
pairs = [(i, j) for i in 1:d for j in (i + 1):d]
pair_labels = reshape(["$(param_names[i]) & $(param_names[j])" for (i, j) in pairs], 1, :)
param_labels = reshape(param_names, 1, :)   # Plots wants a 1 x n row of labels


# -- 8. Compute and plot, one state at a time ----------------------------------
for (name, output_function) in outputs
    println("Computing time-resolved Sobol indices for ", name)

    res = gsa(output_function, Sobol(order = [0, 1, 2]), A, B)

    # Clamp Monte Carlo noise into [0, 1].
    S1 = clamp.(res.S1, 0, 1)       # (n_times x d): column i = parameter i over time
    ST = clamp.(res.ST, 0, 1)       # (n_times x d)

    # Collect S2 into (n_times x n_pairs): column = one pair over time.
    S2 = clamp.(reduce(hcat, [res.S2[i, j, :] for (i, j) in pairs]), 0, 1)

        # Plain-language description of each output, for the titles.
    state_desc = Dict("U1" => "metabolite 1 (U1)",
                      "U2" => "metabolite 2 (U2)")[name]

    # Settings shared by all three panels.
    common = (xlabel        = "time t",
              lw            = 2,
              legend        = :outerbottom,   # legend below each panel
              legend_columns = 3,
              titlefontsize = 11,
              guidefontsize = 10)

    p1 = plot(t_save, S1; common...,
              label  = param_labels,
              ylims  = (0, 0.50),
              title  = "First-order index S1\n(effect of each parameter alone)",
              ylabel = "S1 = share of Var[$name(t)]")


    # -- Second-order panel: distinct color + marker + line style per pair ---------------------------------
    # High-contrast, colorblind-friendly palette (Okabe-Ito based), 10 colors.
    pair_colors = ["#000000", "#E69F00", "#0072B2", "#D55E00", "#009E73",
                   "#CC79A7", "#56B4E9", "#8C564B", "#6A3D9A", "#7F7F7F"]
    pair_markers = [:circle, :rect, :utriangle, :diamond, :star5,
                    :xcross, :dtriangle, :pentagon, :hexagon, :cross]
    pair_styles  = [:solid, :dash, :dot, :dashdot]    # cycles over the 10 pairs

    # Put markers on every 10th time point only (100 markers per line is clutter).
    marker_idx = 1:10:length(t_save)

    # Empty panel with titles/axes first, then add one pair at a time.
    p2 = plot(; common...,
              legend_columns = 2,
              ylims  = (0, 0.50),
              title  = "Second-order index S2\n(extra effect of parameter pairs)",
              ylabel = "S2 = share of Var[$name(t)]")

    for k in 1:length(pairs)
        c  = pair_colors[k]
        m  = pair_markers[k]
        ls = pair_styles[mod1(k, length(pair_styles))]

        # Full-resolution line (no legend entry).
        plot!(p2, t_save, S2[:, k]; color = c, lw = 2, linestyle = ls, label = "")

        # Markers at a subset of times; this series carries the legend entry,
        # so the legend shows the marker shape in the pair's color.
        scatter!(p2, t_save[marker_idx], S2[marker_idx, k];
                 color = c, markershape = m, markersize = 6,
                 markerstrokewidth = 0.5, markerstrokecolor = :black,
                 label = pair_labels[k])
    end

    # -----------------------------------------------------------------------------------------------


    p3 = plot(t_save, ST; common...,
              label  = param_labels,
              ylims  = (0, 0.75),
              title  = "Total-order index ST\n(alone + all interactions)",
              ylabel = "ST = share of Var[$name(t)]")

    fig = plot(p1, p2, p3;
               layout        = (1, 3),        # 1 row, 3 columns = horizontal
               size          = (1800, 650),
               plot_title    = "Time-resolved Sobol indices: proportion unlabeled of $state_desc",
               plot_titlefontsize = 14,
               left_margin   = 8Plots.mm,
               bottom_margin = 10Plots.mm,
               top_margin    = 5Plots.mm)
    display(fig)
    savefig(fig, joinpath(@__DIR__, "sobol_time_$name.png"))
end