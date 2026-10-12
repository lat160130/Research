# -- Calculate the First, Second, and Total order Sobol indices for the
# -- Guppy Mitchell Taylor (GMT) labeled/unlabeled flux model (CPU, threaded).

# -- Load Packages ------------------------------------------------------------- LoadPackage
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra, Statistics
using OrdinaryDiffEq, SciMLBase, GlobalSensitivity, QuasiMonteCarlo, StaticArrays
using Plots, StatsPlots
# ------------------------------------------------------------------------------ LoadPackage

# -- Cite Packages ------------------------------------------------------------- CitePackage
# https://docs.sciml.ai/GlobalSensitivity/stable/
# https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/
# ------------------------------------------------------------------------------ CitePackage

println("Running with $(Threads.nthreads()) threads")

# -- Model, Eq. (16): dU = K*A*U + alpha ---------------------------------------- Model
function f(u, p, t)
    alpha1, alpha2, beta12, k1, k2 = p
    du1 = k1 * (-u[1] + beta12*u[2]      + alpha1)
    du2 = k2 * ((1 - alpha2)*u[1] - u[2] + alpha2)
    return SVector{2}(du1, du2)
end
# ------------------------------------------------------------------------------ Model

# -- Initial Variables --------------------------------------------------------- InitVar
param_names  = ["alpha1", "alpha2", "beta12", "k1", "k2"]
output_names = ["mean U1", "mean U2"]
p_nominal    = @SVector [3.0, 3.0, 1.0, 1.0, 1.0]
u0           = @SVector [1.0,1.0]
prob         = ODEProblem{false}(f, u0, (0.0, 80.0), p_nominal)
# ------------------------------------------------------------------------------ InitVar

# -- Single trajectory and plot ------------------------------------------------ PlotODE
sol = solve(prob, Tsit5())
fig_ode = plot(sol; idxs = [1, 2], lw = 2, label = ["U1(t)" "U2(t)"],
               xlabel = "t", ylabel = "concentration", title = "GMT flux model")
display(fig_ode)
savefig(fig_ode, joinpath(@__DIR__, "ode_solution.png"))
# ------------------------------------------------------------------------------ PlotODE

# -- Batched quantity of interest (threaded CPU ensemble) ---------------------- QoI
t_save   = collect(range(0.0, 10.0; length = 200))
prob_gsa = remake(prob; tspan = (0.0, 10.0))

# P is (n_params x n_samples); returns (n_outputs x n_samples)
function qoi_batch(P)
    N = size(P, 2)
    prob_func = (pr, ctx) ->
        remake(pr; p = SVector{5, Float64}(ntuple(j -> P[j, ctx.sim_id], 5)))
    ens = EnsembleProblem(prob_gsa; prob_func = prob_func, safetycopy = false)
    sim = solve(ens, Tsit5(), EnsembleThreads(); trajectories = N, saveat = t_save)

    nfail = count(s -> !SciMLBase.successful_retcode(s), sim.u)
    nfail > 0 && @warn "$nfail of $N trajectories failed"

    out = Matrix{Float64}(undef, 2, N)
    for i in 1:N
        u = sim.u[i].u
        out[1, i] = mean(x[1] for x in u)
        out[2, i] = mean(x[2] for x in u)
    end
    return out
end
# ------------------------------------------------------------------------------ QoI

lb = fill(1.0, 5)
ub = fill(5.0, 5)

# -- Sanity check before GSA --------------------------------------------------- Check
Ytest = qoi_batch(QuasiMonteCarlo.sample(500, lb, ub, SobolSample()))
@assert size(Ytest) == (2, 500)
@assert all(isfinite, Ytest) "non-finite outputs - check model/bounds"
# ------------------------------------------------------------------------------ Check

# -- Regression GSA ------------------------------------------------------------ RegGSA
reg_sens = gsa(qoi_batch, RegressionGSA(true), [[lb[i], ub[i]] for i in 1:5];
               samples = 5_000, batch = true)

h1 = heatmap(param_names, output_names, reg_sens.partial_correlation;
             title = "Partial correlation", clims = (-1, 1), c = :balance)
h2 = heatmap(param_names, output_names, reg_sens.standard_regression;
             title = "Standard regression", c = :balance)
fig_reg = plot(h1, h2; layout = (2, 1), size = (650, 550))
display(fig_reg)
savefig(fig_reg, joinpath(@__DIR__, "gsa_regression.png"))
# ------------------------------------------------------------------------------ RegGSA

# -- Sobol GSA (first, second, total) ------------------------------------------ SobolGSA
N_sobol = 2^12
A, B = QuasiMonteCarlo.generate_design_matrices(N_sobol, lb, ub, SobolSample(), 2)
sobol_sens = gsa(qoi_batch, Sobol(order = [0, 1, 2]), A, B; batch = true)

S1 = clamp.(sobol_sens.S1, 0, 1)   # (n_outputs x n_params)
ST = clamp.(sobol_sens.ST, 0, 1)

bars = [groupedbar(hcat(S1[j, :], ST[j, :]); label = ["S1" "ST"],
                   xticks = (1:5, param_names), ylims = (0, 1.05),
                   ylabel = "Sobol index [-]", title = output_names[j])
        for j in 1:2]
fig_sobol = plot(bars...; layout = (2, 1), size = (700, 650))
display(fig_sobol)
savefig(fig_sobol, joinpath(@__DIR__, "gsa_sobol.png"))

println("Second-order indices S2:")
display(sobol_sens.S2)
# ------------------------------------------------------------------------------ SobolGSA