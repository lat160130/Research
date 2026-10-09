# -- Calculate the First, Second, and Total order sobol Indicies for the 
# -- Guppy Mitchell Taylor (GMT) labeled/unlabeled flux model.




# -- Load Packages ------------------------------------------------------------- LoadPackage
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using GlobalSensitivity, Statistics, OrdinaryDiffEq, QuasiMonteCarlo
using SciMLBase: EnsembleProblem, EnsembleThreads
using DiffEqGPU, CUDA, OrdinaryDiffEqCore, StaticArrays
using Plots, StatsPlots, LaTeXStrings, CairoMakie
# ------------------------------------------------------------------------------ LoadPackage

# -- Cite Packages ------------------------------------------------------------- CitePackage
# https://github.com/SciML/DiffEqGPU.jl - using diffeq gpu
# https://docs.sciml.ai/GlobalSensitivity/stable/ -- using Global sensitivity
# ------------------------------------------------------------------------------ CitePackage

#= FIX THIS LATER FOR NxN system of ODES
# -- Create Function to solve the system of ODEs ------------------------------- FsolODE
function f_flux_model!(dU, U, p, t)
# Inputs: 
# X - vector of N metabolites and their concentrations
# K - diagonal matrix of k_i 
# p -   


# Outputs:
# dX - this is a "vertical vector" of dX/dt - the left handside of equation (16)
# of https://link.springer.com/article/10.1007/s11538-024-01386-x

# Equation 16: dX = K*A*X + alpha
# Where A is a 2x2 matrix: (recall in Julia, array indexing starts at 1 by default)
# [-1,           beta12]
# [1 - alpha(2),     -1]

# define matrix A
A = [-1 beta12; (1- alpha(2)) -1]
dU = K*A*X + alpha

end 
# ------------------------------------------------------------------------------ FsolODE
=#



function f!(du, u, p, t)
    du[1] = k1 * (-u[1]           + p[3]*u[2] + u[1])
    du[2] = k2 * ((1 - p[2])*u[1] - u[2]      + u[2])

end

# -- Initial Variables --------------------------------------------------------- InitVar
tspan = (0.0, 80)
#=
X0 = [1.0; 1.0] # starting concentrations for 
K = [1 0; 0 1] # K11 = k1, K22 = k2 in equation 16.
alpha = [3 3];
=#


alpha1 = 3
alpha2 = 3
beta12 = 1
k1     = 1
k2     = 1
p = [alpha1; alpha2; beta12; k1; k2]
u0 = [1.0; 1.0]

# ------------------------------------------------------------------------------ InitVar

prob = ODEProblem(f!, u0, tspan, p)
t = collect(range(0, stop = 10, length = 200))

f1 = function (p)
    prob1 = remake(prob; p = p)
    sol = solve(prob1, Tsit5(); saveat = t)
    return [mean(sol[1, :]), maximum(sol[2, :])]
end

bounds = [[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]]

reg_sens = gsa(f1, RegressionGSA(true), bounds, samples = 200)
fig = Figure(resolution = (600, 400))
ax,
hm = CairoMakie.heatmap(fig[1, 1],
    reg_sens.partial_correlation,
    axis = (xticksvisible = false, yticksvisible = false, yticklabelsvisible = false,
        xticklabelsvisible = false, title = "Partial correlation"))
Colorbar(fig[1, 2], hm)
ax,
hm = CairoMakie.heatmap(fig[2, 1],
    reg_sens.standard_regression,
    axis = (xticksvisible = false, yticksvisible = false, yticklabelsvisible = false,
        xticklabelsvisible = false, title = "Standard regression"))
Colorbar(fig[2, 2], hm)
fig



