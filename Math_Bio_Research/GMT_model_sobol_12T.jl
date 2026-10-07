# -- Calculate the First, Second, and Total order sobol Indicies for the 
# -- Guppy Mitchell Taylor (GMT) labeled/unlabeled flux model.




# -- Load Packages ------------------------------------------------------------- LoadPackage
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using GlobalSensitivity, Statistics, OrdinaryDiffEq, QuasiMonteCarlo, Plots
using SciMLBase: EnsembleProblem, EnsembleThreads
using DiffEqGPU, CUDA, OrdinaryDiffEqCore, StaticArrays
# ------------------------------------------------------------------------------ LoadPackage

# -- Cite Packages ------------------------------------------------------------- CitePackage
# https://github.com/SciML/DiffEqGPU.jl - using diffeq gpu
# https://docs.sciml.ai/GlobalSensitivity/stable/ -- using Global sensitivity
# ------------------------------------------------------------------------------ CitePackage


# -- Create Function to solve the system of ODEs ------------------------------- FsolODE
# function f_flux_model(dX, X, alpha, beta, k)
# Inputs:

# Outputs:


# ------------------------------------------------------------------------------ FsolODE

# -- Initial Variables --------------------------------------------------------- InitVar
tspan = (0.0, 10.0)
# ------------------------------------------------------------------------------ InitVar


