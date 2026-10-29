# =============================================================================
# GMT (Guppy-Mitchell-Taylor) chain model, Eq. 20 (Eq. 16 when Nsp = 2)
# Sobol first- and total-order indices, ODE ensemble solved on an NVIDIA GPU.
# https://link.springer.com/article/10.1007/s11538-024-01386-x
#
# One-time setup:
#   using Pkg
#   Pkg.add(["DiffEqGPU", "OrdinaryDiffEq", "CUDA", "StaticArrays",
#            "GlobalSensitivity", "QuasiMonteCarlo", "StatsPlots"])
# =============================================================================

using DiffEqGPU, OrdinaryDiffEq, CUDA, StaticArrays
using GlobalSensitivity, QuasiMonteCarlo, Statistics, StatsPlots

CUDA.functional() || error("No usable NVIDIA GPU found.")

# -- Precision -----------------------------------------------------------------
# Float64 on data-center GPUs (A100/V100/H100).
# Float32 on consumer GPUs (RTX etc.), where Float64 is 32-64x slower.
const T = Float64

# -- Model size ----------------------------------------------------------------
const Nsp = 5               # number of species (>= 2); Nsp = 2 is Eq. 16
const P   = 3Nsp - 1        # parameters: k1..kN, b1..b(N-1), alpha1..alphaN

# -- Parameters and bounds (same as the MATLAB script) --------------------------
pnames = vcat(["k$i" for i in 1:Nsp],
              ["b$i" for i in 1:Nsp-1],         # b_i = beta_{i,i+1}
              ["alpha$i" for i in 1:Nsp])
lb = zeros(P)
ub = vcat(fill(3.0, Nsp), fill(1.0, Nsp - 1), fill(1.0, Nsp))

# -- Eq. 20 right-hand side ------------------------------------------------------
# Out-of-place with static arrays: required by EnsembleGPUKernel.
#   x1' = k1 (            - x1 + b1 x2      + a1)
#   xi' = ki ((1-ai-bi) x_{i-1} - xi + bi x_{i+1} + ai)
#   xN' = kN ((1-aN)    x_{N-1} - xN            + aN)
function gmt_chain(u::SVector{N,S}, p, t) where {N,S}
    SVector{N,S}(ntuple(Val(N)) do i
        k    = p[i]
        α    = p[2N - 1 + i]
        b    = i < N ? p[N + i] : zero(S)
        up   = i > 1 ? (one(S) - α - b) * u[i-1] : zero(S)
        down = i < N ? b * u[i+1] : zero(S)
        k * (up - u[i] + down + α)
    end)
end

u0    = ones(SVector{Nsp,T})            # every species starts at 1
p0    = zeros(SVector{P,T})             # placeholder; replaced per sample
tspan = (zero(T), T(80))
ts    = T.(1:80)    # t = 0 omitted: all outputs equal 1 there -> zero variance -> NaN indices
prob  = ODEProblem{false}(gmt_chain, u0, tspan, p0)

# -- Batched model: P x ns parameter matrix -> (Nsp*nt) x ns output matrix -------
# Rows are ordered species-fastest: row = (time_index - 1)*Nsp + species.
function model_batch(Pmat)
    ns = size(Pmat, 2)
    prob_func = (prob, i, repeat) -> remake(prob, p = SVector{P,T}(view(Pmat, :, i)))
    eprob = EnsembleProblem(prob; prob_func, safetycopy = false)
    sol = solve(eprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
                trajectories = ns, saveat = ts, dt = T(0.1),
                abstol = T(1e-8), reltol = T(1e-6))
    nt = length(ts)
    Y  = Matrix{T}(undef, Nsp * nt, ns)
    @inbounds for j in 1:ns
        us = sol.u[j].u
        @assert length(us) == nt "saveat returned $(length(us)) points, expected $nt"
        for it in 1:nt
            Y[(it-1)*Nsp+1 : it*Nsp, j] .= us[it]
        end
    end
    return Y
end

# -- Sampling ------------------------------------------------------------------
# Model evaluations = Nsamp * (P + 2), same count as sbiosobol.
Nsamp = 2^12
A, B  = QuasiMonteCarlo.generate_design_matrices(Nsamp, lb, ub, SobolSample())

# -- Warm-up (JIT + GPU kernel compilation; not timed) --------------------------
Aw, Bw = QuasiMonteCarlo.generate_design_matrices(32, lb, ub, SobolSample())
gsa(model_batch, Sobol(), Aw, Bw; batch = true)

# -- Timed run ---------------------------------------------------------------------
stats = @timed gsa(model_batch, Sobol(), A, B; batch = true)
res   = stats.value
println("Sobol computation time (GPU): ", round(stats.time; digits = 2), " s")
println("Host memory allocated:        ", round(stats.bytes / 1e9; digits = 2), " GB")
println("Result object size:           ", round(Base.summarysize(res) / 1e6; digits = 2), " MB")
CUDA.memory_status()

# -- Reshape indices to [species, time, parameter] --------------------------------
S1, ST = res.S1, res.ST
size(S1, 2) == P || (S1 = permutedims(S1); ST = permutedims(ST))
nt  = length(ts)
S1t = reshape(S1, Nsp, nt, P)
STt = reshape(ST, Nsp, nt, P)

# -- Time-averaged indices, one bar chart per species ------------------------------
nanmean(x) = (y = filter(!isnan, x); isempty(y) ? NaN : mean(y))
S1bar = [nanmean(S1t[s, :, j]) for s in 1:Nsp, j in 1:P]
STbar = [nanmean(STt[s, :, j]) for s in 1:Nsp, j in 1:P]

plts = [groupedbar(pnames, hcat(S1bar[s, :], STbar[s, :]);
                   label = ["first order" "total order"], title = "x$s",
                   xrotation = 45, ylims = (0, 1))
        for s in 1:Nsp]
fig = plot(plts...; layout = (Nsp, 1), size = (900, 260Nsp))
savefig(fig, "gmt_sobol_gpu.png")
display(fig)

# Index time courses, e.g. first-order indices of every parameter for species s:
#   plot(ts, S1t[s, :, :]; label = permutedims(pnames), xlabel = "t", ylabel = "S1")
