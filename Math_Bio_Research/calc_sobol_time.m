function [m] = calc_sobol_time(n, nSamples)
% CALC_SOBOL_TIME  Sobol indices for the GMT chain model, with benchmark metrics.
%   m = calc_sobol_time(n)            n species, NumberSamples = 2^12
%   m = calc_sobol_time(n, nSamples)
%
%   Returns a struct m with timing, CPU, memory, data-transfer and
%   index-quality metrics (see "Collect Metrics" at the bottom), plus the
%   full sbiosobol results object in m.res.

% -- Installation notes --------------------------------------------------- IN
% To run this code the following toolboxes are required:
% 1. SimBiology
% 2. Parallel Computing Toolbox
% Use the Matlab Add-On Explorer

% This code was run on MATLAB R2024b and R2025b.  Not back tested for
% other versions of MATLAB.

% Per-worker RAM (wGB) uses MEMORY, which only exists on Windows.  On
% macOS/Linux wGB is returned as NaN and everything else still works.
% ------------------------------------------------------------------------- IN

if nargin < 2
    nSamples = 2^12;
end

% -- Set up Sim Biology Model --------------------------------------------- SBM
GMT_Model_Name = "Guppy-Mitchell-Taylor";
GMT_Model = sbiomodel(GMT_Model_Name); % this declares a "sim biology" object
% ------------------------------------------------------------------------- SBM

% -- Variable Set Up ------------------------------------------------------ VSU
N = n;  % <-- NUMBER OF SPECIES (N >= 2).
        %     N = 2 is Equation 16.
        %     N > 2 is Equation 20, the reversible chain reaction model.

initial_amount_of_species = 1; % quantity as in how much i.e. moles

speciesNames = "x" + (1:N);    % x1, x2, ..., xN
% ------------------------------------------------------------------------- VSU

% -- Parallel Setup ------------------------------------------------------- Parallel Setup
% Starts the pool if the driver has not already.  The first start takes
% ~30 s; the driver should start it once so this is skipped on every call.
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool;
end
nW = pool.NumWorkers;
% Note!  sbiosobol does not support GPUs.
% ------------------------------------------------------------------------- Parallel Setup

% -- Setting up the species ----------------------------------------------- Setting up species
for speciesIndex = 1:N
    addspecies(GMT_Model, speciesNames(speciesIndex), ...
        'InitialAmount', initial_amount_of_species);
end
% ------------------------------------------------------------------------- Setting up species

% -- Defining model parameters -------------------------------------------- Parameters
% https://link.springer.com/article/10.1007/s11538-024-01386-x, Equation 20
% (Equation 16 is the N = 2 case of it).
%
% Parameter naming, in the order they get loaded:
%   k1   ... kN       turnover rates
%   b1   ... b(N-1)   bi is beta_{i,i+1}, the reverse flux ratio
%   alpha1 ... alphaN unlabeled-input flux ratios
% Total parameter count is 3N-1.

nParams         = 3*N - 1;
parameterNames  = strings(1, nParams);
parameterValues = zeros(1, nParams);
bounds          = zeros(nParams, 2);

parameterIndex = 0;

for i = 1:N                                  % turnover rates ki ~ U(0,3)
    parameterIndex = parameterIndex + 1;
    parameterNames(parameterIndex)  = "k" + i;
    parameterValues(parameterIndex) = 3/10;
    bounds(parameterIndex, :)       = [0 3];
end

for i = 1:N-1                                % bi = beta_{i,i+1} ~ U(0,1)
    parameterIndex = parameterIndex + 1;
    parameterNames(parameterIndex)  = "b" + i;
    parameterValues(parameterIndex) = 3/20;
    bounds(parameterIndex, :)       = [0 1];
end

for i = 1:N                                  % alphai ~ U(0,1)
    parameterIndex = parameterIndex + 1;
    parameterNames(parameterIndex)  = "alpha" + i;
    parameterValues(parameterIndex) = 1/4;
    bounds(parameterIndex, :)       = [0 1];
end

% parameterValues are used only by the single-simulation baseline below;
% sbiosobol draws every parameter from "bounds".
for parameterIndex = 1:nParams
    addparameter(GMT_Model, parameterNames(parameterIndex), ...
        parameterValues(parameterIndex));
end

% FEASIBILITY NOTE: (1 - alphai - bi) is beta_{i,i-1}, only meaningful when
% alphai + bi <= 1.  Independent U(0,1) sampling lets it go negative.  To
% stay feasible, set the alpha and b bounds to [0 0.45].
% ------------------------------------------------------------------------- Parameters

% -- Add Rules ------------------------------------------------------------ Rules
% Each rate rule is one row of Equation 20 (dxi/dt on the LHS).

% First species: receives the labeled nutrient input.
addrule(GMT_Model, 'x1 = k1*(-x1 + b1*x2 + alpha1)', 'RuleType', 'rate');

% Interior species (this loop does nothing when N = 2).
for i = 2:N-1
    addrule(GMT_Model, sprintf( ...
        'x%d = k%d*((1-alpha%d-b%d)*x%d - x%d + b%d*x%d + alpha%d)', ...
        i, i, i, i, i-1, i, i, i+1, i), 'RuleType', 'rate');
end

% Last species: no downstream neighbour, so no b term.
addrule(GMT_Model, sprintf( ...
    'x%d = k%d*((1-alpha%d)*x%d - x%d + alpha%d)', ...
    N, N, N, N-1, N, N), 'RuleType', 'rate');
% ------------------------------------------------------------------------- Rules

% -- Shared sbiosobol options --------------------------------------------- Options
% One option list for the warm-up and the timed run, so the warm-up compiles
% exactly what the timed run uses (including Accelerate).
outputTimes = linspace(0, 80, 81);
pNames = cellstr(parameterNames);
sNames = cellstr(speciesNames);
opts   = {'Bounds', bounds, 'OutputTimes', outputTimes, ...
          'ShowWaitbar', false, 'UseParallel', true, 'Accelerate', true};
% ------------------------------------------------------------------------- Options

% -- Serial baseline: one simulation ------------------------------------- Baseline
% Time for a single accelerated simulation at the nominal parameter values,
% used to estimate the 1-core run time and the parallel speedup.
cs = getconfigset(GMT_Model);
cs.StopTime = outputTimes(end);
cs.SolverOptions.OutputTimes = outputTimes;
sbioaccelerate(GMT_Model);
tSingleSim = timeit(@() sbiosimulate(GMT_Model));
% ------------------------------------------------------------------------- Baseline

% -- Warm-up (compilation on workers), timed separately ------------------ Warmup
tw = tic;
sbiosobol(GMT_Model, pNames, sNames, opts{:}, 'NumberSamples', 20);
tWarm = toc(tw);
% ------------------------------------------------------------------------- Warmup

% -- Calculate Sobol Indices (timed) -------------------------------------- Calculate
% Cost: sbiosobol runs (nParams + 2) * nSamples = (3N+1) * nSamples simulations.
cpu0 = fetchOutputs(parfevalOnAll(@cputime, 1));   % nW x 1 worker CPU seconds
ticBytes(pool);

t   = tic;
res = sbiosobol(GMT_Model, pNames, sNames, opts{:}, 'NumberSamples', nSamples);
tSobol = toc(t);

bytes     = tocBytes(pool);                        % nW x 2: [to workers, from workers]
cpu1      = fetchOutputs(parfevalOnAll(@cputime, 1));
workerCPU = cpu1 - cpu0;
% ------------------------------------------------------------------------- Calculate

% -- Memory --------------------------------------------------------------- Memory
% RAM held by each worker after the run (Windows only).
if ispc
    mem = fetchOutputs(parfevalOnAll(@memory, 1));
    wGB = [mem.MemUsedMATLAB]' / 1e9;
else
    wGB = nan(nW, 1);
end
s = whos('res');
% ------------------------------------------------------------------------- Memory

% -- Index quality checks ------------------------------------------------- Quality
% SobolIndices is a struct array (parameters x observables); each entry holds
% FirstOrder and TotalOrder time courses.
SI   = res.SobolIndices;
dims = size(SI);
S1   = reshape([SI.FirstOrder], [], dims(1), dims(2));   % time x dim1 x dim2
ST   = reshape([SI.TotalOrder], [], dims(1), dims(2));
pDim  = find(dims == nParams, 1) + 1;                    % parameter dimension
sumS1 = sum(S1, pDim, 'omitnan');

nInvalid = NaN;
if isprop(res, 'SimulationInfo') && isfield(res.SimulationInfo, 'ValidSample')
    nInvalid = nnz(~res.SimulationInfo.ValidSample);
end
% ------------------------------------------------------------------------- Quality

% -- Collect Metrics ------------------------------------------------------ Metrics
nSims = nSamples * (nParams + 2);

m.N              = N;
m.nParams        = nParams;
m.nSamples       = nSamples;
m.nSims          = nSims;
m.nWorkers       = nW;

% timing
m.tWarm          = tWarm;                          % compilation + startup, s
m.tSobol         = tSobol;                         % wall time of timed run, s
m.wallPerSim     = tSobol / nSims;                 % wall s per simulation
m.simsPerSec     = nSims / tSobol;
m.tSingleSim     = tSingleSim;                     % serial s for one simulation
m.estSerialTime  = tSingleSim * nSims;             % estimated 1-core run time, s
m.estSpeedup     = m.estSerialTime / tSobol;

% CPU
m.workerCPU      = workerCPU;                      % nW x 1, s
m.totalWorkerCPU = sum(workerCPU);
m.cpuPerSim      = sum(workerCPU) / nSims;         % CPU s per simulation
m.efficiency     = sum(workerCPU) / (nW * tSobol); % ~1 = workers busy the whole run

% memory and data transfer
m.wGB            = wGB;                            % nW x 1, GB per worker after run
m.resultMB       = s.bytes / 1e6;
m.MBtoWorkers    = sum(bytes(:,1)) / 1e6;
m.MBfromWorkers  = sum(bytes(:,2)) / 1e6;

% index quality (t = 0 has zero variance, so some NaN indices are expected)
m.nInvalidSims   = nInvalid;                       % failed simulations
m.nNaNIndices    = nnz(isnan(S1)) + nnz(isnan(ST));
m.minS1          = min(S1(:));                     % noticeably < 0 -> too few samples
m.fracS1gtST     = mean(S1(:) > ST(:) + 0.01, 'omitnan');  % should be ~0
m.maxSumS1       = max(sumS1(:));                  % should be <= 1

m.res            = res;                            % full sbiosobol results object

fprintf('N = %d: %.1f s on %d workers, efficiency %.2f\n', ...
    N, tSobol, nW, m.efficiency);
% ------------------------------------------------------------------------- Metrics

end