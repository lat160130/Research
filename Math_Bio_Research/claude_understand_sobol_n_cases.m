% -- Clear Model ---------------------------------------------------------- CM
clc;
clear;
close all;
% ------------------------------------------------------------------------- CM

% -- Installation notes --------------------------------------------------- IN
% To run this code the following toolboxes are required:
% 1. SimBiology
% 2. Parallel Computing Toolbox
% Use the Matlab Add-On Explorer

% This code was run on MATLAB R2024b and R2025b.  Not back tested for
% other versions of MATLAB.

% This script is standalone and can be run on its own.
% ------------------------------------------------------------------------- IN

% -- Set up Profiler ------------------------------------------------------ Profiler
% ------------------------------------------------------------------------- Profiler
% -- Set up Sim Biology Model --------------------------------------------- SBM
GMT_Model_Name = "Guppy-Mitchell-Taylor";
GMT_Model = sbiomodel(GMT_Model_Name); % this declares a "sim biology" object
% ------------------------------------------------------------------------- SBM

% -- Variable Set Up ------------------------------------------------------ VSU
N = 5;  % <-- NUMBER OF SPECIES.  Set this to whatever integer you want (N >= 2).
        %     N = 2 reproduces your original Equation 16 script exactly.
        %     N > 2 is Equation 20, the reversible chain reaction model.

initial_amount_of_species = 1; % quantity as in how much i.e. moles

% Species are now named x1, x2, ..., xN instead of x and y.  For N = 2,
% x1 is your old "x" and x2 is your old "y".
speciesNames = strings(1, N);
for speciesIndex = 1:N
    speciesNames(speciesIndex) = "x" + speciesIndex;
end
% ------------------------------------------------------------------------- VSU

% -- Parallel Setup ------------------------------------------------------- Parallel Setup
% This line turns on the parallel pool i.e. MATLAB is telling your CPUs hey
% I am about to use ALL of you guys get ready.  The first time you run this
% code, starting up the parallel pool takes approximately 30 seconds. After
% that the code will run much faster with any subsequent run.
if isempty(gcp('nocreate'))
    parpool;
end
% Note!  Sbiosobol (at last moment of reading) does not support GPUs, i
% suspect then you should be okay to run this on mac/linux/windows with
% igpu/non nvidia dgpu.
% ------------------------------------------------------------------------- Parallel Setup

% -- Setting up the species ----------------------------------------------- Setting up species
% We are editing values in the sim biology object, formally tracking what
% was initialized in section "Variable Set Up."  This loop replaces the two
% hard-coded addspecies calls -- change N in VSU and this follows along.
for speciesIndex = 1:N
    addspecies(GMT_Model, speciesNames(speciesIndex), ...
        'InitialAmount', initial_amount_of_species);
end
% ------------------------------------------------------------------------- Setting up species


% -- Defining model parameters -------------------------------------------- Parameters
% In the following article: https://link.springer.com/article/10.1007/s11538-024-01386-x
% Go to Equation 20 (Equation 16 is the N = 2 case of it).

% Parameter naming, in the order they get loaded:
%   k1   ... kN       turnover rates
%   b1   ... b(N-1)   bi is beta_{i,i+1}, the reverse flux ratio.
%                     For N = 2, b1 is what you called b12.
%   alpha1 ... alphaN unlabeled-input flux ratios
%
% Total parameter count is 3N-1.  For N = 2 that is 5, same as before.

parameterNames  = strings(1, 3*N-1);
parameterValues = zeros(1, 3*N-1);
bounds          = zeros(3*N-1, 2);

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

% NOTE: parameterValues only matters if you call sbiosimulate yourself.
% sbiosobol ignores them and draws every parameter from "bounds", so you do
% not need to hand-pick nominal values for a bigger N.

% load the parameters into the sbiomodel object
for parameterIndex = 1:numel(parameterNames)
    addparameter(GMT_Model, parameterNames(parameterIndex), ...
        parameterValues(parameterIndex));
end

% The below link explains more in detail about what adding parameters to a
% simbio object does.
% https://www.mathworks.com/help/simbio/ref/simbiology.model.addparameter.html

% FEASIBILITY NOTE (applies to your N = 2 script too): in Equation 20 the
% coefficient (1 - alphai - bi) IS the other reverse flux ratio beta_{i,i-1},
% so it is only physically meaningful when alphai + bi <= 1.  Sampling both
% on U(0,1) independently lets that go negative, and for i = 1 it also lets
% the labeled input (1 - alpha1 - b1)*F1 go negative or zero.  The ODE still
% solves, the answer just stops being a proportion.  If you want to stay
% inside the feasible region, swap the alpha and b bounds to [0 0.45].
% ------------------------------------------------------------------------- Parameters


% -- Add Rules ------------------------------------------------------------ Rules
% in "addrule" dx/dt is the LHS i.e. apply the RHS to species x's rate
% (dx/dt).  Each rule corresponds to another row in your system of
% differential equations, so for N species you build N rules.
% These are the three lines of Equation 20.

% First species: receives the labeled nutrient input.
addrule(GMT_Model, sprintf('x1 = k1*(-x1 + b1*x2 + alpha1)'), ...
    'RuleType', 'rate');

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

% The above rules come from equation 20 in the below article.
% https://link.springer.com/article/10.1007/s11538-024-01386-x
% ------------------------------------------------------------------------- Rules

% -- Profile and Calculate Sobol Indices ---------------------------------- Profile and Calculate

% Cost warning: sbiosobol runs (3N-1 + 2) * NumberSamples simulations.
% N = 2 -> 7*4096 = 28,672.  N = 5 -> 16*4096 = 65,536.  Drop NumberSamples
% to 2^10 while you are testing a larger N.
N = 2^12;

% -- WARMUP - model compilation on workers, not timed. -- V
sbiosobol(GMT_Model, cellstr(parameterNames), cellstr(speciesNames), NumberSamples=20, UseParallel=true);
% -- WARMUP --------------------------------------------- ^
t = tic; % measure computation time
res = sbiosobol(GMT_Model, cellstr(parameterNames), cellstr(speciesNames), ...
      'Bounds', bounds, 'NumberSamples', N, ...
      'OutputTimes', linspace(0,80,81), 'ShowWaitbar', true, 'UseParallel', true, Accelerate=true);
tSobol = toc(t);
fprintf('Sobol computation time (12 workers): %.1f s\n', tSobol);

% RAM used by each worker (Windows only)
if ispc
    f = parfevalOnAll(@memory, 1);
    m = fetchOutputs(f);
    wGB = [m.MemUsedMATLAB]'/1e9;
    fprintf('Per-worker RAM: mean %.2f GB, max %.2f GB, total %.2f GB\n', ...
        mean(wGB), max(wGB), sum(wGB));
end
% ------------------------------------------------------------------------- Profile and Calculate

% -- Plot the indices ----------------------------------------------------- Plot
% The below code plots two figures.  Figure 1 shows 1st and total order
% sobol indices OVER TIME for each of the 3N-1 parameters that are in the
% system of ODEs.
% Figure 2 shows the sobol index as a bar graph.  "Darker colors mean that
% those values occur more often over the whole time course."
% See the following link for more: https://www.mathworks.com/help/simbio/ref/sbiosobol.html#d127e21583
% Do a cntrl-f of the following "Plot the time course of the first- and
% total-order Sobol indices." to read more on what the plots are saying
% exactly.


% plot(res); % plots the individual fractions
bar(res); % plots time integrated first and total order indices, bar chart
res.SobolIndices;
% ------------------------------------------------------------------------- Plot