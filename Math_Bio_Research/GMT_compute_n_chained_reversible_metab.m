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

% This code was run ont MATLAB R2024b and R2025b.  Not back tested for
% other versions of MATLAB.

% This script is standalone and can be run on its own.
% ------------------------------------------------------------------------- IN

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


% -- Set up Sim Biology Model --------------------------------------------- SBM
GMT_Model_Name = "Guppy-Mitchell-Taylor n-chained reversible metabolites";
GMT_Model = sbiomodel(GMT_Model_Name); % this declares a "sim biology object
% ------------------------------------------------------------------------- SBM

% -- Variable Set Up ------------------------------------------------------ VSU
numMetabolites = 5; % this is analogous to n
speciesNames = "x" + (1:numMetabolites); % in this code, i am only using two species, however you can 
cellSpeciesNames = cellstr(speciesNames);
% have more species by extending the vector (although you will have to
% adjust more of the code.
initial_amount_of_species = 1; % quantity as in how much i.e. moles
% ------------------------------------------------------------------------- VSU

% -- Setting up the species ----------------------------------------------- Setting up species
% We are editing values in the sim biology object, formally tracking what
% was initialized in section "Variable Set Up."  If you extend the number
% of species in VSU, then you will need to add additional species here.

for i = 1:numMetabolites
    addspecies(GMT_Model, speciesNames(i), 'InitialAmount', initial_amount_of_species);
end % i = 1:numMetabolites
% ------------------------------------------------------------------------- Setting up species

% -- Defining model parameters -------------------------------------------- Parameters
kNames     = "k" + (1:numMetabolites);
alphaNames = "alpha" + (1:numMetabolites); 
betaNamesMat  = str2sym("beta_" + (1:numMetabolites)' + ...
                            "_" + (1:numMetabolites) );
% ------------------------------------------------------------------------- Parameters


% WORK ON THIS!!!!!!!!!!!!
% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


% -- Nondimensional Reversible Chain Reaction Model ----------------------- nxn matrix non dimensional model
% x1dot = k1(-x1 + beta12*x2 + alpha1)

% for i = 2, ..., numMetabolites - 1
% xidot = ki((1-alphai - betai,i+1)xi-1 - xi-1 + betai,i+1*xi+1 + alphai)

% xn = kN((1-alphaN)*xN-1 - xN - alphaN);

% generate the matrix 

% load the parameters into the sbiomodel object

% edit the bounds vector if you have more parameters to add
% (I don't know how to adjust the bounds to NOT use a uniform distribution).


% ------------------------------------------------------------------------- nxn matrix non dimensional model



% -- Add Rules ------------------------------------------------------------ Rules
% in "addrule" dx/dt is the LHS i.e. apply the RHS to species x's rate
% (dx/dt)


% Example for the toy two metabolite system.
% addrule(GMT_Model, 'x = k1*(-x + b12*y + alpha1)',      'RuleType', 'rate');
% addrule(GMT_Model, 'y = k2*((1-alpha2)*x -y + alpha2)', 'RuleType', 'rate');
% The above two rules come from equation 16 in the below article.
% https://link.springer.com/article/10.1007/s11538-024-01386-x

% Use a loop for an n metabolite system O(n) time complexity

% Each rule corresponds to another row in your system of differential
% equations.

% ------------------------------------------------------------------------- Rules



% -- Plot the indices ---------------------------------------------------- Plot
% The below code plots two figures.  Figure 1 shows 1st and total order
% sobol indices OVER TIME for each of the (in this code) 5 parameters that are in the
% system of ODEs.
% Figure 2 shows the sobol index as a bar graph.  "Darker colors mean that those values occur
% more often over the whole time course."
% See the following link for more: https://www.mathworks.com/help/simbio/ref/sbiosobol.html#d127e21583
% Do a cntrl-f of the following "Plot the time course of the first- and
% total-order Sobol indices." to read more on what the plots are saying
% exactly.
res = sbiosobol(GMT_Model, parameterNames, {'x','y'}, ...
      'Bounds', bounds, 'NumberSamples', 2^12, ...
      'OutputTimes', linspace(0,80,81), 'ShowWaitbar', true, 'UseParallel',true);
plot(res);
bar(res);
res.SobolIndices;
% ------------------------------------------------------------------------- Plot