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
betaNames  = "beta"  + (1:numMetabolites);

% ------------------------------------------------------------------------- Parameters
