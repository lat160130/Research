% -- Clear Model ---------------------------------------------------------- CM
clc;
clear;
close all;
% ------------------------------------------------------------------------- CM

GMT_Model_Name = "Guppy-Mitchell-Taylor";
GMT_Model = sbiomodel(GMT_Model_Name);

% -- Variable Set Up ------------------------------------------------------ VSU
speciesNames = ["x", "y"];
initial_amount_of_species = 0;
% ------------------------------------------------------------------------- VSU

% -- Parallel Setup ------------------------------------------------------- Parallel Setup
parpool;
% ------------------------------------------------------------------------- Parallel Setup
% -- Setting up the species ----------------------------------------------- Setting up species
addspecies(GMT_Model, speciesNames(1), 'InitialAmount', initial_amount_of_species);
addspecies(GMT_Model, speciesNames(2), 'InitialAmount', initial_amount_of_species);
% ------------------------------------------------------------------------- Setting up species



% -- Defining model parameters -------------------------------------------- Parameters
parameterNames = ["k1", "k2", "b12", "alpha1", "alpha2"];
parameterValues = [1, 1, 1, 1, 1];

% load the parameters
for parameterIndex = 1:numel(parameterNames)
    addparameter(GMT_Model, parameterNames(parameterIndex), ...
        parameterValues(parameterIndex));
end

bounds = [0 3; 0 3; 0 1; 0 1; 0 1]; 
% ------------------------------------------------------------------------- Parameters


% -- Add Rules ------------------------------------------------------------ Rules
addrule(GMT_Model, 'x = k1*(-x + b12*y + alpha1)',      'RuleType', 'rate');
addrule(GMT_Model, 'y = k2*((1-alpha2)*x -y + alpha2)', 'RuleType', 'rate');
% ------------------------------------------------------------------------- Rules

% -- Plot the indices ----------------------------------------------------- Plot
res = sbiosobol(GMT_Model, parameterNames, {'x','y'}, ...
      'Bounds', bounds, 'NumberSamples', 2^12, ...
      'OutputTimes', linspace(0,80,81), 'ShowWaitbar', true, 'UseParallel',true);
plot(res);
bar(res);
res.SobolIndices;
% ------------------------------------------------------------------------- Plot