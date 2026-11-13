% declare pool
numCores = 4;

if isempty(gcp('nocreate'))
    parpool('local', numCores);
    pool = gcp;
    
else
    pool = gcp;
end
nw = pool.NumWorkers;


% call claude_understand_sobol_n_cases.m n times as a function
% call it for n = 2, ..., 10, 20, 30, 100,
N_vec = [2:13];
nSamples = 2^12;
len_N = length(N_vec);


% -- Output file name: sobol_sweep_4cores_<timestamp>_N<list>.mat ------------------
stamp = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
nTag  = sprintf('%d-%d', N_vec(1), N_vec(end));
fname = sprintf('sobol_sweep_4cores_%s_N%s.mat', stamp, nTag);
% e.g. sobol_sweep_20261030_223015_N2-3.mat



clock_mat = zeros(len_N,1); % sobol time per N
wGB_mat   = zeros(len_N,nw); % row i = per-worker RAM for N_vec(i)
workerCPU_mat = zeros(len_N, nw); % row i = per-worker CPU s for N_vec(i)
metrics = cell(len_N, 1);
% store each clock time calculation


% Sweep along each calculation
for i = 1:len_N
    fprintf('N = %d ... ', N_vec(i));
    m = calc_sobol_time(N_vec(i), nSamples);
 
    clock_mat(i)        = m.tSobol;
    wGB_mat(i, :)       = m.wGB';
    workerCPU_mat(i, :) = m.workerCPU';
    metrics{i}          = rmfield(m, 'res');   % drop big results object; keep m.res if needed
 
    fprintf('%.1f s, efficiency %.2f\n', m.tSobol, m.efficiency);
    save(fname, 'N_vec', 'nSamples', 'nw', 'clock_mat', ...
         'wGB_mat', 'workerCPU_mat', 'metrics');
end


% -- Summary table (one row per N) -------------------------------------------
M = [metrics{:}];
summary = struct2table(rmfield(M, {'workerCPU', 'wGB'}));
disp(summary(:, {'N', 'nSims', 'tWarm', 'tSobol', 'wallPerSim', 'cpuPerSim', ...
                 'efficiency', 'estSpeedup', 'resultMB', 'MBfromWorkers', ...
                 'nInvalidSims', 'minS1', 'maxSumS1'}));

save(fname, 'summary', '-append');


