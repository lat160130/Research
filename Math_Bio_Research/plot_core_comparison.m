% Compare Sobol sweep runs on 4 vs 12 workers for N = 2:13.
% Reads every sobol_sweep_*.mat in the current folder, keeps N in Nkeep,
% averages repeated runs of the same (workers, N), and plots
% tSobol, resultMB and cpuPerSim against N.
clc; clear; close all;

% -- Settings ------------------------------------------------------------------
%files   = dir('sobol_sweep_2_19.mat');   % or list files explicitly, see below

files = [dir('sobol_sweep_2_19.mat'); dir('sobol_sweep_4cores_20261101_105216_N2-13.mat')];
Nkeep   = 2:13;
cores   = [4 12];
metricsToPlot = {'tSobol',   'Wall-clock time (s)'; ...
                 'resultMB', 'Result size (MB)'; ...
                 'cpuPerSim','CPU time per simulation (s)'};

assert(~isempty(files), 'No sobol_sweep_*.mat files found in %s', pwd);

% -- Gather one row per (file, N) ------------------------------------------------
vars = metricsToPlot(:, 1)';
rows = table();
for f = files'
    S = load(fullfile(f.folder, f.name), 'metrics', 'nw');
    if ~isfield(S, 'metrics') || ~isfield(S, 'nw')
        warning('Skipping %s (no metrics/nw saved).', f.name);
        continue
    end
    for i = 1:numel(S.metrics)
        m = S.metrics{i};
        if isempty(m) || ~ismember(m.N, Nkeep)     % unfinished entry or N outside 2:13
            continue
        end
        vals = nan(1, numel(vars));
        for v = 1:numel(vars)
            if isfield(m, vars{v}), vals(v) = m.(vars{v}); end
        end
        rows = [rows; [table(S.nw, m.N, string(f.name), ...
                       'VariableNames', {'nw', 'N', 'file'}), ...
                       array2table(vals, 'VariableNames', vars)]]; %#ok<AGROW>
    end
end

rows = rows(ismember(rows.nw, cores), :);
assert(~isempty(rows), 'No runs with %s workers and N in %d:%d.', ...
       mat2str(cores), Nkeep(1), Nkeep(end));

% -- Average repeated runs ---------------------------------------------------------
G = groupsummary(rows, {'nw', 'N'}, 'mean', vars);
disp(G)

% -- Plot ------------------------------------------------------------------------------
fig = figure('Position', [100 100 1400 380]);
tl  = tiledlayout(fig, 1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
markers = {'o-', 's-'};

for v = 1:numel(vars)
    nexttile; hold on; box on; grid on
    for c = 1:numel(cores)
        g = G(G.nw == cores(c), :);
        g = sortrows(g, 'N');
        plot(g.N, g.("mean_" + vars{v}), markers{c}, 'LineWidth', 1.5, ...
             'DisplayName', sprintf('%d workers', cores(c)));
    end
    xlabel('Number of species N'); ylabel(metricsToPlot{v, 2});
    title(metricsToPlot{v, 2}); xlim([Nkeep(1) Nkeep(end)]); xticks(Nkeep);
    legend('Location', 'northwest');
end

% Speedup panel: t(4 workers) / t(12 workers) at each N present in both
nexttile; hold on; box on; grid on
g4  = sortrows(G(G.nw == cores(1), :), 'N');
g12 = sortrows(G(G.nw == cores(2), :), 'N');
[Nc, i4, i12] = intersect(g4.N, g12.N);
plot(Nc, g4.mean_tSobol(i4) ./ g12.mean_tSobol(i12), 'd-', 'LineWidth', 1.5, ...
     'DisplayName', 'measured');
yline(cores(2) / cores(1), '--', 'ideal (3\times)', 'DisplayName', 'ideal');
xlabel('Number of species N'); ylabel('t_{4} / t_{12}');
title('Speedup, 12 vs 4 workers'); xlim([Nkeep(1) Nkeep(end)]); xticks(Nkeep);
legend('Location', 'southeast');

title(tl, 'GMT chain model: sbiosobol, 4 vs 12 workers');
exportgraphics(fig, 'sobol_4v12_cores.png', 'Resolution', 200);