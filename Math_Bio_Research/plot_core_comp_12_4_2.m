% Compare Sobol sweep runs on 2, 4 and 12 workers for N = 2:13.
% Reads every sobol_sweep_*.mat in the current folder, keeps N in Nkeep,
% averages repeated runs of the same (workers, N), then plots
%   row 1: tSobol, resultMB, cpuPerSim vs N for each worker count
%   row 2: speedup 12 vs 4, 12 vs 2, 4 vs 2 (measured vs ideal)
clc; clear; close all;

% -- Settings ------------------------------------------------------------------
%files   = dir('sobol_sweep_*.mat');
files = [dir('sobol_sweep_2_19.mat'); dir('sobol_sweep_4cores_20261101_105216_N2-13.mat'); dir('sobol_sweep_2cores_20261101_115647_N2-13.mat')];
Nkeep   = 2:13;
cores   = [2 4 12];
pairs   = [12 4; 12 2; 4 2];          % [more workers, fewer workers]
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
missing = setdiff(cores, unique(rows.nw));
if ~isempty(missing)
    warning('No data for %s workers.', mat2str(missing));
end

% -- Average repeated runs ---------------------------------------------------------
G = groupsummary(rows, {'nw', 'N'}, 'mean', vars);
disp(G)

% -- Figure: 2 rows x 3 columns -------------------------------------------------------
fig = figure('Position', [100 100 1400 760]);
tl  = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
markers = {'o-', 's-', '^-'};
colors  = lines(numel(cores));

% Row 1: metrics vs N
for v = 1:numel(vars)
    nexttile; hold on; box on; grid on
    for c = 1:numel(cores)
        g = sortrows(G(G.nw == cores(c), :), 'N');
        if isempty(g), continue; end
        plot(g.N, g.("mean_" + vars{v}), markers{c}, 'Color', colors(c, :), ...
             'LineWidth', 1.5, 'DisplayName', sprintf('%d workers', cores(c)));
    end
    xlabel('Number of species N'); ylabel(metricsToPlot{v, 2});
    title(metricsToPlot{v, 2}); xlim([Nkeep(1) Nkeep(end)]); xticks(Nkeep);
    legend('Location', 'northwest');
end

% Row 2: speedups t(fewer) / t(more) at each N present in both
speedup = table();
for p = 1:size(pairs, 1)
    hi = pairs(p, 1);  lo = pairs(p, 2);
    nexttile; hold on; box on; grid on

    gHi = sortrows(G(G.nw == hi, :), 'N');
    gLo = sortrows(G(G.nw == lo, :), 'N');
    [Nc, iHi, iLo] = intersect(gHi.N, gLo.N);
    S = gLo.mean_tSobol(iLo) ./ gHi.mean_tSobol(iHi);

    plot(Nc, S, 'd-', 'LineWidth', 1.5, 'DisplayName', 'measured');
    yline(hi / lo, '--', sprintf('ideal (%g\\times)', hi / lo), ...
          'DisplayName', 'ideal', 'LabelHorizontalAlignment', 'left');
    xlabel('Number of species N'); ylabel(sprintf('t_{%d} / t_{%d}', lo, hi));
    title(sprintf('Speedup: %d vs %d workers', hi, lo));
    xlim([Nkeep(1) Nkeep(end)]); xticks(Nkeep);
    ylim([0, max([S; hi / lo]) * 1.15]);
    legend('Location', 'southeast');

    speedup = [speedup; table(repmat(sprintf('%d vs %d', hi, lo), numel(Nc), 1), ...
               Nc, S, S / (hi / lo), 'VariableNames', ...
               {'pair', 'N', 'speedup', 'efficiency'})]; %#ok<AGROW>
end

title(tl, 'GMT chain model: sbiosobol, 2 vs 4 vs 12 workers');
exportgraphics(fig, 'sobol_2v4v12_cores.png', 'Resolution', 200);

% speedup table: efficiency = measured / ideal (1 = perfect scaling)
disp(speedup)