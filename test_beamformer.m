% test_beamformer_simple.m
% Test the simplified beamformer

clear; close all; clc;

fprintf('Testing Simplified Beamformer\n');
fprintf('=============================\n');

% Set random seed
utils.set_random_seed(42);

% Create test data
M = 8;
sources = 2;
array = arrayUtils.ULA_positions(M);
theta_range = linspace(-pi/2, pi/2, 180);
theta_true = [-30, 20] * pi/180;

% Generate signals
A = exp(-1j * pi * array * sin(theta_true));
X = A * (randn(sources, 1000) + 1j*randn(sources, 1000))/sqrt(2);
X = utils.add_awgn(X, 20);

% Run beamformer
tic;
[DoAs, spectrum] = beamformer(X, array, theta_range, sources);
elapsed = toc;

% Display results
fprintf('Array: %d elements\n', M);
fprintf('True DOAs: %.1f°, %.1f°\n', rad2deg(theta_true));
fprintf('Computation time: %.4f s\n', elapsed);

if ~isempty(DoAs)
    fprintf('Estimated DOAs: ');
    fprintf('%.1f° ', rad2deg(theta_range(DoAs)));
    fprintf('\n');
else
    fprintf('No DOAs estimated\n');
end

% Plot
figure;
plot(rad2deg(theta_range), 10*log10(spectrum/max(spectrum)), 'LineWidth', 2);
hold on;
plot(rad2deg(theta_true), zeros(2,1), 'r^', 'MarkerSize', 12, ...
     'LineWidth', 2, 'MarkerFaceColor', 'r');
if ~isempty(DoAs)
    plot(rad2deg(theta_range(DoAs)), ...
         10*log10(spectrum(DoAs)/max(spectrum)), 'ko', ...
         'MarkerSize', 10, 'LineWidth', 2);
end
xlabel('Angle (degrees)');
ylabel('Normalized Spectrum (dB)');
title('Beamformer Spectrum');
grid on;
legend('Spectrum', 'True DOAs', 'Estimated', 'Location', 'best');
xlim([-90, 90]);
set(gcf, 'Color', 'w');

fprintf('\nTest completed successfully!\n');