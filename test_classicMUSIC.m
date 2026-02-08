% test_classicMUSIC.m
% Test the classic MUSIC algorithm

clear; close all; clc;

fprintf('===================================================\n');
fprintf('Classic MUSIC Algorithm Test\n');
fprintf('===================================================\n');

% Add utilities to path if needed
addpath(pwd);

% Set random seed for reproducibility
utils.set_random_seed(42);

%% Test Configuration
M = 10;                  % Array elements
true_sources = 3;        % True number of sources
Nsamples = 2000;         % Snapshots
snr_db = 20;             % SNR in dB

% Array geometry
array = arrayUtils.ULA_positions(M, 0.5);
theta_range = linspace(-pi/2, pi/2, 361);  % Higher resolution

% True DOAs (well-separated)
theta_true = [-50, 0, 40] * pi/180;

fprintf('Test Configuration:\n');
fprintf('  Array elements: %d\n', M);
fprintf('  True sources: %d\n', true_sources);
fprintf('  True DOAs: ');
fprintf('%.1f° ', rad2deg(theta_true));
fprintf('\n');
fprintf('  Snapshots: %d\n', Nsamples);
fprintf('  SNR: %.1f dB\n', snr_db);
fprintf('\n');

%% Generate Test Data
fprintf('Generating test data...\n');

% Signal amplitudes (different for each source)
amplitudes = [1.0, 0.8, 0.6];

% Steering matrix
A = zeros(M, true_sources);
for i = 1:true_sources
    A(:, i) = utils.ULA_action_vector(array, theta_true(i));
end

% Source signals (complex Gaussian)
S = (randn(true_sources, Nsamples) + 1j * randn(true_sources, Nsamples)) / sqrt(2);
S = diag(amplitudes) * S;  % Apply different amplitudes

% Received signal
X = A * S;

% Add noise
X = utils.add_awgn(X, snr_db);

fprintf('  Data generation complete.\n\n');

% In test_classicMUSIC.m, update the two test sections:

%% Test 1: Classic MUSIC with known number of sources
fprintf('1. Classic MUSIC (Known Number of Sources)\n');
fprintf('   --------------------------------------\n');

tic;
[DoAs_known, spectrum_known] = classicMUSIC.estimate(X, array, theta_range, true_sources);
elapsed = toc;

if ~isempty(DoAs_known)
    est_angles = rad2deg(theta_range(DoAs_known));
    fprintf('   Estimated DOAs: ');
    fprintf('%.1f° ', est_angles);
    fprintf('\n');
    
    % Calculate errors
    errors = abs(sort(theta_true) - sort(theta_range(DoAs_known)));
    fprintf('   Estimation errors: ');
    fprintf('%.2f° ', rad2deg(errors));
    fprintf('\n');
    fprintf('   RMS error: %.2f°\n', rad2deg(sqrt(mean(errors.^2))));
else
    fprintf('   No DOAs estimated\n');
end
fprintf('   Computation time: %.4f seconds\n', elapsed);
fprintf('   ✓ Test complete\n\n');

%% Test 2: Classic MUSIC with automatic source estimation
fprintf('2. Classic MUSIC (Automatic Source Estimation)\n');
fprintf('   -------------------------------------------\n');

tic;
% Note: Passing empty array for sources to trigger automatic estimation
[DoAs_auto, spectrum_auto] = classicMUSIC.estimate(X, array, theta_range, []);
elapsed = toc;

if ~isempty(DoAs_auto)
    est_angles = rad2deg(theta_range(DoAs_auto));
    fprintf('   Estimated DOAs: ');
    fprintf('%.1f° ', est_angles);
    fprintf('\n');
    fprintf('   Estimated number of sources: %d\n', length(DoAs_auto));
    
    % Calculate errors (if same number of sources)
    if length(DoAs_auto) == true_sources
        errors = abs(sort(theta_true) - sort(theta_range(DoAs_auto)));
        fprintf('   Estimation errors: ');
        fprintf('%.2f° ', rad2deg(errors));
        fprintf('\n');
        fprintf('   RMS error: %.2f°\n', rad2deg(sqrt(mean(errors.^2))));
    end
else
    fprintf('   No DOAs estimated\n');
end
fprintf('   Computation time: %.4f seconds\n', elapsed);
fprintf('   ✓ Test complete\n\n');

%% Test 3: Eigenvalue clustering analysis
fprintf('3. Eigenvalue Analysis\n');
fprintf('   -------------------\n');

% Calculate covariance matrix
R = utils.correlation_matrix(X);

% Get eigenvalues
[~, D] = utils.eigen_decomposition(R, true_sources);
eigenvalues = diag(D);

fprintf('   Eigenvalue range: [%.4f, %.4f]\n', min(eigenvalues), max(eigenvalues));
fprintf('   Eigenvalue ratio (max/min): %.2f\n', max(eigenvalues)/min(eigenvalues));

% Test clustering function
clustered = classicMUSIC.cluster(eigenvalues);
fprintf('   Clustered eigenvalues (noise): %d out of %d\n', length(clustered), M);
fprintf('   Estimated sources (from clustering): %d\n', M - length(clustered));

% Plot eigenvalues
classicMUSIC.plot_eigenvalues(eigenvalues, true_sources);
fprintf('   ✓ Eigenvalue analysis complete\n\n');

%% Test 4: Compare with other algorithms
fprintf('4. Algorithm Comparison\n');
fprintf('   --------------------\n');

% Run beamformer for comparison
[DoAs_bartlett, spectrum_bartlett] = beamformer_all.bartlett(X, array, theta_range, true_sources);

% Run Capon beamformer
[DoAs_capon, spectrum_capon] = beamformer_all.capon(X, array, theta_range, true_sources, 0.01);

fprintf('   Bartlett beamformer: ');
if ~isempty(DoAs_bartlett)
    fprintf('%.1f° ', rad2deg(theta_range(DoAs_bartlett)));
end
fprintf('\n');

fprintf('   Capon beamformer:    ');
if ~isempty(DoAs_capon)
    fprintf('%.1f° ', rad2deg(theta_range(DoAs_capon)));
end
fprintf('\n');

fprintf('   Classic MUSIC:       ');
if ~isempty(DoAs_known)
    fprintf('%.1f° ', rad2deg(theta_range(DoAs_known)));
end
fprintf('\n');

fprintf('   ✓ Comparison complete\n\n');

%% Test 5: Resolution test (closely spaced sources)
fprintf('5. Resolution Test\n');
fprintf('   ---------------\n');

% Create two closely spaced sources
theta_close = [-10, 10] * pi/180;
A_close = zeros(M, 2);
for i = 1:2
    A_close(:, i) = utils.ULA_action_vector(array, theta_close(i));
end

X_close = A_close * (randn(2, Nsamples) + 1j*randn(2, Nsamples))/sqrt(2);
X_close = utils.add_awgn(X_close, 15);

% Test with classic MUSIC
[DoAs_music_close, spectrum_music_close] = classicMUSIC.estimate(X_close, array, theta_range, 2);

fprintf('   True DOAs: %.1f°, %.1f° (Separation: %.1f°)\n', ...
        rad2deg(theta_close(1)), rad2deg(theta_close(2)), ...
        rad2deg(abs(diff(theta_close))));
    
if length(DoAs_music_close) >= 2
    sep_music = abs(diff(theta_range(DoAs_music_close(1:2))));
    fprintf('   MUSIC estimated separation: %.1f°\n', rad2deg(sep_music));
else
    fprintf('   MUSIC failed to resolve both sources\n');
end

fprintf('   ✓ Resolution test complete\n\n');

%% Visualization
fprintf('Generating comprehensive plots...\n');

figure('Position', [50, 50, 1400, 700], 'Name', 'Classic MUSIC Analysis', 'Color', 'w');

% Plot 1: All Spectra Comparison
subplot(2, 3, 1);
plot(rad2deg(theta_range), 10*log10(spectrum_bartlett/max(spectrum_bartlett)), ...
     'LineWidth', 1.5, 'DisplayName', 'Bartlett');
hold on;
plot(rad2deg(theta_range), 10*log10(spectrum_capon/max(spectrum_capon)), ...
     'LineWidth', 1.5, 'DisplayName', 'Capon');
plot(rad2deg(theta_range), 10*log10(spectrum_known/max(spectrum_known)), ...
     'LineWidth', 2, 'DisplayName', 'Classic MUSIC');
plot(rad2deg(theta_true), zeros(size(theta_true))-5, 'r^', ...
     'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'r', ...
     'DisplayName', 'True DOAs');
title('Algorithm Comparison', 'FontSize', 14);
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Normalized Spectrum (dB)', 'FontSize', 12);
grid on;
legend('Location', 'best', 'FontSize', 10);
xlim([-90, 90]);
ylim([-50, 5]);

% Plot 2: MUSIC Spectrum Zoom
subplot(2, 3, 2);
zoom_range = rad2deg(theta_true(1))-20 : 0.1 : rad2deg(theta_true(end))+20;
idx_zoom = find(rad2deg(theta_range) >= zoom_range(1) & rad2deg(theta_range) <= zoom_range(end));

plot(rad2deg(theta_range(idx_zoom)), 10*log10(spectrum_known(idx_zoom)/max(spectrum_known)), ...
     'LineWidth', 2, 'DisplayName', 'Classic MUSIC');
hold on;
plot(rad2deg(theta_true), zeros(size(theta_true)), 'r^', ...
     'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'r', ...
     'DisplayName', 'True DOAs');
if ~isempty(DoAs_known)
    plot(rad2deg(theta_range(DoAs_known)), ...
         10*log10(spectrum_known(DoAs_known)/max(spectrum_known)), 'ko', ...
         'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Estimated');
end
title('MUSIC Spectrum (Zoom)', 'FontSize', 14);
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Normalized Spectrum (dB)', 'FontSize', 12);
grid on;
legend('Location', 'best', 'FontSize', 10);
xlim([zoom_range(1), zoom_range(end)]);

% Plot 3: Resolution Test
subplot(2, 3, 3);
plot(rad2deg(theta_range), 10*log10(spectrum_music_close/max(spectrum_music_close)), ...
     'LineWidth', 2, 'DisplayName', 'MUSIC Spectrum');
hold on;
plot(rad2deg(theta_close), zeros(size(theta_close)), 'r^', ...
     'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'r', ...
     'DisplayName', 'True DOAs');
if ~isempty(DoAs_music_close)
    plot(rad2deg(theta_range(DoAs_music_close)), ...
         10*log10(spectrum_music_close(DoAs_music_close)/max(spectrum_music_close)), ...
         'ko', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Estimated');
end
title('Resolution Test (20° Separation)', 'FontSize', 14);
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Normalized Spectrum (dB)', 'FontSize', 12);
grid on;
legend('Location', 'best', 'FontSize', 10);
xlim([-30, 30]);

% Plot 4: Eigenvalue Distribution (linear)
subplot(2, 3, 4);
stem(1:M, eigenvalues, 'filled', 'LineWidth', 1.5);
hold on;
plot([true_sources+0.5, true_sources+0.5], [0, max(eigenvalues)*1.1], 'r--', 'LineWidth', 1.5);
plot([0.5, M+0.5], [eigenvalues(end) + 0.4, eigenvalues(end) + 0.4], 'g:', 'LineWidth', 1.5);
title('Eigenvalue Distribution', 'FontSize', 14);
xlabel('Eigenvalue Index', 'FontSize', 12);
ylabel('Eigenvalue', 'FontSize', 12);
grid on;
legend('Eigenvalues', 'Signal/Noise Boundary', 'Clustering Threshold', ...
       'Location', 'best', 'FontSize', 9);
xlim([0.5, M+0.5]);

% Plot 5: Eigenvalue Distribution (log)
subplot(2, 3, 5);
semilogy(1:M, eigenvalues, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
plot([true_sources+0.5, true_sources+0.5], ...
     [min(eigenvalues(eigenvalues>0))*0.1, max(eigenvalues)*10], 'r--', 'LineWidth', 1.5);
plot([0.5, M+0.5], [eigenvalues(end) + 0.4, eigenvalues(end) + 0.4], 'g:', 'LineWidth', 1.5);
title('Eigenvalues (Log Scale)', 'FontSize', 14);
xlabel('Eigenvalue Index', 'FontSize', 12);
ylabel('Eigenvalue (log)', 'FontSize', 12);
grid on;
xlim([0.5, M+0.5]);

% Plot 6: Performance Summary
subplot(2, 3, 6);
if ~isempty(DoAs_known) && length(DoAs_known) == true_sources
    % Calculate metrics
    errors = abs(sort(theta_true) - sort(theta_range(DoAs_known)));
    rms_error = rad2deg(sqrt(mean(errors.^2)));
    max_error = rad2deg(max(errors));
    
    metrics = {'RMS Error (°)', 'Max Error (°)', 'Comp Time (s)'};
    values = [rms_error, max_error, elapsed];
    
    bar(values);
    set(gca, 'XTickLabel', metrics);
    title('Performance Metrics', 'FontSize', 14);
    ylabel('Value', 'FontSize', 12);
    grid on;
    
    % Add value labels
    for i = 1:length(values)
        text(i, values(i), sprintf('%.2f', values(i)), ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom', 'FontSize', 10);
    end
end

% Add overall annotation
annotation('textbox', [0.02, 0.98, 0.4, 0.02], ...
           'String', sprintf('Classic MUSIC Test: M=%d, Sources=%d, SNR=%.1fdB', ...
           M, true_sources, snr_db), ...
           'FontSize', 11, 'FontWeight', 'bold', 'EdgeColor', 'none');

fprintf('   ✓ All plots generated\n\n');

fprintf('===================================================\n');
fprintf('Classic MUSIC tests completed successfully!\n');
fprintf('===================================================\n');