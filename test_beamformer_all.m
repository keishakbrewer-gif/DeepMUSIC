% test_beamformer_all.m
% Comprehensive test of all beamformer algorithms

clear; close all; clc;

fprintf('===================================================\n');
fprintf('Comprehensive Beamformer Algorithm Test Suite\n');
fprintf('===================================================\n');

% Add utilities to path if needed
addpath(pwd);

% Set random seed for reproducibility
utils.set_random_seed(42);

%% Test Configuration
M = 10;                  % Array elements
sources = 3;             % Number of sources
Nsamples = 2000;         % Snapshots
snr_db = 15;             % SNR in dB
diagonal_load = 0.01;    % Diagonal loading factor

% Array geometry
array = arrayUtils.ULA_positions(M, 0.5);
theta_range = linspace(-pi/2, pi/2, 361);  % Higher resolution

% True DOAs (well-separated for clear testing)
theta_true = [-45, 0, 30] * pi/180;

fprintf('Test Configuration:\n');
fprintf('  Array elements: %d\n', M);
fprintf('  Sources: %d\n', sources);
fprintf('  True DOAs: ');
fprintf('%.1f° ', rad2deg(theta_true));
fprintf('\n');
fprintf('  Snapshots: %d\n', Nsamples);
fprintf('  SNR: %.1f dB\n', snr_db);
fprintf('  Angle resolution: %.2f°\n', rad2deg(theta_range(2)-theta_range(1)));
fprintf('\n');

%% Generate Test Data
fprintf('Generating test data...\n');

% Signal amplitudes (different for each source)
amplitudes = [1.0, 0.7, 0.5];

% Steering matrix
A = zeros(M, sources);
for i = 1:sources
    A(:, i) = beamformer_all.steering_matrix(array, theta_true(i));
end

% Source signals (complex Gaussian)
S = (randn(sources, Nsamples) + 1j * randn(sources, Nsamples)) / sqrt(2);
S = diag(amplitudes) * S;  % Apply different amplitudes

% Received signal
X = A * S;

% Add noise
X = utils.add_awgn(X, snr_db);

fprintf('  Signal power: %.4f\n', mean(abs(X(:)).^2));
fprintf('  Data generation complete.\n\n');

%% Test 1: Bartlett (Conventional) Beamformer
fprintf('1. Bartlett (Conventional) Beamformer\n');
fprintf('   ---------------------------------\n');
tic;
[DoAs_bartlett, spectrum_bartlett] = beamformer_all.bartlett(X, array, theta_range, sources);
elapsed = toc;

if ~isempty(DoAs_bartlett)
    est_angles = rad2deg(theta_range(DoAs_bartlett));
    fprintf('   Estimated DOAs: ');
    fprintf('%.1f° ', est_angles);
    fprintf('\n');
    
    % Calculate estimation errors
    errors = beamformer_all.calculate_doa_errors(theta_true, theta_range(DoAs_bartlett));
    fprintf('   Estimation errors: ');
    fprintf('%.2f° ', errors);
    fprintf('\n');
    fprintf('   RMS error: %.2f°\n', sqrt(mean(errors.^2)));
else
    fprintf('   No DOAs estimated\n');
end
fprintf('   Computation time: %.4f seconds\n', elapsed);
fprintf('   ✓ Test complete\n\n');

%% Test 2: Capon (MVDR) Beamformer
fprintf('2. Capon (Minimum Variance) Beamformer\n');
fprintf('   -----------------------------------\n');
tic;
[DoAs_capon, spectrum_capon] = beamformer_all.capon(X, array, theta_range, sources, diagonal_load);
elapsed = toc;

if ~isempty(DoAs_capon)
    est_angles = rad2deg(theta_range(DoAs_capon));
    fprintf('   Estimated DOAs: ');
    fprintf('%.1f° ', est_angles);
    fprintf('\n');
    
    errors = beamformer_all.calculate_doa_errors(theta_true, theta_range(DoAs_capon));
    fprintf('   Estimation errors: ');
    fprintf('%.2f° ', errors);
    fprintf('\n');
    fprintf('   RMS error: %.2f°\n', sqrt(mean(errors.^2)));
else
    fprintf('   No DOAs estimated\n');
end
fprintf('   Computation time: %.4f seconds\n', elapsed);
fprintf('   ✓ Test complete\n\n');

%% Test 3: Steering Matrix Generation
fprintf('3. Steering Matrix Functions\n');
fprintf('   ------------------------\n');

% Test steering matrix generation
test_angles = [0, pi/4, pi/3];
A_test = beamformer_all.steering_matrix(array, test_angles);

fprintf('   Steering matrix size: %d x %d\n', size(A_test, 1), size(A_test, 2));
fprintf('   Column norms: ');
for i = 1:size(A_test, 2)
    fprintf('%.4f ', norm(A_test(:, i)));
end
fprintf('\n');

% Verify orthogonality (should not be orthogonal for arbitrary angles)
orthogonality = norm(A_test' * A_test - size(A_test, 1) * eye(size(A_test, 2)), 'fro');
fprintf('   Orthogonality error: %.2e\n', orthogonality);
fprintf('   ✓ Test complete\n\n');

%% Test 4: Beamformer Weights Computation
fprintf('4. Beamformer Weights Computation\n');
fprintf('   -----------------------------\n');

if ~isempty(DoAs_bartlett)
    doas_est = theta_range(DoAs_bartlett);
    
    % Bartlett weights
    w_bartlett = beamformer_all.compute_weights(array, doas_est, 'bartlett');
    fprintf('   Bartlett weights size: %d x %d\n', size(w_bartlett, 1), size(w_bartlett, 2));
    
    % Capon weights
    w_capon = beamformer_all.compute_weights(array, doas_est, 'capon', X, diagonal_load);
    fprintf('   Capon weights size: %d x %d\n', size(w_capon, 1), size(w_capon, 2));
    
    % LCMV weights (using estimated DOAs as constraints)
    w_lcmv = beamformer_all.compute_weights(array, doas_est, 'lcmv', X, diagonal_load);
    fprintf('   LCMV weights size: %d x %d\n', size(w_lcmv, 1), size(w_lcmv, 2));
    
    % Verify weight properties
    fprintf('   Weight magnitudes:\n');
    fprintf('     Bartlett: min=%.4f, max=%.4f\n', min(abs(w_bartlett(:))), max(abs(w_bartlett(:))));
    fprintf('     Capon:    min=%.4f, max=%.4f\n', min(abs(w_capon(:))), max(abs(w_capon(:))));
    fprintf('     LCMV:     min=%.4f, max=%.4f\n', min(abs(w_lcmv(:))), max(abs(w_lcmv(:))));
    
    fprintf('   ✓ Test complete\n\n');
else
    fprintf('   ⚠ Skipping weights test (no DOAs estimated)\n\n');
end

%% Test 5: LCMV Beamformer with Constraints
fprintf('5. LCMV Beamformer with Linear Constraints\n');
fprintf('   --------------------------------------\n');

% Create constraint matrix (look directions = true DOAs)
C = beamformer_all.steering_matrix(array, theta_true);
f = ones(length(theta_true), 1);  % Unity response in look directions

tic;
% Specify number of peaks to find (should match number of constraints)
num_peaks_lcmv = size(C, 2);
[DoAs_lcmv, spectrum_lcmv, weights_lcmv] = beamformer_all.lcmv(...
    X, array, theta_range, C, f, num_peaks_lcmv);
elapsed = toc;

if ~isempty(DoAs_lcmv)
    est_angles = rad2deg(theta_range(DoAs_lcmv));
    fprintf('   Estimated DOAs: ');
    fprintf('%.1f° ', est_angles);
    fprintf('\n');
    
    % Verify constraints
    response = C' * weights_lcmv;
    constraint_error = norm(response - f);
    fprintf('   Constraint error: %.2e\n', constraint_error);
else
    fprintf('   No DOAs estimated\n');
end
fprintf('   Computation time: %.4f seconds\n', elapsed);
fprintf('   ✓ Test complete\n\n');

%% Test 6: Beampattern Analysis
fprintf('6. Beampattern Analysis\n');
fprintf('   --------------------\n');

if exist('w_capon', 'var')
    % Calculate beampattern for Capon weights (first source)
    [pattern, directivity] = beamformer_all.beampattern(array, w_capon(:, 1), theta_range);
    
    fprintf('   Directivity index: %.2f dB\n', directivity);
    
    % Find mainlobe width at -3dB
    [mainlobe_width, sidelobe_level] = beamformer_all.analyze_beampattern(pattern, theta_range);
    fprintf('   Mainlobe width (-3dB): %.2f°\n', rad2deg(mainlobe_width));
    fprintf('   Peak sidelobe level: %.2f dB\n', sidelobe_level);
    
    fprintf('   ✓ Test complete\n\n');
else
    fprintf('   ⚠ Skipping beampattern test (weights not available)\n\n');
end

%% Test 7: Resolution Test (Two closely spaced sources)
fprintf('7. Resolution Test (Closely Spaced Sources)\n');
fprintf('   ---------------------------------------\n');

% Create two closely spaced sources
theta_close = [-5, 5] * pi/180;
A_close = beamformer_all.steering_matrix(array, theta_close);
X_close = A_close * (randn(2, Nsamples) + 1j*randn(2, Nsamples))/sqrt(2);
X_close = utils.add_awgn(X_close, 20);

% Test resolution with both beamformers
[DoAs_bartlett_close, ~] = beamformer_all.bartlett(X_close, array, theta_range, 2);
[DoAs_capon_close, ~] = beamformer_all.capon(X_close, array, theta_range, 2, diagonal_load);

fprintf('   True DOAs: %.1f°, %.1f° (Separation: %.1f°)\n', ...
        rad2deg(theta_close(1)), rad2deg(theta_close(2)), ...
        rad2deg(abs(diff(theta_close))));
    
if length(DoAs_bartlett_close) >= 2
    sep_bartlett = abs(diff(theta_range(DoAs_bartlett_close(1:2))));
    fprintf('   Bartlett estimated separation: %.1f°\n', rad2deg(sep_bartlett));
else
    fprintf('   Bartlett failed to resolve both sources\n');
end

if length(DoAs_capon_close) >= 2
    sep_capon = abs(diff(theta_range(DoAs_capon_close(1:2))));
    fprintf('   Capon estimated separation: %.1f°\n', rad2deg(sep_capon));
else
    fprintf('   Capon failed to resolve both sources\n');
end

fprintf('   ✓ Test complete\n\n');

%% Visualization
fprintf('Generating comprehensive plots...\n');

figure('Position', [50, 50, 1400, 900], 'Name', 'Beamformer Comprehensive Analysis', ...
       'Color', 'w');

% Plot 1: All Spectra Comparison
subplot(2, 3, 1);
plot(rad2deg(theta_range), 10*log10(spectrum_bartlett/max(spectrum_bartlett)), ...
     'LineWidth', 2, 'DisplayName', 'Bartlett');
hold on;
plot(rad2deg(theta_range), 10*log10(spectrum_capon/max(spectrum_capon)), ...
     'LineWidth', 2, 'DisplayName', 'Capon');
plot(rad2deg(theta_range), 10*log10(spectrum_lcmv/max(spectrum_lcmv)), ...
     'LineWidth', 2, 'DisplayName', 'LCMV');
plot(rad2deg(theta_true), zeros(size(theta_true))-5, 'r^', ...
     'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'r', ...
     'DisplayName', 'True DOAs');
title('Beamformer Spectra Comparison', 'FontSize', 14);
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Normalized Spectrum (dB)', 'FontSize', 12);
grid on;
legend('Location', 'best', 'FontSize', 10);
xlim([-90, 90]);
ylim([-50, 5]);

% Plot 2: Zoom on True DOAs
subplot(2, 3, 2);
zoom_range = rad2deg(theta_true(1))-15 : 0.1 : rad2deg(theta_true(end))+15;
idx_zoom = find(rad2deg(theta_range) >= zoom_range(1) & rad2deg(theta_range) <= zoom_range(end));

plot(rad2deg(theta_range(idx_zoom)), 10*log10(spectrum_bartlett(idx_zoom)/max(spectrum_bartlett)), ...
     'LineWidth', 2, 'DisplayName', 'Bartlett');
hold on;
plot(rad2deg(theta_range(idx_zoom)), 10*log10(spectrum_capon(idx_zoom)/max(spectrum_capon)), ...
     'LineWidth', 2, 'DisplayName', 'Capon');
plot(rad2deg(theta_true), zeros(size(theta_true)), 'r^', ...
     'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'r', ...
     'DisplayName', 'True DOAs');
title('Zoom: DOA Region', 'FontSize', 14);
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Normalized Spectrum (dB)', 'FontSize', 12);
grid on;
legend('Location', 'best', 'FontSize', 10);
xlim([zoom_range(1), zoom_range(end)]);

% Plot 3: Beampatterns
if exist('w_capon', 'var')
    subplot(2, 3, 3);
    
    % Calculate and plot beampatterns for each beamformer
    pattern_bartlett = zeros(size(theta_range));
    pattern_capon = zeros(size(theta_range));
    
    for i = 1:length(theta_range)
        a = utils.ULA_action_vector(array, theta_range(i));
        pattern_bartlett(i) = abs(w_bartlett(:, 1)' * a)^2;
        pattern_capon(i) = abs(w_capon(:, 1)' * a)^2;
    end
    
    plot(rad2deg(theta_range), 10*log10(pattern_bartlett/max(pattern_bartlett)), ...
         'LineWidth', 2, 'DisplayName', 'Bartlett');
    hold on;
    plot(rad2deg(theta_range), 10*log10(pattern_capon/max(pattern_capon)), ...
         'LineWidth', 2, 'DisplayName', 'Capon');
    title('Beamformer Patterns (First Source)', 'FontSize', 14);
    xlabel('Angle (degrees)', 'FontSize', 12);
    ylabel('Normalized Response (dB)', 'FontSize', 12);
    grid on;
    legend('Location', 'best', 'FontSize', 10);
    xlim([-90, 90]);
    ylim([-50, 5]);
end

% Plot 4: Eigenvalue Distribution
subplot(2, 3, 4);
R = utils.correlation_matrix(X);
[~, D] = utils.eigen_decomposition(R, sources);
stem(1:M, 10*log10(diag(D)/max(diag(D))), 'filled', 'LineWidth', 2);
hold on;
plot([sources+0.5, sources+0.5], [-60, 5], 'r--', 'LineWidth', 1.5);
title('Eigenvalue Distribution', 'FontSize', 14);
xlabel('Eigenvalue Index', 'FontSize', 12);
ylabel('Normalized Power (dB)', 'FontSize', 12);
grid on;
xlim([0.5, M+0.5]);
ylim([-60, 5]);

% Plot 5: Resolution Test
subplot(2, 3, 5);
[~, spectrum_close_bartlett] = beamformer_all.bartlett(X_close, array, theta_range, 2);
[~, spectrum_close_capon] = beamformer_all.capon(X_close, array, theta_range, 2, diagonal_load);

plot(rad2deg(theta_range), 10*log10(spectrum_close_bartlett/max(spectrum_close_bartlett)), ...
     'LineWidth', 2, 'DisplayName', 'Bartlett');
hold on;
plot(rad2deg(theta_range), 10*log10(spectrum_close_capon/max(spectrum_close_capon)), ...
     'LineWidth', 2, 'DisplayName', 'Capon');
plot(rad2deg(theta_close), zeros(size(theta_close)), 'r^', ...
     'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'r', ...
     'DisplayName', 'True DOAs');
title('Resolution Test (5° Separation)', 'FontSize', 14);
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Normalized Spectrum (dB)', 'FontSize', 12);
grid on;
legend('Location', 'best', 'FontSize', 10);
xlim([-20, 20]);

% Plot 6: Performance Metrics
subplot(2, 3, 6);
metrics = {'RMS Error (°)', 'Comp Time (s)', 'Resolvability'};
if ~isempty(DoAs_bartlett) && ~isempty(DoAs_capon)
    bartlett_metrics = [sqrt(mean(beamformer_all.calculate_doa_errors(theta_true, theta_range(DoAs_bartlett)).^2)), ...
                        NaN, 1];  % Placeholder
    capon_metrics = [sqrt(mean(beamformer_all.calculate_doa_errors(theta_true, theta_range(DoAs_capon)).^2)), ...
                     NaN, 2];  % Placeholder
    
    bar_data = [bartlett_metrics(1:2); capon_metrics(1:2)];
    bar(bar_data);
    set(gca, 'XTickLabel', {'Bartlett', 'Capon'});
    title('Performance Comparison', 'FontSize', 14);
    ylabel('Metric Value', 'FontSize', 12);
    legend({'RMS Error', 'Comp Time'}, 'Location', 'best');
    grid on;
end

% Add overall annotation
annotation('textbox', [0.02, 0.98, 0.4, 0.02], ...
           'String', sprintf('Beamformer Test: M=%d, Sources=%d, SNR=%.1fdB', M, sources, snr_db), ...
           'FontSize', 11, 'FontWeight', 'bold', 'EdgeColor', 'none');

fprintf('   ✓ All plots generated\n\n');

fprintf('===================================================\n');
fprintf('All beamformer tests completed successfully!\n');
fprintf('===================================================\n');

%% Helper Functions (Add to beamformer_all.m or keep here)
function errors = calculate_doa_errors(true_doas, est_doas)
    % Calculate DOA estimation errors (simple matching)
    true_doas = sort(true_doas(:));
    est_doas = sort(est_doas(:));
    
    if length(true_doas) == length(est_doas)
        errors = abs(true_doas - est_doas);
    else
        errors = Inf(size(true_doas));
    end
end

function [width_rad, sidelobe_db] = analyze_beampattern(pattern, theta_range)
    % Analyze beampattern characteristics
    pattern_norm = pattern / max(pattern);
    
    % Find mainlobe width at -3dB
    peak_idx = find(pattern_norm == 1, 1);
    threshold = 1/sqrt(2);  % -3dB
    
    % Find left -3dB point
    left_idx = peak_idx;
    while left_idx > 1 && pattern_norm(left_idx) > threshold
        left_idx = left_idx - 1;
    end
    
    % Find right -3dB point
    right_idx = peak_idx;
    while right_idx < length(pattern_norm) && pattern_norm(right_idx) > threshold
        right_idx = right_idx + 1;
    end
    
    width_rad = theta_range(right_idx) - theta_range(left_idx);
    
    % Find peak sidelobe level
    % Exclude mainlobe region (± width_rad around peak)
    exclude_region = floor(width_rad / (theta_range(2)-theta_range(1)));
    sidlobe_region = [1:peak_idx-exclude_region, peak_idx+exclude_region:length(pattern_norm)];
    
    if ~isempty(sidlobe_region)
        sidelobe_db = 10*log10(max(pattern_norm(sidlobe_region)));
    else
        sidelobe_db = -Inf;
    end
end