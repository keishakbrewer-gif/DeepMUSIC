% run_all_tests.m
% Run comprehensive tests of the entire DOA system

clear; close all; clc;

fprintf('=========================================\n');
fprintf('DOA Estimation System Integration Test\n');
fprintf('=========================================\n');

% Test 1: Basic utilities
fprintf('\n1. Testing utility functions...\n');
test_utils;  % Run the utility test script

% Test 2: Array utilities
fprintf('\n2. Testing array utilities...\n');
test_array_utils;

% Test 3: Augmented MUSIC
fprintf('\n3. Testing augmented MUSIC algorithm...\n');
main_augMUSIC;

% Test 4: Compare with classical methods
fprintf('\n4. Comparing with classical methods...\n');
test_comparison;

fprintf('\n=========================================\n');
fprintf('All integration tests completed!\n');
fprintf('=========================================\n');


%% Helper functions
function test_array_utils()
    % Test array utility functions
    
    % ULA positions
    M = 8;
    positions = arrayUtils.ULA_positions(M);
    fprintf('   ULA positions (lambda/2 spacing):\n');
    fprintf('   %s\n', mat2str(positions', 2));
    
    % Steering vector
    theta = pi/4;
    a = arrayUtils.steering_vector(positions, theta);
    fprintf('   Steering vector magnitude: %.4f\n', norm(a));
    
    % Beampattern
    weights = ones(M, 1) / M;  % Uniform weights
    theta_range = linspace(-pi/2, pi/2, 100);
    beampattern = arrayUtils.array_beampattern(positions, weights, theta_range);
    
    figure('Name', 'Array Beampattern');
    plot(rad2deg(theta_range), 20*log10(abs(beampattern)), 'LineWidth', 2);
    xlabel('Angle (degrees)');
    ylabel('Gain (dB)');
    title('Uniform Linear Array Beampattern');
    grid on;
    
    fprintf('   ✓ Array utilities tested\n');
end

function test_comparison()
    % Compare different DOA methods
    
    % Generate test scenario
    M = 10;
    sources = 3;
    array = arrayUtils.ULA_positions(M);
    theta_range = linspace(-pi/2, pi/2, 180);
    
    % True DOAs
    theta_true = [-40, 0, 35] * pi/180;
    
    % Generate data
    A = exp(-1j * pi * array * sin(theta_true));
    Nsamples = 1000;
    S = (randn(sources, Nsamples) + 1j * randn(sources, Nsamples)) / sqrt(2);
    X = A * S;
    X = utils.add_awgn(X, 15);  % 15 dB SNR
    
    % Correlation matrix
    R = utils.correlation_matrix(X);
    
    % Classical eigen decomposition
    [U, D, ~, noise_subspace] = utils.eigen_decomposition(R, sources);
    
    % Prepare for MUSIC
    real_part = real(noise_subspace(:))';
    imag_part = imag(noise_subspace(:))';
    En_stack = [real_part, imag_part];
    
    % Run MUSIC
    [DoAs_music, spectrum_music] = augMUSIC(En_stack, array, theta_range, sources);
    
    % Simple beamformer spectrum
    spectrum_bf = zeros(size(theta_range));
    for i = 1:length(theta_range)
        a = utils.ULA_action_vector(array, theta_range(i));
        spectrum_bf(i) = abs(a' * R * a);
    end
    
    % Plot comparison
    figure('Position', [100, 100, 1200, 400]);
    
    % Plot 1: MUSIC
    subplot(1, 3, 1);
    plot(rad2deg(theta_range), 10*log10(spectrum_music/max(spectrum_music)), ...
         'LineWidth', 2);
    hold on;
    plot(rad2deg(theta_true), zeros(size(theta_true)), 'r^', ...
         'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'r');
    if ~isempty(DoAs_music)
        plot(rad2deg(theta_range(DoAs_music)), ...
             10*log10(spectrum_music(DoAs_music)/max(spectrum_music)), ...
             'ko', 'MarkerSize', 10, 'LineWidth', 2);
    end
    title('MUSIC Algorithm', 'FontSize', 14);
    xlabel('Angle (degrees)');
    ylabel('Spectrum (dB)');
    grid on; legend('Spectrum', 'True DOAs', 'Estimated', 'Location', 'best');
    xlim([-90, 90]);
    
    % Plot 2: Beamformer
    subplot(1, 3, 2);
    plot(rad2deg(theta_range), 10*log10(spectrum_bf/max(spectrum_bf)), ...
         'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980]);
    hold on;
    plot(rad2deg(theta_true), zeros(size(theta_true)), 'r^', ...
         'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'r');
    title('Classical Beamformer', 'FontSize', 14);
    xlabel('Angle (degrees)');
    ylabel('Spectrum (dB)');
    grid on; legend('Spectrum', 'True DOAs', 'Location', 'best');
    xlim([-90, 90]);
    
    % Plot 3: Eigenvalues
    subplot(1, 3, 3);
    stem(1:M, 10*log10(diag(D)/max(diag(D))), 'filled', 'LineWidth', 2);
    hold on;
    plot([sources+0.5, sources+0.5], [-50, 5], 'r--', 'LineWidth', 1.5);
    title('Eigenvalue Distribution', 'FontSize', 14);
    xlabel('Eigenvalue Index');
    ylabel('Power (dB)');
    grid on;
    xlim([0.5, M+0.5]);
    
    fprintf('   ✓ Method comparison completed\n');
end