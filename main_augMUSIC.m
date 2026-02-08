% main_augMUSIC.m
% Main script to test augmented MUSIC algorithm

clear; close all; clc;

% Add utilities to path if needed
addpath(pwd);

% Set random seed for reproducibility
utils.set_random_seed(42);

fprintf('=========================================\n');
fprintf('Augmented MUSIC Algorithm Test\n');
fprintf('=========================================\n');

% Create synthetic data for testing
M = 8;  % Number of array elements
sources = 2;
array = arrayUtils.ULA_positions(M, 0.5);  % USING ARRAYUTILS
theta_range = linspace(-pi/2, pi/2, 180);

% Generate synthetic covariance matrix with known DOAs
theta_true = [-30, 20] * pi/180;  % True DOAs in radians

% Create steering matrix for true DOAs
A = exp(-1j * pi * array * sin(theta_true));

% Generate source signals
Nsamples = 1000;
S = sqrt(0.5) * (randn(sources, Nsamples) + 1j * randn(sources, Nsamples));

% Create received signal
X = A * S;

% Add noise with specific SNR
snr_db = 20;  % Signal-to-noise ratio in dB
X = utils.add_awgn(X, snr_db);  % USING UTILS

% Calculate sample correlation matrix USING UTILS
R = utils.correlation_matrix(X);

% Eigen decomposition USING UTILS
[U, D, signal_subspace, noise_subspace] = utils.eigen_decomposition(R, sources);

fprintf('Array: %d elements, %d sources\n', M, sources);
fprintf('True DOAs: %.1f°, %.1f°\n', rad2deg(theta_true(1)), rad2deg(theta_true(2)));
fprintf('SNR: %.1f dB\n', snr_db);
fprintf('Noise subspace size: %d x %d\n', size(noise_subspace, 1), size(noise_subspace, 2));

% Stack real and imaginary parts as 1 x (2*M*(M-sources))
real_part = real(noise_subspace(:))';
imag_part = imag(noise_subspace(:))';
En_stack = [real_part, imag_part];  % 1 x (2*M*(M-sources))

fprintf('En_stack size: 1 x %d\n', length(En_stack));
fprintf('Expected length: 2*M*(M-sources) = 2*%d*%d = %d\n', ...
        M, M-sources, 2*M*(M-sources));

% Run MUSIC
fprintf('\nRunning MUSIC algorithm...\n');
tic;
[DoAs, spectrum] = augMUSIC(En_stack, array, theta_range, sources);
elapsed_time = toc;

fprintf('Computation time: %.4f seconds\n', elapsed_time);

% Display results
fprintf('\nResults:\n');

if ~isempty(DoAs)
    est_angles = rad2deg(theta_range(DoAs));
    fprintf('Estimated DOA indices: %s\n', mat2str(DoAs));
    fprintf('Estimated angles: ');
    for i = 1:length(est_angles)
        fprintf('%.1f° ', est_angles(i));
    end
    fprintf('\n');
    
    % Calculate errors
    if length(est_angles) == length(theta_true)
        % Simple error calculation (for demonstration)
        error_deg = abs(sort(est_angles) - sort(rad2deg(theta_true)));
        fprintf('Estimation errors: ');
        for i = 1:length(error_deg)
            fprintf('%.2f° ', error_deg(i));
        end
        fprintf('\n');
    end
else
    fprintf('No DOAs estimated\n');
end

% Plot results
if ~isempty(spectrum)
    figure('Position', [100, 100, 1000, 400]);
    
    % Plot 1: Spatial Spectrum
    subplot(1, 2, 1);
    plot(rad2deg(theta_range), 10*log10(spectrum/max(spectrum)), ...
         'LineWidth', 1.5, 'Color', [0, 0.4470, 0.7410]);
    xlabel('Angle (degrees)', 'FontSize', 12);
    ylabel('Normalized Spectrum (dB)', 'FontSize', 12);
    title('MUSIC Spatial Spectrum', 'FontSize', 14);
    grid on; hold on;
    
    if ~isempty(DoAs)
        % Mark estimated DOAs
        plot(rad2deg(theta_range(DoAs)), 10*log10(spectrum(DoAs)/max(spectrum)), ...
             'ro', 'MarkerSize', 12, 'LineWidth', 2, ...
             'MarkerFaceColor', 'r', 'DisplayName', 'Estimated DOAs');
    end
    
    % Mark true DOAs
    plot(rad2deg(theta_true), [-5, -5], 'g^', ...
         'MarkerSize', 12, 'LineWidth', 2, ...
         'MarkerFaceColor', 'g', 'DisplayName', 'True DOAs');
    
    legend('Location', 'best', 'FontSize', 10);
    xlim([-90, 90]);
    ylim([-40, 5]);
    
    % Plot 2: Eigenvalues
    subplot(1, 2, 2);
    stem(1:M, 10*log10(diag(D)/max(diag(D))), 'filled', ...
         'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Eigenvalue Index', 'FontSize', 12);
    ylabel('Normalized Power (dB)', 'FontSize', 12);
    title('Eigenvalue Distribution', 'FontSize', 14);
    grid on; hold on;
    
    % Mark signal/noise subspace boundary
    plot([sources+0.5, sources+0.5], [-50, 5], 'r--', 'LineWidth', 1.5);
    text(sources+0.7, -10, 'Noise Subspace', ...
         'Rotation', 90, 'FontSize', 10, 'Color', 'r');
    text(sources-0.3, -10, 'Signal Subspace', ...
         'Rotation', 90, 'FontSize', 10, 'Color', 'b');
    
    xlim([0.5, M+0.5]);
    ylim([-50, 5]);
    
    % Set figure background
    set(gcf, 'Color', 'w', 'Name', 'Augmented MUSIC Results');
    
    % Add SNR info to figure
    annotation('textbox', [0.02, 0.95, 0.2, 0.05], ...
               'String', sprintf('SNR: %.1f dB', snr_db), ...
               'FontSize', 10, 'EdgeColor', 'none', ...
               'BackgroundColor', [0.9, 0.9, 0.9]);
end

% Additional analysis
fprintf('\nAdditional Analysis:\n');
fprintf('Spectrum dynamic range: %.2f dB\n', ...
        10*log10(max(spectrum)/min(spectrum)));
fprintf('Number of spectrum samples: %d\n', length(spectrum));
fprintf('Angle resolution: %.2f degrees\n', ...
        rad2deg(theta_range(2) - theta_range(1)));

fprintf('\n=========================================\n');
fprintf('Test completed successfully!\n');
fprintf('=========================================\n');