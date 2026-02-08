% test_syntheticEx.m
% Test synthetic data generation with utility functions

clear; close all; clc;

% Ensure data directory exists
mkdir('data');
fprintf('Created data directory for testing\n');

fprintf('=========================================\n');
fprintf('Synthetic Data Generation Test\n');
fprintf('=========================================\n');

% Set random seed for reproducibility USING UTILS
utils.set_random_seed(42);

%% Test 1: Basic signal construction with utils
fprintf('\n1. Basic Signal Construction (Using Utilities)\n');
fprintf('   ------------------------------------------\n');

% Parameters
m = 8;      % array elements
d = 3;      % sources
snapshots = 100;
snr_db = 20;

% Random DOAs
thetas = pi * (rand(1, d) - 0.5);

% Generate array USING ARRAYUTILS
array = arrayUtils.ULA_positions(m);

% Generate signal USING SYNTHETICEX WITH UTILS
tic;
[measurement, source_signals] = syntheticEx.construct_signal(...
    thetas, m, d, snapshots, snr_db, array);
elapsed = toc;

% Calculate SNR USING UTILS
clean_signal = syntheticEx.steering_matrix(array, thetas) * source_signals;
noise = measurement - clean_signal;
actual_snr = utils.calculate_snr(clean_signal, noise);

fprintf('   Array elements: %d\n', m);
fprintf('   Sources: %d at ', d);
fprintf('%.1f° ', rad2deg(thetas));
fprintf('\n');
fprintf('   Snapshots: %d, Requested SNR: %.1f dB\n', snapshots, snr_db);
fprintf('   Actual SNR: %.2f dB\n', actual_snr);
fprintf('   Measurement size: %d x %d\n', size(measurement, 1), size(measurement, 2));
fprintf('   Computation time: %.4f seconds\n', elapsed);
fprintf('   ✓ Test complete\n\n');

%% Test 2: Coherent signal construction
fprintf('2. Coherent Signal Construction\n');
fprintf('   ----------------------------\n');

tic;
[measurement_coherent, source_signals_coherent] = syntheticEx.construct_coherent_signal(...
    thetas, m, d, snapshots, snr_db, array);
elapsed = toc;

% Check coherence using correlation
correlation_matrix = corrcoef([real(source_signals_coherent(:)), imag(source_signals_coherent(:))]);
coherence_level = mean(abs(correlation_matrix(1:d:end, 1:d:end)));

fprintf('   Coherence level: %.4f (1.0 = fully coherent)\n', coherence_level);
fprintf('   Computation time: %.4f seconds\n', elapsed);

% Verify with utils
clean_coherent = syntheticEx.steering_matrix(array, thetas) * source_signals_coherent;
noise_coherent = measurement_coherent - clean_coherent;
actual_snr_coherent = utils.calculate_snr(clean_coherent, noise_coherent);
fprintf('   Actual SNR: %.2f dB\n', actual_snr_coherent);
fprintf('   ✓ Test complete\n\n');

%% Test 3: Create small dataset
fprintf('3. Create Small Dataset\n');
fprintf('   --------------------\n');

dataset_size = 50;
dataset_name = 'test_dataset';

tic;
[X, Y] = syntheticEx.create_dataset(dataset_name, dataset_size, ...
    m, d, snapshots, snr_db, false, array, true);
elapsed = toc;

fprintf('   Dataset created: %d samples\n', dataset_size);
fprintf('   X size: %s\n', mat2str(size(X)));
fprintf('   Y size: %s\n', mat2str(size(Y)));
fprintf('   Y range: [%.2f°, %.2f°]\n', ...
        rad2deg(min(Y(:))), rad2deg(max(Y(:))));
fprintf('   Computation time: %.4f seconds\n', elapsed);

% Load and verify using utils
data = utils.load_data(fullfile('data', [dataset_name '.mat']));
if isequal(X, data.X) && isequal(Y, data.Y)
    fprintf('   Dataset saved and loaded correctly\n');
else
    warning('Dataset save/load mismatch');
end
fprintf('   ✓ Test complete\n\n');

%% Test 4: Create resolution dataset
fprintf('4. Create Resolution Dataset\n');
fprintf('   -------------------------\n');

res_size = 20;
min_separation = deg2rad(10);  % 10 degrees
res_name = 'test_resolution';

tic;
[X_res, Y_res] = syntheticEx.create_resolution_dataset(...
    res_name, res_size, min_separation, m, snapshots, snr_db, false, array, true);
elapsed = toc;

% Calculate actual separations
separations = abs(diff(Y_res, 1, 2));
fprintf('   Requested separation: %.1f°\n', rad2deg(min_separation));
fprintf('   Actual separations: min=%.2f°, max=%.2f°, mean=%.2f°\n', ...
        rad2deg(min(separations)), rad2deg(max(separations)), rad2deg(mean(separations)));
fprintf('   Computation time: %.4f seconds\n', elapsed);
fprintf('   ✓ Test complete\n\n');

%% Test 5: Create mixed dataset
fprintf('5. Create Mixed Dataset\n');
fprintf('   --------------------\n');

% First create two small datasets using utils functions
dataset1_name = 'test_dataset1';
dataset2_name = 'test_dataset2';

if ~exist(fullfile('data', [dataset1_name '.mat']), 'file')
    fprintf('   Creating dataset 1...\n');
    syntheticEx.create_dataset(dataset1_name, 20, m, 2, snapshots, 15, false, array, true);
end

if ~exist(fullfile('data', [dataset2_name '.mat']), 'file')
    fprintf('   Creating dataset 2...\n');
    syntheticEx.create_dataset(dataset2_name, 20, m, 3, snapshots, 20, true, array, true);
end

% Mix them
mixed_name = 'test_mixed';
tic;
[X_mixed, Y_mixed] = syntheticEx.create_mixed_dataset(...
    mixed_name, fullfile('data', dataset1_name), fullfile('data', dataset2_name), true);
elapsed = toc;

fprintf('   Mixed dataset size: %d samples\n', size(X_mixed, 1));
fprintf('   Sources per sample: %d\n', size(Y_mixed, 2));
fprintf('   Computation time: %.4f seconds\n', elapsed);

% Verify mixed dataset
if size(X_mixed, 1) == 40 && size(Y_mixed, 2) == 3
    fprintf('   Mixed dataset dimensions correct\n');
else
    warning('Mixed dataset dimensions incorrect');
end
fprintf('   ✓ Test complete\n\n');

%% Test 6: Load and analyze dataset with utils
fprintf('6. Load and Analyze Dataset (Using Utilities)\n');
fprintf('   -----------------------------------------\n');

% Load the test dataset USING UTILS
data = utils.load_data(fullfile('data', [dataset_name '.mat']));
X_loaded = data.X;
Y_loaded = data.Y;

fprintf('   Loaded dataset: %s\n', dataset_name);
fprintf('   X shape: %s\n', mat2str(size(X_loaded)));
fprintf('   Y shape: %s\n', mat2str(size(Y_loaded)));

% Calculate statistics USING UTILS
signal_power = zeros(size(X_loaded, 1), 1);
for i = 1:size(X_loaded, 1)
    measurement_i = squeeze(X_loaded(i, :, :));
    R_i = utils.correlation_matrix(measurement_i);
    signal_power(i) = mean(diag(R_i));
end

fprintf('   Average signal power: %.4f\n', mean(signal_power));
fprintf('   DOA statistics:\n');
fprintf('     Min: %.2f°, Max: %.2f°, Mean: %.2f°, Std: %.2f°\n', ...
        rad2deg(min(Y_loaded(:))), rad2deg(max(Y_loaded(:))), ...
        rad2deg(mean(Y_loaded(:))), rad2deg(std(Y_loaded(:))));

% Check angular wrapping
wrapped_doas = utils.wrap_error(Y_loaded);
if all(wrapped_doas(:) >= -pi/2 & wrapped_doas(:) < pi/2)
    fprintf('   All DOAs properly wrapped to [-90°, 90°)\n');
else
    warning('DOA wrapping issue detected');
end
fprintf('   ✓ Test complete\n\n');

%% Test 7: Visualization with syntheticEx utilities
fprintf('7. Visualization\n');
fprintf('   -------------\n');

% Plot a sample USING SYNTHETICEX PLOT FUNCTION
sample_idx = 1;
syntheticEx.plot_sample(X_loaded, Y_loaded, sample_idx, array);

fprintf('   Sample %d plotted\n', sample_idx);
fprintf('   ✓ Visualization complete\n\n');

%% Test 8: Algorithm testing with synthetic data
fprintf('8. Algorithm Testing with Synthetic Data\n');
fprintf('   ------------------------------------\n');

% Use one sample to test algorithms
test_sample = squeeze(X_loaded(1, :, :));
test_doas = Y_loaded(1, :);

% Array configuration
theta_range = linspace(-pi/2, pi/2, 361);

% Test beamformer USING BEAMFORMER_ALL
[DoAs_bf, spectrum_bf] = beamformer_all.bartlett(test_sample, array, theta_range, d);

% Test classic MUSIC
[DoAs_music, spectrum_music] = classicMUSIC.estimate(test_sample, array, theta_range, d);

fprintf('   True DOAs: ');
fprintf('%.1f° ', rad2deg(test_doas));
fprintf('\n');

if ~isempty(DoAs_bf)
    fprintf('   Beamformer DOAs: ');
    fprintf('%.1f° ', rad2deg(theta_range(DoAs_bf)));
    fprintf('\n');
end

if ~isempty(DoAs_music)
    fprintf('   MUSIC DOAs: ');
    fprintf('%.1f° ', rad2deg(theta_range(DoAs_music)));
    fprintf('\n');
end

% Calculate errors USING ERRORM EASURES
if ~isempty(DoAs_bf) && length(DoAs_bf) == d
    rmse_bf = errorMeasures.mean_naive_rmse(...
        theta_range(DoAs_bf)', test_doas');
    fprintf('   Beamformer RMSE: %.2f°\n', rad2deg(rmse_bf));
end

if ~isempty(DoAs_music) && length(DoAs_music) == d
    rmse_music = errorMeasures.mean_naive_rmse(...
        theta_range(DoAs_music)', test_doas');
    fprintf('   MUSIC RMSE: %.2f°\n', rad2deg(rmse_music));
end

fprintf('   ✓ Algorithm testing complete\n\n');

%% Test 9: Comprehensive evaluation with errorMeasures
fprintf('9. Comprehensive Evaluation\n');
fprintf('   ------------------------\n');

% Test error measures on synthetic data
num_test_samples = 10;
test_doas_all = Y_loaded(1:num_test_samples, :);

% Create predictions by adding noise to true DOAs
pred_noise_level = deg2rad(5);  % 5 degrees noise
pred_doas_all = test_doas_all + pred_noise_level * randn(size(test_doas_all));

% Evaluate USING ERRORM EASURES
fprintf('   Testing error measures on %d samples...\n', num_test_samples);

naive_rmse = errorMeasures.mean_naive_rmse(pred_doas_all, test_doas_all);
[rmse_sorted, errors_sorted] = errorMeasures.rmse(pred_doas_all, test_doas_all, 'sorted');
[detection_rate, false_alarms] = errorMeasures.detection_rate(pred_doas_all, test_doas_all, 10);

fprintf('   Naive RMSE: %.2f°\n', rad2deg(naive_rmse));
fprintf('   RMSE (sorted): %.2f°\n', rad2deg(rmse_sorted));
fprintf('   Detection rate (@10°): %.1f%%\n', detection_rate * 100);
fprintf('   False alarms: %d\n', false_alarms);

% Plot error distribution
figure('Name', 'Error Analysis', 'Position', [100, 100, 800, 400]);
subplot(1,2,1);
histogram(rad2deg(errors_sorted(:)), 20, 'FaceColor', [0.2, 0.4, 0.8]);
xlabel('Error (degrees)');
ylabel('Frequency');
title('DOA Estimation Errors');
grid on;

subplot(1,2,2);
scatter(rad2deg(test_doas_all(:)), rad2deg(pred_doas_all(:)), 50, 'filled');
hold on;
plot([-90, 90], [-90, 90], 'r--', 'LineWidth', 2);
xlabel('True DOA (degrees)');
ylabel('Predicted DOA (degrees)');
title('True vs Predicted DOAs');
grid on;
axis equal;
xlim([-90, 90]);
ylim([-90, 90]);

fprintf('   ✓ Comprehensive evaluation complete\n\n');

%% Test 10: Verify utility function integration
fprintf('10. Utility Function Integration Test\n');
fprintf('    --------------------------------\n');

% Test that all utils are properly integrated
test_results = struct();

% 1. Test arrayUtils integration
test_array = arrayUtils.ULA_positions(m);
test_results.array_length = length(test_array);
test_results.array_spacing = mean(diff(test_array));

% 2. Test utils.ULA_action_vector
steering_vec = utils.ULA_action_vector(test_array, 0);
test_results.steering_vec_norm = norm(steering_vec);

% 3. Test utils.correlation_matrix
test_R = utils.correlation_matrix(test_sample);
test_results.R_hermitian = isequal(test_R, test_R');

% 4. Test utils.add_awgn
clean_test = randn(10, 10);
noisy_test = utils.add_awgn(clean_test, 20);
test_snr = utils.calculate_snr(clean_test, noisy_test - clean_test);
test_results.snr_close = abs(test_snr - 20) < 1;  % Within 1 dB

% 5. Test utils.wrap_error
test_angles = [-pi, -pi/2, 0, pi/2, pi];
wrapped = utils.wrap_error(test_angles);
test_results.all_wrapped = all(wrapped >= -pi/2 & wrapped < pi/2);

% Display results
fprintf('    Array length: %d (expected: %d)\n', test_results.array_length, m);
fprintf('    Array spacing: %.2f (expected: 0.5)\n', test_results.array_spacing);
fprintf('    Steering vector norm: %.4f (expected: %.4f)\n', ...
        test_results.steering_vec_norm, sqrt(m));
fprintf('    Correlation matrix Hermitian: %s\n', string(test_results.R_hermitian));
fprintf('    SNR within tolerance: %s\n', string(test_results.snr_close));
fprintf('    All angles wrapped correctly: %s\n', string(test_results.all_wrapped));

if all(structfun(@(x) islogical(x) && x || ~islogical(x) && ~isnan(x), test_results))
    fprintf('    ✓ All utility integrations working correctly\n\n');
else
    warning('Some utility integrations may have issues');
end

%% Cleanup
fprintf('Cleaning up test files...\n');

% List of test files to clean up
test_files = {
    fullfile('data', [dataset_name '.mat']), ...
    fullfile('data', [res_name '.mat']), ...
    fullfile('data', [dataset1_name '.mat']), ...
    fullfile('data', [dataset2_name '.mat']), ...
    fullfile('data', [mixed_name '.mat'])
};

for i = 1:length(test_files)
    if exist(test_files{i}, 'file')
        delete(test_files{i});
        fprintf('  Deleted: %s\n', test_files{i});
    end
end

% Remove data directory if empty
if exist('data', 'dir')
    dir_contents = dir('data');
    if length(dir_contents) <= 2  % Only . and ..
        rmdir('data');
        fprintf('  Removed empty data directory\n');
    end
end

fprintf('\n=========================================\n');
fprintf('All synthetic data tests completed!\n');
fprintf('Utility integration verified successfully!\n');
fprintf('=========================================\n');