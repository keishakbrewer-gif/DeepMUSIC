% test_errorMeasures.m
% Test the error measures functions

clear; close all; clc;

fprintf('=========================================\n');
fprintf('Error Measures Test\n');
fprintf('=========================================\n');

% Set random seed for reproducibility
utils.set_random_seed(42);

%% Test 1: Basic error calculations
fprintf('\n1. Basic Error Calculations\n');
fprintf('   ------------------------\n');

% Create test data
num_samples = 50;
num_sources = 2;

% True DOAs (within [-pi/2, pi/2])
trueDoA = (rand(num_samples, num_sources) - 0.5) * pi;

% Add noise to create predictions
noise_level = 0.1;  % radians (~5.7 degrees)
predDoA = trueDoA + noise_level * randn(num_samples, num_sources);

% Wrap predictions to valid range
predDoA = mod(predDoA + pi/2, pi) - pi/2;

fprintf('   Samples: %d, Sources: %d\n', num_samples, num_sources);
fprintf('   True DOA range: [%.2f°, %.2f°]\n', ...
        rad2deg(min(trueDoA(:))), rad2deg(max(trueDoA(:))));
fprintf('   Noise level: %.2f rad (%.1f°)\n', noise_level, rad2deg(noise_level));
fprintf('\n');

%% Test 2: Naive RMSE
fprintf('2. Naive RMSE Test\n');
fprintf('   ---------------\n');

tic;
naive_rmse = errorMeasures.mean_naive_rmse(predDoA, trueDoA);
elapsed = toc;

fprintf('   Naive RMSE: %.4f rad (%.2f°)\n', naive_rmse, rad2deg(naive_rmse));
fprintf('   Computation time: %.4f seconds\n', elapsed);
fprintf('   ✓ Test complete\n\n');

%% Test 3: Minimal Permutation RMSE (for small number of sources)
fprintf('3. Minimal Permutation RMSE Test\n');
fprintf('   -----------------------------\n');

if num_sources <= 4
    tic;
    minperm_rmse = errorMeasures.mean_min_perm_rmse(predDoA, trueDoA);
    elapsed = toc;
    
    fprintf('   MinPerm RMSE: %.4f rad (%.2f°)\n', minperm_rmse, rad2deg(minperm_rmse));
    fprintf('   Computation time: %.4f seconds\n', elapsed);
else
    fprintf('   Skipped (computationally expensive for %d sources)\n', num_sources);
end
fprintf('   ✓ Test complete\n\n');

%% Test 4: Different RMSE modes
fprintf('4. RMSE with Different Matching Strategies\n');
fprintf('   ---------------------------------------\n');

[rmse_sorted, errors_sorted] = errorMeasures.rmse(predDoA, trueDoA, 'sorted');
[rmse_nearest, errors_nearest] = errorMeasures.rmse(predDoA, trueDoA, 'nearest');

fprintf('   RMSE (sorted): %.4f rad (%.2f°)\n', rmse_sorted, rad2deg(rmse_sorted));
fprintf('   RMSE (nearest): %.4f rad (%.2f°)\n', rmse_nearest, rad2deg(rmse_nearest));

% Calculate difference
rmse_diff = abs(rmse_sorted - rmse_nearest);
fprintf('   Difference: %.4f rad (%.2f°)\n', rmse_diff, rad2deg(rmse_diff));
fprintf('   ✓ Test complete\n\n');

%% Test 5: MAE
fprintf('5. Mean Absolute Error Test\n');
fprintf('   -----------------------\n');

mae_sorted = errorMeasures.mae(predDoA, trueDoA, 'sorted');
mae_nearest = errorMeasures.mae(predDoA, trueDoA, 'nearest');

fprintf('   MAE (sorted): %.4f rad (%.2f°)\n', mae_sorted, rad2deg(mae_sorted));
fprintf('   MAE (nearest): %.4f rad (%.2f°)\n', mae_nearest, rad2deg(mae_nearest));
fprintf('   ✓ Test complete\n\n');

%% Test 6: Detection Rate
fprintf('6. Detection Rate Test\n');
fprintf('   -------------------\n');

threshold_deg = 5;  % 5 degree threshold
[detection_rate, false_alarms] = errorMeasures.detection_rate(predDoA, trueDoA, threshold_deg);

fprintf('   Detection threshold: %.1f°\n', threshold_deg);
fprintf('   Detection rate: %.2f%%\n', detection_rate * 100);
fprintf('   False alarms: %d\n', false_alarms);
fprintf('   ✓ Test complete\n\n');

%% Test 7: Angular Wrap Function
fprintf('7. Angular Wrap Function Test\n');
fprintf('   --------------------------\n');

% Test various angles
test_angles = [-pi, -pi/2, 0, pi/2, pi, 3*pi/2, 2*pi];
wrapped = errorMeasures.wrap_error(test_angles);

fprintf('   Test angles (rad): %s\n', mat2str(test_angles, 2));
fprintf('   Wrapped (rad):     %s\n', mat2str(wrapped, 2));
fprintf('   All in [-pi/2, pi/2): %s\n', string(all(wrapped >= -pi/2 & wrapped < pi/2)));
fprintf('   ✓ Test complete\n\n');

%% Test 8: Confidence Intervals
fprintf('8. Confidence Intervals\n');
fprintf('   --------------------\n');

all_errors = errors_sorted(:);
[ci_lower, ci_upper] = errorMeasures.confidence_intervals(all_errors, 0.95);

fprintf('   Number of errors: %d\n', length(all_errors));
fprintf('   Mean error: %.4f rad (%.2f°)\n', mean(all_errors), rad2deg(mean(all_errors)));
fprintf('   95%% Confidence Interval:\n');
fprintf('     [%.4f, %.4f] rad\n', ci_lower, ci_upper);
fprintf('     [%.2f°, %.2f°]\n', rad2deg(ci_lower), rad2deg(ci_upper));
fprintf('   ✓ Test complete\n\n');

%% Test 9: Comprehensive Evaluation
fprintf('9. Comprehensive Evaluation\n');
fprintf('   ------------------------\n');

results = errorMeasures.evaluate_all(predDoA, trueDoA, 5);

%% Test 10: Edge Cases
fprintf('10. Edge Cases Test\n');
fprintf('    ---------------\n');

% Test with single source
true_single = (rand(10, 1) - 0.5) * pi;
pred_single = true_single + 0.1 * randn(10, 1);

rmse_single = errorMeasures.mean_naive_rmse(pred_single, true_single);
fprintf('    Single source RMSE: %.4f rad (%.2f°)\n', rmse_single, rad2deg(rmse_single));

% Test with mismatched number of sources
true_2src = (rand(5, 2) - 0.5) * pi;
pred_3src = (rand(5, 3) - 0.5) * pi;

try
    rmse_mismatch = errorMeasures.mean_naive_rmse(pred_3src, true_2src);
    fprintf('    Mismatched sources RMSE: %.4f rad\n', rmse_mismatch);
catch ME
    fprintf('    Expected error: %s\n', ME.message);
end

fprintf('    ✓ Edge cases tested\n\n');

%% Visualization
fprintf('Generating error visualization...\n');

figure('Position', [100, 100, 1200, 500], 'Name', 'Error Analysis', 'Color', 'w');

% Plot 1: Error histogram
subplot(1, 3, 1);
histogram(rad2deg(errors_sorted(:)), 20, 'FaceColor', [0.2, 0.4, 0.8], 'EdgeColor', 'black');
hold on;
plot(rad2deg([ci_lower, ci_lower]), ylim, 'r--', 'LineWidth', 1.5);
plot(rad2deg([ci_upper, ci_upper]), ylim, 'r--', 'LineWidth', 1.5);
xlabel('Error (degrees)', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
title('Error Distribution', 'FontSize', 14);
grid on;
legend('Errors', '95% CI', 'Location', 'best');

% Plot 2: True vs Predicted
subplot(1, 3, 2);
for i = 1:min(20, num_samples)
    plot(rad2deg(trueDoA(i, :)), rad2deg(predDoA(i, :)), 'o', 'MarkerSize', 8);
    if i == 1
        hold on;
    end
end
plot([-90, 90], [-90, 90], 'r--', 'LineWidth', 1.5);
xlabel('True DOA (degrees)', 'FontSize', 12);
ylabel('Predicted DOA (degrees)', 'FontSize', 12);
title('True vs Predicted', 'FontSize', 14);
grid on;
xlim([-90, 90]);
ylim([-90, 90]);
axis equal;

% Plot 3: Performance metrics
subplot(1, 3, 3);
metrics = {'Naive RMSE', 'RMSE (sorted)', 'RMSE (nearest)', 'MAE', 'Detect Rate'};
values = [rad2deg(naive_rmse), rad2deg(rmse_sorted), rad2deg(rmse_nearest), ...
          rad2deg(mae_sorted), detection_rate*100];

bar(values);
set(gca, 'XTickLabel', metrics, 'XTickLabelRotation', 45);
ylabel('Value', 'FontSize', 12);
title('Performance Metrics', 'FontSize', 14);
grid on;

% Add value labels
for i = 1:length(values)
    if i < 5
        text(i, values(i), sprintf('%.2f°', values(i)), ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom', 'FontSize', 10);
    else
        text(i, values(i), sprintf('%.1f%%', values(i)), ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom', 'FontSize', 10);
    end
end

fprintf('   ✓ Visualization complete\n\n');

fprintf('=========================================\n');
fprintf('All error measure tests completed!\n');
fprintf('=========================================\n');