%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    errorMeasures.m                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
% Authors: J. M.                                                               %
%                                                                              %
% Created: 27/04/21                                                            %
%                                                                              %
% Purpose: Definitions of custom error measures used to evaluate DoA           %
%          estimation algorithms.                                              %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef errorMeasures
    % ERRORM EASURES Custom error metrics for DOA evaluation
    
    methods(Static)
        
        %**********************#
        %   simple mean rmse   #
        %**********************#
        function mean_rmse = mean_naive_rmse(predDoA, trueDoA)
            % Calculates the mean of all the samples of a naive rmse, i.e. an rmse that
            % takes the squared error by padding/truncating and then sorting the DoA.
            %
            % @param predDoA -- The estimated DoA angles in radians [num_samples x num_sources]
            % @param trueDoA -- The ground truth DoA angles in radians [num_samples x num_sources]
            %
            % @returns -- The mean of the naive rmse.
            
            if nargin < 2
                error('Both predicted and true DOAs required');
            end
            
            % Check dimensions
            [num_samples_pred, num_sources_pred] = size(predDoA);
            [num_samples_true, num_sources_true] = size(trueDoA);
            
            if num_samples_pred ~= num_samples_true
                error('Number of samples must match: pred has %d, true has %d', ...
                      num_samples_pred, num_samples_true);
            end
            
            num_samples = num_samples_true;
            
            % If number of sources doesn't match, pad/truncate
            if num_sources_pred ~= num_sources_true
                warning('Number of sources mismatch: pred has %d, true has %d. Padding/truncating.', ...
                        num_sources_pred, num_sources_true);
                min_sources = min(num_sources_pred, num_sources_true);
                predDoA = predDoA(:, 1:min_sources);
                trueDoA = trueDoA(:, 1:min_sources);
            end
            
            % Calculate RMSE for each sample
            allRMSE = zeros(num_samples, 1);
            
            for i = 1:num_samples
                % Sort both predicted and true DOAs
                pred_sorted = sort(predDoA(i, :));
                true_sorted = sort(trueDoA(i, :));
                
                % Angular difference with modulo operation
                % Equivalent to Python: ((diff + pi/2) % pi) - pi/2
                diff = mod((pred_sorted - true_sorted) + pi/2, pi) - pi/2;
                
                % RMSE for this sample
                allRMSE(i) = sqrt(mean(diff .^ 2));
            end
            
            % Mean RMSE across all samples
            mean_rmse = mean(allRMSE);
        end
        
        
        %***********************************#
        %   mean minimal permutation rmse   #
        %***********************************#
        function mean_rmse = mean_min_perm_rmse(predDoA, trueDoA)
            % Calculates the mean of all the samples of the minimal rmse of
            % (all permutations of) the predicted DoA and the true DoA.
            %
            % @param predDoA -- The estimated DoA angles in radians [num_samples x num_sources]
            % @param trueDoA -- The ground truth DoA angles in radians [num_samples x num_sources]
            %
            % @returns -- The mean of minimal rmse.
            
            if nargin < 2
                error('Both predicted and true DOAs required');
            end
            
            % Check dimensions
            [num_samples_pred, num_sources_pred] = size(predDoA);
            [num_samples_true, num_sources_true] = size(trueDoA);
            
            if num_samples_pred ~= num_samples_true
                error('Number of samples must match: pred has %d, true has %d', ...
                      num_samples_pred, num_samples_true);
            end
            
            num_samples = num_samples_true;
            
            % If number of sources doesn't match, pad/truncate
            if num_sources_pred ~= num_sources_true
                warning('Number of sources mismatch: pred has %d, true has %d. Padding/truncating.', ...
                        num_sources_pred, num_sources_true);
                min_sources = min(num_sources_pred, num_sources_true);
                predDoA = predDoA(:, 1:min_sources);
                trueDoA = trueDoA(:, 1:min_sources);
                num_sources = min_sources;
            else
                num_sources = num_sources_pred;
            end
            
            % Pre-calculate number of permutations (for progress display)
            num_permutations = factorial(num_sources);
            fprintf('Calculating minimal permutation RMSE for %d sources (%d permutations per sample)...\n', ...
                    num_sources, num_permutations);
            
            % Calculate minimal RMSE for each sample
            allMinRMSE = zeros(num_samples, 1);
            
            for i = 1:num_samples
                % Get current predicted and true DOAs
                pred_current = predDoA(i, :);
                true_current = trueDoA(i, :);
                
                % Get all permutations of predicted DOAs
                % Note: We need to convert to cell for permutations function
                pred_cell = num2cell(pred_current);
                perms_cell = utils.permutations(pred_cell);
                
                % Initialize array for RMSE of each permutation
                perms_rmse = zeros(length(perms_cell), 1);
                
                % Calculate RMSE for each permutation
                for j = 1:length(perms_cell)
                    % Convert cell array back to numeric
                    perm_doa = cell2mat(perms_cell{j});
                    
                    % Angular difference with modulo operation
                    diff = mod((perm_doa - true_current) + pi/2, pi) - pi/2;
                    
                    % RMSE for this permutation
                    perms_rmse(j) = sqrt(mean(diff .^ 2));
                end
                
                % Choose minimal RMSE as error for this sample
                allMinRMSE(i) = min(perms_rmse);
                
                % Progress display
                if mod(i, max(1, floor(num_samples/10))) == 0
                    fprintf('  Processed %d/%d samples\n', i, num_samples);
                end
            end
            
            % Mean minimal RMSE across all samples
            mean_rmse = mean(allMinRMSE);
            
            fprintf('  Done! Mean minimal permutation RMSE: %.4f radians (%.2f°)\n', ...
                    mean_rmse, rad2deg(mean_rmse));
        end
        
        
        %***********************************#
        %   root mean square error (RMSE)   #
        %***********************************#
        function [rmse, errors] = rmse(predDoA, trueDoA, mode)
            % Calculate RMSE with different matching strategies.
            %
            % @param predDoA -- Estimated DOAs [num_samples x num_sources]
            % @param trueDoA -- True DOAs [num_samples x num_sources]
            % @param mode -- Matching mode: 'sorted', 'nearest', 'minperm' (default: 'sorted')
            %
            % @returns -- RMSE and individual errors
            
            if nargin < 3
                mode = 'sorted';  % Default to sorted matching
            end
            
            [num_samples, num_sources] = size(trueDoA);
            errors = zeros(num_samples, num_sources);
            
            switch lower(mode)
                case 'sorted'
                    % Simple sorted matching (naive approach)
                    for i = 1:num_samples
                        pred_sorted = sort(predDoA(i, :));
                        true_sorted = sort(trueDoA(i, :));
                        diff = errorMeasures.wrap_error(pred_sorted - true_sorted);
                        errors(i, :) = diff;
                    end
                    
                case 'nearest'
                    % Nearest neighbor matching
                    for i = 1:num_samples
                        pred_current = predDoA(i, :);
                        true_current = trueDoA(i, :);
                        
                        % For each true DOA, find closest predicted DOA
                        for j = 1:num_sources
                            % Find nearest predicted DOA
                            diffs = errorMeasures.wrap_error(pred_current - true_current(j));
                            [min_err, min_idx] = min(abs(diffs));
                            errors(i, j) = diffs(min_idx);
                        end
                    end
                    
                case 'minperm'
                    % Minimal permutation (computationally expensive)
                    for i = 1:num_samples
                        pred_current = predDoA(i, :);
                        true_current = trueDoA(i, :);
                        
                        % Get all permutations
                        pred_cell = num2cell(pred_current);
                        perms_cell = utils.permutations(pred_cell);
                        
                        best_perm = [];
                        best_rmse = Inf;
                        
                        % Find permutation with minimal RMSE
                        for j = 1:length(perms_cell)
                            perm_doa = cell2mat(perms_cell{j});
                            diff = errorMeasures.wrap_error(perm_doa - true_current);
                            perm_rmse = sqrt(mean(diff .^ 2));
                            
                            if perm_rmse < best_rmse
                                best_rmse = perm_rmse;
                                best_perm = perm_doa;
                            end
                        end
                        
                        errors(i, :) = errorMeasures.wrap_error(best_perm - true_current);
                    end
                    
                otherwise
                    error('Unknown mode: %s. Use ''sorted'', ''nearest'', or ''minperm''', mode);
            end
            
            % Calculate RMSE
            rmse = sqrt(mean(errors(:) .^ 2));
        end
        
        
        %***********************************#
        %   mean absolute error (MAE)       #
        %***********************************#
        function mae = mae(predDoA, trueDoA, mode)
            % Calculate Mean Absolute Error.
            %
            % @param predDoA -- Estimated DOAs
            % @param trueDoA -- True DOAs
            % @param mode -- Matching mode (default: 'sorted')
            %
            % @returns -- MAE
            
            if nargin < 3
                mode = 'sorted';
            end
            
            [~, errors] = errorMeasures.rmse(predDoA, trueDoA, mode);
            mae = mean(abs(errors(:)));
        end
        
        
        %***********************************#
        %   angular wrap error              #
        %***********************************#
        function wrapped_error = wrap_error(angular_error)
            % Wrap angular error to range [-pi/2, pi/2).
            % Equivalent to Python: ((error + pi/2) % pi) - pi/2
            %
            % @param angular_error -- Angular error in radians
            %
            % @returns -- Wrapped angular error
            
            wrapped_error = mod(angular_error + pi/2, pi) - pi/2;
        end
        
        
        %***********************************#
        %   detection rate                  #
        %***********************************#
        function [detection_rate, false_alarms] = detection_rate(predDoA, trueDoA, threshold_deg)
            % Calculate detection rate and false alarm rate.
            %
            % @param predDoA -- Estimated DOAs
            % @param trueDoA -- True DOAs
            % @param threshold_deg -- Detection threshold in degrees (default: 5°)
            %
            % @returns -- Detection rate and false alarm count
            
            if nargin < 3
                threshold_deg = 5;  % 5 degree threshold
            end
            
            threshold_rad = deg2rad(threshold_deg);
            [num_samples, num_sources] = size(trueDoA);
            
            true_detections = 0;
            total_true = num_samples * num_sources;
            false_alarms = 0;
            
            for i = 1:num_samples
                pred_current = predDoA(i, :);
                true_current = trueDoA(i, :);
                
                % Track which true DOAs have been detected
                detected_true = false(1, num_sources);
                
                % For each predicted DOA, check if it matches any true DOA
                for j = 1:length(pred_current)
                    pred_angle = pred_current(j);
                    
                    % Find closest true DOA
                    diffs = errorMeasures.wrap_error(pred_angle - true_current);
                    [min_err, min_idx] = min(abs(diffs));
                    
                    if min_err <= threshold_rad
                        % Detection!
                        if ~detected_true(min_idx)
                            detected_true(min_idx) = true;
                            true_detections = true_detections + 1;
                        else
                            % Multiple predictions for same true DOA = false alarm
                            false_alarms = false_alarms + 1;
                        end
                    else
                        % No close true DOA = false alarm
                        false_alarms = false_alarms + 1;
                    end
                end
            end
            
            detection_rate = true_detections / total_true;
        end
        
        
        %***********************************#
        %   confidence intervals            #
        %***********************************#
        function [ci_lower, ci_upper] = confidence_intervals(errors, confidence)
            % Calculate confidence intervals for errors.
            %
            % @param errors -- Error values
            % @param confidence -- Confidence level (0-1, default: 0.95)
            %
            % @returns -- Lower and upper confidence bounds
            
            if nargin < 2
                confidence = 0.95;
            end
            
            errors = errors(:);  % Ensure column vector
            
            % Calculate mean and standard error
            mean_error = mean(errors);
            std_error = std(errors);
            n = length(errors);
            
            % Z-score for confidence level (approximate for large n)
            z = norminv(1 - (1 - confidence)/2);
            
            % Confidence interval
            margin = z * std_error / sqrt(n);
            ci_lower = mean_error - margin;
            ci_upper = mean_error + margin;
        end
        
        
        %***********************************#
        %   comprehensive evaluation        #
        %***********************************#
        function results = evaluate_all(predDoA, trueDoA, threshold_deg)
            % Comprehensive evaluation with multiple metrics.
            %
            % @param predDoA -- Estimated DOAs
            % @param trueDoA -- True DOAs
            % @param threshold_deg -- Detection threshold (default: 5°)
            %
            % @returns -- Structure with all evaluation metrics
            
            if nargin < 3
                threshold_deg = 5;
            end
            
            fprintf('\n=========================================\n');
            fprintf('Comprehensive DOA Evaluation\n');
            fprintf('=========================================\n');
            
            % Basic statistics
            [num_samples, num_sources] = size(trueDoA);
            fprintf('Samples: %d, Sources per sample: %d\n', num_samples, num_sources);
            
            % Calculate all metrics
            results = struct();
            
            % Naive RMSE (sorted matching)
            tic;
            results.naive_rmse_rad = errorMeasures.mean_naive_rmse(predDoA, trueDoA);
            results.naive_rmse_deg = rad2deg(results.naive_rmse_rad);
            elapsed = toc;
            fprintf('Naive RMSE: %.4f rad (%.2f°) [%.3f s]\n', ...
                    results.naive_rmse_rad, results.naive_rmse_deg, elapsed);
            
            % Minimal permutation RMSE (if sources <= 4, otherwise too slow)
            if num_sources <= 4
                tic;
                results.minperm_rmse_rad = errorMeasures.mean_min_perm_rmse(predDoA, trueDoA);
                results.minperm_rmse_deg = rad2deg(results.minperm_rmse_rad);
                elapsed = toc;
                fprintf('MinPerm RMSE: %.4f rad (%.2f°) [%.3f s]\n', ...
                        results.minperm_rmse_rad, results.minperm_rmse_deg, elapsed);
            else
                warning('Skipping minperm RMSE for %d sources (computationally expensive)', num_sources);
                results.minperm_rmse_rad = NaN;
                results.minperm_rmse_deg = NaN;
            end
            
            % Standard RMSE with different matching strategies
            [results.rmse_sorted_rad, errors_sorted] = errorMeasures.rmse(predDoA, trueDoA, 'sorted');
            results.rmse_sorted_deg = rad2deg(results.rmse_sorted_rad);
            
            [results.rmse_nearest_rad, errors_nearest] = errorMeasures.rmse(predDoA, trueDoA, 'nearest');
            results.rmse_nearest_deg = rad2deg(results.rmse_nearest_rad);
            
            fprintf('RMSE (sorted): %.4f rad (%.2f°)\n', ...
                    results.rmse_sorted_rad, results.rmse_sorted_deg);
            fprintf('RMSE (nearest): %.4f rad (%.2f°)\n', ...
                    results.rmse_nearest_rad, results.rmse_nearest_deg);
            
            % MAE
            results.mae_sorted_rad = errorMeasures.mae(predDoA, trueDoA, 'sorted');
            results.mae_sorted_deg = rad2deg(results.mae_sorted_rad);
            
            results.mae_nearest_rad = errorMeasures.mae(predDoA, trueDoA, 'nearest');
            results.mae_nearest_deg = rad2deg(results.mae_nearest_rad);
            
            fprintf('MAE (sorted): %.4f rad (%.2f°)\n', ...
                    results.mae_sorted_rad, results.mae_sorted_deg);
            fprintf('MAE (nearest): %.4f rad (%.2f°)\n', ...
                    results.mae_nearest_rad, results.mae_nearest_deg);
            
            % Detection rate
            [results.detection_rate, results.false_alarms] = ...
                errorMeasures.detection_rate(predDoA, trueDoA, threshold_deg);
            
            fprintf('Detection rate (@%.1f°): %.2f%%\n', ...
                    threshold_deg, results.detection_rate * 100);
            fprintf('False alarms: %d\n', results.false_alarms);
            
            % Error statistics
            all_errors = errors_sorted(:);
            results.mean_error_rad = mean(all_errors);
            results.mean_error_deg = rad2deg(results.mean_error_rad);
            results.std_error_rad = std(all_errors);
            results.std_error_deg = rad2deg(results.std_error_rad);
            results.max_error_rad = max(abs(all_errors));
            results.max_error_deg = rad2deg(results.max_error_rad);
            
            % Confidence intervals
            [ci_lower, ci_upper] = errorMeasures.confidence_intervals(all_errors, 0.95);
            results.ci_95_lower_rad = ci_lower;
            results.ci_95_upper_rad = ci_upper;
            results.ci_95_lower_deg = rad2deg(ci_lower);
            results.ci_95_upper_deg = rad2deg(ci_upper);
            
            fprintf('\nError Statistics:\n');
            fprintf('  Mean error: %.4f rad (%.2f°)\n', results.mean_error_rad, results.mean_error_deg);
            fprintf('  Std error: %.4f rad (%.2f°)\n', results.std_error_rad, results.std_error_deg);
            fprintf('  Max error: %.4f rad (%.2f°)\n', results.max_error_rad, results.max_error_deg);
            fprintf('  95%% CI: [%.4f, %.4f] rad ([%.2f°, %.2f°])\n', ...
                    ci_lower, ci_upper, rad2deg(ci_lower), rad2deg(ci_upper));
            
            fprintf('\n=========================================\n');
        end
        
    end
end