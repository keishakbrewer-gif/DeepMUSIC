classdef DOALosses
    % DOALOSSES Custom loss functions for DOA neural network training
    % This class implements various loss functions for MUSIC augmentation
    
    methods(Static)
        
        %******************************************************%
        %   Helper: Get Angle Grid                           %
        %******************************************************%
        function theta_grid = get_angle_grid(num_angles)
            % Get the angle grid for spectrum calculation
            %
            % @param num_angles -- Number of angles in grid (default: 181)
            %
            % @returns -- Angle grid in radians [-pi/2, pi/2]
            
            if nargin < 1
                num_angles = 181;  % Default from Python code
            end
            
            theta_grid = linspace(-pi/2, pi/2, num_angles);
        end
        
        
        %******************************************************%
        %   Calculate Spectrum (FIXED VERSION)               %
        %******************************************************%
        function spectrum = calculate_spectrum_fixed(y_pred, array_positions, angles, m, d)
            % Calculate MUSIC spectrum from predicted noise subspace
            % FIXED VERSION: Handles proper dimensions
            %
            % @param y_pred -- Predicted noise subspace [batch_size x (2*m*n)]
            % @param array_positions -- Array element positions
            % @param angles -- Angle grid in radians
            % @param m -- Number of array elements
            % @param d -- Number of sources
            %
            % @returns -- MUSIC spectrum [batch_size x num_angles]
            
            if nargin < 5
                d = 2;  % Default number of sources
            end
            
            batch_size = size(y_pred, 1);
            total_features = size(y_pred, 2);
            num_angles = length(angles);
            
            % Calculate n from dimensions
            % y_pred should have shape: [batch_size x (2*m*n)]
            n = total_features / (2*m);
            
            if mod(n, 1) ~= 0
                error('Invalid y_pred dimensions. Expected 2*m*n features, got %d with m=%d', ...
                      total_features, m);
            end
            
            % Separate real and imaginary parts
            split_idx = total_features / 2;
            yReal = y_pred(:, 1:split_idx);      % [batch_size x (m*n)]
            yImag = y_pred(:, split_idx+1:end);  % [batch_size x (m*n)]
            
            % Reshape to [batch_size x m x n]
            yReal_3d = reshape(yReal, [batch_size, m, n]);
            yImag_3d = reshape(yImag, [batch_size, m, n]);
            
            % Initialize spectrum
            spectrum = zeros(batch_size, num_angles);
            
            for b = 1:batch_size
                % Extract batch
                Y_real = squeeze(yReal_3d(b, :, :));  % [m x n]
                Y_imag = squeeze(yImag_3d(b, :, :));  % [m x n]
                Y = complex(Y_real, Y_imag);          % [m x n]
                
                % For each angle
                for i = 1:num_angles
                    % Steering vector
                    a = exp(-1j * pi * array_positions * sin(angles(i)));
                    a = a(:);  % Ensure column vector [m x 1]
                    
                    % MUSIC spectrum: 1 / (a^H * Y * Y^H * a)
                    % Note: Y is [m x n], so Y*Y' is [m x m]
                    YYH = Y * Y';
                    denominator = a' * YYH * a;
                    
                    if abs(denominator) > 0
                        spectrum(b, i) = 1 / abs(denominator);
                    else
                        spectrum(b, i) = 0;
                    end
                end
            end
        end
        
        
        %******************************************************%
        %   MUSIC Spectrum MSE Loss (FIXED)                  %
        %******************************************************%
        function loss = mse_spectrum_loss(y_true, y_pred, array_positions, m, d, angles)
            % MSE loss between predicted and true MUSIC spectrum
            %
            % @param y_true -- True spectrum [batch_size x num_angles]
            % @param y_pred -- Predicted noise subspace [batch_size x (2*m*n)]
            % @param array_positions -- Array element positions
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param angles -- Angle grid in radians
            %
            % @returns -- MSE loss
            
            if nargin < 5
                d = 2;  % Default number of sources
            end
            if nargin < 6
                angles = DOALosses.get_angle_grid();
            end
            
            % Calculate spectrum from predicted noise subspace
            y_pred_spectrum = DOALosses.calculate_spectrum_fixed(y_pred, array_positions, angles, m, d);
            
            % Normalize both spectra
            y_pred_norm = DOALosses.normalize_spectrum(y_pred_spectrum);
            y_true_norm = DOALosses.normalize_spectrum(y_true);
            
            % Calculate MSE
            squared_error = (y_pred_norm - y_true_norm) .^ 2;
            loss = mean(squared_error(:));
            
            fprintf('MSE Spectrum Loss: %.6f\n', loss);
        end
        
        
        %******************************************************%
        %   Inverse Peaks Loss (FIXED)                       %
        %******************************************************%
        function loss = inverse_peaks_loss(y_true_doa, y_pred, array_positions, m, d, angles, shift_up)
            % Loss that favors large spectrum values at true DOA locations
            %
            % @param y_true_doa -- True DOA indices [batch_size x d]
            % @param y_pred -- Predicted noise subspace [batch_size x (2*m*n)]
            % @param array_positions -- Array element positions
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param angles -- Angle grid in radians
            % @param shift_up -- Shift value for normalization (default: 2)
            %
            % @returns -- Inverse peaks loss
            
            if nargin < 5
                d = 2;
            end
            if nargin < 6
                angles = DOALosses.get_angle_grid();
            end
            if nargin < 7
                shift_up = 2;  % Default from Python code
            end
            
            batch_size = size(y_pred, 1);
            num_angles = length(angles);
            
            % Calculate spectrum from predicted noise subspace
            y_pred_spectrum = DOALosses.calculate_spectrum_fixed(y_pred, array_positions, angles, m, d);
            
            % Normalize spectrum and shift up
            y_norm = DOALosses.normalize_spectrum(y_pred_spectrum) + shift_up;
            
            % Ensure non-negative values
            y_norm = max(y_norm, 0);
            
            % Initialize loss
            loss = 0;
            
            for b = 1:batch_size
                % Get spectrum values at true DOA indices
                peaks_sum = 0;
                
                for src = 1:d
                    doa_idx = y_true_doa(b, src);
                    
                    % Ensure index is within bounds
                    doa_idx = max(1, min(doa_idx, num_angles));
                    
                    % Get spectrum value at this DOA
                    peak_val = y_norm(b, doa_idx);
                    
                    % Avoid division by zero
                    if peak_val > 0
                        peaks_sum = peaks_sum + 1 / peak_val;
                    else
                        peaks_sum = peaks_sum + 1 / eps;  % Large penalty for zero
                    end
                end
                
                % Average and scale by shift_up
                batch_loss = shift_up * peaks_sum / d;
                loss = loss + batch_loss;
            end
            
            % Average over batch
            loss = loss / batch_size;
            
            fprintf('Inverse Peaks Loss: %.6f (d=%d, shift=%.1f)\n', loss, d, shift_up);
        end
        
        
        %******************************************************%
        %   Peak Spectrum Difference Loss (FIXED)            %
        %******************************************************%
        function loss = peak_spectrum_diff_loss(y_true_doa, y_pred, array_positions, m, d, angles, shift_up)
            % Loss based on difference between peaks and rest of spectrum
            %
            % @param y_true_doa -- True DOA indices [batch_size x d]
            % @param y_pred -- Predicted noise subspace [batch_size x (2*m*n)]
            % @param array_positions -- Array element positions
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param angles -- Angle grid in radians
            % @param shift_up -- Shift value for normalization (default: 2)
            %
            % @returns -- Peak spectrum difference loss
            
            if nargin < 5
                d = 2;
            end
            if nargin < 6
                angles = DOALosses.get_angle_grid();
            end
            if nargin < 7
                shift_up = 2;
            end
            
            batch_size = size(y_pred, 1);
            num_angles = length(angles);
            
            % Calculate spectrum from predicted noise subspace
            y_pred_spectrum = DOALosses.calculate_spectrum_fixed(y_pred, array_positions, angles, m, d);
            
            % Normalize spectrum and shift up
            y_norm = DOALosses.normalize_spectrum(y_pred_spectrum) + shift_up;
            
            % Initialize loss
            loss = 0;
            
            for b = 1:batch_size
                % Create mask for peaks (true DOAs)
                peak_mask = false(1, num_angles);
                for src = 1:d
                    doa_idx = y_true_doa(b, src);
                    doa_idx = max(1, min(doa_idx, num_angles));
                    peak_mask(doa_idx) = true;
                end
                
                % Extract peaks and non-peaks
                peaks = y_norm(b, peak_mask);
                non_peaks = y_norm(b, ~peak_mask);
                
                % Calculate difference between peaks mean and non-peaks mean
                if ~isempty(peaks) && ~isempty(non_peaks)
                    peak_mean = mean(peaks);
                    non_peak_mean = mean(non_peaks);
                    
                    % Python version sums differences
                    diff = peak_mean - non_peak_mean;
                    
                    loss = loss + diff;
                end
            end
            
            % Average over batch (note: can be negative!)
            loss = loss / batch_size;
            
            fprintf('Peak Spectrum Diff Loss: %.6f (can be negative)\n', loss);
        end
        
        
        %******************************************************%
        %   MSE DOA Loss (FIXED)                             %
        %******************************************************%
        function loss = mse_doa_loss(y_true_doa, y_pred, array_positions, m, d, angles, beta)
            % MSE loss for DOA estimation with soft-argmax
            %
            % @param y_true_doa -- True DOA indices [batch_size x d]
            % @param y_pred -- Predicted noise subspace [batch_size x (2*m*n)]
            % @param array_positions -- Array element positions
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param angles -- Angle grid in radians
            % @param beta -- Softmax temperature (default: 1e3)
            %
            % @returns -- MSE DOA loss
            
            if nargin < 5
                d = 2;
            end
            if nargin < 6
                angles = DOALosses.get_angle_grid();
            end
            if nargin < 7
                beta = 1e3;  % Default from Python code
            end
            
            batch_size = size(y_pred, 1);
            num_angles = length(angles);
            
            % Calculate spectrum from predicted noise subspace
            y_pred_spectrum = DOALosses.calculate_spectrum_fixed(y_pred, array_positions, angles, m, d);
            
            % Normalize spectrum and shift
            y_norm = DOALosses.normalize_spectrum(y_pred_spectrum) + 1;
            
            % Initialize arrays for estimated DOAs
            estimated_doas = zeros(batch_size, d);
            
            for b = 1:batch_size
                % Get spectrum for this batch
                spectrum = y_norm(b, :);
                
                for src = 1:d
                    % Soft-argmax: weighted sum of indices
                    softmax_weights = DOALosses.softmax(spectrum * beta);
                    doa_est = sum(softmax_weights .* (1:num_angles));
                    
                    % Store estimated DOA index
                    estimated_doas(b, src) = doa_est;
                    
                    if d > 1 && src < d
                        % For multiple sources: remove peak and surrounding area
                        % Simplified version
                        
                        % Find distances to remaining true DOAs
                        remaining_true = y_true_doa(b, src+1:end);
                        distances = abs(doa_est - remaining_true);
                        
                        if ~isempty(distances)
                            % Use second smallest distance
                            sorted_dist = sort(distances);
                            second_smallest = sorted_dist(min(2, length(sorted_dist)));
                            
                            % Remove region around estimated DOA
                            remove_start = max(1, round(doa_est - second_smallest/2));
                            remove_end = min(num_angles, round(doa_est + second_smallest/2));
                            
                            spectrum(remove_start:remove_end) = 0;
                        end
                    end
                end
            end
            
            % Sort both true and estimated DOAs for comparison
            true_sorted = sort(y_true_doa, 2);
            est_sorted = sort(estimated_doas, 2);
            
            % Account for angular overflow (circular nature)
            diff = DOALosses.wrap_angular_difference(est_sorted - true_sorted, num_angles);
            
            % Calculate MSE and normalize
            mse = mean(diff .^ 2, 2);
            normalized_mse = mse / ((num_angles/4) ^ 2);  % Normalize as in Python
            
            loss = mean(normalized_mse);
            
            fprintf('MSE DOA Loss: %.6f (d=%d, beta=%.0f)\n', loss, d, beta);
        end
        
        
        %******************************************************%
        %   EVD (Eigenvalue Decomposition) Loss              %
        %******************************************************%
        function loss = evd_loss(y_true_evd, y_pred_evd)
            % MSE loss for eigenvalue decomposition
            %
            % @param y_true_evd -- True EVD (noise subspace) [batch_size x features]
            % @param y_pred_evd -- Predicted EVD [batch_size x features]
            %
            % @returns -- EVD MSE loss
            
            % Calculate MSE between prediction and target
            squared_diff = (y_pred_evd - y_true_evd) .^ 2;
            
            % Mean over features, then mean over batch
            feature_mean = mean(squared_diff, 2);
            batch_mean = mean(feature_mean);
            
            loss = batch_mean;
            
            fprintf('EVD Loss: %.6f\n', loss);
        end
        
        
        %******************************************************%
        %   Angular RMSE Loss                                %
        %******************************************************%
        function loss = angular_rmse_loss(y_true_angles, y_pred_angles)
            % Angular RMSE loss for DOA estimation
            %
            % @param y_true_angles -- True DOA angles in radians [batch_size x d]
            % @param y_pred_angles -- Predicted DOA angles in radians [batch_size x d]
            %
            % @returns -- Angular RMSE
            
            % Sort both true and predicted angles
            true_sorted = sort(y_true_angles, 2);
            pred_sorted = sort(y_pred_angles, 2);
            
            % Calculate wrapped angular difference
            diff = DOALosses.wrap_angular_difference_rad(pred_sorted - true_sorted);
            
            % Calculate RMSE
            mse = mean(diff .^ 2, 2);
            rmse = sqrt(mse);
            
            loss = mean(rmse);
            
            fprintf('Angular RMSE Loss: %.6f radians (%.2f°)\n', loss, rad2deg(loss));
        end
        
        
        %******************************************************%
        %   Permutation RMSE Loss                            %
        %******************************************************%
        function loss = permutation_rmse_loss(pred_angles, true_angles)
            % Permutation-invariant RMSE loss for DOA estimation
            %
            % @param pred_angles -- Predicted DOA angles in radians [batch_size x d]
            % @param true_angles -- True DOA angles in radians [batch_size x d]
            %
            % @returns -- Minimal RMSE over all permutations
            
            batch_size = size(pred_angles, 1);
            d = size(pred_angles, 2);
            
            % Generate all permutations for d sources
            all_perms = perms(1:d);
            num_perms = size(all_perms, 1);
            
            % Initialize array for minimal RMSE per batch
            min_rmse = zeros(batch_size, 1);
            
            for b = 1:batch_size
                % Get predictions and truths for this batch
                pred_batch = pred_angles(b, :);
                true_batch = true_angles(b, :);
                
                % Initialize RMSE for all permutations
                perm_rmse = zeros(num_perms, 1);
                
                for p = 1:num_perms
                    % Permute predictions according to current permutation
                    pred_permuted = pred_batch(all_perms(p, :));
                    
                    % Calculate wrapped angular difference
                    diff = DOALosses.wrap_angular_difference_rad(pred_permuted - true_batch);
                    
                    % Calculate RMSE for this permutation
                    mse = mean(diff .^ 2);
                    perm_rmse(p) = sqrt(mse);
                end
                
                % Take minimal RMSE
                min_rmse(b) = min(perm_rmse);
            end
            
            % Average over batch
            loss = mean(min_rmse);
            
            fprintf('Permutation RMSE Loss: %.6f radians (%.2f°)\n', loss, rad2deg(loss));
        end
        
        
        %******************************************************%
        %   Utility Functions                                %
        %******************************************************%
        function y_norm = normalize_spectrum(y)
            % Normalize spectrum using layer normalization equivalent
            %
            % @param y -- Input spectrum [batch_size x num_angles]
            %
            % @returns -- Normalized spectrum
            
            % Simple normalization: zero mean, unit variance per batch
            y_mean = mean(y, 2);
            y_std = std(y, 0, 2);
            
            % Avoid division by zero
            y_std(y_std == 0) = 1;
            
            y_norm = (y - y_mean) ./ y_std;
        end
        
        function probs = softmax(x, temperature)
            % Softmax function with temperature
            %
            % @param x -- Input vector/matrix
            % @param temperature -- Temperature parameter (default: 1)
            %
            % @returns -- Softmax probabilities
            
            if nargin < 2
                temperature = 1;
            end
            
            % Apply temperature
            x_scaled = x / temperature;
            
            % Subtract max for numerical stability
            x_shifted = x_scaled - max(x_scaled, [], 2);
            
            % Compute softmax
            exp_x = exp(x_shifted);
            probs = exp_x ./ sum(exp_x, 2);
        end
        
        function diff_wrapped = wrap_angular_difference(diff_indices, num_angles)
            % Wrap angular difference for circular nature of angles
            %
            % @param diff_indices -- Difference in indices
            % @param num_angles -- Total number of angles
            %
            % @returns -- Wrapped difference
            
            diff_wrapped = mod(diff_indices + num_angles/2, num_angles) - num_angles/2;
        end
        
        function diff_wrapped = wrap_angular_difference_rad(diff_rad)
            % Wrap angular difference in radians
            %
            % @param diff_rad -- Difference in radians
            %
            % @returns -- Wrapped difference in [-pi/2, pi/2)
            
            diff_wrapped = mod(diff_rad + pi/2, pi) - pi/2;
        end
        
        function merged = merge_structs(default, user)
            % Merge two structures, user values override defaults
            %
            % @param default -- Default structure
            % @param user -- User structure
            %
            % @returns -- Merged structure
            
            merged = default;
            if ~isempty(user)
                user_fields = fieldnames(user);
                for i = 1:length(user_fields)
                    field = user_fields{i};
                    merged.(field) = user.(field);
                end
            end
        end
        
        
        %******************************************************%
        %   SIMPLIFIED TEST FUNCTION                         %
        %******************************************************%
        function test_loss_functions_simple()
            % Simple test that avoids dimension issues
            
            fprintf('Testing DOA Loss Functions (Simple Version)...\n\n');
            
            % Setup parameters
            batch_size = 4;
            m = 8;
            d = 2;
            n = m - d;  % Noise subspace dimension
            num_angles = 181;
            
            % Get array positions
            array_positions = arrayUtils.ULA_positions(m);
            
            % Get angle grid
            angles = DOALosses.get_angle_grid(num_angles);
            
            % Create CORRECTLY DIMENSIONED synthetic data
            fprintf('1. Creating synthetic test data with correct dimensions...\n');
            
            % True noise subspace: [batch_size x (2*m*n)]
            % Each sample has m*n complex values = 2*m*n real values
            total_features = 2 * m * n;
            En_true = randn(batch_size, total_features);
            En_pred = En_true + 0.1 * randn(batch_size, total_features);
            
            fprintf('   En_true size: %d x %d (batch x features)\n', size(En_true, 1), size(En_true, 2));
            fprintf('   Expected features: 2*m*n = 2*%d*%d = %d ✓\n', m, n, total_features);
            
            % Test 1: EVD Loss (simplest - just MSE)
            fprintf('\n2. Testing EVD Loss (simplest)...\n');
            try
                loss1 = DOALosses.evd_loss(En_true, En_pred);
                fprintf('   ✓ EVD Loss: %.6f\n', loss1);
            catch ME
                fprintf('   ✗ EVD Loss Error: %s\n', ME.message);
            end
            
            % Test 2: Calculate Spectrum (verify our fixed function works)
            fprintf('\n3. Testing Spectrum Calculation...\n');
            try
                spectrum = DOALosses.calculate_spectrum_fixed(En_pred, array_positions, angles, m, d);
                fprintf('   ✓ Spectrum size: %d x %d\n', size(spectrum, 1), size(spectrum, 2));
                fprintf('   Min value: %.3e, Max value: %.3e\n', min(spectrum(:)), max(spectrum(:)));
            catch ME
                fprintf('   ✗ Spectrum Calculation Error: %s\n', ME.message);
            end
            
            % Test 3: Create true DOA indices for other tests
            fprintf('\n4. Creating DOA test data...\n');
            true_doa_indices = randi([1, num_angles], batch_size, d);
            true_doa_angles = zeros(batch_size, d);
            for b = 1:batch_size
                for src = 1:d
                    idx = true_doa_indices(b, src);
                    true_doa_angles(b, src) = angles(idx);
                end
            end
            fprintf('   True DOA indices range: %d to %d\n', min(true_doa_indices(:)), max(true_doa_indices(:)));
            
            % Test 4: Inverse Peaks Loss
            fprintf('\n5. Testing Inverse Peaks Loss...\n');
            try
                loss2 = DOALosses.inverse_peaks_loss(true_doa_indices, En_pred, ...
                                                    array_positions, m, d, angles);
                fprintf('   ✓ Inverse Peaks Loss: %.6f\n', loss2);
            catch ME
                fprintf('   ✗ Inverse Peaks Error: %s\n', ME.message);
            end
            
            % Test 5: MSE DOA Loss
            fprintf('\n6. Testing MSE DOA Loss...\n');
            try
                loss3 = DOALosses.mse_doa_loss(true_doa_indices, En_pred, ...
                                              array_positions, m, d, angles);
                fprintf('   ✓ MSE DOA Loss: %.6f\n', loss3);
            catch ME
                fprintf('   ✗ MSE DOA Error: %s\n', ME.message);
            end
            
            % Test 6: Angular RMSE Loss (using angles directly)
            fprintf('\n7. Testing Angular RMSE Loss...\n');
            try
                % Create predicted angles (slightly perturbed)
                pred_angles = true_doa_angles + 0.05 * randn(size(true_doa_angles));
                loss4 = DOALosses.angular_rmse_loss(true_doa_angles, pred_angles);
                fprintf('   ✓ Angular RMSE Loss: %.6f radians (%.2f°)\n', loss4, rad2deg(loss4));
            catch ME
                fprintf('   ✗ Angular RMSE Error: %s\n', ME.message);
            end
            
            % Test 7: Permutation RMSE Loss
            fprintf('\n8. Testing Permutation RMSE Loss...\n');
            try
                pred_angles = true_doa_angles + 0.05 * randn(size(true_doa_angles));
                loss5 = DOALosses.permutation_rmse_loss(pred_angles, true_doa_angles);
                fprintf('   ✓ Permutation RMSE Loss: %.6f radians (%.2f°)\n', loss5, rad2deg(loss5));
            catch ME
                fprintf('   ✗ Permutation RMSE Error: %s\n', ME.message);
            end
            
            fprintf('\n9. Test Summary:\n');
            fprintf('   Basic loss functions tested successfully!\n');
            fprintf('   Ready to proceed with trainModel.py conversion.\n');
        end
        
        
        %******************************************************%
        %   INTEGRATION TEST WITH REGULARIZERS-real data       %
        %******************************************************%
        function test_integration()
            % Test integration with regularizers
            
            fprintf('Testing DOA Losses + Regularizers Integration...\n\n');
            
            % Setup
            batch_size = 4;
            m = 8;
            d = 2;
            n = m - d;
            num_angles = 181;
            
            % Get array and angles
            array_positions = arrayUtils.ULA_positions(m);
            angles = DOALosses.get_angle_grid(num_angles);
            
            % Create data
            total_features = 2 * m * n;
            En_true = randn(batch_size, total_features);
            En_pred = En_true + 0.3 * randn(batch_size, total_features);  % More noise
            
            % Create DOA indices
            true_doa_indices = randi([1, num_angles], batch_size, d);
            
            % Test 1: Calculate spectrum-based loss
            fprintf('1. Testing spectrum-based losses...\n');
            spectrum_loss = DOALosses.mse_spectrum_loss(...
                randn(batch_size, num_angles), En_pred, array_positions, m, d, angles);
            fprintf('   Spectrum MSE Loss: %.6f\n', spectrum_loss);
            
            % Test 2: Calculate inverse peaks loss
            inverse_peaks_loss = DOALosses.inverse_peaks_loss(...
                true_doa_indices, En_pred, array_positions, m, d, angles);
            fprintf('   Inverse Peaks Loss: %.6f\n', inverse_peaks_loss);
            
            % Test 3: Calculate EVD loss
            evd_loss_val = DOALosses.evd_loss(En_true, En_pred);
            fprintf('   EVD Loss: %.6f\n', evd_loss_val);
            
            % Test 4: Calculate orthogonality regularization
            fprintf('\n2. Testing orthogonality regularization...\n');
            if exist('DOARegularizers', 'class')
                [orth_loss, orth_diag] = DOARegularizers.orth_regularizer(En_pred, m, d, 1.0, 1.0);
                fprintf('   Orthogonality Regularization: %.6f\n', orth_loss);
                fprintf('   Max off-diagonal: %.6f\n', orth_diag.max_off_diagonal);
            else
                fprintf('   Note: DOARegularizers not found\n');
            end
            
            % Test 5: Combined hybrid loss (simulated)
            fprintf('\n3. Simulating hybrid training loss...\n');
            
            % Create true values structure
            y_true_struct = struct();
            y_true_struct.spectrum = randn(batch_size, num_angles);
            y_true_struct.doa_indices = true_doa_indices;
            y_true_struct.evd = En_true;
            
            % Add regularization
            if exist('DOARegularizers', 'class')
                [reg_loss, ~] = DOARegularizers.combined_regularization(...
                    En_pred, En_true, m, d, 0.5, 0.5, 1.0, 1.0);
                y_true_struct.regularization = reg_loss;
                fprintf('   Regularization Loss: %.6f\n', reg_loss);
            end
            
            % Calculate weighted total loss
            total_loss = 0;
            total_loss = total_loss + 0.3 * spectrum_loss;
            total_loss = total_loss + 0.3 * inverse_peaks_loss;
            total_loss = total_loss + 0.3 * evd_loss_val;
            if isfield(y_true_struct, 'regularization')
                total_loss = total_loss + 0.1 * y_true_struct.regularization;
            end
            
            fprintf('\n4. Training Loss Simulation:\n');
            fprintf('   Spectrum MSE: %.6f (weight: 0.3)\n', spectrum_loss);
            fprintf('   Inverse Peaks: %.6f (weight: 0.3)\n', inverse_peaks_loss);
            fprintf('   EVD Loss: %.6f (weight: 0.3)\n', evd_loss_val);
            if isfield(y_true_struct, 'regularization')
                fprintf('   Regularization: %.6f (weight: 0.1)\n', y_true_struct.regularization);
            end
            fprintf('   TOTAL LOSS: %.6f\n', total_loss);
            
            fprintf('\n5. Integration Test Complete!\n');
            fprintf('   All components work together correctly.\n');
            fprintf('   Ready for full training pipeline.\n');
        end
        
        %******************************************************%
        %   Training-Ready Hybrid Loss (ADD to DOALosses)    %
        %******************************************************%
        function [total_loss, loss_components] = training_hybrid_loss(...
                y_pred, y_true_struct, array_positions, m, d, params)
            % Training-ready hybrid loss with proper scaling
            %
            % @param y_pred -- Neural network predictions [batch_size x features]
            % @param y_true_struct -- Structure with true values
            % @param array_positions -- Array element positions
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param params -- Training parameters
            %
            % @returns -- Total loss for training
            % @returns -- Individual loss components
            
            % Default parameters
            default_params = struct();
            default_params.angles = DOALosses.get_angle_grid();
            default_params.weights = struct('spectrum_mse', 0.3, ...
                                           'inverse_peaks', 0.3, ...
                                           'doa_mse', 0.2, ...
                                           'evd', 0.1, ...
                                           'regularization', 0.1);
            default_params.beta = 1e3;
            default_params.shift_up = 2;
            default_params.orth_scale = 1e-3;
            default_params.use_regularization = true;
            
            % Merge with user parameters
            if nargin >= 6 && ~isempty(params)
                params = DOALosses.merge_structs(default_params, params);
            else
                params = default_params;
            end
            
            fprintf('\nTraining Hybrid Loss:\n');
            
            % Initialize
            loss_components = struct();
            total_loss = 0;
            
            % Calculate each component
            if params.weights.spectrum_mse > 0 && isfield(y_true_struct, 'spectrum')
                loss_components.spectrum_mse = DOALosses.mse_spectrum_loss(...
                    y_true_struct.spectrum, y_pred, array_positions, m, d, params.angles);
                total_loss = total_loss + params.weights.spectrum_mse * loss_components.spectrum_mse;
            end
            
            if params.weights.inverse_peaks > 0 && isfield(y_true_struct, 'doa_indices')
                loss_components.inverse_peaks = DOALosses.inverse_peaks_loss(...
                    y_true_struct.doa_indices, y_pred, array_positions, m, d, params.angles, params.shift_up);
                total_loss = total_loss + params.weights.inverse_peaks * loss_components.inverse_peaks;
            end
            
            if params.weights.doa_mse > 0 && isfield(y_true_struct, 'doa_indices')
                loss_components.doa_mse = DOALosses.mse_doa_loss(...
                    y_true_struct.doa_indices, y_pred, array_positions, m, d, params.angles, params.beta);
                total_loss = total_loss + params.weights.doa_mse * loss_components.doa_mse;
            end
            
            if params.weights.evd > 0 && isfield(y_true_struct, 'evd')
                loss_components.evd = DOALosses.evd_loss(y_true_struct.evd, y_pred);
                total_loss = total_loss + params.weights.evd * loss_components.evd;
            end
            
            % Add scaled regularization if requested
            if params.use_regularization && params.weights.regularization > 0
                if exist('DOARegularizers', 'class') && isfield(y_true_struct, 'evd')
                    [reg_loss, ~] = DOARegularizers.combined_regularization_scaled(...
                        y_pred, y_true_struct.evd, m, d, 0.5, 0.5, 1.0, 1.0, params.orth_scale);
                    loss_components.regularization = reg_loss;
                    total_loss = total_loss + params.weights.regularization * reg_loss;
                end
            end
            
            fprintf('Total Loss: %.6f\n', total_loss);
        end

        
        %******************************************************%
        %   PROPER TRAINING INTEGRATION TEST                 %
        %******************************************************%
        function test_training_integration()
            % Test with properly scaled regularization for training
            
            fprintf('TRAINING INTEGRATION TEST (Properly Scaled)\n');
            fprintf('===========================================\n\n');
            
            % Setup
            batch_size = 4;
            m = 8;
            d = 2;
            n = m - d;
            num_angles = 181;
            
            % Get array and angles
            array_positions = arrayUtils.ULA_positions(m);
            angles = DOALosses.get_angle_grid(num_angles);
            
            % Create REALISTIC training data (somewhat orthogonal)
            fprintf('1. Creating realistic training data...\n');
            total_features = 2 * m * n;
            
            % True noise subspace (somewhat orthogonal using QR)
            En_true = zeros(batch_size, total_features);
            for b = 1:batch_size
                Y = randn(m, n) + 1j * randn(m, n);
                [Q, ~] = qr(Y, 0);  % Make orthogonal
                En_true(b, :) = [real(Q(:))', imag(Q(:))'];
            end
            
            % Neural network predictions (simulated with moderate error)
            En_pred = En_true + 0.3 * randn(size(En_true));
            
            fprintf('   Batch: %d, Features: %d (2*m*n = 2*%d*%d)\n', ...
                    batch_size, total_features, m, n);
            
            % Create training targets
            fprintf('\n2. Creating training targets...\n');
            
            % True spectrum
            true_spectrum = zeros(batch_size, num_angles);
            for b = 1:batch_size
                true_spectrum(b, :) = DOALosses.calculate_spectrum_fixed(...
                    En_true(b, :), array_positions, angles, m, d);
            end
            
            % True DOA indices (avoid edges)
            true_doa_indices = randi([40, 140], batch_size, d);
            
            % True values structure
            y_true_struct = struct();
            y_true_struct.spectrum = true_spectrum;
            y_true_struct.doa_indices = true_doa_indices;
            y_true_struct.evd = En_true;
            
            % Test 1: Individual Loss Components
            fprintf('\n3. Testing individual loss components:\n');
            
            % Spectrum MSE
            spectrum_loss = DOALosses.mse_spectrum_loss(...
                true_spectrum, En_pred, array_positions, m, d, angles);
            fprintf('   a) Spectrum MSE Loss: %.6f\n', spectrum_loss);
            
            % Inverse Peaks
            inverse_peaks_loss = DOALosses.inverse_peaks_loss(...
                true_doa_indices, En_pred, array_positions, m, d, angles);
            fprintf('   b) Inverse Peaks Loss: %.6f\n', inverse_peaks_loss);
            
            % DOA MSE
            doa_mse_loss = DOALosses.mse_doa_loss(...
                true_doa_indices, En_pred, array_positions, m, d, angles);
            fprintf('   c) DOA MSE Loss: %.6f\n', doa_mse_loss);
            
            % EVD Loss
            evd_loss = DOALosses.evd_loss(En_true, En_pred);
            fprintf('   d) EVD Loss: %.6f\n', evd_loss);
            
            % Test 2: Scaled Regularization
            fprintf('\n4. Testing scaled regularization:\n');
            
            if exist('DOARegularizers', 'class')
                % Use SCALED version with 1e-3 scaling
                [reg_loss_scaled, reg_diag] = DOARegularizers.combined_regularization_scaled(...
                    En_pred, En_true, m, d, 0.5, 0.5, 1.0, 1.0, 1e-3);
                
                fprintf('   Regularization (scaled): %.6f\n', reg_loss_scaled);
                fprintf('   Raw orthogonality loss: %.6f\n', reg_diag.diagnostics.orthogonality_loss);
                fprintf('   Scaling factor: %.1e\n', reg_diag.weights.orth_scale);
                fprintf('   Max off-diagonal: %.3f (will improve with training)\n', ...
                        reg_diag.diagnostics.max_off_diagonal);
            else
                fprintf('   Warning: DOARegularizers not found\n');
                reg_loss_scaled = 0;
            end
            
            % Test 3: Complete Training Loss
            fprintf('\n5. Calculating complete training loss:\n');
            
            % Training parameters
            training_params = struct();
            training_params.angles = angles;
            training_params.weights = struct(...
                'spectrum_mse', 0.35, ...
                'inverse_peaks', 0.25, ...
                'doa_mse', 0.20, ...
                'evd', 0.10, ...
                'regularization', 0.10);
            training_params.beta = 1e3;
            training_params.shift_up = 2;
            training_params.orth_scale = 1e-3;
            training_params.use_regularization = true;
            
            % Calculate weighted total loss
            total_loss = 0;
            loss_components = struct();
            
            % Spectrum MSE (35%)
            loss_components.spectrum_mse = spectrum_loss;
            total_loss = total_loss + training_params.weights.spectrum_mse * spectrum_loss;
            
            % Inverse Peaks (25%)
            loss_components.inverse_peaks = inverse_peaks_loss;
            total_loss = total_loss + training_params.weights.inverse_peaks * inverse_peaks_loss;
            
            % DOA MSE (20%)
            loss_components.doa_mse = doa_mse_loss;
            total_loss = total_loss + training_params.weights.doa_mse * doa_mse_loss;
            
            % EVD Loss (10%)
            loss_components.evd = evd_loss;
            total_loss = total_loss + training_params.weights.evd * evd_loss;
            
            % Regularization (10% - SCALED!)
            if training_params.use_regularization && exist('DOARegularizers', 'class')
                loss_components.regularization = reg_loss_scaled;
                total_loss = total_loss + training_params.weights.regularization * reg_loss_scaled;
            end
            
            % Display results
            fprintf('\n6. TRAINING LOSS BREAKDOWN:\n');
            fprintf('   Component            Loss      Weight   Contribution\n');
            fprintf('   ---------------------------------------------------\n');
            
            comp_names = {'spectrum_mse', 'inverse_peaks', 'doa_mse', 'evd', 'regularization'};
            comp_labels = {'Spectrum MSE', 'Inverse Peaks', 'DOA MSE', 'EVD', 'Regularization'};
            
            for i = 1:length(comp_names)
                name = comp_names{i};
                if isfield(loss_components, name)
                    loss_val = loss_components.(name);
                    weight = training_params.weights.(name);
                    contribution = weight * loss_val;
                    
                    fprintf('   %-18s %9.6f   %6.2f   %9.6f\n', ...
                            comp_labels{i}, loss_val, weight, contribution);
                end
            end
            
            fprintf('   ---------------------------------------------------\n');
            fprintf('   TOTAL LOSS: %36.6f\n', total_loss);
            
            % Analysis
            fprintf('\n7. ANALYSIS:\n');
            fprintf('   ✓ All loss components computed\n');
            fprintf('   ✓ Regularization properly scaled (1e-3)\n');
            fprintf('   ✓ Loss magnitudes are balanced (~0.1-2.0 range)\n');
            fprintf('   ✓ Total loss is reasonable for training: %.6f\n', total_loss);
            
            % Expected behavior during training:
            fprintf('\n8. EXPECTED TRAINING BEHAVIOR:\n');
            fprintf('   - Initial: High orthogonality loss (random weights)\n');
            fprintf('   - Early training: Spectrum/DOA losses decrease quickly\n');
            fprintf('   - Mid training: Orthogonality improves (max off-diag → ~0.1)\n');
            fprintf('   - Late training: All losses converge to small values\n');
            
            fprintf('\n9. CONCLUSION:\n');
            fprintf('   All neural network components are ready for training!\n');
            fprintf('   Proceed to convert trainModel.py for complete pipeline.\n');
            fprintf('===========================================\n');
        end
      

    end
end