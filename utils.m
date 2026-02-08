%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     utils.m                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
% Authors: J. M.                                                               %
%                                                                              %
% Created: 27/04/21                                                            %
%                                                                              %
% Purpose: Definitions of helpful functions for DOA estimation.                %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef utils
    % UTILS Utility functions for DOA estimation
    
    properties(Constant)
        % Global random seed for reproducibility
        SEED = 42;
    end
    
    methods(Static)
        
        %**********************************%
        %   calculate the MUSIC spectrum   %
        %**********************************%
        function spectrum = calculate_spectrum(y_pred, array, angles, m)
            % Calculates the MUSIC spectrum according to P = 1 / (a^H En En^H a).
            %
            % @param y_pred -- The estimated noise space vectors.
            %                  Size: [batch_size, 2*m] where m = array elements
            % @param array -- Array element positions (m x 1)
            % @param angles -- The continuum of angles to evaluate (1 x num_samples)
            % @param m -- Number of array elements
            %
            % @returns -- The estimated spectrum (batch_size x num_samples).
            
            [batch_size, pred_cols] = size(y_pred);
            num_samples = length(angles);
            
            % DEBUG: Check dimensions
            fprintf('DEBUG: y_pred size: %d x %d\n', batch_size, pred_cols);
            fprintf('DEBUG: m = %d, expected cols = %d\n', m, 2*m);
            
            if pred_cols ~= 2*m
                error('y_pred should have %d columns (2*m), but has %d', 2*m, pred_cols);
            end
            
            % Extract real and imaginary parts
            EnReal = y_pred(:, 1:m);
            EnImag = y_pred(:, m+1:end);
            
            % Form complex noise subspace vectors
            % Each row is a noise subspace vector of length m
            En = EnReal + 1j * EnImag;  % batch_size x m
            
            % Initialize spectrum matrix
            spectrum = zeros(batch_size, num_samples);
            
            % Calculate spatial spectrum for each angle
            for i = 1:num_samples
                % Establish array steering vector
                a = utils.ULA_action_vector(array, angles(i));
                a = a(:);  % Ensure column vector (m x 1)
                
                % For each batch sample
                for b = 1:batch_size
                    % Extract noise subspace vector for this batch
                    en_vec = En(b, :).';  % m x 1
                    
                    % Compute En * En^H (outer product for rank-1 case)
                    EnEnH = en_vec * en_vec';
                    
                    % MUSIC spectrum: 1 / |a^H * En * En^H * a|
                    H = a' * EnEnH * a;
                    spectrum(b, i) = 1 / abs(H);
                end
            end
        end
        
        
        %****************************#
        %   calculate permutations   #
        %****************************#
        function perms = permutations(predDoA)
            % Calculates all permutations of the given list.
            %
            % @param predDoA -- The estimated DoA angles to be permuted.
            %
            % @returns -- All permutations of the estimated DoA.
            
            if isempty(predDoA)
                perms = {};
                return;
            end
            
            if length(predDoA) == 1
                perms = {predDoA};
                return;
            end
            
            perms = {};
            for i = 1:length(predDoA)
                remaining = predDoA([1:i-1, i+1:end]);
                
                sub_perms = utils.permutations(remaining);
                for j = 1:length(sub_perms)
                    perms{end+1} = [predDoA(i), sub_perms{j}];
                end
            end
        end
        
        
        %*******************************************#
        %   uniform linear array steering vector    %
        %*******************************************#
        function a = ULA_action_vector(array, theta)
            % Establish the possible mode vectors (steering vectors) given the
            % positions of a uniform linear array.
            %
            % @param array -- Holds the positions of the array elements.
            % @param theta -- The value of the given axis to be evaluated.
            %
            % @returns -- The action vector.
            
            a = exp(-1j * pi * array * sin(theta));
            a = a(:);  % Ensure column vector
        end
        
        
        %**************************#
        %   generate random seed   #
        %**************************#
        function set_random_seed(seed)
            % Set random seeds for reproducibility
            %
            % @param seed -- Random seed value
            
            if nargin < 1
                seed = utils.SEED;
            end
            
            rng(seed, 'twister');  % MATLAB random number generator
            
            % Note: For Deep Learning, additional seeding is needed
            % when we convert the neural network components
        end
        
        
        %**********************************#
        %   normalize complex data [0,1]   #
        %**********************************#
        function data_norm = normalize_complex(data)
            % Normalize complex data to range [0, 1]
            %
            % @param data -- Complex-valued data
            %
            % @returns -- Normalized data
            
            if isreal(data)
                % Real data normalization
                data_min = min(data(:));
                data_max = max(data(:));
                if data_max ~= data_min
                    data_norm = (data - data_min) / (data_max - data_min);
                else
                    data_norm = zeros(size(data));
                end
            else
                % Complex data: normalize magnitude and phase separately
                mag = abs(data);
                phase = angle(data);
                
                % Normalize magnitude [0, 1]
                mag_min = min(mag(:));
                mag_max = max(mag(:));
                if mag_max ~= mag_min
                    mag_norm = (mag - mag_min) / (mag_max - mag_min);
                else
                    mag_norm = zeros(size(mag));
                end
                
                % Normalize phase [-pi, pi] to [0, 1]
                phase_norm = (phase + pi) / (2*pi);
                
                % Reconstruct complex data
                data_norm = mag_norm .* exp(1j * 2*pi * phase_norm);
            end
        end
        
        
        %************************************#
        %   calculate correlation matrix     #
        %************************************#
        function R = correlation_matrix(X)
            % Calculate sample correlation matrix
            %
            % @param X -- Measurement matrix [array_elements x snapshots]
            %
            % @returns -- Correlation matrix
            
            [M, N] = size(X);
            R = (X * X') / N;
        end
        
        
        %********************************#
        %   eigenvalue decomposition     #
        %********************************#
        function [U, D, signal_subspace, noise_subspace] = eigen_decomposition(R, num_sources)
            % Eigenvalue decomposition for subspace methods
            %
            % @param R -- Correlation matrix
            % @param num_sources -- Number of signal sources
            %
            % @returns -- Eigenvectors, eigenvalues, signal and noise subspaces
            
            [U, D] = eig(R);
            [D_sorted, idx] = sort(diag(D), 'descend');
            U = U(:, idx);
            
            M = size(R, 1);
            if num_sources < M
                signal_subspace = U(:, 1:num_sources);
                noise_subspace = U(:, num_sources+1:end);
                D = diag(D_sorted);
            else
                signal_subspace = U;
                noise_subspace = [];
                warning('Number of sources >= array elements');
            end
        end
        
        
        %**********************************#
        %   find peaks in spectrum         #
        %**********************************#
        function [peak_locs, peak_vals] = find_spectrum_peaks(spectrum, num_peaks, min_distance)
            % Find peaks in spatial spectrum
            %
            % @param spectrum -- Spatial spectrum
            % @param num_peaks -- Maximum number of peaks to find (default: Inf)
            % @param min_distance -- Minimum distance between peaks (default: 1)
            %
            % @returns -- Peak locations and values
            
            if nargin < 2 || isempty(num_peaks)
                num_peaks = Inf;  % Find all peaks by default
            end
            
            if nargin < 3
                min_distance = 1;
            end
            
            % Ensure num_peaks is integer if not Inf
            if ~isinf(num_peaks)
                num_peaks = floor(num_peaks);
            end
            
            % Use MATLAB's findpeaks with proper parameters
            if isinf(num_peaks)
                [peak_vals, peak_locs] = findpeaks(spectrum, ...
                    'MinPeakDistance', min_distance, ...
                    'SortStr', 'descend');
            else
                [peak_vals, peak_locs] = findpeaks(spectrum, ...
                    'MinPeakDistance', min_distance, ...
                    'SortStr', 'descend', ...
                    'NPeaks', num_peaks);
            end
            
            % Sort by location (ascending)
            if ~isempty(peak_locs)
                [peak_locs, idx] = sort(peak_locs);
                peak_vals = peak_vals(idx);
            end
        end     
        
        %**********************************#
        %   calculate SNR                  #
        %**********************************#
        function snr_value = calculate_snr(signal, noise)
            % Calculate Signal-to-Noise Ratio
            %
            % @param signal -- Clean signal
            % @param noise -- Noise component
            %
            % @returns -- SNR in dB
            
            signal_power = mean(abs(signal(:)).^2);
            noise_power = mean(abs(noise(:)).^2);
            
            if noise_power > 0
                snr_value = 10 * log10(signal_power / noise_power);
            else
                snr_value = Inf;
            end
        end
        
        
        %**********************************#
        %   add white gaussian noise       #
        %**********************************#
        function X_noisy = add_awgn(X, snr_db)
            % Add Additive White Gaussian Noise
            %
            % @param X -- Clean signal
            % @param snr_db -- Desired SNR in dB
            %
            % @returns -- Noisy signal
            
            signal_power = mean(abs(X(:)).^2);
            noise_power = signal_power / (10^(snr_db/10));
            
            % Generate complex Gaussian noise
            noise = sqrt(noise_power/2) * (randn(size(X)) + 1j * randn(size(X)));
            
            X_noisy = X + noise;
        end
        
        
        %**********************************#
        %   save/load data utilities       #
        %**********************************#
        function save_data(filename, data_struct)
            % Save data to .mat file
            %
            % @param filename -- Output filename
            % @param data_struct -- Structure containing data
            
            save(filename, '-struct', 'data_struct', '-v7.3');
        end
        
        function data = load_data(filename)
            % Load data from .mat file
            %
            % @param filename -- Input filename
            %
            % @returns -- Loaded data structure
            
            data = load(filename);
        end
        
        
        %**********************************#
        %   timing utility                 #
        %**********************************#
        function elapsed = time_function(func_handle, varargin)
            % Time the execution of a function
            %
            % @param func_handle -- Function handle to time
            % @param varargin -- Arguments to pass to function
            %
            % @returns -- Elapsed time in seconds
            
            tic;
            func_handle(varargin{:});
            elapsed = toc;
        end
        
        %**********************************#
        %         wrap error               #
        %**********************************#
        function wrapped_error = wrap_error(angular_error)
            % Wrap angular error to range [-pi/2, pi/2).
            % Equivalent to Python: ((error + pi/2) % pi) - pi/2
            %
            % @param angular_error -- Angular error in radians
            %
            % @returns -- Wrapped angular error
            
            wrapped_error = mod(angular_error + pi/2, pi) - pi/2;
        end

    end
end