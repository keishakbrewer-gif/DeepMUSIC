%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              beamformer_all.m                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
% Authors: J. M.                                                               %
%                                                                              %
% Created: 22/04/21                                                            %
%                                                                              %
% Purpose: Implementation of classical beamforming algorithms using utilities.  %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef beamformer_all
    % BEAMFORMER Classical beamforming algorithms for DOA estimation
    
    methods(Static)
        
        %******************************%
        %   Bartlett Beamformer        %
        %******************************%
        function [DoAs, spectrum] = bartlett(incident, array, continuum, sources)
            % The classical Bartlett (conventional) beamformer.
            % Spectrum: P(theta) = a(theta)^H * R * a(theta) / ||a(theta)||^2
            %
            % @param incident -- Measured waveforms [M x N]
            % @param array -- Array element positions [M x 1]
            % @param continuum -- Angles to evaluate [1 x K]
            % @param sources -- Number of signal sources (default: 2)
            %
            % @returns -- DOA indices and spatial spectrum
            
            if nargin < 4
                sources = 2;
            end
            
            % Calculate covariance matrix USING UTILS
            R = utils.correlation_matrix(incident);
            
            % Calculate spatial spectrum
            numSamples = length(continuum);
            spectrum = zeros(1, numSamples);
            
            for i = 1:numSamples
                % Steering vector USING UTILS
                a = utils.ULA_action_vector(array, continuum(i));
                
                % Bartlett spectrum
                spectrum(i) = real(a' * R * a) / norm(a)^2;
            end
            
            % Find peaks USING UTILS
            [peak_locs, ~] = utils.find_spectrum_peaks(spectrum, sources, 1);
            DoAs = beamformer_all.select_largest_peaks(spectrum, peak_locs, sources);
        end
        
        
        %******************************%
        %   Capon (MVDR) Beamformer    %
        %******************************%
        function [DoAs, spectrum] = capon(incident, array, continuum, sources, diagonal_load)
            % Minimum Variance Distortionless Response (Capon) beamformer.
            % Spectrum: P(theta) = 1 / (a(theta)^H * R^{-1} * a(theta))
            %
            % @param incident -- Measured waveforms [M x N]
            % @param array -- Array element positions
            % @param continuum -- Angles to evaluate
            % @param sources -- Number of sources
            % @param diagonal_load -- Diagonal loading factor (default: 0.01)
            %
            % @returns -- DOA indices and spectrum
            
            if nargin < 5
                diagonal_load = 0.01;
            end
            
            if nargin < 4
                sources = 2;
            end
            
            % Covariance matrix USING UTILS
            R = utils.correlation_matrix(incident);
            M = size(R, 1);
            
            % Regularize with diagonal loading
            R_reg = R + diagonal_load * trace(R) / M * eye(M);
            
            % Invert matrix
            R_inv = inv(R_reg);
            
            % Calculate spectrum
            numSamples = length(continuum);
            spectrum = zeros(1, numSamples);
            
            for i = 1:numSamples
                a = utils.ULA_action_vector(array, continuum(i));
                spectrum(i) = 1 / real(a' * R_inv * a);
            end
            
            % Find peaks USING UTILS
            [peak_locs, ~] = utils.find_spectrum_peaks(spectrum, sources, 1);
            DoAs = beamformer_all.select_largest_peaks(spectrum, peak_locs, sources);
        end
        
        
        %**********************************%
        %   Helper: Select largest peaks   %
        %**********************************%
        function DoAs = select_largest_peaks(spectrum, peak_locs, num_peaks)
            % Select the largest peaks from detected peak locations
            %
            % @param spectrum -- Spatial spectrum
            % @param peak_locs -- Detected peak locations
            % @param num_peaks -- Number of peaks to select
            %
            % @returns -- Selected peak locations
            
            if isempty(peak_locs)
                DoAs = [];
                return;
            end
            
            if length(peak_locs) <= num_peaks
                DoAs = sort(peak_locs);
                return;
            end
            
            % Get peak values
            peak_vals = spectrum(peak_locs);
            
            % Sort by value (descending)
            [~, sort_idx] = sort(peak_vals, 'descend');
            
            % Select largest peaks
            selected_locs = peak_locs(sort_idx(1:num_peaks));
            DoAs = sort(selected_locs);
        end
        
        
        %**********************************%
        %   LCMV Beamformer               %
        %**********************************%
        function [DoAs, spectrum, weights] = lcmv(incident, array, continuum, ...
                                                  constraint_matrix, response_vector, num_peaks)
            % Linearly Constrained Minimum Variance beamformer.
            %
            % @param incident -- Measured waveforms
            % @param array -- Array element positions
            % @param continuum -- Angles to evaluate
            % @param constraint_matrix -- Constraint matrix C [M x P]
            % @param response_vector -- Desired response f [P x 1]
            % @param num_peaks -- Number of peaks to find (default: size of constraint matrix)
            %
            % @returns -- DOA indices, spectrum, and beamformer weights
            
            if nargin < 5
                response_vector = 1;  % Default unity response
            end
            
            if nargin < 6
                num_peaks = size(constraint_matrix, 2);  % Default: number of constraints
            end
            
            % Covariance matrix USING UTILS
            R = utils.correlation_matrix(incident);
            M = size(R, 1);
            
            % Add small diagonal loading for stability
            R_reg = R + 0.01 * trace(R) / M * eye(M);
            R_inv = inv(R_reg);
            
            % LCMV weights: w = R^{-1} * C * (C^H * R^{-1} * C)^{-1} * f
            C = constraint_matrix;
            weights = R_inv * C / (C' * R_inv * C) * response_vector;
            
            % Calculate spectrum
            numSamples = length(continuum);
            spectrum = zeros(1, numSamples);
            
            for i = 1:numSamples
                a = utils.ULA_action_vector(array, continuum(i));
                spectrum(i) = abs(weights' * a)^2;
            end
            
            % Find peaks USING UTILS - specify number of peaks
            [DoAs, ~] = utils.find_spectrum_peaks(spectrum, num_peaks, 1);
        end
        
        
        %**********************************%
        %   Compute Beamformer Weights    %
        %**********************************%
        function w = compute_weights(array, doas, beamformer_type, incident, diagonal_load)
            % Compute beamformer weights for given DOAs.
            %
            % @param array -- Array element positions
            % @param doas -- Directions of arrival (radians)
            % @param beamformer_type -- 'bartlett', 'capon', or 'lcmv'
            % @param incident -- Measurement matrix (for adaptive beamformers)
            % @param diagonal_load -- Diagonal loading factor
            %
            % @returns -- Beamformer weights
            
            if nargin < 5
                diagonal_load = 0.01;
            end
            
            M = length(array);
            num_doas = length(doas);
            
            % Steering matrix
            A = beamformer_all.steering_matrix(array, doas);
            
            switch lower(beamformer_type)
                case 'bartlett'
                    % Conventional beamformer: uniform weights
                    w = A / num_doas;
                    
                case 'capon'
                    % MVDR beamformer
                    if nargin < 4 || isempty(incident)
                        error('Incident data required for Capon beamformer');
                    end
                    
                    R = utils.correlation_matrix(incident);
                    R_reg = R + diagonal_load * trace(R) / M * eye(M);
                    R_inv = inv(R_reg);
                    
                    w = zeros(M, num_doas);
                    for i = 1:num_doas
                        a = A(:, i);
                        w(:, i) = (R_inv * a) / (a' * R_inv * a);
                    end
                    
                case 'lcmv'
                    % LCMV beamformer
                    if nargin < 4 || isempty(incident)
                        error('Incident data required for LCMV beamformer');
                    end
                    
                    R = utils.correlation_matrix(incident);
                    R_reg = R + diagonal_load * trace(R) / M * eye(M);
                    R_inv = inv(R_reg);
                    
                    % Use steering matrix as constraint
                    w = R_inv * A / (A' * R_inv * A);
                    
                otherwise
                    error('Unknown beamformer type: %s', beamformer_type);
            end
        end
        
        
        %**********************************%
        %   Steering Matrix Generator     %
        %**********************************%
        function A = steering_matrix(array, angles)
            % Generate steering matrix for multiple angles.
            %
            % @param array -- Array element positions
            % @param angles -- Angles (radians)
            %
            % @returns -- Steering matrix [M x num_angles]
            
            M = length(array);
            num_angles = length(angles);
            A = zeros(M, num_angles);
            
            for i = 1:num_angles
                A(:, i) = utils.ULA_action_vector(array, angles(i));
            end
        end
        
        
        %**********************************%
        %   Beampattern Calculator        %
        %**********************************%
        function [pattern, directivity] = beampattern(array, weights, theta_range)
            % Calculate array beampattern for given weights.
            %
            % @param array -- Array element positions
            % @param weights -- Beamformer weights [M x 1]
            % @param theta_range -- Angles to evaluate
            %
            % @returns -- Beampattern and directivity index
            
            pattern = zeros(size(theta_range));
            
            for i = 1:length(theta_range)
                a = utils.ULA_action_vector(array, theta_range(i));
                pattern(i) = abs(weights' * a)^2;
            end
            
            % Calculate directivity (approximate)
            pattern_norm = pattern / max(pattern);
            directivity = 10 * log10(2*pi / sum(pattern_norm * (theta_range(2)-theta_range(1))));
        end
    
            %**********************************%
            %   Calculate DOA Errors          %
            %**********************************%
            function errors = calculate_doa_errors(true_doas, est_doas)
                % Calculate DOA estimation errors with optimal matching.
                %
                % @param true_doas -- True DOAs (radians)
                % @param est_doas -- Estimated DOAs (radians)
                %
                % @returns -- Absolute errors for matched DOA pairs
                
                true_doas = sort(true_doas(:));
                est_doas = sort(est_doas(:));
                
                if length(true_doas) == length(est_doas)
                    % Simple 1-to-1 matching
                    errors = abs(true_doas - est_doas);
                else
                    % Handle mismatched number of DOAs
                    n = min(length(true_doas), length(est_doas));
                    errors = abs(true_doas(1:n) - est_doas(1:n));
                    if length(errors) < length(true_doas)
                        errors = [errors; Inf(length(true_doas)-length(errors), 1)];
                    end
                end
            end
            
            %**********************************%
            %   Analyze Beampattern           %
            %**********************************%
            function [width_rad, sidelobe_db] = analyze_beampattern(pattern, theta_range)
                % Analyze beampattern characteristics.
                %
                % @param pattern -- Beampattern array
                % @param theta_range -- Corresponding angles (radians)
                %
                % @returns -- Mainlobe width (radians) and peak sidelobe level (dB)
                
                pattern_norm = pattern / max(pattern);
                
                % Find main peak
                [~, peak_idx] = max(pattern_norm);
                
                % Find -3dB points
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
                
                % Linear interpolation for more accurate width
                if left_idx > 1 && right_idx < length(pattern_norm)
                    width_rad = theta_range(right_idx) - theta_range(left_idx);
                else
                    width_rad = theta_range(min(right_idx, length(theta_range))) - ...
                               theta_range(max(left_idx, 1));
                end
                
                % Find peak sidelobe level (excluding mainlobe)
                exclude_width = 2 * width_rad;  % Exclude region around mainlobe
                exclude_start = max(1, floor(peak_idx - exclude_width/(2*(theta_range(2)-theta_range(1)))));
                exclude_end = min(length(pattern_norm), ceil(peak_idx + exclude_width/(2*(theta_range(2)-theta_range(1)))));
                
                sidlobe_indices = [1:exclude_start-1, exclude_end+1:length(pattern_norm)];
                
                if ~isempty(sidlobe_indices)
                    sidelobe_linear = max(pattern_norm(sidlobe_indices));
                    sidelobe_db = 10 * log10(sidelobe_linear);
                else
                    sidelobe_db = -Inf;
                end
            end

    end
end