classdef classicMUSIC
    % CLASSICMUSIC Traditional MUSIC algorithm for DOA estimation
    
    methods(Static)
        
        %**********************************************************************%
        %   the classic MUSIC algorithm                                       %
        %**********************************************************************%
        function [DoA, spectrum] = estimate(incident, array, continuum, sources)
            % The classic MUSIC algorithm calculates the spatial spectrum,
            % which is used to estimate the directions of arrival of the
            % incident signals by finding its d peaks.
            %
            % @param incident -- The measured waveforms [M x N]
            % @param array -- Holds the positions of the array elements.
            % @param continuum -- The continuum of all possible mode vectors (angles)
            % @param sources -- The number of signal sources (optional).
            %
            % @returns -- The d locations of the spatial spectrum peaks.
            % @returns -- The spatial spectrum.
            
            % Calculate covariance matrix USING UTILS
            covariance = utils.correlation_matrix(incident);
            M = size(covariance, 1);
            
            % Determine number of sources if not provided
            if nargin < 4 || isempty(sources)
                % First get all eigenvalues
                [eigenvectors, eigenvalues] = eig(covariance);
                eigenvalues = diag(eigenvalues);
                [eigenvalues, idx] = sort(eigenvalues, 'descend');
                eigenvectors = eigenvectors(:, idx);
                
                % Estimate from eigenvalue clustering
                d = classicMUSIC.estimate_num_sources(eigenvalues, M);
                fprintf('Estimated number of sources: %d\n', d);
                
                % Get noise subspace
                noise_subspace = eigenvectors(:, d+1:end);
            else
                d = sources;
                % Get decomposition with known number of sources
                [eigenvectors, eigenvalues, ~, noise_subspace] = ...
                    utils.eigen_decomposition(covariance, d);
            end
            
            % Validate noise subspace
            if isempty(noise_subspace)
                error('Noise subspace is empty. Check number of sources.');
            end
            
            fprintf('Array elements: %d, Sources: %d, Noise subspace: %d x %d\n', ...
                    M, d, size(noise_subspace, 1), size(noise_subspace, 2));
            
            % Calculate spatial spectrum
            numSamples = length(continuum);
            spectrum = zeros(1, numSamples);
            
            % Pre-compute En * En' for efficiency
            En = noise_subspace;
            EnEnH = En * En';
            
            % Loop through all continuum points
            for i = 1:numSamples
                % Establish array steering vector USING UTILS
                a = utils.ULA_action_vector(array, continuum(i));
                
                % MUSIC spectrum: 1/(a^H * En * En^H * a)
                denominator = a' * EnEnH * a;
                spectrum(i) = 1 / abs(denominator);
            end
            
            % Find peaks USING UTILS
            [peak_locs, ~] = utils.find_spectrum_peaks(spectrum, d, 1);
            
            % Only keep d largest peaks (match Python behavior)
            if ~isempty(peak_locs)
                if length(peak_locs) > d
                    % Get peak values
                    peak_vals = spectrum(peak_locs);
                    % Sort and keep largest d peaks
                    [~, sort_idx] = sort(peak_vals, 'descend');
                    DoA = peak_locs(sort_idx(1:d));
                else
                    DoA = peak_locs;
                end
                DoA = sort(DoA);
            else
                DoA = [];
                warning('No peaks found in MUSIC spectrum');
            end
        end        
        
        %*******************************#
        %   cluster small eigenvalues   #
        %*******************************#
        function clustered_evs = cluster(evs)
            % Estimates multiplicity of smallest eigenvalue.
            %
            % @param evs -- The eigenvalues in descending order.
            %
            % @returns -- The eigenvalues similar or equal to the smallest eigenvalue.
            
            % Ensure eigenvalues are sorted descending
            evs = sort(evs, 'descend');
            
            % Python method: simplest clustering with threshold
            threshold = 0.4;
            
            % Find eigenvalues close to smallest eigenvalue
            smallest_ev = abs(evs(end));
            clustered_idx = abs(evs) < smallest_ev + threshold;
            clustered_evs = evs(clustered_idx);
        end
        
        
        %**********************************#
        %   estimate number of sources     #
        %**********************************#
        function d = estimate_num_sources(eigenvalues, M)
            % Estimate number of sources from eigenvalue distribution.
            %
            % @param eigenvalues -- Sorted eigenvalues (descending)
            % @param M -- Number of array elements
            %
            % @returns -- Estimated number of sources
            
            % Sort eigenvalues descending (if not already)
            eigenvalues = sort(eigenvalues, 'descend');
            
            % Method 1: Python clustering approach
            clustered = classicMUSIC.cluster(eigenvalues);
            n = length(clustered);   % multiplicity of smallest eigenvalue
            d_est1 = M - n;          % number of signal sources
            
            % Method 2: MDL (Minimum Description Length) criterion
            d_est2 = classicMUSIC.mdl_criterion(eigenvalues, M);
            
            % Method 3: AIC (Akaike Information Criterion)
            d_est3 = classicMUSIC.aic_criterion(eigenvalues, M);
            
            % Use consensus or Python method
            fprintf('Source estimation methods:\n');
            fprintf('  Clustering: %d sources\n', d_est1);
            fprintf('  MDL: %d sources\n', d_est2);
            fprintf('  AIC: %d sources\n', d_est3);
            
            % Default to Python clustering method
            d = d_est1;
            
            % Validate
            if d < 1 || d >= M
                warning('Estimated sources (%d) out of range [1, %d). Using default: 1', d, M);
                d = 1;
            end
        end
        
        
        %**********************************#
        %   MDL criterion                  #
        %**********************************#
        function d = mdl_criterion(eigenvalues, M)
            % Minimum Description Length criterion for source number estimation.
            %
            % @param eigenvalues -- Sorted eigenvalues
            % @param M -- Number of array elements
            %
            % @returns -- Estimated number of sources
            
            N = length(eigenvalues);  % Should equal M
            
            mdl_values = zeros(1, M-1);
            
            for k = 0:M-2
                % Likelihood function
                lambda_k = eigenvalues(k+1:end);
                arithmetic_mean = mean(lambda_k);
                geometric_mean = prod(lambda_k)^(1/length(lambda_k));
                
                if arithmetic_mean > 0 && geometric_mean > 0
                    L = ((M - k) * N) * log(arithmetic_mean / geometric_mean);
                    penalty = 0.5 * k * (2*M - k) * log(N);
                    mdl_values(k+1) = -L + penalty;
                else
                    mdl_values(k+1) = Inf;
                end
            end
            
            [~, d] = min(mdl_values);
            d = d - 1;  % Convert from 1-based to 0-based
        end
        
        
        %**********************************#
        %   AIC criterion                  #
        %**********************************#
        function d = aic_criterion(eigenvalues, M)
            % Akaike Information Criterion for source number estimation.
            %
            % @param eigenvalues -- Sorted eigenvalues
            % @param M -- Number of array elements
            %
            % @returns -- Estimated number of sources
            
            N = length(eigenvalues);  % Should equal M
            
            aic_values = zeros(1, M-1);
            
            for k = 0:M-2
                % Likelihood function
                lambda_k = eigenvalues(k+1:end);
                arithmetic_mean = mean(lambda_k);
                geometric_mean = prod(lambda_k)^(1/length(lambda_k));
                
                if arithmetic_mean > 0 && geometric_mean > 0
                    L = ((M - k) * N) * log(arithmetic_mean / geometric_mean);
                    penalty = k * (2*M - k);
                    aic_values(k+1) = -2 * L + 2 * penalty;
                else
                    aic_values(k+1) = Inf;
                end
            end
            
            [~, d] = min(aic_values);
            d = d - 1;  % Convert from 1-based to 0-based
        end
        
        
        %**********************************#
        %   eigenvalue analysis plot       #
        %**********************************#
        function plot_eigenvalues(eigenvalues, sources)
            % Plot eigenvalue distribution for analysis.
            %
            % @param eigenvalues -- Sorted eigenvalues
            % @param sources -- Number of signal sources (optional)
            
            eigenvalues = sort(eigenvalues, 'descend');
            M = length(eigenvalues);
            
            figure('Position', [100, 100, 800, 400]);
            
            % Plot 1: Linear scale
            subplot(1, 2, 1);
            stem(1:M, eigenvalues, 'filled', 'LineWidth', 1.5);
            hold on;
            
            if nargin >= 2 && sources > 0 && sources < M
                plot([sources+0.5, sources+0.5], [0, max(eigenvalues)*1.1], ...
                     'r--', 'LineWidth', 1.5);
                text(sources+0.7, max(eigenvalues)*0.8, 'Noise Subspace', ...
                     'Rotation', 90, 'FontSize', 10, 'Color', 'r');
                text(sources-0.3, max(eigenvalues)*0.8, 'Signal Subspace', ...
                     'Rotation', 90, 'FontSize', 10, 'Color', 'b');
            end
            
            title('Eigenvalue Distribution', 'FontSize', 14);
            xlabel('Eigenvalue Index', 'FontSize', 12);
            ylabel('Eigenvalue', 'FontSize', 12);
            grid on;
            xlim([0.5, M+0.5]);
            
            % Plot 2: Logarithmic scale (better for seeing small eigenvalues)
            subplot(1, 2, 2);
            semilogy(1:M, eigenvalues, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8);
            hold on;
            
            if nargin >= 2 && sources > 0 && sources < M
                plot([sources+0.5, sources+0.5], [min(eigenvalues(eigenvalues>0))*0.1, max(eigenvalues)*10], ...
                     'r--', 'LineWidth', 1.5);
            end
            
            % Plot clustering threshold
            if M > 1
                smallest_ev = eigenvalues(end);
                threshold = smallest_ev + 0.4;
                plot([0.5, M+0.5], [threshold, threshold], 'g:', 'LineWidth', 1.5);
                legend('Eigenvalues', 'Signal/Noise Boundary', 'Clustering Threshold', ...
                       'Location', 'best');
            else
                legend('Eigenvalues', 'Location', 'best');
            end
            
            title('Eigenvalues (Log Scale)', 'FontSize', 14);
            xlabel('Eigenvalue Index', 'FontSize', 12);
            ylabel('Eigenvalue (log)', 'FontSize', 12);
            grid on;
            xlim([0.5, M+0.5]);
            
            set(gcf, 'Color', 'w');
        end
        
    end
end