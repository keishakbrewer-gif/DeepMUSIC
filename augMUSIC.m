%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               augMUSIC.m                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
% Authors: J. M.                                                               %
%                                                                              %
% Created: 23/03/21                                                            %
%                                                                              %
% Purpose: Implementation of the augmented MUSIC algorithm.                    %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [DoAsMUSIC, spectrum] = augMUSIC(En_stack, array, continuum, sources)
    % The MUSIC algorithm calculates the spatial spectrum, which is used
    % to estimate the directions of arrival of the incident signals by
    % finding its d peaks.
    %
    % @param En_stack -- The estimated noise subspace (real and imag stacked).
    %                    Should be a 1 x (2*M*(M-sources)) vector that can be
    %                    reshaped into an M x (M-sources) complex matrix.
    % @param array -- Holds the positions of the array elements.
    % @param continuum -- The continuum of all possible mode vectors
    % @param sources -- The number of signal sources.
    %
    % @returns -- The d locations of the spatial spectrum peaks.
    % @returns -- The spatial spectrum
    
    M = length(array);  % Number of array elements
    N = M - sources;    % Dimension of noise subspace
    
    % RESHAPE En_stack into proper matrix using utils
    total_elements = 2 * M * N;
    
    if length(En_stack) ~= total_elements
        fprintf('Warning: En_stack length = %d, expected %d\n', ...
                length(En_stack), total_elements);
        % Try to reshape anyway with what we have
        N_est = floor(length(En_stack) / (2 * M));
        if N_est < 1
            error('En_stack too small to form noise subspace matrix');
        end
        N = N_est;
    end
    
    % Extract and reshape real and imaginary parts
    real_part = reshape(En_stack(1:M*N), M, N);
    imag_part = reshape(En_stack(M*N+1:end), M, N);
    
    % Form complex noise subspace matrix (M x N)
    En = real_part + 1j * imag_part;
    
    fprintf('Noise subspace size: %d x %d\n', size(En, 1), size(En, 2));
    
    % Calculate spatial spectrum
    numSamples = length(continuum);
    spectrum = zeros(1, numSamples);
    
    % Pre-compute En * En' for efficiency
    EnEnH = En * En';
    
    % Loop through all continuum points
    for i = 1:numSamples
        % Establish array steering vector USING UTILS
        theta = continuum(i);
        a = utils.ULA_action_vector(array, theta);
        
        % MUSIC spectrum: 1/(a^H * En * En^H * a)
        denominator = a' * EnEnH * a;
        spectrum(i) = 1 / abs(denominator);
    end
    
    % Find peaks using UTILS
    [peak_locs, ~] = utils.find_spectrum_peaks(spectrum, sources, 1);
    DoAsMUSIC = peak_locs;
    
    % Only keep d largest peaks
    if length(DoAsMUSIC) > sources
        % Get peak values
        peak_vals = spectrum(DoAsMUSIC);
        % Sort and keep largest 'sources' peaks
        [~, sort_idx] = sort(peak_vals, 'descend');
        DoAsMUSIC = DoAsMUSIC(sort_idx(1:sources));
    end
    
    % Sort DoAs by index
    DoAsMUSIC = sort(DoAsMUSIC);
end