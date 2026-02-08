%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    beamformer.m                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
% Purpose: Python-equivalent beamformer using utility functions.               %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [DoAs, spectrum] = beamformer(incident, array, continuum, sources)
    % Python-equivalent beamformer algorithm.
    % Spectrum: P(theta) = a(theta)^H * R * a(theta) / ||a(theta)||^2
    %
    % @param incident -- The measured waveforms [M x N]
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
        
        % Bartlett spectrum (same as Python beamformer)
        spectrum(i) = real(a' * R * a) / norm(a)^2;
    end
    
    % Find peaks USING UTILS
    [peak_locs, ~] = utils.find_spectrum_peaks(spectrum, sources, 1);
    
    % Select largest peaks (Python behavior)
    if ~isempty(peak_locs)
        if length(peak_locs) > sources
            peak_vals = spectrum(peak_locs);
            [~, sort_idx] = sort(peak_vals, 'descend');
            DoAs = peak_locs(sort_idx(1:sources));
        else
            DoAs = peak_locs;
        end
        DoAs = sort(DoAs);
    else
        DoAs = [];
    end
end