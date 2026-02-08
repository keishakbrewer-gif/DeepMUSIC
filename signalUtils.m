classdef signalUtils
    % SIGNALUTILS Signal processing utilities
    
    methods(Static)
        
        function [freq, spectrum] = compute_spectrum(signal, fs)
            % Compute frequency spectrum
            %
            % @param signal -- Time-domain signal
            % @param fs -- Sampling frequency
            %
            % @returns -- Frequency vector and spectrum
            
            N = length(signal);
            spectrum = fft(signal);
            freq = (0:N-1) * fs / N;
            
            % Return single-sided spectrum
            if nargout > 0
                freq = freq(1:floor(N/2));
                spectrum = abs(spectrum(1:floor(N/2)));
            end
        end
        
        function filtered = apply_filter(signal, b, a)
            % Apply IIR filter
            %
            % @param signal -- Input signal
            % @param b -- Numerator coefficients
            % @param a -- Denominator coefficients
            %
            % @returns -- Filtered signal
            
            filtered = filter(b, a, signal);
        end
        
    end
end