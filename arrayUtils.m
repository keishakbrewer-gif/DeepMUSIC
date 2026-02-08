classdef arrayUtils
    % ARRAYUTILS Array geometry and steering vector utilities
    
    methods(Static)
        
        function positions = ULA_positions(num_elements, spacing)
            % Generate ULA element positions
            %
            % @param num_elements -- Number of array elements
            % @param spacing -- Element spacing (default: 0.5 lambda)
            %
            % @returns -- Array element positions
            
            if nargin < 2
                spacing = 0.5;  % lambda/2 spacing
            end
            
            positions = (0:num_elements-1)' * spacing;
        end
        
        function a = steering_vector(array_positions, theta, wavelength)
            % General steering vector for arbitrary array
            %
            % @param array_positions -- N x 2 matrix of [x, y] positions
            % @param theta -- Azimuth angle (radians)
            % @param wavelength -- Signal wavelength (default: 1)
            %
            % @returns -- Steering vector
            
            if nargin < 3
                wavelength = 1;
            end
            
            if size(array_positions, 2) == 1
                % 1D array (ULA)
                a = exp(-1j * 2*pi/wavelength * array_positions * sin(theta));
            else
                % 2D array
                k = 2*pi/wavelength * [sin(theta); 0];  % Assuming elevation = 0
                a = exp(-1j * array_positions * k);
            end
            
            a = a(:);
        end
        
        function beampattern = array_beampattern(array_positions, weights, theta_range)
            % Calculate array beampattern
            %
            % @param array_positions -- Array element positions
            % @param weights -- Beamformer weights
            % @param theta_range -- Angles to evaluate
            %
            % @returns -- Beampattern
            
            beampattern = zeros(size(theta_range));
            
            for i = 1:length(theta_range)
                a = arrayUtils.steering_vector(array_positions, theta_range(i));
                beampattern(i) = abs(weights' * a)^2;
            end
        end
        
    end
end