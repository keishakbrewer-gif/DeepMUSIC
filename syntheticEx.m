classdef syntheticEx
    % SYNTHETICEX Synthetic data generation for DOA testing
    
    properties(Constant)
        % Default parameters
        DEFAULT_SOURCES = 5;      % number of sources (d)
        DEFAULT_ELEMENTS = 8;     % number of array elements (m)
        DEFAULT_SNR = 10;         % signal to noise ratio (dB)
        DEFAULT_SNAPSHOTS = 200;  % number of snapshots
        
        % Signal parameters
        MEAN_SIGNAL_POWER = 0;
        VAR_SIGNAL_POWER = 1;
        MEAN_NOISE = 0;
        VAR_NOISE = 1;
    end
    
    methods(Static)
        
        %***********************#
        %   construct signals   %
        %***********************#
        function [measurement, source_signals] = construct_signal(thetas, m, d, ...
                snapshots, snr_db, array_positions)
            % Construct a signal with the given initializations.
            %
            % @param thetas -- The DOA angles of the sources (radians) [1 x d]
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param snapshots -- Number of snapshots
            % @param snr_db -- Signal-to-noise ratio (dB)
            % @param array_positions -- Array element positions [m x 1]
            %
            % @returns -- The measurement vector [m x snapshots]
            % @returns -- The source signals [d x snapshots]
            
            % Set defaults using UTILS approach
            if nargin < 6
                array_positions = arrayUtils.ULA_positions(m);  % USING ARRAYUTILS
            end
            
            if nargin < 5
                snr_db = syntheticEx.DEFAULT_SNR;
            end
            
            if nargin < 4
                snapshots = syntheticEx.DEFAULT_SNAPSHOTS;
            end
            
            if nargin < 3
                d = length(thetas);
            end
            
            if nargin < 2
                m = syntheticEx.DEFAULT_ELEMENTS;
            end
            
            % Generate source signals USING UTILS approach
            signal_power = syntheticEx.VAR_SIGNAL_POWER * (10^(snr_db/10));
            source_signals = sqrt(signal_power/2) * ...
                (randn(d, snapshots) + 1j * randn(d, snapshots)) + ...
                syntheticEx.MEAN_SIGNAL_POWER;
            
            % Create steering matrix A USING UTILS
            A = syntheticEx.steering_matrix(array_positions, thetas);
            
            % Generate noise USING UTILS function
            noise_power = syntheticEx.VAR_NOISE;
            noise = utils.add_awgn(zeros(m, snapshots), ...
                10*log10(signal_power/noise_power));  % USING UTILS
            
            % Measurement: A^H * S + noise
            measurement = A * source_signals + noise;
        end
        
        
        %********************************#
        %   construct coherent signals   %
        %********************************#
        function [measurement, source_signals] = construct_coherent_signal(thetas, m, d, ...
                snapshots, snr_db, array_positions)
            % Construct a coherent signal with the given initializations.
            %
            % @param thetas -- The DOA angles of the sources (radians) [1 x d]
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param snapshots -- Number of snapshots
            % @param snr_db -- Signal-to-noise ratio (dB)
            % @param array_positions -- Array element positions
            %
            % @returns -- The measurement vector [m x snapshots]
            % @returns -- The source signals [d x snapshots]
            
            % Set defaults
            if nargin < 6
                array_positions = arrayUtils.ULA_positions(m);  % USING ARRAYUTILS
            end
            
            if nargin < 5
                snr_db = syntheticEx.DEFAULT_SNR;
            end
            
            if nargin < 4
                snapshots = syntheticEx.DEFAULT_SNAPSHOTS;
            end
            
            if nargin < 3
                d = length(thetas);
            end
            
            if nargin < 2
                m = syntheticEx.DEFAULT_ELEMENTS;
            end
            
            % Generate a single source signal
            signal_power = syntheticEx.VAR_SIGNAL_POWER * (10^(snr_db/10));
            single_signal = sqrt(signal_power/2) * ...
                (randn(1, snapshots) + 1j * randn(1, snapshots)) + ...
                syntheticEx.MEAN_SIGNAL_POWER;
            
            % All signals receive same amplitude and phase (coherent)
            source_signals = repmat(single_signal, d, 1);
            
            % Create steering matrix A USING UTILS
            A = syntheticEx.steering_matrix(array_positions, thetas);
            
            % Generate noise USING UTILS
            noise_power = syntheticEx.VAR_NOISE;
            noise = utils.add_awgn(zeros(m, snapshots), ...
                10*log10(signal_power/noise_power));  % USING UTILS
            
            % Measurement: A^H * S + noise
            measurement = A * source_signals + noise;
        end
        
        
        %********************#
        %   create dataset   %
        %********************#
        function [X, Y] = create_dataset(name, size, m, d, snapshots, snr_db, ...
                coherent, array_positions, save_data)
            % Creates dataset of given size with the above initializations.
            %
            % @param name -- The name (and path) of the dataset file
            % @param size -- The size of the dataset (number of samples)
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param snapshots -- Number of snapshots
            % @param snr_db -- Signal-to-noise ratio (dB)
            % @param coherent -- If true, the signals are coherent
            % @param array_positions -- Array element positions
            % @param save_data -- If true, save to .mat file
            %
            % @returns -- Measurements X [size x m x snapshots]
            % @returns -- DOAs Y [size x d]
            
            % Set defaults
            if nargin < 9
                save_data = true;
            end
            
            if nargin < 8
                array_positions = arrayUtils.ULA_positions(m);
            end
            
            if nargin < 7
                coherent = false;
            end
            
            if nargin < 6
                snr_db = syntheticEx.DEFAULT_SNR;
            end
            
            if nargin < 5
                snapshots = syntheticEx.DEFAULT_SNAPSHOTS;
            end
            
            if nargin < 4
                d = syntheticEx.DEFAULT_SOURCES;
            end
            
            if nargin < 3
                m = syntheticEx.DEFAULT_ELEMENTS;
            end
            
            % Set random seed
            utils.set_random_seed(42);
            
            % Initialize arrays
            X = zeros(size, m, snapshots, 'like', 1j);  % Complex array
            Y = zeros(size, d);                         % Real array
            
            fprintf('Creating dataset: %s\n', name);
            fprintf('  Samples: %d, Elements: %d, Sources: %d\n', size, m, d);
            fprintf('  Snapshots: %d, SNR: %.1f dB, Coherent: %s\n', ...
                    snapshots, snr_db, string(coherent));
            
            % Progress display
            progress_step = max(1, floor(size/10));
            
            % Generate samples
            for i = 1:size
                % Random source directions in [-pi/2, pi/2]
                thetas = pi * (rand(1, d) - 0.5);
                
                if coherent
                    [measurement, ~] = syntheticEx.construct_coherent_signal(...
                        thetas, m, d, snapshots, snr_db, array_positions);
                else
                    [measurement, ~] = syntheticEx.construct_signal(...
                        thetas, m, d, snapshots, snr_db, array_positions);
                end
                
                % Store
                X(i, :, :) = measurement;
                Y(i, :) = thetas;
                
                % Progress display
                if mod(i, progress_step) == 0
                    fprintf('  Generated %d/%d samples\n', i, size);
                end
            end
            
            % Save to file if requested
            if save_data
                % BULLETPROOF directory creation
                data_dir = 'data';
                
                % Check if directory exists
                if exist(data_dir, 'dir') ~= 7  % 7 means it's a directory
                    fprintf('  Creating directory: %s\n', data_dir);
                    try
                        mkdir(data_dir);
                        fprintf('  Directory created successfully\n');
                    catch ME
                        fprintf('  Warning: Could not create directory "%s": %s\n', ...
                                data_dir, ME.message);
                        fprintf('  Saving to current directory instead\n');
                        data_dir = '.';  % Save to current directory
                    end
                end
                
                % Save metadata
                metadata = struct();
                metadata.m = m;
                metadata.d = d;
                metadata.snapshots = snapshots;
                metadata.snr_db = snr_db;
                metadata.coherent = coherent;
                metadata.array_positions = array_positions;
                metadata.creation_date = datestr(now);
                
                % Construct filename
                if strcmp(data_dir, '.')
                    filename = [name '.mat'];
                else
                    filename = fullfile(data_dir, [name '.mat']);
                end
                
                % Save the file
                fprintf('  Saving to: %s\n', filename);
                try
                    save(filename, 'X', 'Y', 'metadata', '-v7.3');
                    fprintf('  File saved successfully\n');
                catch ME
                    fprintf('  Error saving file: %s\n', ME.message);
                    fprintf('  Trying alternative save method...\n');
                    
                    % Alternative: Save without metadata
                    try
                        save(filename, 'X', 'Y', '-v7.3');
                        fprintf('  File saved (without metadata)\n');
                    catch ME2
                        fprintf('  Could not save file: %s\n', ME2.message);
                        fprintf('  Dataset will not be saved to disk\n');
                    end
                end
            end
            
            fprintf('  Dataset creation complete!\n');
        end
        
        
        %**************************#
        %   create mixed dataset   %
        %**************************#
        function [X_mixed, Y_mixed] = create_mixed_dataset(name, first_file, ...
                second_file, save_data)
            % Creates mixed dataset from two existing datasets.
            %
            % @param name -- Name of the new dataset
            % @param first_file -- Path to first dataset (.mat file)
            % @param second_file -- Path to second dataset (.mat file)
            % @param save_data -- If true, save to file
            %
            % @returns -- Mixed measurements and DOAs
            
            if nargin < 4
                save_data = true;
            end
            
            fprintf('Creating mixed dataset: %s\n', name);
            
            % Load datasets
            data1 = utils.load_data(first_file);
            data2 = utils.load_data(second_file);
            
            if ~isfield(data1, 'X') || ~isfield(data1, 'Y')
                error('Invalid dataset format in: %s', first_file);
            end
            
            if ~isfield(data2, 'X') || ~isfield(data2, 'Y')
                error('Invalid dataset format in: %s', second_file);
            end
            
            X1 = data1.X;
            Y1 = data1.Y;
            X2 = data2.X;
            Y2 = data2.Y;
            
            % Check if number of sources is different
            num_sources1 = size(Y1, 2);
            num_sources2 = size(Y2, 2);
            
            if num_sources1 ~= num_sources2
                fprintf('  Warning: Different number of sources (%d vs %d)\n', ...
                        num_sources1, num_sources2);
                fprintf('  Padding with NaNs to match dimensions\n');
                
                % Pad the dataset with fewer sources with NaNs
                max_sources = max(num_sources1, num_sources2);
                
                if num_sources1 < max_sources
                    % Pad Y1 with NaNs
                    Y1_padded = [Y1, NaN(size(Y1, 1), max_sources - num_sources1)];
                    Y1 = Y1_padded;
                end
                
                if num_sources2 < max_sources
                    % Pad Y2 with NaNs
                    Y2_padded = [Y2, NaN(size(Y2, 1), max_sources - num_sources2)];
                    Y2 = Y2_padded;
                end
            end
            
            % Check other compatibility
            if size(X1, 2) ~= size(X2, 2) || size(X1, 3) ~= size(X2, 3)
                error('Dataset dimensions mismatch: X1 size = %s, X2 size = %s', ...
                      mat2str(size(X1)), mat2str(size(X2)));
            end
            
            % Concatenate along first dimension
            X_mixed = cat(1, X1, X2);
            Y_mixed = cat(1, Y1, Y2);
            
            % Shuffle the combined dataset
            [X_mixed, Y_mixed] = syntheticEx.shuffle_data(X_mixed, Y_mixed);
            
            fprintf('  Mixed dataset size: %d samples\n', size(X_mixed, 1));
            fprintf('  Sources per sample: %d (padded to %d)\n', ...
                    max(num_sources1, num_sources2), size(Y_mixed, 2));
            
            % Save if requested
            if save_data
                % Create data directory if it doesn't exist
                data_dir = 'data';
                if ~exist(data_dir, 'dir')
                    try
                        mkdir(data_dir);
                    catch
                        data_dir = '.';
                    end
                end
                
                % Preserve metadata if available
                metadata = struct();
                if isfield(data1, 'metadata')
                    metadata.dataset1 = data1.metadata;
                end
                if isfield(data2, 'metadata')
                    metadata.dataset2 = data2.metadata;
                end
                metadata.num_sources_original = [num_sources1, num_sources2];
                metadata.creation_date = datestr(now);
                metadata.mixed = true;
                metadata.padded = (num_sources1 ~= num_sources2);
                
                filename = fullfile(data_dir, [name '.mat']);
                save(filename, 'X_mixed', 'Y_mixed', 'metadata', '-v7.3');
                fprintf('  Saved to: %s\n', filename);
            end
            
            fprintf('  Mixed dataset creation complete!\n');
        end
        
        
        %*******************************#
        %   create resolution dataset   %
        %*******************************#
        function [X, Y] = create_resolution_dataset(name, size, min_separation, ...
                m, snapshots, snr_db, coherent, array_positions, save_data)
            % Creates dataset for testing resolution capabilities.
            %
            % @param name -- Name of the dataset
            % @param size -- Number of samples
            % @param min_separation -- Minimum angular separation (radians)
            % @param m -- Number of array elements
            % @param snapshots -- Number of snapshots
            % @param snr_db -- Signal-to-noise ratio (dB)
            % @param coherent -- If true, signals are coherent
            % @param array_positions -- Array element positions
            % @param save_data -- If true, save to file
            %
            % @returns -- Measurements and DOAs for two closely spaced sources
            
            % Set defaults
            if nargin < 9
                save_data = true;
            end
            
            if nargin < 8
                array_positions = arrayUtils.ULA_positions(m);  % USING ARRAYUTILS
            end
            
            if nargin < 7
                coherent = false;
            end
            
            if nargin < 6
                snr_db = syntheticEx.DEFAULT_SNR;
            end
            
            if nargin < 5
                snapshots = syntheticEx.DEFAULT_SNAPSHOTS;
            end
            
            if nargin < 4
                m = syntheticEx.DEFAULT_ELEMENTS;
            end
            
            if nargin < 3
                min_separation = deg2rad(5);  % 5 degrees default
            end
            
            % Set random seed USING UTILS
            utils.set_random_seed(42);
            
            % Initialize arrays (2 sources for resolution testing)
            d = 2;
            X = zeros(size, m, snapshots, 'like', 1j);
            Y = zeros(size, d);
            
            fprintf('Creating resolution dataset: %s\n', name);
            fprintf('  Samples: %d, Elements: %d\n', size, m);
            fprintf('  Separation: %.2f rad (%.1f°)\n', ...
                    min_separation, rad2deg(min_separation));
            fprintf('  SNR: %.1f dB, Coherent: %s\n', snr_db, string(coherent));
            
            % Progress display
            progress_step = max(1, floor(size/10));
            
            % Generate samples
            for i = 1:size
                % Random first source direction in [-pi/2, pi/2]
                theta1 = pi * (rand() - 0.5);
                
                % Second source at specified separation
                theta2 = theta1 + min_separation;
                
                % Wrap to [-pi/2, pi/2] range USING UTILS
                theta2 = utils.wrap_error(theta2 - theta1) + theta1;
                
                thetas = [theta1, theta2];
                
                if coherent
                    [measurement, ~] = syntheticEx.construct_coherent_signal(...
                        thetas, m, d, snapshots, snr_db, array_positions);
                else
                    [measurement, ~] = syntheticEx.construct_signal(...
                        thetas, m, d, snapshots, snr_db, array_positions);
                end
                
                % Store
                X(i, :, :) = measurement;
                Y(i, :) = thetas;
                
                % Progress display
                if mod(i, progress_step) == 0
                    fprintf('  Generated %d/%d samples\n', i, size);
                end
            end
            
            % Calculate actual separations USING UTILS
            actual_separations = abs(diff(Y, 1, 2));
            fprintf('  Actual separation range: [%.3f°, %.3f°]\n', ...
                    rad2deg(min(actual_separations)), ...
                    rad2deg(max(actual_separations)));
            
            % Save if requested
            if save_data
                % Create data directory if it doesn't exist
                data_dir = 'data';
                if ~exist(data_dir, 'dir')
                    try
                        mkdir(data_dir);
                    catch
                        data_dir = '.';  % Use current directory
                    end
                end
                
                % Save metadata
                metadata = struct();
                metadata.m = m;
                metadata.d = d;
                metadata.snapshots = snapshots;
                metadata.snr_db = snr_db;
                metadata.coherent = coherent;
                metadata.min_separation = min_separation;
                metadata.array_positions = array_positions;
                metadata.creation_date = datestr(now);
                
                filename = fullfile(data_dir, [name '.mat']);
                save(filename, 'X', 'Y', 'metadata', '-v7.3');
                fprintf('  Saved to: %s\n', filename);
            end
            
            fprintf('  Resolution dataset creation complete!\n');
        end
        
        
        %**************************#
        %   helper functions      %
        %**************************#
        
        function A = steering_matrix(array_positions, thetas)
            % Create steering matrix for given array and DOAs.
            %
            % @param array_positions -- Array element positions [m x 1]
            % @param thetas -- DOA angles [1 x d]
            %
            % @returns -- Steering matrix [m x d]
            
            m = length(array_positions);
            d = length(thetas);
            A = zeros(m, d, 'like', 1j);
            
            for j = 1:d
                A(:, j) = utils.ULA_action_vector(array_positions, thetas(j));  % USING UTILS
            end
        end
        
        function [X_shuffled, Y_shuffled] = shuffle_data(X, Y)
            % Shuffle dataset while maintaining X-Y correspondence.
            %
            % @param X -- Measurements
            % @param Y -- DOAs
            %
            % @returns -- Shuffled X and Y
            
            num_samples = size(X, 1);
            shuffle_idx = randperm(num_samples);
            
            X_shuffled = X(shuffle_idx, :, :);
            Y_shuffled = Y(shuffle_idx, :);
        end
        
        function save_dataset(filename, X, Y, metadata)
            % Save dataset to file USING UTILS approach.
            %
            % @param filename -- Output filename
            % @param X -- Measurements
            % @param Y -- DOAs
            % @param metadata -- Metadata structure
            
            utils.save_data(filename, struct('X', X, 'Y', Y, 'metadata', metadata));  % USING UTILS
        end
        
        function data = load_dataset(filename)
            % Load dataset from file USING UTILS.
            %
            % @param filename -- Input filename
            %
            % @returns -- Loaded data structure
            
            data = utils.load_data(filename);  % USING UTILS
        end
        
        function plot_sample(X, Y, sample_idx, array_positions)
            % Plot a sample from the dataset.
            %
            % @param X -- Measurements
            % @param Y -- DOAs
            % @param sample_idx -- Index of sample to plot
            % @param array_positions -- Array positions
            
            if nargin < 4
                m = size(X, 2);
                array_positions = arrayUtils.ULA_positions(m);  % USING ARRAYUTILS
            end
            
            if sample_idx > size(X, 1)
                error('Sample index out of range');
            end
            
            % Extract sample
            measurement = squeeze(X(sample_idx, :, :));
            doas = Y(sample_idx, :);
            
            % Calculate correlation matrix USING UTILS
            R = utils.correlation_matrix(measurement);  % USING UTILS
            
            % Create figure
            figure('Position', [100, 100, 1200, 400], 'Name', ...
                   sprintf('Dataset Sample %d', sample_idx), 'Color', 'w');
            
            % Plot 1: Array snapshot (first 50 snapshots)
            subplot(1, 3, 1);
            num_snapshots_plot = min(50, size(measurement, 2));
            t = 1:num_snapshots_plot;
            
            plot(t, real(measurement(1, 1:num_snapshots_plot)), 'b-', ...
                 'DisplayName', 'Real (Elem 1)', 'LineWidth', 1.5);
            hold on;
            plot(t, imag(measurement(1, 1:num_snapshots_plot)), 'r-', ...
                 'DisplayName', 'Imag (Elem 1)', 'LineWidth', 1.5);
            plot(t, real(measurement(end, 1:num_snapshots_plot)), 'b--', ...
                 'DisplayName', 'Real (Elem M)', 'LineWidth', 1);
            plot(t, imag(measurement(end, 1:num_snapshots_plot)), 'r--', ...
                 'DisplayName', 'Imag (Elem M)', 'LineWidth', 1);
            
            xlabel('Snapshot Index', 'FontSize', 12);
            ylabel('Amplitude', 'FontSize', 12);
            title('Array Measurements', 'FontSize', 14);
            legend('Location', 'best', 'FontSize', 10);
            grid on;
            
            % Plot 2: Correlation matrix magnitude
            subplot(1, 3, 2);
            imagesc(abs(R));
            colorbar;
            xlabel('Array Element', 'FontSize', 12);
            ylabel('Array Element', 'FontSize', 12);
            title('Correlation Matrix |R|', 'FontSize', 14);
            axis equal tight;
            
            % Plot 3: DOA visualization
            subplot(1, 3, 3);
            theta_range = linspace(-pi/2, pi/2, 361);
            
            % Calculate beampattern USING ARRAYUTILS
            weights = ones(size(array_positions)) / length(array_positions);
            pattern = arrayUtils.array_beampattern(array_positions, weights, theta_range);  % USING ARRAYUTILS
            
            plot(rad2deg(theta_range), 10*log10(pattern/max(pattern)), ...
                 'b-', 'LineWidth', 1.5);
            hold on;
            
            % Mark true DOAs
            for i = 1:length(doas)
                plot(rad2deg([doas(i), doas(i)]), [-30, 0], 'r--', ...
                     'LineWidth', 1.5, 'DisplayName', 'True DOA');
            end
            
            xlabel('Angle (degrees)', 'FontSize', 12);
            ylabel('Normalized Response (dB)', 'FontSize', 12);
            title('Array Beampattern with DOAs', 'FontSize', 14);
            grid on;
            xlim([-90, 90]);
            ylim([-30, 5]);
            
            % Add info text
            info_str = sprintf('Sample %d:\n', sample_idx);
            info_str = [info_str sprintf('DOAs: ')];
            for i = 1:length(doas)
                info_str = [info_str sprintf('%.1f° ', rad2deg(doas(i)))];
            end
            info_str = [info_str sprintf('\nElements: %d\nSnapshots: %d', ...
                         size(measurement, 1), size(measurement, 2))];
            
            annotation('textbox', [0.02, 0.98, 0.3, 0.05], ...
                       'String', info_str, 'FontSize', 10, ...
                       'EdgeColor', 'none', 'BackgroundColor', [0.9, 0.9, 0.9]);
        end
        
    end
end