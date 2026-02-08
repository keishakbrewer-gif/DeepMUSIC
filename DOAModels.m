classdef DOAModels
    % DOAMODELS Neural network architectures for augmented MUSIC algorithm
    % This class contains various neural network models for DOA estimation
    
    properties(Constant)
        % Default architecture parameters (can be overridden)
        DEFAULT_M = 8;          % Number of array elements
        DEFAULT_D = 2;          % Number of sources
        DEFAULT_SNAPSHOTS = 200; % Number of snapshots
        DEFAULT_ANGLES = 181;   % Angle grid size (resolution)
    end
    
    methods(Static)
        
        %******************************************************%
        %   Helper method to get noise subspace dimension      %
        %******************************************************%
        function n = get_noise_subspace_dim(m, d)
            % Calculate dimension of noise subspace
            %
            % @param m -- Number of array elements
            % @param d -- Number of sources
            %
            % @returns -- Dimension of noise subspace
            if nargin < 1
                m = DOAModels.DEFAULT_M;
            end
            if nargin < 2
                d = DOAModels.DEFAULT_D;
            end
            n = m - d;
        end
        
        
        %******************************************************%
        %   Simple CNN Model                                  %
        %******************************************************%
        function [lgraph, inputName, outputName] = create_model_simpleCNN(m, d, snapshots)
            % Simple CNN model for noise subspace estimation
            %
            % @param m -- Number of array elements (default: 8)
            % @param d -- Number of sources (default: 2)
            % @param snapshots -- Number of snapshots (default: 200)
            %
            % @returns -- Layer graph, input layer name, output layer name
            
            % Set defaults
            if nargin < 1
                m = DOAModels.DEFAULT_M;
            end
            if nargin < 2
                d = DOAModels.DEFAULT_D;
            end
            if nargin < 3
                snapshots = DOAModels.DEFAULT_SNAPSHOTS;
            end
            
            n = DOAModels.get_noise_subspace_dim(m, d);  % Noise subspace dimension
            
            % Input layer
            input_layer = imageInputLayer([2*m, snapshots, 1], 'Name', 'input', 'Normalization', 'none');
            
            % Create layer array
            layers = [
                input_layer
                flattenLayer('Name', 'flatten1')
                batchNormalizationLayer('Name', 'batch_norm1')
                
                % Feature extraction layers
                fullyConnectedLayer(1024, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                dropoutLayer(0.2, 'Name', 'dropout1')
                
                fullyConnectedLayer(1024, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                dropoutLayer(0.2, 'Name', 'dropout2')
                
                fullyConnectedLayer(1024, 'Name', 'fc3')
                reluLayer('Name', 'relu3')
                dropoutLayer(0.2, 'Name', 'dropout3')
                
                % Final output: noise subspace coefficients
                fullyConnectedLayer(2*m*n, 'Name', 'output_fc')
                ];
            
            % For older MATLAB versions without reshapeLayer,
            % we'll use a custom function layer or process outside network
            % Add a custom function layer to reshape
            layers = [
                layers
                functionLayer(@(x) reshape(x, 2*m, n, []), 'Name', 'reshape_output')
                ];
            
            % Create layer graph
            lgraph = layerGraph(layers);
            inputName = 'input';
            outputName = 'reshape_output';
        end
        
        
        %******************************************************%
        %   CNN with Eigenvalue Decomposition Model          %
        %******************************************************%
        function [lgraph, inputName, outputName] = create_model_CNNwithEVD(m, d, snapshots)
            % CNN model with eigenvalue decomposition
            %
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param snapshots -- Number of snapshots
            %
            % @returns -- Layer graph, input layer name, output layer name
            
            if nargin < 1
                m = DOAModels.DEFAULT_M;
            end
            if nargin < 2
                d = DOAModels.DEFAULT_D;
            end
            if nargin < 3
                snapshots = DOAModels.DEFAULT_SNAPSHOTS;
            end
            
            n = DOAModels.get_noise_subspace_dim(m, d);
            
            % Create layers
            layers = [
                imageInputLayer([2*m, snapshots, 1], 'Name', 'input', 'Normalization', 'none')
                flattenLayer('Name', 'flatten1')
                batchNormalizationLayer('Name', 'batch_norm1')
                
                % Feature extraction layers
                fullyConnectedLayer(1024, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                dropoutLayer(0.2, 'Name', 'dropout1')
                
                fullyConnectedLayer(1024, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                dropoutLayer(0.2, 'Name', 'dropout2')
                
                fullyConnectedLayer(1024, 'Name', 'fc3')
                reluLayer('Name', 'relu3')
                dropoutLayer(0.2, 'Name', 'dropout3')
                
                batchNormalizationLayer('Name', 'batch_norm2')
                
                % Output: predicted covariance matrix (real and imag stacked)
                fullyConnectedLayer(2*m*m, 'Name', 'cov_output')
                functionLayer(@(x) reshape(x, 2*m, m, []), 'Name', 'reshape_cov')
                ];
            
            lgraph = layerGraph(layers);
            inputName = 'input';
            outputName = 'reshape_cov';
        end
        
        
        %******************************************************%
        %   End-to-End Simple CNN Model                       %
        %******************************************************%
        function [lgraph, inputName, outputName] = create_model_E2E_simpleCNN(m, d, snapshots)
            % End-to-end simple CNN model for direct DOA estimation
            %
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param snapshots -- Number of snapshots
            %
            % @returns -- Layer graph, input layer name, output layer name
            
            if nargin < 1
                m = DOAModels.DEFAULT_M;
            end
            if nargin < 2
                d = DOAModels.DEFAULT_D;
            end
            if nargin < 3
                snapshots = DOAModels.DEFAULT_SNAPSHOTS;
            end
            
            layers = [
                imageInputLayer([2*m, snapshots, 1], 'Name', 'input', 'Normalization', 'none')
                flattenLayer('Name', 'flatten1')
                batchNormalizationLayer('Name', 'batch_norm1')
                
                % Feature extraction
                fullyConnectedLayer(1024, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                dropoutLayer(0.2, 'Name', 'dropout1')
                
                fullyConnectedLayer(1024, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                dropoutLayer(0.2, 'Name', 'dropout2')
                
                fullyConnectedLayer(1024, 'Name', 'fc3')
                reluLayer('Name', 'relu3')
                dropoutLayer(0.2, 'Name', 'dropout3')
                
                batchNormalizationLayer('Name', 'batch_norm2')
                
                flattenLayer('Name', 'flatten2')
                
                % Direct DOA estimation (d angles)
                fullyConnectedLayer(d, 'Name', 'doa_output')
                regressionLayer('Name', 'output')  % For regression task
                ];
            
            lgraph = layerGraph(layers);
            inputName = 'input';
            outputName = 'output';
        end
        
        
        %******************************************************%
        %   RNN Model (GRU-based)                            %
        %******************************************************%
        function [lgraph, inputName, outputName] = create_model_RNN(m, d, snapshots)
            % RNN model with GRU for sequence processing
            %
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param snapshots -- Number of snapshots
            %
            % @returns -- Layer graph, input layer name, output layer name
            
            if nargin < 1
                m = DOAModels.DEFAULT_M;
            end
            if nargin < 2
                d = DOAModels.DEFAULT_D;
            end
            if nargin < 3
                snapshots = DOAModels.DEFAULT_SNAPSHOTS;
            end
            
            n = DOAModels.get_noise_subspace_dim(m, d);
            
            layers = [
                % Input as sequence (features x sequence_length)
                sequenceInputLayer(2*m, 'Name', 'input', 'Normalization', 'none')
                batchNormalizationLayer('Name', 'batch_norm1')
                
                % GRU layer for sequence processing
                gruLayer(2*m*m, 'Name', 'gru1', 'OutputMode', 'last')
                
                % Reshape to predicted covariance matrix
                fullyConnectedLayer(2*m*m, 'Name', 'fc_reshape')
                functionLayer(@(x) reshape(x, 2*m, m, []), 'Name', 'reshape_cov')
                ];
            
            lgraph = layerGraph(layers);
            inputName = 'input';
            outputName = 'reshape_cov';
        end
        
        
        %******************************************************%
        %   End-to-End RNN Model                              %
        %******************************************************%
        function [lgraph, inputName, outputName] = create_model_E2E_RNN(m, d, snapshots)
            % End-to-end RNN model for direct DOA estimation
            %
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param snapshots -- Number of snapshots
            %
            % @returns -- Layer graph, input layer name, output layer name
            
            if nargin < 1
                m = DOAModels.DEFAULT_M;
            end
            if nargin < 2
                d = DOAModels.DEFAULT_D;
            end
            if nargin < 3
                snapshots = DOAModels.DEFAULT_SNAPSHOTS;
            end
            
            layers = [
                sequenceInputLayer(2*m, 'Name', 'input', 'Normalization', 'none')
                batchNormalizationLayer('Name', 'batch_norm1')
                
                % GRU layer
                gruLayer(2*m*m, 'Name', 'gru1', 'OutputMode', 'last')
                
                % Direct DOA estimation
                fullyConnectedLayer(d, 'Name', 'doa_output')
                regressionLayer('Name', 'output')
                ];
            
            lgraph = layerGraph(layers);
            inputName = 'input';
            outputName = 'output';
        end
        
        
        %******************************************************%
        %   Deep RNN Model (multiple GRU layers)              %
        %******************************************************%
        function [lgraph, inputName, outputName] = create_model_deepRNN(m, d, snapshots)
            % Deep RNN model with multiple GRU layers
            %
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param snapshots -- Number of snapshots
            %
            % @returns -- Layer graph, input layer name, output layer name
            
            if nargin < 1
                m = DOAModels.DEFAULT_M;
            end
            if nargin < 2
                d = DOAModels.DEFAULT_D;
            end
            if nargin < 3
                snapshots = DOAModels.DEFAULT_SNAPSHOTS;
            end
            
            n = DOAModels.get_noise_subspace_dim(m, d);
            
            layers = [
                sequenceInputLayer(2*m, 'Name', 'input', 'Normalization', 'none')
                batchNormalizationLayer('Name', 'batch_norm1')
                
                % Multiple GRU layers
                gruLayer(2*m, 'Name', 'gru1', 'OutputMode', 'sequence')
                gruLayer(2*m, 'Name', 'gru2', 'OutputMode', 'sequence')
                gruLayer(2*m, 'Name', 'gru3', 'OutputMode', 'last')
                
                % Predict covariance matrix
                fullyConnectedLayer(2*m*m, 'Name', 'fc_cov')
                functionLayer(@(x) reshape(x, 2*m, m, []), 'Name', 'reshape_cov')
                ];
            
            lgraph = layerGraph(layers);
            inputName = 'input';
            outputName = 'reshape_cov';
        end
        
        
        %******************************************************%
        %   Simple RNN Model (no EVD)                        %
        %******************************************************%
        function [lgraph, inputName, outputName] = create_model_simpleRNN_noEVD(m, d, snapshots)
            % Simple RNN model without eigenvalue decomposition
            %
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param snapshots -- Number of snapshots
            %
            % @returns -- Layer graph, input layer name, output layer name
            
            if nargin < 1
                m = DOAModels.DEFAULT_M;
            end
            if nargin < 2
                d = DOAModels.DEFAULT_D;
            end
            if nargin < 3
                snapshots = DOAModels.DEFAULT_SNAPSHOTS;
            end
            
            n = DOAModels.get_noise_subspace_dim(m, d);
            
            layers = [
                sequenceInputLayer(2*m, 'Name', 'input', 'Normalization', 'none')
                batchNormalizationLayer('Name', 'batch_norm1')
                
                % Single GRU layer
                gruLayer(2*m, 'Name', 'gru1', 'OutputMode', 'last')
                
                % Direct noise subspace prediction
                fullyConnectedLayer(2*m*n, 'Name', 'fc_noise')
                functionLayer(@(x) reshape(x, 2*m, n, []), 'Name', 'reshape_output')
                ];
            
            lgraph = layerGraph(layers);
            inputName = 'input';
            outputName = 'reshape_output';
        end
        
        
        %******************************************************%
        %   Alternative Model (from paper)                   %
        %******************************************************%
        function [lgraph, inputName, outputName] = create_model_alternative(m, d, snapshots)
            % Alternative model based on "DEEP AUGMENTED MUSIC ALGORITHM"
            %
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param snapshots -- Number of snapshots
            %
            % @returns -- Layer graph, input layer name, output layer name
            
            if nargin < 1
                m = DOAModels.DEFAULT_M;
            end
            if nargin < 2
                d = DOAModels.DEFAULT_D;
            end
            if nargin < 3
                snapshots = DOAModels.DEFAULT_SNAPSHOTS;
            end
            
            layers = [
                sequenceInputLayer(2*m, 'Name', 'input', 'Normalization', 'none')
                batchNormalizationLayer('Name', 'batch_norm1')
                
                % GRU layer
                gruLayer(2*m, 'Name', 'gru1', 'OutputMode', 'last')
                
                % Predict covariance matrix
                fullyConnectedLayer(2*m*m, 'Name', 'fc_cov')
                functionLayer(@(x) reshape(x, 2*m, m, []), 'Name', 'reshape_cov')
                
                % Additional processing layers
                fullyConnectedLayer(2*m, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                fullyConnectedLayer(2*m, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(2*m, 'Name', 'fc3')
                reluLayer('Name', 'relu3')
                
                % Final DOA estimation
                fullyConnectedLayer(d, 'Name', 'doa_output')
                regressionLayer('Name', 'output')
                ];
            
            lgraph = layerGraph(layers);
            inputName = 'input';
            outputName = 'output';
        end
        
        
        %******************************************************%
        %   Model Summary and Selection                      %
        %******************************************************%
        function summary = get_model_summary()
            % Get summary of available models
            %
            % @returns -- Structure array with model information
            
            summary = [
                struct('name', 'simpleCNN', ...
                       'description', 'Simple CNN for noise subspace estimation', ...
                       'requires_evd', false, ...
                       'e2e', false), ...
                struct('name', 'CNNwithEVD', ...
                       'description', 'CNN with eigenvalue decomposition', ...
                       'requires_evd', true, ...
                       'e2e', false), ...
                struct('name', 'E2E_simpleCNN', ...
                       'description', 'End-to-end simple CNN for direct DOA', ...
                       'requires_evd', false, ...
                       'e2e', true), ...
                struct('name', 'RNN', ...
                       'description', 'RNN (GRU) model', ...
                       'requires_evd', true, ...
                       'e2e', false), ...
                struct('name', 'E2E_RNN', ...
                       'description', 'End-to-end RNN for direct DOA', ...
                       'requires_evd', false, ...
                       'e2e', true), ...
                struct('name', 'deepRNN', ...
                       'description', 'Deep RNN with multiple GRU layers', ...
                       'requires_evd', true, ...
                       'e2e', false), ...
                struct('name', 'simpleRNN_noEVD', ...
                       'description', 'Simple RNN without EVD', ...
                       'requires_evd', false, ...
                       'e2e', false), ...
                struct('name', 'alternative', ...
                       'description', 'Alternative model from paper', ...
                       'requires_evd', true, ...
                       'e2e', false)
                ];
        end
        
        
        %******************************************************%
        %   Create Model by Name                             %
        %******************************************************%
        function [lgraph, inputName, outputName] = create_model_by_name(model_name, m, d, snapshots)
            % Create model by name
            %
            % @param model_name -- Name of the model to create
            % @param m -- Number of array elements
            % @param d -- Number of sources
            % @param snapshots -- Number of snapshots
            %
            % @returns -- Layer graph, input layer name, output layer name
            
            if nargin < 2
                m = DOAModels.DEFAULT_M;
            end
            if nargin < 3
                d = DOAModels.DEFAULT_D;
            end
            if nargin < 4
                snapshots = DOAModels.DEFAULT_SNAPSHOTS;
            end
            
            % Map model names to creation functions
            switch lower(model_name)
                case 'simplecnn'
                    [lgraph, inputName, outputName] = ...
                        DOAModels.create_model_simpleCNN(m, d, snapshots);
                case 'cnnwithevd'
                    [lgraph, inputName, outputName] = ...
                        DOAModels.create_model_CNNwithEVD(m, d, snapshots);
                case 'e2e_simplecnn'
                    [lgraph, inputName, outputName] = ...
                        DOAModels.create_model_E2E_simpleCNN(m, d, snapshots);
                case 'rnn'
                    [lgraph, inputName, outputName] = ...
                        DOAModels.create_model_RNN(m, d, snapshots);
                case 'e2e_rnn'
                    [lgraph, inputName, outputName] = ...
                        DOAModels.create_model_E2E_RNN(m, d, snapshots);
                case 'deeprnn'
                    [lgraph, inputName, outputName] = ...
                        DOAModels.create_model_deepRNN(m, d, snapshots);
                case 'simplernn_noevd'
                    [lgraph, inputName, outputName] = ...
                        DOAModels.create_model_simpleRNN_noEVD(m, d, snapshots);
                case 'alternative'
                    [lgraph, inputName, outputName] = ...
                        DOAModels.create_model_alternative(m, d, snapshots);
                otherwise
                    error('Unknown model name: %s', model_name);
            end
            
            fprintf('Created model: %s\n', model_name);
            fprintf('  Input size: [%d, %d]\n', 2*m, snapshots);
            fprintf('  Output size: varies by model\n');
        end
        
        
        %******************************************************%
        %   Test Model Creation                              %
        %******************************************************%
        function test_model_creation()
            % Test function to verify all models can be created
            
            fprintf('Testing DOA model creation...\n\n');
            
            % Test parameters
            m = 8;
            d = 2;
            snapshots = 200;
            
            % Get model summary
            summary = DOAModels.get_model_summary();
            
            for i = 1:length(summary)
                model_info = summary(i);
                fprintf('Testing model: %s\n', model_info.name);
                fprintf('  Description: %s\n', model_info.description);
                
                try
                    % Create model
                    [lgraph, inputName, outputName] = ...
                        DOAModels.create_model_by_name(model_info.name, m, d, snapshots);
                    
                    % Display layer summary
                    fprintf('  Successfully created layer graph with %d layers\n', ...
                            numel(lgraph.Layers));
                    fprintf('  Input: %s, Output: %s\n\n', inputName, outputName);
                    
                catch ME
                    fprintf('  ERROR creating model: %s\n\n', ME.message);
                end
            end
            
            fprintf('Model creation test complete.\n');
        end
        
        
        %******************************************************%
        %   Custom Function Layer Helper                     %
        %******************************************************%
        function reshaped = custom_reshape(x, output_shape)
            % Custom reshape function for older MATLAB versions
            %
            % @param x -- Input tensor
            % @param output_shape -- Desired shape [rows, cols, batch_size]
            %
            % @returns -- Reshaped tensor
            
            if ndims(x) == 2
                % For 2D input (batch_size x features)
                batch_size = size(x, 1);
                reshaped = reshape(x', [output_shape(1), output_shape(2), batch_size]);
            else
                % Already in correct format or 3D
                reshaped = reshape(x, output_shape);
            end
        end
        
        
        %******************************************************%
        %   Complex Number Processing                        %
        %******************************************************%
        function [real_part, imag_part] = separate_complex_channels(data)
            % Separate complex data into real and imaginary channels
            %
            % @param data -- Complex data [batch, features, ...]
            %
            % @returns -- Real part and imaginary part
            
            real_part = real(data);
            imag_part = imag(data);
        end
        
        
        function complex_data = combine_complex_channels(real_part, imag_part)
            % Combine real and imaginary parts into complex data
            %
            % @param real_part -- Real part
            % @param imag_part -- Imaginary part
            %
            % @returns -- Complex data
            
            complex_data = complex(real_part, imag_part);
        end
        
    end
end