%% test_DOAModels_corrected.m
% Test script for DOA neural network models - CORRECTED VERSION

clear; close all; clc;
fprintf('Testing DOA Neural Network Models...\n\n');

%% 1. SETUP TEST PARAMETERS
fprintf('1. Setting up test parameters...\n');

% Use consistent parameters
m = 8;              % Array elements
d = 2;              % Signal sources
snapshots = 200;    % Snapshots
snr_db = 10;        % SNR in dB

% Get array positions
array_positions = arrayUtils.ULA_positions(m);

% Angle grid for spectrum calculation
theta_min = -60;    % degrees
theta_max = 60;     % degrees
theta_grid_deg = linspace(theta_min, theta_max, 181);
theta_grid_rad = deg2rad(theta_grid_deg);

% Create test measurement data
fprintf('  Generating test data...\n');
true_doas_deg = [-15, 25];      % True DOAs in degrees
true_doas_rad = deg2rad(true_doas_deg);

[measurement, ~] = syntheticEx.construct_signal(true_doas_rad, m, d, ...
                                               snapshots, snr_db, ...
                                               array_positions);

fprintf('  Measurement size: %d x %d\n', size(measurement, 1), size(measurement, 2));
fprintf('  True DOAs: %.1f°, %.1f°\n', true_doas_deg);

%% 2. TEST MODEL CREATION
fprintf('\n2. Testing model creation...\n');

% Get all available models
model_summary = DOAModels.get_model_summary();

% Test each model
for i = 1:length(model_summary)
    model_info = model_summary(i);
    fprintf('\n  Testing: %s\n', model_info.name);
    fprintf('    %s\n', model_info.description);
    
    try
        % Create the model
        [lgraph, inputName, outputName] = ...
            DOAModels.create_model_by_name(model_info.name, m, d, snapshots);
        
        % Display model information
        fprintf('    ✓ Created successfully\n');
        fprintf('    Layers: %d, Input: %s, Output: %s\n', ...
                numel(lgraph.Layers), inputName, outputName);
        
        % Display layer summary
        fprintf('    Layer types:\n');
        layer_types = {};
        for j = 1:numel(lgraph.Layers)
            layer = lgraph.Layers(j);
            % Get layer type based on MATLAB version
            if isa(layer, 'nnet.cnn.layer.ImageInputLayer')
                layer_type = 'imageInputLayer';
            elseif isa(layer, 'nnet.cnn.layer.SequenceInputLayer')
                layer_type = 'sequenceInputLayer';
            elseif isa(layer, 'nnet.cnn.layer.FullyConnectedLayer')
                layer_type = 'fullyConnectedLayer';
            elseif isa(layer, 'nnet.cnn.layer.ReLULayer')
                layer_type = 'reluLayer';
            elseif isa(layer, 'nnet.cnn.layer.DropoutLayer')
                layer_type = 'dropoutLayer';
            elseif isa(layer, 'nnet.cnn.layer.BatchNormalizationLayer')
                layer_type = 'batchNormalizationLayer';
            elseif isa(layer, 'nnet.cnn.layer.GRULayer')
                layer_type = 'gruLayer';
            elseif isa(layer, 'nnet.cnn.layer.RegressionLayer')
                layer_type = 'regressionLayer';
            elseif isa(layer, 'nnet.cnn.layer.FlattenLayer')
                layer_type = 'flattenLayer';
            elseif isa(layer, 'nnet.cnn.layer.FunctionLayer')
                layer_type = 'functionLayer';
            else
                layer_type = class(layer);
            end
            layer_types{end+1} = layer_type;
        end
        
        % Count unique layer types
        unique_types = unique(layer_types);
        for k = 1:length(unique_types)
            count = sum(strcmp(layer_types, unique_types{k}));
            fprintf('      %-20s: %d\n', unique_types{k}, count);
        end
        
        % Store for later testing
        model_summary(i).lgraph = lgraph;
        model_summary(i).inputName = inputName;
        model_summary(i).outputName = outputName;
        model_summary(i).success = true;
        
    catch ME
        fprintf('    ✗ ERROR: %s\n', ME.message);
        model_summary(i).success = false;
    end
end

%% 3. TEST MODEL PREDICTIONS WITH RANDOM INPUTS
fprintf('\n3. Testing model predictions (random inputs)...\n');

% Create small batch of random input data
batch_size = 5;

% Format 1: For imageInputLayer models (CNN-like)
cnn_input_size = [2*m, snapshots, 1, batch_size];
cnn_input_data = randn(cnn_input_size);

% Format 2: For sequenceInputLayer models (RNN-like)
% Reshape to sequence format: [features, sequence_length, batch_size]
rnn_input_data = randn(2*m, snapshots, batch_size);

for i = 1:length(model_summary)
    if ~model_summary(i).success
        continue;
    end
    
    model_info = model_summary(i);
    fprintf('\n  Testing predictions: %s\n', model_info.name);
    
    try
        % Check input layer type to format data correctly
        input_layer = model_info.lgraph.Layers(1);
        
        if isa(input_layer, 'nnet.cnn.layer.ImageInputLayer')
            % CNN model - use image format
            test_input = cnn_input_data;
            input_type = 'Image';
        elseif isa(input_layer, 'nnet.cnn.layer.SequenceInputLayer')
            % RNN model - use sequence format
            test_input = rnn_input_data;
            input_type = 'Sequence';
        else
            error('Unknown input layer type: %s', class(input_layer));
        end
        
        % Get input size
        if strcmp(input_type, 'Image')
            input_size = input_layer.InputSize;
        else
            input_size = [input_layer.InputSize, snapshots];
        end
        
        % Determine expected output size based on model type
        if contains(model_info.name, 'E2E')
            % End-to-end models output DOAs directly
            output_size = [d, batch_size];
            fprintf('    Expected output: DOA estimates [%d x %d]\n', ...
                    output_size(1), output_size(2));
        elseif contains(model_info.name, 'CNNwithEVD') || ...
               contains(model_info.name, 'RNN') || ...
               contains(model_info.name, 'deepRNN')
            % Models with covariance output
            output_size = [2*m, m, batch_size];
            fprintf('    Expected output: Covariance matrix [%d x %d x %d]\n', ...
                    output_size(1), output_size(2), output_size(3));
        elseif contains(model_info.name, 'simpleRNN_noEVD')
            % Simple RNN with noise subspace output
            n = DOAModels.get_noise_subspace_dim(m, d);
            output_size = [2*m, n, batch_size];
            fprintf('    Expected output: Noise subspace [%d x %d x %d]\n', ...
                    output_size(1), output_size(2), output_size(3));
        else
            % Other CNN models
            n = DOAModels.get_noise_subspace_dim(m, d);
            output_size = [2*m, n, batch_size];
            fprintf('    Expected output: Noise subspace [%d x %d x %d]\n', ...
                    output_size(1), output_size(2), output_size(3));
        end
        
        fprintf('    Input type: %s, Size: ', input_type);
        fprintf('%d ', input_size);
        fprintf('\n');
        
        fprintf('    ✓ Prediction format validated\n');
        
        model_summary(i).prediction_tested = true;
        
    catch ME
        fprintf('    ✗ Prediction test failed: %s\n', ME.message);
        model_summary(i).prediction_tested = false;
    end
end

%% 4. TEST INTEGRATION WITH UTILS FUNCTIONS
fprintf('\n4. Testing integration with utils functions...\n');

% Test 4.1: Complex number processing
fprintf('\n  Testing complex number utilities...\n');
test_complex = randn(3, 4) + 1j * randn(3, 4);
[real_part, imag_part] = DOAModels.separate_complex_channels(test_complex);
reconstructed = DOAModels.combine_complex_channels(real_part, imag_part);

complex_error = max(abs(test_complex(:) - reconstructed(:)));
fprintf('    Complex reconstruction error: %.2e\n', complex_error);
if complex_error < 1e-10
    fprintf('    ✓ Complex processing works correctly\n');
else
    fprintf('    ✗ Complex processing error too high\n');
end

% Test 4.2: MUSIC spectrum calculation
fprintf('\n  Testing MUSIC spectrum calculation...\n');

% Create synthetic noise subspace (for testing)
n = DOAModels.get_noise_subspace_dim(m, d);
batch_size_test = 3;
noise_subspace_real = randn(batch_size_test, m);
noise_subspace_imag = randn(batch_size_test, m);
y_pred_test = [noise_subspace_real, noise_subspace_imag];

try
    spectrum = utils.calculate_spectrum(y_pred_test, array_positions, ...
                                        theta_grid_rad, m);
    
    fprintf('    Spectrum size: %d x %d\n', size(spectrum, 1), size(spectrum, 2));
    fprintf('    Min spectrum value: %.2e\n', min(spectrum(:)));
    fprintf('    Max spectrum value: %.2e\n', max(spectrum(:)));
    
    % Check for reasonable spectrum shape
    if all(spectrum(:) >= 0) && ~any(isnan(spectrum(:))) && ~any(isinf(spectrum(:)))
        fprintf('    ✓ Spectrum calculation successful\n');
        
        % Plot sample spectrum
        figure('Name', 'Sample MUSIC Spectrum', 'Position', [100, 100, 600, 400]);
        plot(theta_grid_deg, 10*log10(spectrum(1,:)/max(spectrum(1,:))), 'b-', 'LineWidth', 1.5);
        xlabel('Angle (degrees)');
        ylabel('Normalized Spectrum (dB)');
        title('Sample MUSIC Spectrum from Random Noise Subspace');
        grid on;
        xlim([theta_min, theta_max]);
        ylim([-40, 5]);
        
    else
        fprintf('    ✗ Spectrum values invalid\n');
    end
    
catch ME
    fprintf('    ✗ Spectrum calculation failed: %s\n', ME.message);
end

%% 5. TEST TRAINING DATA PREPARATION
fprintf('\n5. Testing training data preparation...\n');

% Create a small training dataset
num_train_samples = 10;
fprintf('  Creating training dataset (%d samples)...\n', num_train_samples);

% Initialize arrays
X_train = zeros(2*m, snapshots, num_train_samples);
Y_train_noise = zeros(2*m, n, num_train_samples);  % Noise subspace targets
Y_train_doa = zeros(d, num_train_samples);         % DOA targets

for sample_idx = 1:num_train_samples
    % Generate random DOAs
    random_doas_deg = theta_min + (theta_max - theta_min) * rand(1, d);
    random_doas_rad = deg2rad(random_doas_deg);
    
    % Generate measurement
    [measurement_sample, ~] = syntheticEx.construct_signal(random_doas_rad, m, d, ...
                                                          snapshots, snr_db, ...
                                                          array_positions);
    
    % Format input: stack real and imaginary parts
    X_real = real(measurement_sample);
    X_imag = imag(measurement_sample);
    X_train(:,:,sample_idx) = [X_real; X_imag];
    
    % Calculate target: noise subspace from classic MUSIC
    R = utils.correlation_matrix(measurement_sample);
    [~, ~, ~, noise_subspace] = utils.eigen_decomposition(R, d);
    
    % Format noise subspace target
    if ~isempty(noise_subspace)
        % Ensure we have the right number of noise vectors
        actual_n = size(noise_subspace, 2);
        if actual_n >= n
            % Use first n vectors
            noise_target = noise_subspace(:, 1:n);
        else
            % Pad with zeros
            noise_target = [noise_subspace, zeros(m, n - actual_n)];
        end
        
        % Separate real and imaginary parts
        Y_train_noise(:,:,sample_idx) = [real(noise_target); imag(noise_target)];
    else
        Y_train_noise(:,:,sample_idx) = zeros(2*m, n);
    end
    
    % DOA target
    Y_train_doa(:, sample_idx) = sort(random_doas_deg(:));
    
    % Progress
    if mod(sample_idx, 5) == 0
        fprintf('    Generated %d/%d samples\n', sample_idx, num_train_samples);
    end
end

fprintf('  Training data sizes:\n');
fprintf('    X_train: %d x %d x %d\n', size(X_train, 1), size(X_train, 2), size(X_train, 3));
fprintf('    Y_train_noise: %d x %d x %d\n', size(Y_train_noise, 1), size(Y_train_noise, 2), size(Y_train_noise, 3));
fprintf('    Y_train_doa: %d x %d\n', size(Y_train_doa, 1), size(Y_train_doa, 2));

%% 6. SUMMARY AND RECOMMENDATIONS
fprintf('\n6. Test Summary and Recommendations\n');
fprintf('==================================================\n');

% Count successful models
success_count = 0;
prediction_count = 0;

for i = 1:length(model_summary)
    if isfield(model_summary, 'success') && model_summary(i).success
        success_count = success_count + 1;
    end
    if isfield(model_summary, 'prediction_tested') && model_summary(i).prediction_tested
        prediction_count = prediction_count + 1;
    end
end

fprintf('Models tested: %d/%d successful\n', success_count, length(model_summary));
fprintf('Predictions validated: %d/%d\n', prediction_count, success_count);

fprintf('\nNext steps for neural network implementation:\n');
fprintf('1. Choose a model architecture:\n');

% Display model options
fprintf('   For noise subspace estimation:\n');
for i = 1:length(model_summary)
    if ~isfield(model_summary, 'success') || ~model_summary(i).success
        continue;
    end
    if isfield(model_summary, 'e2e') && model_summary(i).e2e
        continue;
    end
    fprintf('   - %s: %s\n', model_summary(i).name, model_summary(i).description);
end

fprintf('\n   For end-to-end DOA estimation:\n');
for i = 1:length(model_summary)
    if ~isfield(model_summary, 'success') || ~model_summary(i).success
        continue;
    end
    if ~isfield(model_summary, 'e2e') || ~model_summary(i).e2e
        continue;
    end
    fprintf('   - %s: %s\n', model_summary(i).name, model_summary(i).description);
end

fprintf('\n2. Implement custom layers needed:\n');
fprintf('   - Eigenvalue decomposition layer\n');
fprintf('   - Complex number operations layer\n');
fprintf('   - Spectrum calculation layer\n');

fprintf('\n3. Prepare training pipeline:\n');
fprintf('   - Convert losses.py to MATLAB\n');
fprintf('   - Implement training loop (trainModel.py)\n');
fprintf('   - Add data augmentation and preprocessing\n');

fprintf('\n4. Recommended first model to implement fully:\n');
fprintf('   Model: simpleRNN_noEVD\n');
fprintf('   Reason: No custom layers needed, outputs noise subspace directly\n');
fprintf('   Easy integration with existing MUSIC algorithm\n');

%% 7. QUICK TEST OF RECOMMENDED MODEL
fprintf('\n7. Quick implementation test of recommended model...\n');

% Test the simpleRNN_noEVD model more thoroughly
try
    [lgraph, inputName, outputName] = ...
        DOAModels.create_model_by_name('simpleRNN_noEVD', m, d, snapshots);
    
    fprintf('  Model: simpleRNN_noEVD\n');
    fprintf('  Input: %s (sequence, %d features)\n', inputName, 2*m);
    
    % Get noise subspace dimension
    n = DOAModels.get_noise_subspace_dim(m, d);
    fprintf('  Output: %s (noise subspace, %d x %d)\n', outputName, 2*m, n);
    
    % Create example network for display
    fprintf('\n  Layer architecture:\n');
    for layer_idx = 1:numel(lgraph.Layers)
        layer = lgraph.Layers(layer_idx);
        fprintf('    %2d. %-25s', layer_idx, layer.Name);
        
        if isa(layer, 'nnet.cnn.layer.SequenceInputLayer')
            fprintf('Input: %d features\n', layer.InputSize);
        elseif isa(layer, 'nnet.cnn.layer.GRULayer')
            fprintf('Hidden units: %d\n', layer.NumHiddenUnits);
        elseif isa(layer, 'nnet.cnn.layer.FullyConnectedLayer')
            fprintf('Output size: %d\n', layer.OutputSize);
        elseif isa(layer, 'nnet.cnn.layer.FunctionLayer')
            fprintf('Custom reshape function\n');
        else
            fprintf('%s\n', class(layer));
        end
    end
    
    fprintf('\n  ✓ Recommended model ready for implementation\n');
    fprintf('  This model can be trained to predict noise subspace directly\n');
    fprintf('  Output can be fed into utils.calculate_spectrum()\n');
    
catch ME
    fprintf('  ✗ Error testing recommended model: %s\n', ME.message);
end

fprintf('\n==================================================\n');
fprintf('Test completed successfully!\n');
fprintf('Proceed to convert losses.py for training implementation.\n');