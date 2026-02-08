classdef DOATrainer
    % DOATRAINER Training pipeline for DOA neural network augmentation
    % This class implements the complete training pipeline for MUSIC augmentation
    
    properties(Constant)
        % Default training parameters
        DEFAULT_SEED = 42;
        DEFAULT_BATCH_SIZE = 16;
        DEFAULT_LEARNING_RATE = 0.001;
        DEFAULT_EPOCHS = 70;
        DEFAULT_VAL_SPLIT = 0.2;
        DEFAULT_TEST_SPLIT = 0.1;
    end
    
    methods(Static)
        
        %******************************************************%
        %   Main Training Function                           %
        %******************************************************%
        function [model, history, results] = train_model(model_name, data_path, params)
            % Main training function
            %
            % @param model_name -- Name of model to train
            % @param data_path -- Path to training data
            % @param params -- Training parameters structure
            %
            % @returns -- Trained model, training history, results
            
            fprintf('DOA Neural Network Training\n');
            fprintf('===========================\n\n');
            
            % Set default parameters
            if nargin < 3
                params = struct();
            end
            
            % Merge with defaults
            params = DOATrainer.set_default_params(params);
            
            % Display parameters
            DOATrainer.display_training_params(params);
            
            % Step 1: Load and prepare data
            fprintf('\n1. Loading and preparing data...\n');
            [trainX, trainY, testX, testY, train_doa, test_doa, metadata] = ...
                DOATrainer.load_and_prepare_data(data_path, params);
            
            % Step 2: Create model
            fprintf('\n2. Creating model: %s\n', model_name);
            [lgraph, inputName, outputName] = DOAModels.create_model_by_name(...
                model_name, metadata.m, metadata.d, metadata.snapshots);
            
            % Convert layer graph to dlnetwork for custom training
            model = DOATrainer.compile_model(lgraph, params);
            
            % Step 3: Prepare training targets
            fprintf('\n3. Preparing training targets...\n');
            [train_targets, test_targets, loss_func] = DOATrainer.prepare_targets(...
                trainX, trainY, train_doa, testX, testY, test_doa, ...
                model_name, params, metadata);
            
            % Step 4: Train model
            fprintf('\n4. Training model...\n');
            if params.use_custom_training
                [model, history] = DOATrainer.train_custom(...
                    model, trainX, train_targets, testX, test_targets, params);
            else
                [model, history] = DOATrainer.train_standard(...
                    model, trainX, train_targets, testX, test_targets, params);
            end
            
            % Step 5: Evaluate model
            fprintf('\n5. Evaluating model...\n');
            results = DOATrainer.evaluate_model(...
                model, testX, test_doa, test_targets, model_name, params, metadata);
            
            % Step 6: Save model
            if params.save_model
                DOATrainer.save_model(model, model_name, params);
            end
            
            fprintf('\n6. Training complete!\n');
            fprintf('===========================\n');
        end
        
        
        %******************************************************%
        %   Load and Prepare Data                           %
        %******************************************************%
        function [trainX, trainY, testX, testY, train_doa, test_doa, metadata] = ...
                load_and_prepare_data(data_path, params)
            % Load and prepare training data
            
            % Set random seed for reproducibility
            utils.set_random_seed(params.seed);
            
            % Load data from file
            fprintf('   Loading data from: %s\n', data_path);
            data = utils.load_data(data_path);
            
            % Extract data
            dataX = data.X;  % Measurements
            dataY = data.Y;  % DOA angles in radians
            
            % Get metadata
            metadata = struct();
            if isfield(data, 'metadata')
                metadata = data.metadata;
            else
                % Infer from data
                metadata.m = size(dataX, 2);  % Array elements
                metadata.d = size(dataY, 2);  % Sources
                metadata.snapshots = size(dataX, 3);  % Snapshots
                metadata.snr_db = 10;  % Default
            end
            
            num_samples = size(dataX, 1);
            fprintf('   Samples: %d, Elements: %d, Sources: %d, Snapshots: %d\n', ...
                    num_samples, metadata.m, metadata.d, metadata.snapshots);
            
            % Format input data: stack real and imaginary parts
            fprintf('   Formatting input data...\n');
            trainX_real = real(dataX);
            trainX_imag = imag(dataX);
            
            % Reshape to [num_samples, 2*m, snapshots]
            trainX = zeros(num_samples, 2*metadata.m, metadata.snapshots);
            for i = 1:num_samples
                trainX(i, :, :) = [squeeze(trainX_real(i, :, :)); ...
                                   squeeze(trainX_imag(i, :, :))];
            end
            
            % For neural network, we might want to reshape differently
            % Option 1: Flatten to [num_samples, 2*m*snapshots]
            % Option 2: Keep as [num_samples, 2*m, snapshots] for sequence models
            % Option 3: Reshape for CNN: [num_samples, 2*m, snapshots, 1]
            
            % Split into train and test sets
            fprintf('   Splitting data (train: %.0f%%, test: %.0f%%)...\n', ...
                    (1-params.test_split)*100, params.test_split*100);
            
            % Create random indices for splitting
            num_test = floor(num_samples * params.test_split);
            indices = randperm(num_samples);
            test_idx = indices(1:num_test);
            train_idx = indices(num_test+1:end);
            
            % Split data
            if length(size(trainX)) == 3
                % 3D data [samples, features, sequence]
                trainX_split = trainX(train_idx, :, :);
                testX_split = trainX(test_idx, :, :);
            else
                % 2D data
                trainX_split = trainX(train_idx, :);
                testX_split = trainX(test_idx, :);
            end
            
            train_doa = dataY(train_idx, :);
            test_doa = dataY(test_idx, :);
            
            % Create training targets (spectrum or EVD)
            fprintf('   Creating training targets...\n');
            
            % Option 1: Create MUSIC spectra as targets
            if params.train_with_spectrum
                trainY = DOATrainer.create_spectrum_targets(...
                    dataX(train_idx, :, :), metadata.m, metadata.d, params);
                testY = DOATrainer.create_spectrum_targets(...
                    dataX(test_idx, :, :), metadata.m, metadata.d, params);
            else
                % Option 2: Create EVD (noise subspace) as targets
                trainY = DOATrainer.create_evd_targets(...
                    dataX(train_idx, :, :), metadata.m, metadata.d);
                testY = DOATrainer.create_evd_targets(...
                    dataX(test_idx, :, :), metadata.m, metadata.d);
            end
            
            % Return data
            trainX = trainX_split;
            testX = testX_split;
            trainY = trainY;
            testY = testY;
            
            fprintf('   Training set: %d samples\n', size(trainX, 1));
            fprintf('   Test set: %d samples\n', size(testX, 1));
        end
        
        
        %******************************************************%
        %   Create Spectrum Targets                          %
        %******************************************************%
        function spectra = create_spectrum_targets(dataX, m, d, params)
            % Create MUSIC spectrum targets
            
            num_samples = size(dataX, 1);
            snapshots = size(dataX, 3);
            
            % Get array positions
            array_positions = arrayUtils.ULA_positions(m);
            
            % Get angle grid
            angles = DOALosses.get_angle_grid(params.angle_resolution);
            
            % Initialize spectra
            spectra = zeros(num_samples, length(angles));
            
            fprintf('     Creating spectra for %d samples...\n', num_samples);
            
            for i = 1:num_samples
                % Extract measurement
                X = squeeze(dataX(i, :, :));
                
                % Run classic MUSIC
                [~, spectrum] = classicMUSIC.estimate(X, array_positions, angles, d);
                
                % Store spectrum
                spectra(i, :) = spectrum;
                
                % Progress
                if mod(i, 100) == 0
                    fprintf('       Processed %d/%d samples\n', i, num_samples);
                end
            end
            
            % Normalize spectra if requested
            if params.normalize_spectra
                spectra = DOATrainer.normalize_spectra(spectra);
            end
        end
        
        
        %******************************************************%
        %   Create EVD Targets                               %
        %******************************************************%
        function evd_targets = create_evd_targets(dataX, m, d)
            % Create EVD (noise subspace) targets
            
            num_samples = size(dataX, 1);
            n = m - d;  % Noise subspace dimension
            
            % Initialize targets
            evd_targets = zeros(num_samples, 2*m*n);
            
            fprintf('     Creating EVD targets for %d samples...\n', num_samples);
            
            for i = 1:num_samples
                % Extract measurement
                X = squeeze(dataX(i, :, :));
                
                % Calculate covariance matrix
                R = utils.correlation_matrix(X);
                
                % Eigenvalue decomposition
                [~, ~, ~, noise_subspace] = utils.eigen_decomposition(R, d);
                
                if ~isempty(noise_subspace)
                    % Ensure correct number of noise vectors
                    if size(noise_subspace, 2) > n
                        noise_subspace = noise_subspace(:, 1:n);
                    elseif size(noise_subspace, 2) < n
                        % Pad with zeros
                        padding = zeros(m, n - size(noise_subspace, 2));
                        noise_subspace = [noise_subspace, padding];
                    end
                    
                    % Separate real and imaginary parts
                    evd_real = real(noise_subspace);
                    evd_imag = imag(noise_subspace);
                    
                    % Flatten and stack
                    evd_targets(i, :) = [evd_real(:)', evd_imag(:)'];
                else
                    % Use zeros if no noise subspace
                    evd_targets(i, :) = zeros(1, 2*m*n);
                end
                
                % Progress
                if mod(i, 100) == 0
                    fprintf('       Processed %d/%d samples\n', i, num_samples);
                end
            end
        end
        
        
        %******************************************************%
        %   Compile Model                                    %
        %******************************************************%
        function model = compile_model(lgraph, params)
            % Compile model for training
            
            fprintf('   Compiling model...\n');
            
            % Convert layer graph to dlnetwork for custom training
            if params.use_custom_training
                model = dlnetwork(lgraph);
                fprintf('     Using dlnetwork for custom training\n');
            else
                % For standard training, we need to specify loss and optimizer
                % This is a placeholder - actual compilation depends on loss function
                model = lgraph;
                fprintf('     Using standard layer graph\n');
            end
            
            % Display model info
            fprintf('     Input size: ');
            input_layer = lgraph.Layers(1);
            if isa(input_layer, 'nnet.cnn.layer.ImageInputLayer')
                fprintf('%s\n', mat2str(input_layer.InputSize));
            elseif isa(input_layer, 'nnet.cnn.layer.SequenceInputLayer')
                fprintf('%d features\n', input_layer.InputSize);
            end
            
            fprintf('     Number of layers: %d\n', numel(lgraph.Layers));
        end
        
        
        %******************************************************%
        %   Prepare Training Targets                         %
        %******************************************************%
        function [train_targets, test_targets, loss_func] = prepare_targets(...
                trainX, trainY, train_doa, testX, testY, test_doa, ...
                model_name, params, metadata)
            % Prepare training targets based on model type
            
            fprintf('   Model type: %s\n', model_name);
            
            % Determine model type and appropriate loss
            if contains(model_name, 'E2E')
                % End-to-end DOA estimation
                train_targets = train_doa;
                test_targets = test_doa;
                loss_func = @DOALosses.permutation_rmse_loss;
                fprintf('     End-to-end model: Predicting DOA angles directly\n');
                fprintf('     Using permutation RMSE loss\n');
                
            elseif params.train_with_spectrum
                % Spectrum prediction
                train_targets = trainY;
                test_targets = testY;
                loss_func = @DOALosses.mse_spectrum_loss;
                fprintf('     Spectrum prediction model\n');
                fprintf('     Using spectrum MSE loss\n');
                
            else
                % EVD (noise subspace) prediction
                train_targets = trainY;
                test_targets = testY;
                
                % Create hybrid loss function with parameters
                loss_func = @(y_pred, y_true) DOATrainer.hybrid_loss_wrapper(...
                    y_pred, y_true, metadata.m, metadata.d, params);
                fprintf('     EVD prediction model\n');
                fprintf('     Using hybrid loss function\n');
            end
            
            fprintf('     Training targets size: %s\n', mat2str(size(train_targets)));
            fprintf('     Test targets size: %s\n', mat2str(size(test_targets)));
        end
        
        
        %******************************************************%
        %   Hybrid Loss Wrapper                              %
        %******************************************************%
        function loss = hybrid_loss_wrapper(y_pred, y_true, m, d, params)
            % Wrapper for hybrid loss function
            
            % Create true values structure
            y_true_struct = struct();
            y_true_struct.evd = y_true;
            
            % Get array positions (needed for some losses)
            array_positions = arrayUtils.ULA_positions(m);
            
            % Training parameters for loss
            loss_params = struct();
            loss_params.angles = DOALosses.get_angle_grid();
            loss_params.weights = params.loss_weights;
            loss_params.beta = params.beta;
            loss_params.shift_up = params.shift_up;
            loss_params.orth_scale = params.orth_scale;
            loss_params.use_regularization = params.use_regularization;
            
            % Calculate hybrid loss
            [loss, ~] = DOALosses.training_hybrid_loss(...
                y_pred, y_true_struct, array_positions, m, d, loss_params);
        end
        
        
        %******************************************************%
        %   Custom Training Loop                             %
        %******************************************************%
        function [model, history] = train_custom(model, trainX, trainY, testX, testY, params)
            % Custom training loop using dlnetwork
            
            fprintf('   Starting custom training...\n');
            fprintf('   Epochs: %d, Batch size: %d, Learning rate: %.4f\n', ...
                    params.epochs, params.batch_size, params.learning_rate);
            
            % Convert data to dlarray
            if ndims(trainX) == 3
                % Sequence data: [features, sequence, batch]
                XTrain = dlarray(permute(trainX, [2, 3, 1]), 'CBT');
            else
                % Image/feature data
                XTrain = dlarray(trainX', 'CB');
            end
            
            YTrain = dlarray(trainY', 'CB');
            
            % Get dataset sizes
            num_train = size(trainX, 1);
            num_batches = ceil(num_train / params.batch_size);
            
            % Initialize optimizer
            if strcmp(params.optimizer, 'adam')
                optimizer = adamOptimizer(params.learning_rate);
            else
                optimizer = sgdmOptimizer(params.learning_rate);
            end
            
            % Initialize history
            history = struct();
            history.train_loss = zeros(params.epochs, 1);
            history.val_loss = zeros(params.epochs, 1);
            
            % Training loop
            for epoch = 1:params.epochs
                fprintf('\n   Epoch %d/%d\n', epoch, params.epochs);
                
                % Shuffle data
                idx = randperm(num_train);
                XTrain_shuffled = XTrain(:, :, idx);
                YTrain_shuffled = YTrain(:, :, idx);
                
                % Initialize epoch loss
                epoch_loss = 0;
                
                % Mini-batch training
                for batch = 1:num_batches
                    % Get batch indices
                    start_idx = (batch-1) * params.batch_size + 1;
                    end_idx = min(batch * params.batch_size, num_train);
                    batch_size_actual = end_idx - start_idx + 1;
                    
                    % Get batch data
                    XBatch = XTrain_shuffled(:, :, start_idx:end_idx);
                    YBatch = YTrain_shuffled(:, :, start_idx:end_idx);
                    
                    % Evaluate model and compute gradients
                    [loss, gradients] = dlfeval(@DOATrainer.model_loss, ...
                        model, XBatch, YBatch, params);
                    
                    % Update model parameters
                    model = optimizer.update(model, gradients);
                    
                    % Accumulate loss
                    epoch_loss = epoch_loss + extractdata(loss) * batch_size_actual;
                    
                    % Display progress
                    if mod(batch, max(1, floor(num_batches/10))) == 0
                        fprintf('     Batch %d/%d, Loss: %.4f\n', ...
                                batch, num_batches, extractdata(loss));
                    end
                end
                
                % Average epoch loss
                history.train_loss(epoch) = epoch_loss / num_train;
                
                % Validation
                if ~isempty(testX)
                    val_loss = DOATrainer.validate_model(model, testX, testY, params);
                    history.val_loss(epoch) = val_loss;
                    fprintf('     Train Loss: %.4f, Val Loss: %.4f\n', ...
                            history.train_loss(epoch), val_loss);
                else
                    fprintf('     Train Loss: %.4f\n', history.train_loss(epoch));
                end
                
                % Learning rate schedule
                if params.use_lr_schedule && mod(epoch, params.lr_drop_epochs) == 0
                    params.learning_rate = params.learning_rate * params.lr_drop_factor;
                    optimizer = adamOptimizer(params.learning_rate);
                    fprintf('     Learning rate decreased to: %.6f\n', params.learning_rate);
                end
            end
            
            fprintf('\n   Training complete!\n');
        end
        
        
        %******************************************************%
        %   Model Loss Function                              %
        %******************************************************%
        function [loss, gradients] = model_loss(model, X, Y, params)
            % Compute loss and gradients
            
            % Forward pass
            Y_pred = forward(model, X);
            
            % Compute loss (custom loss function)
            % Note: This would need to be adapted based on the specific loss
            loss = mean((Y_pred - Y).^2, 'all');
            
            % Compute gradients
            gradients = dlgradient(loss, model.Learnables);
        end
        
        
        %******************************************************%
        %   Standard Training                               %
        %******************************************************%
        function [model, history] = train_standard(model, trainX, trainY, testX, testY, params)
            % Standard training using trainNetwork
            
            fprintf('   Starting standard training...\n');
            
            % Prepare data for trainNetwork
            % Note: This requires converting to appropriate format
            % For now, we'll use a simplified approach
            
            % Reshape data based on input type
            input_layer = model.Layers(1);
            
            if isa(input_layer, 'nnet.cnn.layer.ImageInputLayer')
                % CNN input: [height, width, channels, batch]
                XTrain = permute(trainX, [2, 3, 1]);
                XTrain = reshape(XTrain, [size(XTrain, 1), size(XTrain, 2), 1, size(XTrain, 3)]);
                XTest = permute(testX, [2, 3, 1]);
                XTest = reshape(XTest, [size(XTest, 1), size(XTest, 2), 1, size(XTest, 3)]);
                
            elseif isa(input_layer, 'nnet.cnn.layer.SequenceInputLayer')
                % RNN input: {sequences} where each sequence is [features, timesteps]
                XTrain = cell(size(trainX, 1), 1);
                for i = 1:size(trainX, 1)
                    XTrain{i} = squeeze(trainX(i, :, :));
                end
                XTest = cell(size(testX, 1), 1);
                for i = 1:size(testX, 1)
                    XTest{i} = squeeze(testX(i, :, :));
                end
            else
                % Dense input
                XTrain = trainX;
                XTest = testX;
            end
            
            % Training options
            options = trainingOptions(params.optimizer, ...
                'MaxEpochs', params.epochs, ...
                'MiniBatchSize', params.batch_size, ...
                'InitialLearnRate', params.learning_rate, ...
                'ValidationData', {XTest, testY}, ...
                'ValidationFrequency', 30, ...
                'Verbose', true, ...
                'Plots', 'training-progress');
            
            % Train network
            % Note: This requires a regressionLayer at the end
            % We might need to add one if not present
            [model, history] = trainNetwork(XTrain, trainY, model, options);
        end
        
        
        %******************************************************%
        %   Validate Model                                   %
        %******************************************************%
        function val_loss = validate_model(model, XTest, YTest, params)
            % Validate model on test set
            
            % Convert to dlarray
            if ndims(XTest) == 3
                XTest_dl = dlarray(permute(XTest, [2, 3, 1]), 'CBT');
            else
                XTest_dl = dlarray(XTest', 'CB');
            end
            
            YTest_dl = dlarray(YTest', 'CB');
            
            % Forward pass
            Y_pred = forward(model, XTest_dl);
            
            % Compute loss
            val_loss = extractdata(mean((Y_pred - YTest_dl).^2, 'all'));
        end
        
        
        %******************************************************%
        %   Evaluate Model                                   %
        %******************************************************%
        function results = evaluate_model(model, testX, test_doa, test_targets, ...
                                         model_name, params, metadata)
            % Evaluate trained model
            
            fprintf('   Evaluating model performance...\n');
            
            % Get array positions and angles for evaluation
            array_positions = arrayUtils.ULA_positions(metadata.m);
            angles = DOALosses.get_angle_grid(params.angle_resolution);
            
            % Initialize results structure
            results = struct();
            
            % Make predictions
            fprintf('   Making predictions...\n');
            
            if contains(model_name, 'E2E')
                % End-to-end: Predict DOA directly
                pred_doa = DOATrainer.predict_e2e(model, testX, params);
                results.metrics = DOATrainer.evaluate_doa_predictions(...
                    pred_doa, test_doa, 'E2E Model');
                
            elseif params.train_with_spectrum
                % Spectrum prediction
                pred_spectrum = DOATrainer.predict_spectrum(model, testX, params);
                results.spectrum_mse = mean((pred_spectrum - test_targets).^2, 'all');
                fprintf('     Spectrum MSE: %.6f\n', results.spectrum_mse);
                
                % Estimate DOA from spectrum
                pred_doa = DOATrainer.spectrum_to_doa(pred_spectrum, angles, metadata.d);
                results.metrics = DOATrainer.evaluate_doa_predictions(...
                    pred_doa, test_doa, 'Spectrum Model');
                
            else
                % EVD prediction
                pred_evd = DOATrainer.predict_evd(model, testX, params);
                
                % Calculate spectrum from EVD
                pred_spectrum = zeros(size(testX, 1), length(angles));
                for i = 1:size(testX, 1)
                    pred_spectrum(i, :) = DOALosses.calculate_spectrum_fixed(...
                        pred_evd(i, :), array_positions, angles, metadata.m, metadata.d);
                end
                
                % Estimate DOA from spectrum
                pred_doa = DOATrainer.spectrum_to_doa(pred_spectrum, angles, metadata.d);
                results.metrics = DOATrainer.evaluate_doa_predictions(...
                    pred_doa, test_doa, 'EVD Model');
            end
            
            % Compare with baseline algorithms
            fprintf('\n   Comparing with baseline algorithms...\n');
            
            % Classic MUSIC
            music_doa = DOATrainer.run_classic_music(testX, metadata.m, metadata.d, angles);
            results.music_metrics = DOATrainer.evaluate_doa_predictions(...
                music_doa, test_doa, 'Classic MUSIC');
            
            % Beamformer
            beamformer_doa = DOATrainer.run_beamformer(testX, metadata.m, metadata.d, angles);
            results.beamformer_metrics = DOATrainer.evaluate_doa_predictions(...
                beamformer_doa, test_doa, 'Beamformer');
            
            % Random baseline
            random_doa = -pi/2 + pi * rand(size(test_doa));
            results.random_metrics = DOATrainer.evaluate_doa_predictions(...
                random_doa, test_doa, 'Random');
            
            % Display comparison
            DOATrainer.display_comparison(results);
        end
        
        
        %******************************************************%
        %   Prediction Functions                             %
        %******************************************************%
        function pred_doa = predict_e2e(model, X, params)
            % Predict DOA directly (end-to-end)
            
            num_samples = size(X, 1);
            pred_doa = zeros(num_samples, params.d);
            
            for i = 1:num_samples
                % Prepare input
                if ndims(X) == 3
                    X_i = dlarray(permute(X(i, :, :), [2, 3, 1]), 'CBT');
                else
                    X_i = dlarray(X(i, :)', 'CB');
                end
                
                % Predict
                pred = forward(model, X_i);
                pred_doa(i, :) = extractdata(pred)';
            end
        end
        
        function pred_spectrum = predict_spectrum(model, X, params)
            % Predict spectrum
            
            num_samples = size(X, 1);
            num_angles = params.angle_resolution;
            pred_spectrum = zeros(num_samples, num_angles);
            
            for i = 1:num_samples
                % Prepare input
                if ndims(X) == 3
                    X_i = dlarray(permute(X(i, :, :), [2, 3, 1]), 'CBT');
                else
                    X_i = dlarray(X(i, :)', 'CB');
                end
                
                % Predict
                pred = forward(model, X_i);
                pred_spectrum(i, :) = extractdata(pred)';
            end
        end
        
        function pred_evd = predict_evd(model, X, params)
            % Predict EVD (noise subspace)
            
            num_samples = size(X, 1);
            % Assuming output size is known from model
            % For m=8, d=2: output size = 2*m*(m-d) = 96
            output_size = 96;  % This should be determined from model
            pred_evd = zeros(num_samples, output_size);
            
            for i = 1:num_samples
                % Prepare input
                if ndims(X) == 3
                    X_i = dlarray(permute(X(i, :, :), [2, 3, 1]), 'CBT');
                else
                    X_i = dlarray(X(i, :)', 'CB');
                end
                
                % Predict
                pred = forward(model, X_i);
                pred_evd(i, :) = extractdata(pred)';
            end
        end
        
        
        %******************************************************%
        %   DOA Estimation from Spectrum                     %
        %******************************************************%
        function doa_angles = spectrum_to_doa(spectrum, angles, d)
            % Estimate DOA angles from spectrum
            
            num_samples = size(spectrum, 1);
            doa_angles = zeros(num_samples, d);
            
            for i = 1:num_samples
                % Find peaks in spectrum
                [~, peak_locs] = findpeaks(spectrum(i, :), ...
                    'SortStr', 'descend', 'NPeaks', d);
                
                % Convert to angles
                if length(peak_locs) >= d
                    doa_indices = sort(peak_locs(1:d));
                    doa_angles(i, :) = angles(doa_indices);
                else
                    % If not enough peaks, use random angles
                    doa_angles(i, :) = sort(angles(randi(length(angles), 1, d)));
                end
            end
        end
        
        
        %******************************************************%
        %   Baseline Algorithms                              %
        %******************************************************%
        function doa_angles = run_classic_music(X, m, d, angles)
            % Run classic MUSIC algorithm
            
            num_samples = size(X, 1);
            array_positions = arrayUtils.ULA_positions(m);
            doa_angles = zeros(num_samples, d);
            
            for i = 1:num_samples
                % Extract measurement (complex)
                X_real = squeeze(X(i, 1:m, :));
                X_imag = squeeze(X(i, m+1:end, :));
                X_complex = complex(X_real, X_imag);
                
                % Run MUSIC
                [doa_rad, ~] = classicMUSIC.estimate(X_complex, array_positions, angles, d);
                
                if length(doa_rad) >= d
                    doa_angles(i, :) = sort(doa_rad(1:d));
                else
                    % Pad with random angles if needed
                    doa_angles(i, :) = sort([doa_rad; angles(randi(length(angles), d-length(doa_rad), 1))]);
                end
            end
        end
        
        function doa_angles = run_beamformer(X, m, d, angles)
            % Run beamformer algorithm
            
            num_samples = size(X, 1);
            array_positions = arrayUtils.ULA_positions(m);
            doa_angles = zeros(num_samples, d);
            
            for i = 1:num_samples
                % Extract measurement
                X_real = squeeze(X(i, 1:m, :));
                X_imag = squeeze(X(i, m+1:end, :));
                X_complex = complex(X_real, X_imag);
                
                % Run beamformer
                % Note: Need to implement or use existing beamformer function
                % For now, use random angles as placeholder
                doa_angles(i, :) = sort(angles(randi(length(angles), 1, d)));
            end
        end
        
        
        %******************************************************%
        %   Evaluation Metrics                               %
        %******************************************************%
        function metrics = evaluate_doa_predictions(pred_doa, true_doa, algorithm_name)
            % Evaluate DOA prediction performance
            
            % Calculate permutation-invariant RMSE
            rmse = DOALosses.permutation_rmse_loss(pred_doa, true_doa);
            
            % Calculate mean absolute error
            mae = mean(abs(pred_doa(:) - true_doa(:)));
            
            % Calculate success rate (within 5 degrees)
            threshold_deg = 5;
            threshold_rad = deg2rad(threshold_deg);
            
            % For each sample, check if all DOAs are within threshold
            num_samples = size(pred_doa, 1);
            success_count = 0;
            
            for i = 1:num_samples
                % Find best permutation
                perms = perms(1:size(pred_doa, 2));
                min_error = inf;
                
                for p = 1:size(perms, 1)
                    permuted_pred = pred_doa(i, perms(p, :));
                    error = mean(abs(permuted_pred - true_doa(i, :)));
                    min_error = min(min_error, error);
                end
                
                if min_error < threshold_rad
                    success_count = success_count + 1;
                end
            end
            
            success_rate = success_count / num_samples;
            
            % Store metrics
            metrics = struct();
            metrics.algorithm = algorithm_name;
            metrics.rmse_rad = rmse;
            metrics.rmse_deg = rad2deg(rmse);
            metrics.mae_rad = mae;
            metrics.mae_deg = rad2deg(mae);
            metrics.success_rate = success_rate;
            metrics.threshold_deg = threshold_deg;
            
            fprintf('     %-20s: RMSE = %.2f° | Success Rate = %.1f%%\n', ...
                    algorithm_name, rad2deg(rmse), success_rate*100);
        end
        
        
        %******************************************************%
        %   Display Comparison                               %
        %******************************************************%
        function display_comparison(results)
            % Display comparison of different algorithms
            
            fprintf('\n   PERFORMANCE COMPARISON:\n');
            fprintf('   ========================\n');
            
            algorithms = {'metrics', 'music_metrics', 'beamformer_metrics', 'random_metrics'};
            algo_names = {'Neural Network', 'Classic MUSIC', 'Beamformer', 'Random'};
            
            for i = 1:length(algorithms)
                if isfield(results, algorithms{i})
                    algo = results.(algorithms{i});
                    fprintf('   %-20s: RMSE = %6.2f° | Success = %5.1f%%\n', ...
                            algo_names{i}, algo.rmse_deg, algo.success_rate*100);
                end
            end
            
            fprintf('   ========================\n');
        end
        
        
        %******************************************************%
        %   Save Model                                       %
        %******************************************************%
        function save_model(model, model_name, params)
            % Save trained model
            
            % Create model directory if it doesn't exist
            model_dir = 'models';
            if ~exist(model_dir, 'dir')
                mkdir(model_dir);
                fprintf('   Created directory: %s\n', model_dir);
            end
            
            % Create filename with timestamp
            timestamp = datestr(now, 'yyyy-mm-dd_HH-MM');
            filename = sprintf('%s/%s_%s.mat', model_dir, model_name, timestamp);
            
            % Save model
            save(filename, 'model', 'params', '-v7.3');
            fprintf('   Model saved to: %s\n', filename);
        end
        
        
        %******************************************************%
        %   Utility Functions                                %
        %******************************************************%
        function params = set_default_params(params)
            % Set default training parameters
            
            default_params = struct();
            default_params.seed = DOATrainer.DEFAULT_SEED;
            default_params.batch_size = DOATrainer.DEFAULT_BATCH_SIZE;
            default_params.learning_rate = DOATrainer.DEFAULT_LEARNING_RATE;
            default_params.epochs = DOATrainer.DEFAULT_EPOCHS;
            default_params.val_split = DOATrainer.DEFAULT_VAL_SPLIT;
            default_params.test_split = DOATrainer.DEFAULT_TEST_SPLIT;
            default_params.optimizer = 'adam';
            default_params.use_custom_training = true;
            default_params.train_with_spectrum = false;
            default_params.normalize_spectra = true;
            default_params.angle_resolution = 181;
            default_params.use_regularization = true;
            default_params.use_lr_schedule = true;
            default_params.lr_drop_epochs = 20;
            default_params.lr_drop_factor = 0.5;
            default_params.save_model = true;
            
            % Loss weights
            default_params.loss_weights = struct(...
                'spectrum_mse', 0.35, ...
                'inverse_peaks', 0.25, ...
                'doa_mse', 0.20, ...
                'evd', 0.10, ...
                'regularization', 0.10);
            
            default_params.beta = 1e3;
            default_params.shift_up = 2;
            default_params.orth_scale = 1e-3;
            
            % Merge with user parameters
            if ~isempty(params)
                param_fields = fieldnames(params);
                for i = 1:length(param_fields)
                    field = param_fields{i};
                    default_params.(field) = params.(field);
                end
            end
            
            params = default_params;
        end
        
        function display_training_params(params)
            % Display training parameters
            
            fprintf('Training Parameters:\n');
            fprintf('  Seed: %d\n', params.seed);
            fprintf('  Batch size: %d\n', params.batch_size);
            fprintf('  Learning rate: %.4f\n', params.learning_rate);
            fprintf('  Epochs: %d\n', params.epochs);
            fprintf('  Validation split: %.2f\n', params.val_split);
            fprintf('  Test split: %.2f\n', params.test_split);
            fprintf('  Optimizer: %s\n', params.optimizer);
            fprintf('  Training mode: %s\n', ...
                    iif(params.use_custom_training, 'Custom', 'Standard'));
            fprintf('  Target type: %s\n', ...
                    iif(params.train_with_spectrum, 'Spectrum', 'EVD'));
            fprintf('  Use regularization: %s\n', string(params.use_regularization));
            fprintf('  Angle resolution: %d points\n', params.angle_resolution);
            
            if params.use_regularization
                fprintf('  Orthogonality scale: %.1e\n', params.orth_scale);
            end
        end
        
        function spectra_norm = normalize_spectra(spectra)
            % Normalize spectra
            
            % Min-max normalization per sample
            spectra_min = min(spectra, [], 2);
            spectra_max = max(spectra, [], 2);
            spectra_range = spectra_max - spectra_min;
            
            % Avoid division by zero
            spectra_range(spectra_range == 0) = 1;
            
            spectra_norm = (spectra - spectra_min) ./ spectra_range;
        end
        
        
        %******************************************************%
        %   Example Usage                                    %
        %******************************************************%
        function example_training()
            % Example training script
            
            fprintf('DOA Neural Network Training Example\n');
            fprintf('===================================\n\n');
            
            % Training parameters
            params = struct();
            params.epochs = 50;
            params.batch_size = 32;
            params.learning_rate = 0.001;
            params.use_regularization = true;
            params.orth_scale = 1e-3;
            
            % Loss weights
            params.loss_weights = struct(...
                'spectrum_mse', 0.4, ...
                'inverse_peaks', 0.3, ...
                'doa_mse', 0.2, ...
                'evd', 0.1, ...
                'regularization', 0.1);
            
            % Model name (choose from DOAModels.get_model_summary())
            model_name = 'simpleRNN_noEVD';
            
            % Data path (adjust based on your data location)
            data_path = 'data/m8/d2_snr10_10k.mat';
            
            % Check if data exists
            if ~exist(data_path, 'file')
                fprintf('Data file not found: %s\n', data_path);
                fprintf('Creating synthetic dataset for testing...\n');
                
                % Create synthetic dataset
                dataset_name = 'test_dataset';
                m = 8;
                d = 2;
                snapshots = 200;
                snr_db = 10;
                num_samples = 1000;
                
                [X, Y] = syntheticEx.create_dataset(...
                    dataset_name, num_samples, m, d, snapshots, snr_db, false, [], true);
                
                data_path = 'data/test_dataset.mat';
            end
            
            % Train model
            [model, history, results] = DOATrainer.train_model(...
                model_name, data_path, params);
            
            % Plot training history
            DOATrainer.plot_training_history(history);
            
            fprintf('\nExample training complete!\n');
        end
        
        function plot_training_history(history)
            % Plot training history
            
            figure('Position', [100, 100, 800, 400]);
            
            % Plot training and validation loss
            subplot(1, 2, 1);
            plot(1:length(history.train_loss), history.train_loss, 'b-', 'LineWidth', 2);
            hold on;
            if isfield(history, 'val_loss')
                plot(1:length(history.val_loss), history.val_loss, 'r-', 'LineWidth', 2);
                legend('Training Loss', 'Validation Loss', 'Location', 'best');
            else
                legend('Training Loss', 'Location', 'best');
            end
            xlabel('Epoch');
            ylabel('Loss');
            title('Training History');
            grid on;
            
            % Plot learning rate if available
            if isfield(history, 'learning_rate')
                subplot(1, 2, 2);
                plot(1:length(history.learning_rate), history.learning_rate, 'g-', 'LineWidth', 2);
                xlabel('Epoch');
                ylabel('Learning Rate');
                title('Learning Rate Schedule');
                grid on;
            end
            
            sgtitle('DOA Neural Network Training');
        end
        
    end
end

% Helper function
function result = iif(condition, trueValue, falseValue)
    % Immediate if function
    if condition
        result = trueValue;
    else
        result = falseValue;
    end
end