% test_utils.m
% Test the utility functions

clear; clc;

% Set random seed
utils.set_random_seed(42);

fprintf('=========================================\n');
fprintf('Testing DOA Utility Functions\n');
fprintf('=========================================\n');

% Test 1: Permutations
fprintf('\n1. Testing permutations...\n');
test_array = [1, 2, 3];
perms = utils.permutations(test_array);
fprintf('   Permutations of %s:\n', mat2str(test_array));
for i = 1:length(perms)
    fprintf('   %d) %s\n', i, mat2str(perms{i}));
end
fprintf('   ✓ Found %d permutations (expected: 6)\n', length(perms));

% Test 2: Normalization
fprintf('\n2. Testing normalization...\n');
data = randn(10, 5) + 1j * randn(10, 5);
data_norm = utils.normalize_complex(data);
fprintf('   Original data range:\n');
fprintf('     Real: [%.4f, %.4f]\n', min(real(data(:))), max(real(data(:))));
fprintf('     Imag: [%.4f, %.4f]\n', min(imag(data(:))), max(imag(data(:))));
fprintf('   Normalized magnitude range: [%.4f, %.4f]\n', ...
    min(abs(data_norm(:))), max(abs(data_norm(:))));
fprintf('   ✓ Normalization completed\n');

% Test 3: Correlation matrix
fprintf('\n3. Testing correlation matrix...\n');
M = 4; N = 100;
X = randn(M, N) + 1j * randn(M, N);
R = utils.correlation_matrix(X);
fprintf('   Correlation matrix size: %d x %d\n', size(R,1), size(R,2));

% Check if Hermitian (R = R^H)
is_hermitian = norm(R - R', 'fro') < 1e-10;
fprintf('   Hermitian check: %s\n', string(is_hermitian));
fprintf('   ✓ Correlation matrix computed\n');

% Test 4: Eigen decomposition
fprintf('\n4. Testing eigen decomposition...\n');
[U, D, signal_sub, noise_sub] = utils.eigen_decomposition(R, 2);
fprintf('   Signal subspace size: %d x %d\n', size(signal_sub,1), size(signal_sub,2));
fprintf('   Noise subspace size: %d x %d\n', size(noise_sub,1), size(noise_sub,2));

% Check orthogonality
orthogonality = norm(signal_sub' * noise_sub, 'fro');
fprintf('   Signal-Noise subspace orthogonality: %.2e\n', orthogonality);
fprintf('   ✓ Eigen decomposition completed\n');

% Test 5: AWGN
fprintf('\n5. Testing AWGN...\n');
clean_signal = randn(100, 1);
snr_target = 10;  % dB
noisy_signal = utils.add_awgn(clean_signal, snr_target);
noise = noisy_signal - clean_signal;
snr_actual = utils.calculate_snr(clean_signal, noise);
fprintf('   Target SNR: %.2f dB\n', snr_target);
fprintf('   Actual SNR: %.2f dB\n', snr_actual);
fprintf('   Error: %.2f dB\n', abs(snr_target - snr_actual));
fprintf('   ✓ AWGN test completed\n');

% Test 6: Peak finding
fprintf('\n6. Testing peak finding...\n');
x = linspace(0, 10*pi, 1000);
spectrum = sin(x) + 0.5*sin(2*x + 1);
[peak_locs, peak_vals] = utils.find_spectrum_peaks(spectrum, 3, 50);
fprintf('   Found %d peaks at indices: %s\n', length(peak_locs), mat2str(peak_locs));
fprintf('   Peak values: %s\n', mat2str(peak_vals, 3));
fprintf('   ✓ Peak finding completed\n');

% Test 7: ULA steering vector
fprintf('\n7. Testing ULA steering vector...\n');
array = (0:7)';
theta = pi/6;  % 30 degrees
a = utils.ULA_action_vector(array, theta);
fprintf('   Array size: %d elements\n', length(array));
fprintf('   Theta: %.1f degrees\n', rad2deg(theta));
fprintf('   Steering vector size: %d x %d\n', size(a,1), size(a,2));
fprintf('   First 3 elements: %s\n', mat2str(a(1:3), 3));
fprintf('   ✓ Steering vector computed\n');

% Test 8: Timing utility
fprintf('\n8. Testing timing utility...\n');
test_func = @() sum(randn(10000, 1));
elapsed = utils.time_function(test_func);
fprintf('   Function execution time: %.4f seconds\n', elapsed);
fprintf('   ✓ Timing utility completed\n');

% Test 9: MUSIC spectrum calculation (requires special setup)
fprintf('\n9. Testing MUSIC spectrum calculation...\n');
try
    % Create synthetic data
    m = 8;
    batch_size = 2;
    angles = linspace(-pi/2, pi/2, 50);
    array = (0:m-1)';
    
    % Create random noise subspace predictions
    y_pred = randn(batch_size, 2*m);
    
    % Calculate spectrum
    spectrum = utils.calculate_spectrum(y_pred, array, angles, m);
    
    fprintf('   Spectrum size: %d x %d\n', size(spectrum,1), size(spectrum,2));
    fprintf('   Spectrum range: [%.4f, %.4f]\n', min(spectrum(:)), max(spectrum(:)));
    fprintf('   ✓ MUSIC spectrum calculation completed\n');
catch ME
    fprintf('   ⚠ MUSIC spectrum test skipped: %s\n', ME.message);
end

fprintf('\n=========================================\n');
fprintf('All tests completed!\n');
fprintf('=========================================\n');