%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    plotFigures.m                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
% Authors: J. M.                                                               %
%                                                                              %
% Created: 26/03/21                                                            %
%                                                                              %
% Purpose: Plot synthetic examples to test correctness and performance of      %
%          algorithms estimating directions of arrival (DoA) of multiple       %
%          signals.                                                            %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef plotFigures
    % PLOTFIGURES Visualization tools for DOA algorithm analysis
    
    properties(Constant)
        % Default plotting styles
        LINE_WIDTH = 1.5;
        MARKER_SIZE = 10;
        FONT_SIZE = 12;
        TITLE_FONT_SIZE = 14;
        LEGEND_FONT_SIZE = 11;
        
        % Color scheme
        COLORS = struct(...
            'beamformer', [0.2, 0.2, 0.2], ...      % Dark gray
            'classic_music', [0, 0.4470, 0.7410], ... % Blue
            'aug_music', [0.8500, 0.3250, 0.0980], ... % Orange
            'true_doa', [0.4660, 0.6740, 0.1880], ...  % Green
            'estimated', [0.6350, 0.0780, 0.1840] ...  % Red
        );
    end
    
    methods(Static)
        
        %***********************#
        %   plot BF vs. MUSIC   #
        %***********************#
        function fig = plotBFvMUSIC(spectrumBF, DoABF, true_doas, theta_range, ...
                spectrumMUSIC, DoAMUSIC, fig_title)
            % Plot Beamformer vs. MUSIC comparison.
            %
            % @param spectrumBF -- Beamformer spectrum
            % @param DoABF -- Beamformer DOA indices
            % @param true_doas -- True DOAs (radians)
            % @param theta_range -- Angle range (radians)
            % @param spectrumMUSIC -- MUSIC spectrum (optional)
            % @param DoAMUSIC -- MUSIC DOA indices (optional)
            % @param fig_title -- Figure title (optional)
            %
            % @returns -- Figure handle
            
            if nargin < 7
                fig_title = 'Beamformer vs. MUSIC Comparison';
            end
            
            fig = figure('Position', [100, 100, 800, 500], ...
                         'Name', fig_title, 'Color', 'w');
            
            % Plot Beamformer spectrum
            plot(rad2deg(theta_range), 10*log10(spectrumBF/max(spectrumBF)), ...
                 '--', 'Color', plotFigures.COLORS.beamformer, ...
                 'LineWidth', plotFigures.LINE_WIDTH, ...
                 'DisplayName', 'Beamformer');
            hold on;
            
            % Plot MUSIC spectrum if provided
            if nargin >= 5 && ~isempty(spectrumMUSIC)
                plot(rad2deg(theta_range), 10*log10(spectrumMUSIC/max(spectrumMUSIC)), ...
                     '-', 'Color', plotFigures.COLORS.classic_music, ...
                     'LineWidth', plotFigures.LINE_WIDTH, ...
                     'DisplayName', 'Classic MUSIC');
            end
            
            % Mark true DOAs
            plot(rad2deg(true_doas), zeros(size(true_doas)) - 5, '*', ...
                 'Color', plotFigures.COLORS.true_doa, ...
                 'MarkerSize', plotFigures.MARKER_SIZE + 4, ...
                 'LineWidth', 2, 'MarkerFaceColor', plotFigures.COLORS.true_doa, ...
                 'DisplayName', 'True DOA');
            
            % Mark estimated DOAs
            if ~isempty(DoABF)
                plot(rad2deg(theta_range(DoABF)), ...
                     10*log10(spectrumBF(DoABF)/max(spectrumBF)), 'x', ...
                     'Color', plotFigures.COLORS.estimated, ...
                     'MarkerSize', plotFigures.MARKER_SIZE, ...
                     'LineWidth', 2, 'DisplayName', 'Beamformer Est.');
            end
            
            if nargin >= 6 && ~isempty(DoAMUSIC)
                plot(rad2deg(theta_range(DoAMUSIC)), ...
                     10*log10(spectrumMUSIC(DoAMUSIC)/max(spectrumMUSIC)), 's', ...
                     'Color', plotFigures.COLORS.classic_music, ...
                     'MarkerSize', plotFigures.MARKER_SIZE - 2, ...
                     'LineWidth', 2, 'MarkerFaceColor', plotFigures.COLORS.classic_music, ...
                     'DisplayName', 'MUSIC Est.');
            end
            
            % Formatting
            xlabel('Angle (degrees)', 'FontSize', plotFigures.FONT_SIZE);
            ylabel('Normalized Spectrum (dB)', 'FontSize', plotFigures.FONT_SIZE);
            title(fig_title, 'FontSize', plotFigures.TITLE_FONT_SIZE);
            legend('Location', 'best', 'FontSize', plotFigures.LEGEND_FONT_SIZE);
            grid on;
            xlim([-90, 90]);
            ylim([-40, 5]);
            
            % Add grid with different style
            set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1);
        end
        
        
        %************************#
        %   plot classic MUSIC   #
        %************************#
        function fig = plotMUSIC(spectrum, DoAMUSIC, true_doas, theta_range, fig_title)
            % Plot classic MUSIC results.
            %
            % @param spectrum -- MUSIC spectrum
            % @param DoAMUSIC -- MUSIC DOA indices
            % @param true_doas -- True DOAs (radians)
            % @param theta_range -- Angle range (radians)
            % @param fig_title -- Figure title (optional)
            %
            % @returns -- Figure handle
            
            if nargin < 5
                fig_title = 'Classic MUSIC Algorithm';
            end
            
            fig = figure('Position', [100, 100, 800, 500], ...
                         'Name', fig_title, 'Color', 'w');
            
            % Plot MUSIC spectrum
            plot(rad2deg(theta_range), 10*log10(spectrum/max(spectrum)), ...
                 '-', 'Color', plotFigures.COLORS.classic_music, ...
                 'LineWidth', plotFigures.LINE_WIDTH);
            hold on;
            
            % Mark estimated DOAs
            if ~isempty(DoAMUSIC)
                plot(rad2deg(theta_range(DoAMUSIC)), ...
                     10*log10(spectrum(DoAMUSIC)/max(spectrum)), 'x', ...
                     'Color', plotFigures.COLORS.estimated, ...
                     'MarkerSize', plotFigures.MARKER_SIZE, ...
                     'LineWidth', 2);
            end
            
            % Mark true DOAs
            plot(rad2deg(true_doas), zeros(size(true_doas)) - 5, '*', ...
                 'Color', plotFigures.COLORS.true_doa, ...
                 'MarkerSize', plotFigures.MARKER_SIZE + 4, ...
                 'LineWidth', 2, 'MarkerFaceColor', plotFigures.COLORS.true_doa);
            
            % Formatting
            xlabel('Angle (degrees)', 'FontSize', plotFigures.FONT_SIZE);
            ylabel('Normalized Spectrum (dB)', 'FontSize', plotFigures.FONT_SIZE);
            title(fig_title, 'FontSize', plotFigures.TITLE_FONT_SIZE);
            
            % Create custom legend
            legend_items = {'Classic MUSIC'};
            if ~isempty(DoAMUSIC)
                legend_items{end+1} = 'Estimated DOA';
            end
            legend_items{end+1} = 'True DOA';
            
            legend(legend_items, 'Location', 'best', ...
                   'FontSize', plotFigures.LEGEND_FONT_SIZE);
            
            grid on;
            xlim([-90, 90]);
            ylim([-40, 5]);
        end
        
        
        %************************#
        %   plot est. spectrum   #
        %************************#
        function fig = plotAugMUSIC(pred_spec, DoA, true_doas, theta_range, fig_title)
            % Plot augmented MUSIC results.
            %
            % @param pred_spec -- Augmented MUSIC spectrum
            % @param DoA -- Estimated DOA indices
            % @param true_doas -- True DOAs (radians)
            % @param theta_range -- Angle range (radians)
            % @param fig_title -- Figure title (optional)
            %
            % @returns -- Figure handle
            
            if nargin < 5
                fig_title = 'Augmented MUSIC Algorithm';
            end
            
            fig = figure('Position', [100, 100, 800, 500], ...
                         'Name', fig_title, 'Color', 'w');
            
            % Plot augmented MUSIC spectrum
            plot(rad2deg(theta_range), 10*log10(pred_spec/max(pred_spec)), ...
                 '-', 'Color', plotFigures.COLORS.aug_music, ...
                 'LineWidth', plotFigures.LINE_WIDTH);
            hold on;
            
            % Mark estimated DOAs
            if ~isempty(DoA)
                plot(rad2deg(theta_range(DoA)), ...
                     10*log10(pred_spec(DoA)/max(pred_spec)), 'x', ...
                     'Color', plotFigures.COLORS.estimated, ...
                     'MarkerSize', plotFigures.MARKER_SIZE, ...
                     'LineWidth', 2);
            end
            
            % Mark true DOAs
            plot(rad2deg(true_doas), zeros(size(true_doas)) - 5, '*', ...
                 'Color', plotFigures.COLORS.true_doa, ...
                 'MarkerSize', plotFigures.MARKER_SIZE + 4, ...
                 'LineWidth', 2, 'MarkerFaceColor', plotFigures.COLORS.true_doa);
            
            % Formatting
            xlabel('Angle (degrees)', 'FontSize', plotFigures.FONT_SIZE);
            ylabel('Normalized Spectrum (dB)', 'FontSize', plotFigures.FONT_SIZE);
            title(fig_title, 'FontSize', plotFigures.TITLE_FONT_SIZE);
            
            % Create custom legend
            legend_items = {'Aug MUSIC'};
            if ~isempty(DoA)
                legend_items{end+1} = 'Estimated DOA';
            end
            legend_items{end+1} = 'True DOA';
            
            legend(legend_items, 'Location', 'best', ...
                   'FontSize', plotFigures.LEGEND_FONT_SIZE);
            
            grid on;
            xlim([-90, 90]);
            ylim([-40, 5]);
        end
        
        
        %**************************#
        %   plot all algorithms    #
        %**************************#
        function fig = plotAllAlgorithms(x, array, theta_range, true_doas, d, fig_title)
            % Run and plot all algorithms on given data.
            %
            % @param x -- Measurement data [m x snapshots]
            % @param array -- Array element positions
            % @param theta_range -- Angle range (radians)
            % @param true_doas -- True DOAs (radians)
            % @param d -- Number of sources
            % @param fig_title -- Figure title (optional)
            %
            % @returns -- Figure handle and results structure
            
            if nargin < 6
                fig_title = 'DOA Algorithm Comparison';
            end
            
            % Run all algorithms
            fprintf('Running algorithms for comparison...\n');
            
            % Beamformer
            tic;
            [DoABF, spectrumBF] = beamformer_all.bartlett(x, array, theta_range, d);
            bf_time = toc;
            fprintf('  Beamformer: %.4f s\n', bf_time);
            
            % Classic MUSIC
            tic;
            [DoAMUSIC, spectrumMUSIC] = classicMUSIC.estimate(x, array, theta_range, d);
            music_time = toc;
            fprintf('  Classic MUSIC: %.4f s\n', music_time);
            
            % Note: Augmented MUSIC requires neural network model
            % Will plot placeholder if model not available
            
            % Create comparison plot
            fig = figure('Position', [100, 100, 1200, 600], ...
                         'Name', fig_title, 'Color', 'w');
            
            % Plot 1: All spectra
            subplot(1, 2, 1);
            
            % Beamformer spectrum
            plot(rad2deg(theta_range), 10*log10(spectrumBF/max(spectrumBF)), ...
                 '--', 'Color', plotFigures.COLORS.beamformer, ...
                 'LineWidth', plotFigures.LINE_WIDTH, 'DisplayName', 'Beamformer');
            hold on;
            
            % MUSIC spectrum
            plot(rad2deg(theta_range), 10*log10(spectrumMUSIC/max(spectrumMUSIC)), ...
                 '-', 'Color', plotFigures.COLORS.classic_music, ...
                 'LineWidth', plotFigures.LINE_WIDTH, 'DisplayName', 'Classic MUSIC');
            
            % True DOAs
            plot(rad2deg(true_doas), zeros(size(true_doas)) - 5, '*', ...
                 'Color', plotFigures.COLORS.true_doa, ...
                 'MarkerSize', plotFigures.MARKER_SIZE + 4, ...
                 'LineWidth', 2, 'MarkerFaceColor', plotFigures.COLORS.true_doa, ...
                 'DisplayName', 'True DOA');
            
            % Estimated DOAs
            if ~isempty(DoABF)
                plot(rad2deg(theta_range(DoABF)), ...
                     10*log10(spectrumBF(DoABF)/max(spectrumBF)), 'x', ...
                     'Color', plotFigures.COLORS.beamformer, ...
                     'MarkerSize', plotFigures.MARKER_SIZE, ...
                     'LineWidth', 2, 'DisplayName', 'BF Est.');
            end
            
            if ~isempty(DoAMUSIC)
                plot(rad2deg(theta_range(DoAMUSIC)), ...
                     10*log10(spectrumMUSIC(DoAMUSIC)/max(spectrumMUSIC)), 's', ...
                     'Color', plotFigures.COLORS.classic_music, ...
                     'MarkerSize', plotFigures.MARKER_SIZE - 2, ...
                     'LineWidth', 2, 'MarkerFaceColor', plotFigures.COLORS.classic_music, ...
                     'DisplayName', 'MUSIC Est.');
            end
            
            xlabel('Angle (degrees)', 'FontSize', plotFigures.FONT_SIZE);
            ylabel('Normalized Spectrum (dB)', 'FontSize', plotFigures.FONT_SIZE);
            title('Algorithm Spectra', 'FontSize', plotFigures.TITLE_FONT_SIZE);
            legend('Location', 'best', 'FontSize', plotFigures.LEGEND_FONT_SIZE);
            grid on;
            xlim([-90, 90]);
            ylim([-40, 5]);
            
            % Plot 2: Performance metrics
            subplot(1, 2, 2);
            
            % Calculate errors
            if ~isempty(DoABF) && length(DoABF) == length(true_doas)
                bf_errors = abs(sort(theta_range(DoABF)) - sort(true_doas));
                bf_rmse = sqrt(mean(bf_errors.^2));
            else
                bf_rmse = NaN;
            end
            
            if ~isempty(DoAMUSIC) && length(DoAMUSIC) == length(true_doas)
                music_errors = abs(sort(theta_range(DoAMUSIC)) - sort(true_doas));
                music_rmse = sqrt(mean(music_errors.^2));
            else
                music_rmse = NaN;
            end
            
            % Bar chart of RMSE
            algorithms = {'Beamformer', 'MUSIC'};
            rmse_values = [rad2deg(bf_rmse), rad2deg(music_rmse)];
            times = [bf_time, music_time];
            
            % Create bar chart
            bar_data = [rmse_values; times];
            h = bar(bar_data');
            
            % Set colors
            h(1).FaceColor = plotFigures.COLORS.beamformer;
            h(2).FaceColor = plotFigures.COLORS.classic_music;
            
            set(gca, 'XTickLabel', algorithms);
            ylabel('Value', 'FontSize', plotFigures.FONT_SIZE);
            title('Performance Metrics', 'FontSize', plotFigures.TITLE_FONT_SIZE);
            legend({'RMSE (°)', 'Time (s)'}, 'Location', 'best', ...
                   'FontSize', plotFigures.LEGEND_FONT_SIZE);
            grid on;
            
            % Add value labels
            for i = 1:length(algorithms)
                text(i, rmse_values(i) + 0.1*max(rmse_values), ...
                     sprintf('%.2f°', rmse_values(i)), ...
                     'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'bottom', 'FontSize', 10);
                text(i, times(i) + 0.1*max(times), ...
                     sprintf('%.3f s', times(i)), ...
                     'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'bottom', 'FontSize', 10);
            end
            
            % Add info annotation
            info_str = sprintf('Array: %d elements\nSources: %d\nSnapshots: %d', ...
                               length(array), d, size(x, 2));
            annotation('textbox', [0.02, 0.95, 0.2, 0.05], ...
                       'String', info_str, 'FontSize', 10, ...
                       'EdgeColor', 'none', 'BackgroundColor', [0.9, 0.9, 0.9]);
            
            % Store results
            results = struct();
            results.DoABF = DoABF;
            results.spectrumBF = spectrumBF;
            results.DoAMUSIC = DoAMUSIC;
            results.spectrumMUSIC = spectrumMUSIC;
            results.bf_rmse = bf_rmse;
            results.music_rmse = music_rmse;
            results.bf_time = bf_time;
            results.music_time = music_time;
            
            fprintf('Comparison complete!\n');
        end
        
        
        %*****************************#
        %   plot error distribution   #
        %*****************************#
        function fig = plotErrorDistribution(pred_doas, true_doas, theta_range, fig_title)
            % Plot error distribution for DOA estimation.
            %
            % @param pred_doas -- Predicted DOA indices [num_samples x num_sources]
            % @param true_doas -- True DOAs [num_samples x num_sources]
            % @param theta_range -- Angle range (radians)
            % @param fig_title -- Figure title (optional)
            %
            % @returns -- Figure handle
            
            if nargin < 4
                fig_title = 'DOA Error Distribution';
            end
            
            % Convert indices to angles
            pred_angles = theta_range(pred_doas);
            
            % Calculate errors
            errors_deg = zeros(size(pred_angles));
            for i = 1:size(pred_angles, 1)
                pred_sorted = sort(pred_angles(i, :));
                true_sorted = sort(true_doas(i, :));
                errors_deg(i, :) = rad2deg(abs(pred_sorted - true_sorted));
            end
            
            % Flatten errors
            errors_flat = errors_deg(:);
            errors_flat = errors_flat(~isnan(errors_flat));  % Remove NaNs
            
            % Create figure
            fig = figure('Position', [100, 100, 1000, 400], ...
                         'Name', fig_title, 'Color', 'w');
            
            % Plot 1: Histogram
            subplot(1, 2, 1);
            histogram(errors_flat, 30, 'FaceColor', [0.2, 0.4, 0.8], ...
                      'EdgeColor', 'black', 'FaceAlpha', 0.7);
            
            % Calculate statistics
            mean_error = mean(errors_flat);
            median_error = median(errors_flat);
            std_error = std(errors_flat);
            max_error = max(errors_flat);
            
            % Add statistics lines
            hold on;
            plot([mean_error, mean_error], ylim, 'r-', 'LineWidth', 2, ...
                 'DisplayName', sprintf('Mean: %.2f°', mean_error));
            plot([median_error, median_error], ylim, 'g--', 'LineWidth', 2, ...
                 'DisplayName', sprintf('Median: %.2f°', median_error));
            
            xlabel('Error (degrees)', 'FontSize', plotFigures.FONT_SIZE);
            ylabel('Frequency', 'FontSize', plotFigures.FONT_SIZE);
            title('Error Distribution', 'FontSize', plotFigures.TITLE_FONT_SIZE);
            legend('Location', 'best', 'FontSize', plotFigures.LEGEND_FONT_SIZE);
            grid on;
            
            % Plot 2: Box plot
            subplot(1, 2, 2);
            boxplot(errors_flat, 'Orientation', 'horizontal');
            
            % Add statistics text
            stats_str = {sprintf('Mean: %.2f°', mean_error), ...
                         sprintf('Std: %.2f°', std_error), ...
                         sprintf('Median: %.2f°', median_error), ...
                         sprintf('Max: %.2f°', max_error), ...
                         sprintf('N = %d', length(errors_flat))};
            
            text(max_error * 0.7, 1.3, stats_str, ...
                 'FontSize', 10, 'VerticalAlignment', 'top', ...
                 'BackgroundColor', [0.95, 0.95, 0.95]);
            
            xlabel('Error (degrees)', 'FontSize', plotFigures.FONT_SIZE);
            title('Error Statistics', 'FontSize', plotFigures.TITLE_FONT_SIZE);
            grid on;
            
            % Overall title
            sgtitle(fig_title, 'FontSize', plotFigures.TITLE_FONT_SIZE + 2, ...
                    'FontWeight', 'bold');
        end
        
        
        %**********************************#
        %   plot resolution capability     #
        %**********************************#
        function fig = plotResolutionTest(x_close, array, theta_range, ...
                true_doas_close, separation_deg, fig_title)
            % Plot resolution test for closely spaced sources.
            %
            % @param x_close -- Measurement data for close sources
            % @param array -- Array element positions
            % @param theta_range -- Angle range (radians)
            % @param true_doas_close -- True DOAs for close sources
            % @param separation_deg -- Source separation in degrees
            % @param fig_title -- Figure title (optional)
            %
            % @returns -- Figure handle
            
            if nargin < 6
                fig_title = sprintf('Resolution Test (%.1f° Separation)', separation_deg);
            end
            
            d_close = length(true_doas_close);
            
            % Run algorithms
            [DoABF, spectrumBF] = beamformer_all.bartlett(x_close, array, theta_range, d_close);
            [DoAMUSIC, spectrumMUSIC] = classicMUSIC.estimate(x_close, array, theta_range, d_close);
            
            % Create figure
            fig = figure('Position', [100, 100, 1000, 400], ...
                         'Name', fig_title, 'Color', 'w');
            
            % Plot 1: Spectra
            subplot(1, 2, 1);
            
            % Plot spectra
            plot(rad2deg(theta_range), 10*log10(spectrumBF/max(spectrumBF)), ...
                 '--', 'Color', plotFigures.COLORS.beamformer, ...
                 'LineWidth', plotFigures.LINE_WIDTH, 'DisplayName', 'Beamformer');
            hold on;
            
            plot(rad2deg(theta_range), 10*log10(spectrumMUSIC/max(spectrumMUSIC)), ...
                 '-', 'Color', plotFigures.COLORS.classic_music, ...
                 'LineWidth', plotFigures.LINE_WIDTH, 'DisplayName', 'MUSIC');
            
            % Mark true DOAs
            plot(rad2deg(true_doas_close), zeros(size(true_doas_close)) - 5, '*', ...
                 'Color', plotFigures.COLORS.true_doa, ...
                 'MarkerSize', plotFigures.MARKER_SIZE + 6, ...
                 'LineWidth', 2, 'MarkerFaceColor', plotFigures.COLORS.true_doa, ...
                 'DisplayName', 'True DOA');
            
            % Zoom in on region of interest
            zoom_margin = separation_deg * 2;
            zoom_min = min(rad2deg(true_doas_close)) - zoom_margin;
            zoom_max = max(rad2deg(true_doas_close)) + zoom_margin;
            
            xlabel('Angle (degrees)', 'FontSize', plotFigures.FONT_SIZE);
            ylabel('Normalized Spectrum (dB)', 'FontSize', plotFigures.FONT_SIZE);
            title('Resolution Test Spectra', 'FontSize', plotFigures.TITLE_FONT_SIZE);
            legend('Location', 'best', 'FontSize', plotFigures.LEGEND_FONT_SIZE);
            grid on;
            xlim([zoom_min, zoom_max]);
            ylim([-40, 5]);
            
            % Plot 2: Mainlobe comparison
            subplot(1, 2, 2);
            
            % Find and plot mainlobes
            [bf_width, bf_sidelobe] = plotFigures.analyze_mainlobe(spectrumBF, theta_range);
            [music_width, music_sidelobe] = plotFigures.analyze_mainlobe(spectrumMUSIC, theta_range);
            
            metrics = {'Mainlobe Width (°)', 'Sidelobe Level (dB)'};
            bf_metrics = [bf_width, bf_sidelobe];
            music_metrics = [music_width, music_sidelobe];
            
            bar_data = [bf_metrics; music_metrics];
            h = bar(bar_data);
            
            h(1).FaceColor = plotFigures.COLORS.beamformer;
            h(2).FaceColor = plotFigures.COLORS.classic_music;
            
            set(gca, 'XTickLabel', {'Beamformer', 'MUSIC'});
            ylabel('Value', 'FontSize', plotFigures.FONT_SIZE);
            title('Resolution Metrics', 'FontSize', plotFigures.TITLE_FONT_SIZE);
            legend(metrics, 'Location', 'best', 'FontSize', plotFigures.LEGEND_FONT_SIZE);
            grid on;
            
            % Add info
            info_str = sprintf('Separation: %.1f°\nArray: %d elements', ...
                               separation_deg, length(array));
            annotation('textbox', [0.02, 0.95, 0.2, 0.05], ...
                       'String', info_str, 'FontSize', 10, ...
                       'EdgeColor', 'none', 'BackgroundColor', [0.9, 0.9, 0.9]);
        end
        
        
        %**************************#
        %   helper functions      #
        %**************************#
        function [width_deg, sidelobe_db] = analyze_mainlobe(spectrum, theta_range)
            % Analyze mainlobe characteristics.
            %
            % @param spectrum -- Spatial spectrum
            % @param theta_range -- Angle range
            %
            % @returns -- Mainlobe width (degrees) and peak sidelobe level (dB)
            
            % Find main peak
            [~, peak_idx] = max(spectrum);
            
            % Find -3dB points
            threshold = spectrum(peak_idx) / sqrt(2);  % -3dB
            
            % Find left -3dB point
            left_idx = peak_idx;
            while left_idx > 1 && spectrum(left_idx) > threshold
                left_idx = left_idx - 1;
            end
            
            % Find right -3dB point
            right_idx = peak_idx;
            while right_idx < length(spectrum) && spectrum(right_idx) > threshold
                right_idx = right_idx + 1;
            end
            
            % Calculate width
            width_deg = rad2deg(theta_range(right_idx) - theta_range(left_idx));
            
            % Find peak sidelobe (excluding mainlobe)
            exclude_region = max(1, left_idx - 5):min(length(spectrum), right_idx + 5);
            sidelobe_region = setdiff(1:length(spectrum), exclude_region);
            
            if ~isempty(sidelobe_region)
                sidelobe_linear = max(spectrum(sidelobe_region));
                sidelobe_db = 10 * log10(sidelobe_linear / spectrum(peak_idx));
            else
                sidelobe_db = -Inf;
            end
        end
        
        
        %**************************#
        %   export figure         #
        %**************************#
        function exportFigure(fig, filename, format, dpi)
            % Export figure to file.
            %
            % @param fig -- Figure handle
            % @param filename -- Output filename (without extension)
            % @param format -- File format: 'png', 'pdf', 'eps', 'fig'
            % @param dpi -- Resolution (default: 300)
            
            if nargin < 4
                dpi = 300;
            end
            
            if nargin < 3
                format = 'png';
            end
            
            % Ensure figure is current
            figure(fig);
            
            % Create export directory if needed
            [filepath, ~, ~] = fileparts(filename);
            if ~isempty(filepath) && ~exist(filepath, 'dir')
                mkdir(filepath);
            end
            
            % Add extension if not present
            if ~endsWith(filename, ['.' format])
                filename = [filename '.' format];
            end
            
            % Export based on format
            switch lower(format)
                case 'png'
                    print(fig, filename, '-dpng', ['-r' num2str(dpi)]);
                case 'pdf'
                    print(fig, filename, '-dpdf', '-bestfit');
                case 'eps'
                    print(fig, filename, '-depsc', ['-r' num2str(dpi)]);
                case 'fig'
                    savefig(fig, filename);
                otherwise
                    error('Unsupported format: %s', format);
            end
            
            fprintf('Figure exported to: %s\n', filename);
        end
        
    end
end