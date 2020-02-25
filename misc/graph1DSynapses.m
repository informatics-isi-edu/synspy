%% Graphing 1-Dimensional Synapse Data
% This script takes 1 dimensional data from a field of real data synapses
% and graphs what would be selected by the synapse identification software
% (synspy) as a candidate synapse.
%
% Note that here, one must manually determine the peaks using assistance
% from graphs, just to make things straightforward. If one of the peaks is
% a local peak, then it can be selected. This peak selection process is
% repeated until the entire line of intensity data (1D) is traversed.
%
% Note that the software will crash if there are no synapses or peaks
% within the 1D line segment that was selected.
%

function [] = graph1DSynapses(fileName_raw,fileName_blur,filePath,...
    Cache_Loc,backgroundThres)
% fileName_raw: .csv file containing the 1D profile of intensity data
%       (using "line" and "Analyze->Plot Profile" tools in fiji)
% fileName_blur: .csv file containing the 1D profile of intensity data
%       after applying a 2.16 pixel blur (e.g., 0.5625 µm for SPIM data)
% filePath: The file path leading to the .csv files to be graphed
% Cache_Loc: Path to where you want to output the .csv files and graphs
% pixSize: This is the size of the pixels in the x dimension (should be
%       apparent from the result of the "plot profile" analysis in fiji
% backgroundThres: A variable that is used to estimate how close to
%       background the signal should be outside of the radius of an
%       identified synapse (e.g., 0.5, for within 50% of background)
%% Important Variables
% Graphing
  % Set default linewidth to a certain value;
  set(groot,'defaultLineLineWidth',2.0)
  % Color for individual datapoints (black, red , green, yellow, cyan)
  colors = [0,0,0; 0.8 0 0; 0 0.8 0; 0.8 0.8 0; 0 0.8 0.8];
  
  % The next variable represents half of the width of the kernel that 
  % scans looking for candidate synapses. As long as there is not a higher
  % value within this range when the candidate synapse (a local peak) is
  % centered within the range, then it is a potential synapse. If there is
  % a local higher value, then it is not considered as a synapse. Value of
  % 1.3/2 is imposed by the synspy software.
  binRad = 1.3/2; % Radius (um) of bins around each possible candidate
  % Note that this is dependent on the pixel size of the image. For this
  % paper, the pixel size for SPIM imaging was 0.2600036 µm. This value can
  % be adjusted (keeping track in synspy) for images with different pixel 
  % sizes.
  
  % Core and Hollow radii represent the shapes that are pulled out of each
  % image after identification. They are imposed by the synspy software.
  coreRad = 1.5/2; % microns
  
  
%% Input Data
% Make sure the table contains just the X and Y columns from the fiji plot
% profile function. Preferably, the proper pixel size should be
% incorporated into the x value of the plot (the spatial dimension) while
% the Y column represents intensity data. If not, make sure to incorporate
% that before running this script.
rawData = readtable([filePath,fileName_raw]);
blurData = readtable([filePath,fileName_blur]);

%% Manually Determining Peaks

% Note: in this section, graphs will appear. A grader must choose whether a
% particular peak is a local maximum within a certain range. The range is
% indicated by red vertical lines on the left and right hand sides of each
% graph. The peak is indicated by a blue arrow, and a green horizontal line
% emanates on either side of the peak. As long as the light blue plotted
% line does not cross from below to above the green horizontal line 
% while still being within the red line boundaries, then it is a true local
% maximum within the binRad definition.

% Raw data
  % Find peaks to add to plot
  [posPeaksRaw,peakValsRaw] = goThruPeaks(rawData.X,rawData.Y,colors,...
      binRad);
    PeakPositionsRawData = posPeaksRaw';
    PeakValuesRawData = peakValsRaw';
  rawTable = table(PeakPositionsRawData,PeakValuesRawData);
  writetable(rawTable,[Cache_Loc,'CalculatedPeakValues_RawData.csv']);
  
% Blurred data
  % Find peaks to add to plot
  [posPeaksBlur,peakValsBlur] = ...
      goThruPeaks(blurData.X,blurData.Y,colors,binRad);
    PeakPositionsGaussianBlurData = posPeaksBlur';
    PeakValuesGaussianBlurData = peakValsBlur';
  blurTable = table(PeakPositionsGaussianBlurData,...
      PeakValuesGaussianBlurData);
  writetable(blurTable,...
      [Cache_Loc,'CalculatedPeakValues_2pt16Pixel_GaussianBlurData.csv']);

%% Identifying Synapses from Peaks
% Now, choosing which data would actually be segmented
    % Using other peaks in raw value, add to the plot
    [posChosenPeaks,chosenPeakVals] = ...
        chooseSyns(blurTable.PeakPositionsGaussianBlurData,...
        blurTable.PeakValuesGaussianBlurData,coreRad,...
        rawData.X,rawData.Y,backgroundThres);
      PeakPositionsOfChosenSynapses = posChosenPeaks';
      PeakValuesOfChosenSynapses = chosenPeakVals';
    chosenSynsTable = table(PeakPositionsOfChosenSynapses,...
        PeakValuesOfChosenSynapses);
    writetable(chosenSynsTable,...
        [Cache_Loc,'ChosenSynapses.csv']);

%% Displaying final plots for the accepted peaks
% Raw data
figRaw = figure,
linePlotter(rawData.X,rawData.Y,colors)
figRaw.OuterPosition = [140 60 410 280];
saveas(figRaw,[Cache_Loc,'RawDataProfile.svg'],'svg')
% Now, just save the peaks that would be segmented
coreFill(PeakPositionsOfChosenSynapses,coreRad,rawData.X,rawData.Y);
linePlotter(rawData.X,rawData.Y,colors)
  saveas(figRaw,[Cache_Loc,'RawDataProfile_withFill.svg'],'svg')

hold off
% Blurred data
figBlur = figure,
linePlotter(blurData.X,blurData.Y,colors)
figBlur.OuterPosition = [140 60 410 280];
saveas(figBlur,[Cache_Loc,'GaussianBlurDataProfile.svg'],'svg')
hold off
end
%% Helper Functions
    function [] = linePlotter(xVal,yVal,colors)
        % Plots the data in a line where xVal are the x positions and yVal
        % are the intensities
        linew = 2;
        
        HH = plot(xVal',yVal','linewidth',linew);

        hold on
        set(HH, {'color'}, num2cell(colors(1,:),2));
        set(gca,'linewidth',2)
        set(gca,'box','off')
        g = gca;
        g.YLim = [0 max(max(yVal))*1.1];
        g.XLim = [0 max(xVal)+1];
        g.FontSize=24;
        g.FontWeight='bold';
        output_args = [];
        hold on
    end
    
    function [positionsOfTruePeaks,valuesOfTruePeaks] = ...
        goThruPeaks(xVal,yVal,colors,binRadius)
    % Plot the peaks for the yValues relative to the xVal positions
    % This enables individual manual selection of whether something is a
    % local peak or not. If it is a local peak, the user should select it.
    % If not, then the user should select "no". Peaks will ultimately be
    % highlighted in a red background color. binRadius refers to the size
    % of the bins in the sliding window where the scan for local maxima is
    % performed.
    
    handleFig = figure, findpeaks(yVal,xVal);
    handleFig.Position = [1 168 560 420];
    yl = get(gca,'YLim'); % Get the Y limits of the graph for help lines
    xl = get(gca,'XLim'); % Get the X limits of the graph for help lines
    % Store peaks and positions
    [peakVals,locs] = findpeaks(yVal,xVal);
    counter = 1;
    g = gca;
    for i = 1:numel(locs)
        g.XLim = [locs(i)-1.05*binRadius locs(i)+1.05*binRadius];
        g.YLim = [0 1.1*peakVals(i)];
        % Helper Vertical Lines
        line([locs(i)+binRadius,locs(i)+binRadius],yl,...
            'LineStyle','-',...
            'Color',colors(2,:),'LineWidth',2);
        line([locs(i)-binRadius,locs(i)-binRadius],yl,...
            'LineStyle','-',...
            'Color',colors(2,:),'LineWidth',2)
        % Helper Horizontal Line
        
        line(xl,[peakVals(i) peakVals(i)],...
            'LineStyle','-',...
            'Color',colors(3,:),'LineWidth',1)
        list = {'Yes','No'};
        [indx,~] = listdlg('PromptString',...
            'Is this a candidate synapse? (no higher value in the range)',...
            'SelectionMode','Single','ListString',list);
        if indx == 1
            positionsOfTruePeaks(counter) = locs(i);
            valuesOfTruePeaks(counter) = peakVals(i);
            counter = counter + 1;
        end
        children = get(gca,'children');
        delete(children(1));
        delete(children(2));
        delete(children(3));
    end
    close(gcf)
    end
    
    function [] = coreFill(xVal,coreRadius,xSpace,ySpace)
    % After they are identified, selected "synaptic" peaks are filled in
    % with a light red color using this function. xVal is a vector
    % containing the positions of possible synapses. xSpace and ySpace
    % are the full data range. coreRadius is the radius around the center
    % xVal where the red fill occurs.
    
     alphaVal = 0.3; % Transparency value
    for i = 1:numel(xVal)
        % Fill around the identified synapse
        % First, find the nearest value from the xVals outside of the 
        % limits of the coreRadius
        [~,closestIndexLow] = ...
            min(abs(xSpace-(xVal(i)-coreRadius)));
        [~,closestIndexHi] = ...
            min(abs(xSpace-(xVal(i)+coreRadius)));
        % Now, get ready for interpolation for the y values
        posForInterpolation = ...
            xSpace(closestIndexLow):.0100:xSpace(closestIndexHi);
        % intp is just an arbitrary variable standing for the interpolation
        intp = interp1(xSpace(closestIndexLow:closestIndexHi),...
            ySpace(closestIndexLow:closestIndexHi),...
            posForInterpolation,'linear');
        % Fill the interpolated values
        x_Fill = [xSpace(closestIndexLow),posForInterpolation,...
            xSpace(closestIndexHi),xSpace(closestIndexLow)];
        y_Fill = [0,intp,0,0];
        h = fill(x_Fill,y_Fill,'r');
        set(h,'EdgeColor','none')
        set(h,'facealpha',alphaVal); % Adjust transparency
    end
    end
    
    function [chosenPos,chosenPeak] = chooseSyns(xVal,yVal,coreRadius,...
        xSpace,ySpace,bcTh)
    % After the potential peak locations are chosen (xVals) using the 
    % blurred data (corresponds to the peak values after blurring is done),
    % determine whether each are true synapses by noticing 
    % whether the intensity at +/-coreRadius from a synapse centroid
    % is close to the level of background, using bcTh as a parameter for
    % the background threshold value input into the function originally. 
    % xSpace and ySpace are the full data range. This will be used to
    % select synapses instead of nuclei. yVal are the peak values for the
    % given xVal potential synapse positions.
    % IMPORTANT NOTE: This is NOT how the 3D classification
    % algorithm is performed in synspy. This is just a 1D example of how
    % one could threshold potential candidate synapses. In the actual 3D
    % classification, blinded human participants would validate synapse
    % choices after classification by the algorithm.
        % Now, find the mean of the lowest 10% of intensity values in the
        % line of data to get a conservative estimate of the background
        sort_ySpace = sort(ySpace,1,'ascend');
        bk_ySpace = mean(...
            sort_ySpace(1:round(size(sort_ySpace)*.10),:),1);
        counter = 1;
    for i = 1:numel(xVal)
        % See whether there is a local minimum around xVal by looking at
        % local minima of the ySpace parameters within a coreRadius range
        % around the potential peak locations
        dummyPos = find(xSpace>(xVal(i)-coreRadius)&xSpace<...
            (xVal(i)+coreRadius));
        % We only care about the first and last value for the range
        dummyPos = [dummyPos(1),dummyPos(end)];
        % Now, calculate how far from background the signal outside of the
        % coreRadius actually is and whether it is within bcTh*dummyF0 of
        % bck_ySpace. Since synapses are more or less symmetric, average
        % the data on either side of the potential synapse peak.
        dummyBc = (ySpace(dummyPos(1))+ySpace(dummyPos(end)))/2;
        % IF the background around the candidate synapse is within 20% of
        % the background of the 1D line of intensity data, then it is
        % classified as a synapse. Otherwise, it is scrapped.
        if dummyBc < bk_ySpace*(1+bcTh)
            chosenPos(counter) = xVal(i);
            chosenPeak(counter) = yVal(i);
            counter = counter+1;
        end
    end
    end