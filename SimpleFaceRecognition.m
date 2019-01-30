%% Load Image Information from Face Database
clc;
clear all;
faceDatabase = imageSet('OurClass','recursive'); 
% Training & Test Sets
training = faceDatabase;
test = imageSet('testClass','recursive');

%% ---------Extract and display Histogram of Oriented Gradient Features for single face ------------
person = 1;
[hogFeature, visualization]= extractHOGFeatures(read(training(person),1));
figure;
subplot(2,1,1);
imshow(read(training(person),1));
title('Input Face');
subplot(2,1,2);
plot(visualization);
title('HoG Feature');


%% ----------------------------Extract HOG Features for training set -----------------------------------
trainingFeatures = zeros(size(training,2)*training(1).Count,10404);
featureCount = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
        sizeNormalizedImage = rgb2gray(read(training(i),j));
        trainingFeatures(featureCount,:) = extractHOGFeatures(sizeNormalizedImage);
        trainingLabel{featureCount} = training(i).Description;    
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end

%% --------------------------Create class classifier using fitcecoc--------------------------------------
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);
% Test Images from Test Set 
queryFace = read(test,1);

%% --------------Detect Face from test image using  Viola-Jones Algorithm---------------------------------
faceDetector = vision.CascadeObjectDetector;
bbox = faceDetector(queryFace);
queryFace = imcrop(queryFace,bbox);
scaleFactor = 150/size(queryFace,1);
queryFace = imresize(queryFace, scaleFactor);
queryNormalizedImage = rgb2gray(queryFace);
queryFeatures = extractHOGFeatures(queryNormalizedImage);
personLabel = predict(faceClassifier,queryFeatures);

% Map back to training set to find identity 
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
subplot(1,2,1);imshow(queryNormalizedImage);title('Query Face');
subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched Class');

 %% -------------------------------------------Test from Test Set-------------------------------------------
figure;
figureNum = 1;
    for j = 1:test.Count
        queryFace = read(test,j);
        faceDetector = vision.CascadeObjectDetector;
        bbox = faceDetector(queryFace);
        try ne(size(bbox,1),0)
            queryFace = imcrop(queryFace,bbox);
            scaleFactor = 150/size(queryFace,1);
            queryFace = imresize (queryFace, scaleFactor);
            queryNormalizedImage = rgb2gray(queryFace);
        catch ME
            queryNormalizedImage = imresize (rgb2gray(queryFace),[150 150]);
        end
        queryFeatures = extractHOGFeatures(queryNormalizedImage);
        personLabel = predict(faceClassifier,queryFeatures);
        % Map back to training set to find identity
        subplot(round(test.Count/2),4,figureNum);
        imshow(imresize(queryNormalizedImage,3));title('Query Face');
        subplot(round(test.Count/2),4,figureNum+1);text(0.5,0.5,personLabel);
        figureNum = figureNum+2;
    end
    figure;
    figureNum = 1;
    
%% ----------------------------------------Create the face detector object.------------------------------------
faceDetector = vision.CascadeObjectDetector();

% Create the point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Create the webcam object.
cam = webcam();

% Capture first frame
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

%% ------------------------------------------Classifying in frames----------------------------------------------
runLoop = true;
numPts = 0;
frameCount = 0;

while runLoop && frameCount < 200

    % Get the next frame.
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;

    if numPts < 500
        bbox = faceDetector(videoFrameGray);
        if ~isempty(bbox)
            % Find corners inside the ROI
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));

            % Re-initialize the point tracker.
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

            % copy of the points.
            oldPoints = xyPoints;
% -----------------------------------------------------------------------------------------------------------------
% Convert the rectangle represented as [x, y, w, h] into an M-by-2 matrix of [x,y] coordinates of the four corners
% This is needed to be able to transform the bounding box to display the orientation of the face
            
            bboxPoints = bbox2points(bbox(1, :));
            bboxPolygon = reshape(bboxPoints, 1, []);
            %Filtering face from the snapshot
            box = faceDetector(videoFrameGray);
            queryFace = imcrop(videoFrameGray,box);
            scaleFactor = 150/size(queryFace,1);
            queryFace = imresize (queryFace, scaleFactor);
            queryFeatures = extractHOGFeatures(queryFace);
            label = predict(faceClassifier,queryFeatures);
                       
            % Display a bounding box around the detected face
            videoFrame = insertObjectAnnotation(videoFrame,'rectangle',bbox,label);
            % Display detected corners
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end
    else
        % Tracking mode
        [xyPoints, isFound] = pointTracker(videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);
        numPts = size(visiblePoints, 1);

        if numPts >= 500
            % Estimate the geometric transformation between the old points and the new points
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform( oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

            % Apply the transformation to the bounding box
            bboxPoints = transformPointsForward(xform, bboxPoints);

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4] format required by insertShape
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the face being tracked
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
           
            % Display tracked points
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');
            videoFrame = insertText(videoFrame, bboxPolygon(1:2), label);

            % Reset the points
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
            %Filtering face from the snapshot
            box = faceDetector(videoFrameGray);
        end
    end
    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);

    % Check whether the video player window is open
    runLoop = isOpen(videoPlayer);
end

clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);
