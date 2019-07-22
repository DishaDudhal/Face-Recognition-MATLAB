%% Load dataset
clc;
clear all;
faceGal = imageSet('OurClass','recursive');
%% Detect Face
i= 2;
    for j = 1:faceGal(i).Count
        queryFace = read(faceGal(i),j);
        faceDetector = vision.CascadeObjectDetector; %% This step uses the Viola Jones algorithm to detect people's faces, noses, eyes, mouth, upper body...^M
        bbox = faceDetector(queryFace); %% this step returns an M ny 4 matrix containing the biunding box coordinates of the detected faces....^M
        if ne(size(bbox,1),0)
        queryFace = imcrop(queryFace,bbox);
        scaleFactor = 150/size(queryFace,1);
        queryFace = imresize (queryFace, scaleFactor);
        imwrite(queryFace,sprintf('%d%d.jpg',i,j));
        end
    end
