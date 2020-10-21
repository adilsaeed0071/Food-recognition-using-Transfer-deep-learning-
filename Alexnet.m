clc
clear all;
close all;


imds = imageDatastore('D:\ECU\Digital signal processing\SVM\02 Training\training2','IncludeSubfolders',true,'LabelSource','foldernames');

count = imds.countEachLabel;

% % load alexnet;

net = alexnet;


% % % imds.ReadFcn = @(filename)readAndPreprocessImage(filename);


layers=[imageInputLayer([640 640 3])
    
net(2:end-3)
fullyConnectedLayer(10)

softmaxLayer


classificationLayer()
];

% % % training 

opt=trainingOptions('sgdm', 'Maxepoch', 5, 'InitialLearnRate', 0.00001);

training = trainNetwork(imds, layers, opt);


% % % testing

a = imread ('br075.jpg');

out = classify(training, a);

figure, imshow(a)
title(string(out))
