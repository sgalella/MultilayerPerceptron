%%%%% Main %%%%%
%
% Loads the data, trains the network and plot the accuracy and cost through
% the different epochs
%
% Author: sgalella
% https://github.com/sgalella

% Clear command window and workspace
clear; clc;

% Addpath to MNIST dataset
addpath ../data/

% Load the MNIST dataset
% Original dataset from:
% http://yann.lecun.com/exdb/mnist/
% Matlab dataset from:
% https://github.com/daniel-e/mnist_octave
load mnist.mat

% Normalize the pixel values to be between 0 and 1
trainX = double(trainX)/255;
trainY = double(trainY);
testX = double(testX)/255;
testY = double(testY);

% Initialize Neural Network
runSeed = 4371;
neuronsLayer = [784 30 10];
Net = NeuralNet(runSeed, neuronsLayer);

% Prepare training and test set. Sets are formed by cells containing the
% examples and labels separated
trainingSet = cell(1,2);
trainingSet{1,1} = trainX;
trainingSet{1,2} = trainY;

testData = cell(1,2);
testData{1,1} = testX;
testData{1,2} = testY;

% Initialize learning rate (eta), the size of the mini batches and the
% number of epochs and run the stochastic gradient descent
eta = 0.5; % Learning rate
miniBatchSize = 30;
epochs = 20; 
[Net, accuracy, cost] = stochasticGD(runSeed, Net, trainingSet, testData, eta, miniBatchSize, epochs);

% Plot the accuracy and cost over epochs
figure()
sgtitle(['$\eta = ',num2str(eta),'$', ', batch size = ',num2str(miniBatchSize),', epochs = ',num2str(epochs)],'interpreter','latex','fontsize',22)
subplot(2,1,1)
plot(1:epochs, accuracy,'b');
ylim([0 100])
xlim([1, epochs])
ylabel('Accuracy (\%)','interpreter','latex','fontsize',20)
subplot(2,1,2)
plot(1:epochs, cost,'r');
xlim([1 ,epochs])
ylabel('Cost','interpreter','latex','fontsize',20)
ylim([0 1.1*max(cost)])
xlabel('Epochs','interpreter','latex','fontsize',20)
