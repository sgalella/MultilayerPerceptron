function [NN, accuracyEpochs, totalCost] = stochasticGD(runSeed, NN, trainingData, testData, eta, miniBatchSize, epochs)
%
% Function:
% - stochasticGD: Computes the stochastic gradient descent for the network
%
% Inputs:
% - runSeed: Seed to generate stream to generate different training sets
% across minibatches (double)
% - NN: Initialized neural network (NeuralNet)
% - trainingData: Data used to train the newtork (cell of size 1x2)
% - testData: Data used to test the network (cell of size 1x2)
% - eta: Learning rate (double)
% - miniBatchSize: Training examples per mini batch (double)
% - epochs: Number of total epochs (double)
%
% Outputs:
% - NN: Neural network with optimized weights and bias (NeuralNet)
% - accuracyEpochs: Accuracy of the network along epochs (double)
% - totalCost: Cost for all the minibatches along epochs (double)
%
% Author: sgalella
% https://github.com/sgalella
% Based on the 'SGD' function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

% Create random stream using runSeed
s = RandStream.create('mrg32k3a','NumStreams', 1,'Seed',runSeed);

% Define the size of the training set (number of examples)
nTrain = size(trainingData{1,1},1);

% Define the number of total mini batches
numMiniBatches = nTrain/miniBatchSize;

% Initialize the vector for the accuracy, cost per minibatch and total cost
accuracyEpochs = NaN(1,epochs);
costMiniBatch = NaN(1,numMiniBatches);
totalCost = NaN(1,epochs);

% Run the stochastic gradient descent
for epoch = 1:epochs
    
    % Create different minibatches with different training examples
    permutation = randperm(s,nTrain);
    randomTrainingSetX = trainingData{1,1}(permutation,:);
    randomTrainingSetY = trainingData{1,2}(permutation);
    miniBatchesX = cell(numMiniBatches,1);
    miniBatchesY = cell(numMiniBatches,1);
    for j = 1:numMiniBatches
       miniBatchesX{j} = randomTrainingSetX((j-1)*miniBatchSize+1:j*miniBatchSize,:);
       miniBatchesY{j} = randomTrainingSetY((j-1)*miniBatchSize+1:j*miniBatchSize);
    end
    
    % Update the weights and bias of NN through the different mini batches
    for minibatch = 1:length(miniBatchesX) 
        [NN, costMiniBatch(minibatch)] = updateminibatch(NN, miniBatchesX{minibatch},miniBatchesY{minibatch}, eta);
    end
    
    % Compute the accuracy and cost of the epoch
    accuracyEpochs(epoch) = evaluate(NN, testData);
    totalCost(epoch) = mean(costMiniBatch);
    fprintf('Epoch %d/%d. Accuracy: %.3f%%. Cost: %.3f.\n',epoch,epochs,accuracyEpochs(epoch),totalCost(epoch));
end

end

