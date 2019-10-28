function [NN, costMinibatch] = updateminibatch(NN, miniBatchX, miniBatchY, eta)
%
% Function:
% - updatateminibatch: Updates the weights and bias of the NN in a
% mini_batch run
%
% Inputs: 
% - NN: Initialized neural network (NeuralNet)
% - mini_batch_X: Contains training examples of the mini batch 
% (miniBatchSizex784 double)
% - mini_batch_Y: Contains label of the trainig examples of the mini batch
% (miniBatchSizex1 double)
% - eta: Learning rate (double)
% 
% Outputs: 
% - NN: Neural network with optimized weights and bias (NeuralNet)
% - costMiniBatch: Cost of the mini batch (double)
%
% Author: sgalella
% https://github.com/sgalella
% Based on the homonymous function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

% Initialize gradient of the weigths and bias, having the same size as the
% ones in NN
nablaW = NN.weights;
nablab = NN.bias;

% Set them to be all zeros
nablaW = cellfun(@(x) x*0, nablaW,'un',0);
nablab = cellfun(@(x) x*0, nablab,'un',0);

% Initialize the cost of the mini batch
costMinibatch = NaN(1,size(miniBatchX,1));

% Run backpropagation and update the values of the gradients of the weights
% and bias
for j = 1:size(miniBatchX,1)
    [deltaNablaW, deltaNablab, costExample] = backpropagation(NN, miniBatchX(j,:)',miniBatchY(j));
    nablaW = sumcells(nablaW, deltaNablaW);
    nablab = sumcells(nablab, deltaNablab);
    costMinibatch(j) = costExample;
end

% Update the weights and bias of the NN
NN.weights = sumcells(NN.weights, cellfun(@(x) x*(-eta/size(miniBatchX,1)),nablaW,'un',0));
NN.bias = sumcells(NN.bias, cellfun(@(x) x*(-eta/size(miniBatchX,1)),nablab,'un',0));

% Average the cost 
costMinibatch = mean(costMinibatch);

end

