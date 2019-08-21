function [NN, cost_minibatch] = update_mini_batch(NN, mini_batch_X, mini_batch_Y, eta)
%
% Function:
% - updatate_mini_batch: Updates the weights and bias of the NN in a
% mini_batch run
%
% Inputs: 
% - NN: Initialized neural network (NeuralNet)
% - mini_batch_X: Contains training examples of the mini batch 
% (mini_batch_sizex784 double)
% - mini_batch_Y: Contains label of the trainig examples of the mini batch
% (mini_batch_sizex1 double)
% - eta: Learning rate (double)
% 
% Outputs: 
% - NN: Neural network with optimized weights and bias (NeuralNet)
% - cost_minibatch: Cost of the mini batch (double)
%
% Author: sgalella
% https://github.com/sgalella
% Based on the homonymous function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

% Initialize gradient of the weigths and bias, having the same size as the
% ones in NN
nabla_w = NN.weights;
nabla_b = NN.bias;

% Set them to be all zeros
nabla_w = cellfun(@(x) x*0, nabla_w,'un',0);
nabla_b = cellfun(@(x) x*0, nabla_b,'un',0);

% Initialize the cost of the mini batch
cost_minibatch = NaN(1,size(mini_batch_X,1));

% Run backpropagation and update the values of the gradients of the weights
% and bias
for j = 1:size(mini_batch_X,1)
    [delta_nabla_w, delta_nabla_b, cost_example] = backpropagation(NN, mini_batch_X(j,:)',mini_batch_Y(j));
    nabla_w = sum_cells(nabla_w, delta_nabla_w);
    nabla_b = sum_cells(nabla_b, delta_nabla_b);
    cost_minibatch(j) = cost_example;
end

% Update the weights and bias of the NN
NN.weights = sum_cells(NN.weights, cellfun(@(x) x*(-eta/size(mini_batch_X,1)),nabla_w,'un',0));
NN.bias = sum_cells(NN.bias, cellfun(@(x) x*(-eta/size(mini_batch_X,1)),nabla_b,'un',0));

% Average the cost 
cost_minibatch = mean(cost_minibatch);

end

