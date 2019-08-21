function y_pred = feedforward(NN, training_example)
%
% Function:
% - feedforward: Computes the output of the input in the NN
%
% Inputs:
% - NN: rained neural network (NeuralNet)
% - Input: Training example (781x1 double)
%
% Outputs:
% - y_pred: output predicted (10x1 double)
%
% Author: sgalella
% https://github.com/sgalella
% Based on the homonymous function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

% Set the input as the initial activation
a = training_example;

% Check matching between sizes of the weights and the input
assert(isequal(size(NN.weights{1},2),size(a,1)),...
    'Dimension mismatch between a(1) and the input. Assure that the number of columns of W1 equals the number of rows of the input.');

% Compute the feedforward pass
for i = 1:length(NN.weights)
        a = sigmoid(NN.weights{i}*a + NN.bias{i});
end

% Set y to be the last activation
y_pred = a;

end

