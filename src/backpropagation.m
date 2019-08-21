function [nabla_w, nabla_b, cost_example] = backpropagation(NN, x, y)
%
% Function:
% - backpropagation: Computes the backpropagation of the NN with inputs
% mini_batch_X and labels mini_batch_Y
%
% Inputs: 
% - NN: Initialized neural network (NeuralNet)
% - x: Contains one example of the mini batch (784x1 double)
% - y: Contains the label of one example of the mini 
% batch (1x1 double)
%
% Outputs:
% - nabla_w: gradient of the weights of the example
% - nabla_b: gradient of the bias of the example
% - cost_example: quadratic cost of the example
%
% Author: sgalella
% https://github.com/sgalella
% Based on the 'backprop' function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

% Map the output of y to a one hot vector
y = y_map(y);

% Initialize gradient of the weigths and bias, having the same size as the
% ones in NN
nabla_b = NN.bias;
nabla_w = NN.weights;

% Set them to be all zeros
nabla_w = cellfun(@(x) x*0, nabla_w,'un',0);
nabla_b = cellfun(@(x) x*0, nabla_b,'un',0);

% Compute a0 and initialize cell for different a's and z's
activation = x;
activations = cell(NN.num_layers,1);
activations{1} = x;
zs = cell(NN.num_layers-1,1);

% Compute the feedforward run
for i = 1:length(nabla_b)
    z = NN.weights{i}*activation + NN.bias{i};
    zs{i} = z;
    activation = sigmoid(z);
    activations{i+1} = activation;
end

% Compute the delta at the output layer. Update nabla_b and nabla_w
cost_example = quadratic_error(activations{end},y);
delta = cost_derivative(activations{end}, y) .* derivative_sigmoid(zs{end});
nabla_b{end} = delta;
nabla_w{end} = delta*activations{end-1}';

% Compute the delta for the remaining layers. Update nabla_b and nabla_w
for layer = 2:NN.num_layers-1
    z = zs{end + (1-layer)};
    sp = derivative_sigmoid(z);
    delta = NN.weights{end-(layer-2)}'*delta.*sp;
    nabla_b{end + (1-layer)} = delta;
    nabla_w{end + (1-layer)} = delta*activations{end - layer}';
end


end

