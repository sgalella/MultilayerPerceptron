function [nablaW, nablab, costExample] = backpropagation(NN, x, y)
%
% Function:
% - backpropagation: Computes the backpropagation of the NN with inputs
% miniBatchX and labels mini_batch_Y
%
% Inputs: 
% - NN: Initialized neural network (NeuralNet)
% - x: Contains one example of the mini batch (784x1 double)
% - y: Contains the label of one example of the mini 
% batch (1x1 double)
%
% Outputs:
% - nablaW: gradient of the weights of the example
% - nablab: gradient of the bias of the example
% - costExample: quadratic cost of the example
%
% Author: sgalella
% https://github.com/sgalella
% Based on the 'backprop' function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

% Map the output of y to a one hot vector
y = ymap(y);

% Initialize gradient of the weigths and bias, having the same size as the
% ones in NN
nablab = NN.bias;
nablaW = NN.weights;

% Set them to be all zeros
nablaW = cellfun(@(x) x*0, nablaW,'un',0);
nablab = cellfun(@(x) x*0, nablab,'un',0);

% Compute a0 and initialize cell for different a's and z's
activation = x;
activations = cell(NN.numLayers,1);
activations{1} = x;
zs = cell(NN.numLayers-1,1);

% Compute the feedforward run
for i = 1:length(nablab)
    z = NN.weights{i}*activation + NN.bias{i};
    zs{i} = z;
    activation = sigmoid(z);
    activations{i+1} = activation;
end

% Compute the delta at the output layer. Update nabla_b and nabla_w
costExample = quadraticerror(activations{end},y);
delta = costderivative(activations{end}, y) .* derivativesigmoid(zs{end});
nablab{end} = delta;
nablaW{end} = delta*activations{end-1}';

% Compute the delta for the remaining layers. Update nabla_b and nabla_w
for layer = 2:NN.numLayers-1
    z = zs{end + (1-layer)};
    sp = derivativesigmoid(z);
    delta = NN.weights{end-(layer-2)}'*delta.*sp;
    nablab{end + (1-layer)} = delta;
    nablaW{end + (1-layer)} = delta*activations{end - layer}';
end


end

