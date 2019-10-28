classdef NeuralNet
%
% Class:
% - NeuralNet: Feedforward neural network. Four different properties and one method
% for initialize layers, bias and weights
% Author: sgalella
% https://github.com/sgalella

properties
    numLayers
    sizes
    bias
    weights
end

methods
    function NN = NeuralNet(runSeed, arg)
        if nargin == 2
            s = RandStream.create('mrg32k3a','NumStreams', 1,'Seed',runSeed);
            NN.numLayers = length(arg);
            NN.sizes = arg;
            NN.bias = cell(NN.numLayers-1,1);
            NN.weights = cell(NN.numLayers-1,1);
            for i = 2:NN.numLayers
               NN.bias{i-1} = randn(s,[NN.sizes(i) 1]); 
               NN.weights{i-1} = randn(s,[NN.sizes(i) NN.sizes(i-1)]);
            end
        elseif nargin < 2
            error('Missing arguments. Introduce the random seed and a vector containing the number of neurons in each layer.')
        else
            error('Exceding arguments. Introduce the random seed and a vector containing the number of neurons in each layer.')
        end
    end
end

end

