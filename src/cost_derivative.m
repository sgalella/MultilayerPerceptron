function cost_prime = cost_derivative(output_activations, y)
%
% Function:
% - cost_derivative: Computes derivative of the cost in the output layer
%
% Input:
% - output_activations: Activation of the last layer (10x1 double)
% - y: output of the network (10x1 double)
% 
% Output: 
% - cost_prime: Derivative of the cost in the output layer (10x1 double)
%
% Author: sgalella
% https://github.com/sgalellaa
% Based on the homonymous function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

cost_prime = output_activations-y;

end

