function costPrime = costderivative(outputActivations, y)
%
% Function:
% - costDerivative: Computes derivative of the cost in the output layer
%
% Inputs:
% - outputActivations: Activation of the last layer (10x1 double)
% - y: output of the network (10x1 double)
% 
% Outputs 
% - costPrime: Derivative of the cost in the output layer (10x1 double)
%
% Author: sgalella
% https://github.com/sgalella
% Based on the homonymous function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

costPrime = outputActivations-y;

end

