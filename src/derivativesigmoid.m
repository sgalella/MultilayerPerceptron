function sigmoidPrime = derivativesigmoid(z)
%
% Function:
% derivativesigmoid: Computes the derivate of the sigmoid function of z
%
% Inputs:
% z: Weighted input of the output layer (10x1 double)
%
% Outputs:
% sigmoidPrime: Derivative of the sigmoid function of z (10x1 double)
%
% Author: sgalella
% https://github.com/sgalella
% Based on the 'sigmpoid_prime' function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

sigmoidPrime = sigmoid(z) .* (1 - sigmoid(z));


end

