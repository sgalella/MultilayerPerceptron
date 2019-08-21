function sigmoid_prime = derivative_sigmoid(z)
%
% Function:
% derivative_sigmoid: Computes the derivate of the sigmoid function of z
%
% Input:
% z: Weighted input of the output layer (10x1 double)
%
% Output:
% sigmoid_prime: Derivative of the sigmoid function of z (10x1 double)
%
% Author: sgalella
% https://github.com/sgalella
% Based on the 'sigmpoid_prime' function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

sigmoid_prime = sigmoid(z) .* (1 - sigmoid(z));


end

