function s = sigmoid(z)
%
% Function:
% sigmoid: Computes the sigmoid function of z
%
% Inputs:
% z: Weighted input of the output layer (10x1 double)
%
% Outputs:
% s: Sigmoid of z (10x1 double)
%
% Author: sgalella
% https://github.com/sgalella
% Based on the homonymous function written in Python 2.7 by mnielsen:
% https://github.com/mnielsen/neural-networks-and-deep-learning

s = 1./(1 + exp(-z));

end

