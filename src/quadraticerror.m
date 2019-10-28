function quadraticCost = quadraticerror(outputActivations, y)
%
% Function:
% - quadraticerror: Computes the quadratic cost of a training example
%
% Inputs:
% - outputActivations: Activation of the last layer (10x1 double)
% - y: output of the network (10x1 double)
%
% Outputs:
% quadraticCost: Quadratic error of the last layer (double)
%
% Author: sgalella
% https://github.com/sgalella

quadraticCost = 0.5*sum((outputActivations-y).^2);

end

