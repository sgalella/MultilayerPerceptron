function quadratic_cost = quadratic_error(output_activations, y)
%
% Function:
% - quadratic_error: Computes the quadratic cost of a training example
%
% Inputs:
% - output_activations: Activation of the last layer (10x1 double)
% - y: output of the network (10x1 double)
%
% Outputs:
% quadratic_cost: Quadratic error of the last layer (double)
%
% Author: sgalella
% https://github.com/sgalella

quadratic_cost = 0.5*sum((output_activations-y).^2);

end

