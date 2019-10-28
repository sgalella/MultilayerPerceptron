function y2 = ymap(y1)
%
% Function:
% ymap: Maps integer to one hot vector or vector to integer
%
% Input: 
% y1: integer or vector (10x1)
%
% Output:
% y2: vector (10x1) or integer
%
% Author: sgalella
% https://github.com/sgalella

if numel(y1) == 1
    y2 = zeros(10,1);
    y2(y1+1,1) = 1;
else
   [~,idx] = max(y1);
   y2 = idx-1;
end


end

