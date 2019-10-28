function sumCell = sumcells(cell1,cell2)
%
% Function:
% sumcells: sums each component of two different cells
%
% Input:
% cell1: cell (3x1 double)
% cell2: cell (3x1 double)
%
% Output:
% sumCell: sum of components of cell1 and cell2 (3x1 double)
%
% Author: sgalella
% https://github.com/sgalella

% Check size of the cells is the same
assert(isequal(size(cell1),size(cell1)),'Cell dimension must agree.');

% Initialize sumCell
sumCell = cell(size(cell1));

% Add up each component
for i = 1:length(cell1)
    assert(isequal(size(cell1{i}),size(cell2{i})),'Matrix dimension in cells must agree.');
    sumCell{i} = cell1{i} + cell2{i};
end

end

