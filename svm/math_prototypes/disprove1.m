% This disproves my first theory for the annoying
% active set case described in nonmin_grad.m.
% My theory was that, if v1, v2, v3, ..., vn are
% dependent inequality gradients, and the point is
% not a local constrained minimum, then excluding
% vn and stepping in the direction of the first v
% with a positive coefficient (with all the other
% vs besides vn projected out) will result in a
% feasible direction that increases the gradient
% while also increasing the constraint corresponding
% to vn.

function [grad, basis] = disprove1(dim)
  while true
    [grad, basis] = nonmin_grad(dim);
    solution = basis(:, 1:dim-1)\grad;
    for i = 1:dim-1
      projOut = [basis(:, 1:i-1) basis(:, i+1:dim-1)];
      projMatrix = projOut*inv(projOut'*projOut)*projOut';
      if solution(i) >= 0
        projVec1 = projMatrix * basis(:, i);
        projVec2 = projMatrix * basis(:, dim);
        if dot(projVec1, projVec2) < 0
          return
        end
      end
    end
  end
end
