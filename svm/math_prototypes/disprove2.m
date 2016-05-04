% This disproves another silly theory of mine.
% The theory is that, if you have a situation
% like the one described in nonmin_grad.m, with
% constraint gradients v1 through vn, then
% solving grad=a1*v1 + ... + an-1*vn-1 would yield
% at least two ai values > 0.

function [grad, basis] = disprove2(dim)
  while true
    [grad, basis] = nonmin_grad(dim);
    solution = basis(:, 1:dim-1)\grad;
    posCount = 0;
    for x = transpose(solution)
      if x >= 0
        posCount += 1;
      end
    end
    if posCount < 2
      return
    end
  end
end
