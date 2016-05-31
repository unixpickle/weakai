% Generate a KKT situation wherein the
% active inequality constraints are
% linearly dependent but still are not
% at a minimum.

function [gradient, basis] = nonmin_grad(dim)
  basis = constraint_vectors(dim);
  while true
    gradient = rand(dim, 1)*2 - repmat([1], dim, 1);
    gradient = basis * gradient;
    isMinimum = false;
    for i = 1:dim
      subBasis = [basis(:, 1:i-1) basis(:, i+1:dim)];
      solution = subBasis\gradient;
      allNeg = true;
      for j = 1:dim-1
        if solution(j) >= 0
          allNeg = false;
        end
      end
      if allNeg
        isMinimum = true;
      end
    end
    if !isMinimum
      break
    end
  end
end

function [vecs] = constraint_vectors(dim)
  equalityConstraint = repmat([1; -1], 1, dim)(1:dim);
  projSpace = null(equalityConstraint);
  vecs = projSpace*inv(projSpace'*projSpace)*projSpace';
end
