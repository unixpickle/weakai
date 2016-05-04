% "Prove" that, if a set of dependent KKT vectors
% do not represent a local minimum, then there is
% a pair of vectors which, after projecting out the
% other constraint gradients, have a positive dot
% product with each other and with the overall gradient.

function [rightCount, totalCount] = prove1(dim)
  rightCount = 0;
  totalCount = 0;
  for i = 1:1000
    totalCount += 1;
    [grad, basis] = nonmin_grad(dim);
    gotSolution = false;
    for j = 1:dim
      % Delete vector j; treat it as the dependent residue.
      delVec = basis(:, j);
      delBasis = [basis(:, 1:j-1) basis(:, j+1:dim)];
      for k = 1:dim-1
        % Step in the direction of k with the rest of the
        % vectors (besides vector j) projected out.
        others = [delBasis(:, 1:k-1) delBasis(:, k+1:dim-1)];
        othersProj = others * inv(others'*others) * others';
        directionVec = delBasis(:, k);
        directionVec = directionVec - othersProj*directionVec;
        if dot(directionVec,delVec) >= 0 && dot(directionVec,grad) >= 0
          gotSolution = true;
          break;
        end
      end
      if gotSolution
        break;
      end
    end
    if gotSolution
      rightCount += 1;
    end
  end
end
