% "Prove" that, if g is a linear combination
% of independent vectors V1 through Vn, then
% sign(dot(Xi, g))==sign(Ai) where Xi is Vi
% with all the other V's projected out and Ai
% is the coefficient of Vi in the linear
% combination for g.

function [rightCount, totalCount] = prove2(dim)
  rightCount = 0;
  totalCount = 0;
  basis = rand(dim, dim)*2 - repmat([1], dim, dim);
  for i = 1:1000
    g = rand(dim, 1);
    combination = basis\g;
    broken = false;
    for j = 1:dim
      num = combination(j);
      projOutBasis = [basis(:, 1:j-1) basis(:, j+1:dim)];
      projMat = projOutBasis*inv(projOutBasis'*projOutBasis)*projOutBasis';
      basisVec = basis(:, j);
      basisVec = basisVec - projMat*basisVec;
      d = dot(basisVec, g);
      if sign(d) != sign(num)
        broken = true;
        break
      end
    end
    if !broken
      rightCount += 1;
    end
    totalCount += 1;
  end
end
