% make each dimension 0 centered
function X = normZeroMean (X)

  mu = mean(X);
  X = bsxfun(@minus, X, mu);

end
