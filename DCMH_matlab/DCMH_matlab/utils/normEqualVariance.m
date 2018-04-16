% make each dimension to have equal variance (skip constant dimensions)
function X = normEqualVariance (X)

  mu = mean(X);
  X = bsxfun(@minus, X, mu);

  sigma = std(X);
  nz = find(sigma > 0);
  X(:, nz) = bsxfun(@rdivide, X(:, nz), sigma(nz));

  mu = mean(X);
  X = bsxfun(@plus, X, mu);

end
