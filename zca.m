function [Z, Wsep, Wmix, m] = zca(X, portion)

if nargin < 2
    portion = 1;
end

rndidx = randperm(size(X, 1));
X_orig = X;
X = X(rndidx(1:round(size(X,1) * portion)), :);

m = mean(X, 1);
Xc = bsxfun(@minus, X, m);
[U, S, V] = svd(Xc, 0);
Us = std(U, [], 1);
Wsep = V * inv(S .* diag(Us));
Wmix = V';
Z = bsxfun(@minus, X_orig, m) * Wsep * Wmix;





