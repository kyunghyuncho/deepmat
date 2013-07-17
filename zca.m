function [Z, Wsep, Wmix, m, U, S, V] = zca(X, portion)

if nargin < 2
    portion = 1;
end

rndidx = randperm(size(X, 1));
X_orig = X;
X = X(rndidx(1:round(size(X,1) * portion)), :);

m = mean(X, 1);
Xc = bsxfun(@minus, X, m);

sigma = Xc' * Xc / size(Xc, 1);

[U, S, V] = svd(sigma, 0);

Wsep = V / sqrt(S+1e-12);
Wmix = V';
Z = bsxfun(@minus, X_orig, m) * Wsep * Wmix;





