function [Z] = zca_whiten(X, Wsep, Wmix, mX)

Z = bsxfun(@minus, X, mX) * Wsep * Wmix;

