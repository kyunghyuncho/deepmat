function [Y] = padimages(X, szin, cin, k, val)

Xr = reshape(X, [size(X, 1), szin, szin, cin]);
Y = padarray(Xr, [0, k, k, 0], val);
Y = reshape(Y, [size(X, 1), (szin+2*k)*(szin+2*k)*cin]);



