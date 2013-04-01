% simple wrapper for sigmoid function
function [y] = sigmoid(x, use_tanh)

if nargin < 2
    use_tanh = 0;
end

if use_tanh
    y = tanh(x);
else
    y = 1./(1 + exp (-x));
end

