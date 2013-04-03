function [E] = dbm_entropy(h0, hidden)

n_layers = length(h0);
n_samples = size(h0{1}, 1);

E = zeros(n_samples, 1);

for l = 2:n_layers
    if hidden(l) == 0
        E = E + sum((h0{l} .* log(h0{l}) + (1 - h0{l}) .* log(1 - h0{l})), 2);
    end
end

E = -E;





