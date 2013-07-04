% gsn_sample - one-step sampling with GSN
% Copyright (C) 2013 KyungHyun Cho
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%

function [X, occupied] = gsn_sample(G, X, occupied, meanfield)

if nargin < 4
    meanfield = 0;
end

mb_sz = size(X{1}, 1);
 
if G.data.binary == 0 && G.noise.level > 0
    X{1} = X{1} + G.noise.level * randn(size(X{1}));
end

if G.noise.drop > 0
    mask = binornd(1, 1 - G.noise.drop, size(X{1}));
    if G.data.binary
        sandp = binornd(1, 0.5, size(X{1}));
    else
        sandp = zeros(size(X{1}));
    end
    X{1} = X{1} .* mask + (1 - mask) .* sandp;
    clear mask;
end

for l = 1:G.structure.n_layers
    if mod(l, 2) == 1
        continue;
    end

    check = 0;
    if l > 1 && occupied(l-1)
        check = check + 1;
    end
    if l < G.structure.n_layers && occupied(l+1)
        check = check + 1;
    end
    if check == 0;
        continue;
    end

    % odd-numbered hidden layers
    X{l} = 0 * X{l};

    occupied(l) = 1;

    if l < G.structure.n_layers && occupied(l+1) 
        X{l} = X{l} + X{l+1} * G.W{l}';
    end

    if l > 1 && occupied(l-1)
        X{l} = X{l} + X{l-1} * G.W{l-1};
    end

    X{l} = bsxfun(@plus, X{l}, G.biases{l}');

    if ~meanfield
        if l > 1
            if G.hidden.add_noise(l)
                X{l} = X{l} + G.hidden.noise_level * randn(mb_sz, G.structure.layers(l));
            end
        end
    end

    if l > 1
        X{l} = sigmoid(X{l}, G.hidden.use_tanh);
    else
        % visible layer
        if G.data.binary
            X{l} = sigmoid(X{l});
        end
    end

    if ~meanfield
        if l > 1
            if G.hidden.add_noise(l)
                X{l} = X{l} + G.hidden.noise_level * randn(mb_sz, G.structure.layers(l));
            end
        end
    end
end

for l = 1:G.structure.n_layers
    if mod(l, 2) == 0
        continue;
    end

    check = 0;
    if l > 1 && occupied(l-1)
        check = check + 1;
    end
    if l < G.structure.n_layers && occupied(l+1)
        check = check + 1;
    end
    if check == 0;
        continue;
    end

    % even-numbered hidden layers
    X{l} = 0 * X{l};

    if l < G.structure.n_layers && occupied(l+1) 
        X{l} = X{l} + X{l+1} * G.W{l}';
    end

    if l > 1 && occupied(l-1)
        X{l} = X{l} + X{l-1} * G.W{l-1};
    end

    occupied(l) = 1;

    X{l} = bsxfun(@plus, X{l}, G.biases{l}');
    
    if ~meanfield
        if l > 1
            if G.hidden.add_noise(l)
                X{l} = X{l} + G.hidden.noise_level * randn(mb_sz, G.structure.layers(l));
            end
        end
    end

    if l > 1
        X{l} = sigmoid(X{l}, G.hidden.use_tanh);
    else
        % visible layer
        if G.data.binary
            X{l} = sigmoid(X{l});
        end
    end

    if ~meanfield
        if l > 1
            if G.hidden.add_noise(l)
                X{l} = X{l} + G.hidden.noise_level * randn(mb_sz, G.structure.layers(l));
            end
        end
    end
end




