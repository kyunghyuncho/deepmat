% dbm_get_hidden_raw
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
%This program is free software; you can redistribute it and/or
%modify it under the terms of the GNU General Public License
%as published by the Free Software Foundation; either version 2
%of the License, or (at your option) any later version.
%
%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License
%along with this program; if not, write to the Free Software
%Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [h_mf] = dbm_get_hidden_raw(x0, binary, layers, ...
    W, biases, sigmas, ...
    max_iter, tol, reg, ...
    do_centering, centers)

layers = layers;
n_layers = length(layers);

h_mf = cell(n_layers, 1);
h_mf{1} = x0;

for l = 2:n_layers
    if l == n_layers
        mult = 1;
    else
        mult = 2;
    end
    if l == 2
        if binary == 1
            h_mf{l} = sigmoid(bsxfun(@plus, mult * h_mf{l-1} * W{l-1}, biases{l}'));
        else
            h_mf{l} = sigmoid(bsxfun(@plus, mult * bsxfun(@rdivide, h_mf{l-1}, sigmas.^2') * W{l-1}, biases{l}'));
        end
    else
        h_mf{l} = sigmoid(bsxfun(@plus, mult * h_mf{l-1} * W{l-1}, biases{l}'));
    end
end

h_mf_prev = h_mf;

for iter = 1:max_iter
    diff_err = 0;
    for oddeven = [0 1]
        for l = 2:n_layers
            if mod(l, 2) == oddeven
                continue;
            end
            h_mf{l} = h_mf{l} * 0;
            if do_centering
                if l > 1
                    if l == 2
                        if binary == 1
                            h_mf{l} = h_mf{l} + bsxfun(@minus, h_mf{l-1}, centers{l-1}') * W{l-1};
                        else
                            h_mf{l} = h_mf{l} + bsxfun(@rdivide, h_mf{l-1}, sigmas.^2') * W{l-1};
                        end
                    else
                        h_mf{l} = h_mf{l} + bsxfun(@minus, h_mf{l-1}, centers{l-1}') * W{l-1};
                    end
                end

                if l < n_layers
                    h_mf{l} = h_mf{l} + bsxfun(@minus, h_mf{l+1}, centers{l+1}') * W{l}';
                end
            else
                if l > 1
                    if l == 2
                        if binary == 1
                            h_mf{l} = h_mf{l} + h_mf{l-1} * W{l-1};
                        else
                            h_mf{l} = h_mf{l} + bsxfun(@rdivide, h_mf{l-1}, sigmas.^2') * W{l-1};
                        end
                    else
                        h_mf{l} = h_mf{l} + h_mf{l-1} * W{l-1};
                    end
                end

                if l < n_layers
                    h_mf{l} = h_mf{l} + h_mf{l+1} * W{l}';
                end
            end

            h_mf{l} = sigmoid(bsxfun(@plus, h_mf{l}, biases{l}'));

            if reg > 0
                h_mf{l} = max(h_mf{l} - reg, 0);
            end

            diff_err = diff_err + sum(sum((h_mf_prev{l} - h_mf{l}).^2));
        end
    end

    %fprintf(2, '%d\n', diff_err);

    if diff_err < tol
        break;
    end

    h_mf_prev = h_mf;
end

clear h_mf_prev;



