% dbn_sample - Sequential Gibbs sampler
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
function [S] = dbn_sample(x0, D, n, n_inter)

if nargin < 4
    n_inter = 1;
end

layers = D.structure.layers;
n_layers = length(layers);

% initial sample
vh = x0;
for l = 2:n_layers-1
    vh = sigmoid(bsxfun(@plus, vh * D.rec.W{l-1}, D.rec.biases{l}'));
    vh = binornd(1, vh);
end

S = zeros(n, layers(1));

for k=1:n
    for j=1:n_inter
        vh = sigmoid(bsxfun(@plus, vh * D.top.W, D.top.hbias'));
        vh = sigmoid(bsxfun(@plus, vh * D.top.W', D.top.vbias'));
    end

    vs = vh;
    for l = n_layers-2:-1:1
        vs = sigmoid(bsxfun(@plus, vs * D.gen.W{l}', D.gen.biases{l}'));
        if l > 1
            vs = binornd(1, vs);
        end
    end
    
    S(k, :) = vs;

    if D.verbose == 1
        fprintf(2, '.');
    end
end

if D.verbose == 1
    fprintf(2, '\n');
end


