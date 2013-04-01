% fbm_sample - Sequential Gibbs sampler
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
function [S] = rbm_sample(x0, R, n, n_inter)

if nargin < 4
    n_inter = 1;
end

n_visible = R.structure.n_visible;
n_hidden = R.structure.n_hidden;

% initial samples
vh = x0;

S = zeros(size(x0,1), n_visible + n_hidden, n);

for k=1:n
    for j=1:n_inter
        for i = 1:n_visible
            vh(:, i) = sigmoid(vh * R.W(:, i) + R.vbias(i)) > rand(size(vh, 1), 1);
        end
        for j = 1:n_hidden
            vh(:, n_visible + j) = sigmoid(vh * R.W(:, n_visible + j) + R.hbias(j)) > rand(size(vh, 1), 1);
        end
    end
    
    S(:,:,k) = vh;

    if R.verbose == 1
        fprintf(2, '.');
    end
end

if R.verbose == 1
    fprintf(2, '\n');
end

end

