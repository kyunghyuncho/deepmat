% grbm_sample - Gibbs sampler
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
function [S] = grbm_sample(x0, R, n, n_inter)

if nargin < 4
    n_inter = 1;
end

n_visible = size(R.W, 1);
n_hidden = size(R.W, 2);

v1 = x0;

S = zeros(size(x0,1), n_visible, n);

for k=1:n
    for j=1:n_inter
        h1 = sigmoid(bsxfun(@plus, bsxfun(@rdivide, v1, R.sigmas.^2) * R.W, R.hbias'));
        h1 = binornd(1, h1, size(h1,1), size(h1,2));
        v1_mean = bsxfun(@plus, h1 * R.W', R.vbias');
        v1 = normrnd(v1_mean, repmat(R.sigmas, [size(v1_mean,1) 1]));
    end
    
    S(:,:,k) = v1_mean;
end

end

