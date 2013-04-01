% rbm_get_hidden
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
function [h] = rbm_get_hidden(x0, R)

n_visible = size(R.W, 1);
n_hidden = size(R.W, 2);

if R.data.binary == 1
    h = sigmoid(bsxfun(@plus,x0 * R.W, R.hbias'));
else
    h = sigmoid(bsxfun(@plus, bsxfun(@rdivide, x0, R.sigmas.^2) * R.W, R.hbias'));
end

