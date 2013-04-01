% dbm_get_hidden
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
function [h_mf] = dbm_get_hidden(x0, D, max_iter, tol, reg)

if nargin < 3
    max_iter = 10;
end

if nargin < 4
    tol = 1e-6;
end

if nargin < 5
    reg = 0;
end

h_mf = dbm_get_hidden_raw(x0, D.data.binary, D.structure.layers, ...
    D.W, D.biases, D.sigmas, max_iter, tol, reg, D.centering.use, D.centering.centers);

