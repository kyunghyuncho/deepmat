% set_mlp_dbm - prepare mlp to be initialized with dbm
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
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
function [M] = set_mlp_dbm (M)

    layers = M.structure.layers;
    n_layers = length(layers);


    % is it being initialized with a DBM?
    M.dbm.use = 1;

    % initializations
    M.dbm.W = cell(n_layers, 1);
    for l = 2:n_layers-2
        M.dbm.W{l} = 2 * sqrt(6)/sqrt(layers(l)+layers(l+1)) * (rand(layers(l), layers(l+1)) - 0.5);
    end

    % adagrad
    for l = 2:n_layers-2
        if l < n_layers
            M.adagrad.dbm.W{l} = zeros(layers(l), layers(l+1));
        end
    end

