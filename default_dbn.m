% default_dbn - 
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
function [D] = default_dbn (layers)

    % learning parameters
    D.learning.lrate = 1e-3;
    D.learning.lrate0 = 5000;
    D.learning.momentum = 0;
    D.learning.weight_decay = 0;
    D.learning.minibatch_sz = 100;
    D.learning.lrate_anneal = 0.9;

    D.learning.contrastive_step = 1;
    D.learning.persistent_cd = 1;

    D.learning.ffactored = 0;

    % stopping criterion
    % if you happen to know some other criteria, please, do add them.
    % if the criterion is zero, it won't stop unless the whole training epochs were consumed.
    D.stop.criterion = 0;
    % criterion == 1
    D.stop.recon_error.tolerate_count = 1000;

    % structure
    n_layers = length(layers);
    D.structure.layers = layers;

    % initializations
    D.rec.W = cell(n_layers-1, 1);
    D.rec.biases = cell(n_layers-1, 1);
    for l = 1:n_layers-1
        D.rec.biases{l} = zeros(layers(l), 1);
        if l < n_layers
            D.rec.W{l} = 1/sqrt(layers(l)+layers(l+1)) * randn(layers(l), layers(l+1));
        end
    end

    D.gen.W = cell(n_layers-1, 1);
    D.gen.biases = cell(n_layers-1, 1);
    for l = 1:n_layers-1
        D.gen.biases{l} = zeros(layers(l), 1);
        if l < n_layers
            D.gen.W{l} = 1/sqrt(layers(l)+layers(l+1)) * randn(layers(l), layers(l+1));
        end
    end

    D.top.W = 1/sqrt(layers(end-1)+layers(end)) * randn(layers(end-1), layers(end));
    D.top.vbias = zeros(layers(end-1), 1);
    D.top.hbias = zeros(layers(end), 1);

    % iteration
    D.iteration.n_epochs = 100;
    D.iteration.n_updates = 0;

    % learning signals
    D.signals.recon_errors = [];
    D.signals.lrates = [];
    D.signals.n_epochs = 0;

    % debug
    D.verbose = 0;
    D.debug.do_display = 0;
    D.debug.display_interval = 10;
    D.debug.display_fid = 1;
    D.debug.display_function = @visualize_dae;

    % hook
    D.hook.per_epoch = {@save_intermediate, {'dbn.mat'}};
    D.hook.per_update = {@print_n_updates, {}};

