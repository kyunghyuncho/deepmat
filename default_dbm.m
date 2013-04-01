% default_dbm - 
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
function [D] = default_dbm (layers)
    % data type
    D.data.binary = 1;
    %D.data.binary = 0; % for GDBM

    % learning parameters
    D.learning.cd_k = 1;
    D.learning.persistent_cd = 0;
    D.learning.lrate = 1e-3;
    D.learning.lrate0 = 5000;
    D.learning.momentum = 0;
    D.learning.weight_decay = 0;
    D.learning.minibatch_sz = 100;
    D.learning.lrate_anneal = 0.9;

    D.enhanced_grad.use = 1;

    % adaptive learning rate
    D.adaptive_lrate.use = 1;
    D.adaptive_lrate.max_iter_up = 1;
    D.adaptive_lrate.max_iter_down = 1;
    D.adaptive_lrate.exp_up = 1.01;
    D.adaptive_lrate.exp_down = 0.99;
    D.adaptive_lrate.lrate_ub = Inf;
    D.adaptive_lrate.lrate_lb = -Inf;

    % Gaussian-Bernoulli RBM
    D.grbm.do_vsample = 1;
    D.grbm.do_normalize = 1;
    D.grbm.do_normalize_std = 1;
    D.grbm.learn_sigmas = 1;
    D.grbm.sigmas_ub = Inf;
    D.grbm.use_single_sigma = 1;

    D.mf.reg = 0;

    % stopping criterion
    % if you happen to know some other criteria, please, do add them.
    % if the criterion is zero, it won't stop unless the whole training epochs were consumed.
    D.stop.criterion = 0;
    % criterion == 1
    D.stop.recon_error.tolerate_count = 1000;
    % criterion == 2
    D.stop.lrate.lowlrate_threshold = 1e-8;

    % structure
    n_layers = length(layers);
    D.structure.layers = layers;

    % initializations
    D.W = cell(n_layers, 1);
    D.biases = cell(n_layers, 1);
    D.sigmas = ones(layers(1), 1);
    for l = 1:n_layers
        D.biases{l} = zeros(layers(l), 1);
        if l < n_layers
            D.W{l} = 1/sqrt(layers(l)+layers(l+1)) * randn(layers(l), layers(l+1));
        end
    end

    D.centering.use = 0;
    D.centering.centers = cell(n_layers, 1);
    for l = 1:n_layers
        D.centering.centers{l} = sigmoid(D.biases{l});
    end

    % iteration
    D.iteration.n_epochs = 100;
    D.iteration.n_updates = 0;

    % learning signals
    D.signals.recon_errors = [];
    D.signals.lrates = [];
    D.signals.momentums = [];
    D.signals.n_epochs = 0;

    % debug
    D.verbose = 0;
    D.debug.do_display = 0;
    D.debug.display_interval = 10;
    D.debug.display_fid = 1;
    D.debug.display_function = @visualize_dbm;

    % hook
    D.hook.per_epoch = {@save_intermediate, {'dbm.mat'}};
    D.hook.per_update = {@print_n_updates, {}};

