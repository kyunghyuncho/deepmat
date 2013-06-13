% default_mlp - 
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
function [M] = default_mlp (layers)
    % structure
    n_layers = length(layers);
    M.structure.layers = layers;

    % data type
    M.data.binary = 1;
    %M.data.binary = 0;

    % output type
    M.output.binary = 1;% for classification
    %M.output.binary = 0; % for regression

    % nonlinearity: the name of the variable will change in the later revision
    % 0 - sigmoid
    % 1 - tanh
    % 2 - relu
    M.hidden.use_tanh = 0;

    % is it being initialized with a DBM?
    M.dbm.use = 0;

    % learning parameters
    M.learning.lrate = 1e-3;
    M.learning.lrate0 = 5000;
    M.learning.momentum = 0;
    M.learning.weight_decay = 0;
    M.learning.minibatch_sz = 100;
    M.learning.lrate_anneal = 0.9;

    M.valid_min_epochs = 10;

    M.dropout.use = 0;
    % by default
    M.dropout.probs = cell(n_layers, 1);
    for l = 1:n_layers
        M.dropout.probs{l} = 0.5 * ones(layers(l), 1);
    end

    M.do_normalize = 1;
    M.do_normalize_std = 1;

    % stopping criterion
    % if you happen to know some other criteria, please, do add them.
    % if the criterion is zero, it won't stop unless the whole training epochs were consumed.
    M.stop.criterion = 0;
    % criterion == 1
    M.stop.recon_error.tolerate_count = 1000;

    % denoising
    M.noise.drop = 0.1;
    M.noise.level = 0.1;

    % initializations
    M.W = cell(n_layers, 1);
    M.biases = cell(n_layers, 1);
    for l = 1:n_layers
        M.biases{l} = zeros(layers(l), 1);
        if l < n_layers
            %M.W{l} = 1/sqrt(layers(l)+layers(l+1)) * randn(layers(l), layers(l+1));
            M.W{l} = 2 * sqrt(6)/sqrt(layers(l)+layers(l+1)) * (rand(layers(l), layers(l+1)) - 0.5);
        end
    end

    % adagrad
    M.adagrad.use = 0;
    M.adagrad.epsilon = 1e-8;
    M.adagrad.W = cell(n_layers, 1);
    M.adagrad.biases = cell(n_layers, 1);
    for l = 1:n_layers
        M.adagrad.biases{l} = zeros(layers(l), 1);
        if l < n_layers
            M.adagrad.W{l} = zeros(layers(l), layers(l+1));
        end
    end

    M.adadelta.use = 0;
    M.adadelta.momentum = 0.99;
    M.adadelta.epsilon = 1e-6;
    M.adadelta.gW = cell(n_layers, 1);
    M.adadelta.gbiases = cell(n_layers, 1);
    M.adadelta.W = cell(n_layers, 1);
    M.adadelta.biases = cell(n_layers, 1);
    for l = 1:n_layers
        M.adadelta.gbiases{l} = zeros(layers(l), 1);
        M.adadelta.biases{l} = zeros(layers(l), 1);
        if l < n_layers
            M.adadelta.gW{l} = zeros(layers(l), layers(l+1));
            M.adadelta.W{l} = zeros(layers(l), layers(l+1));
        end
    end

    % iteration
    M.iteration.n_epochs = 100;
    M.iteration.n_updates = 0;

    % learning signals
    M.signals.recon_errors = [];
    M.signals.valid_errors = [];
    M.signals.lrates = [];
    M.signals.n_epochs = 0;

    % debug
    M.verbose = 0;
    M.debug.do_display = 0;
    M.debug.display_interval = 10;
    M.debug.display_fid = 1;
    M.debug.display_function = @visualize_dae;

    % hook
    M.hook.per_epoch = {@save_intermediate, {'mlp.mat'}};
    M.hook.per_update = {@print_n_updates, {}};

