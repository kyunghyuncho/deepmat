% default_gsn - 
% Copyright (C) 2013 KyungHyun Cho
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
function [G] = default_gsn (layers)
    % structure
    n_layers = length(layers);
    G.structure.n_layers = n_layers;
    G.structure.layers = layers;

    % data type
    G.data.binary = 1;
    %G.data.binary = 0;

    % nonlinearity
    % 0 - sigmoid
    % 1 - tanh
    % 2 - relu
    G.hidden.use_tanh = 1;

    % noise type in the hidden variables
    % 0 - none
    % 1 - white noise
    % 2 - dropout noise
    G.hidden.noise_type = 1;
    G.hidden.noise_level = 0.1;
    G.hidden.add_noise = ones(n_layers, 1);

    % learning parameters
    G.learning.lrate = 1e-3;
    G.learning.lrate0 = 5000;
    G.learning.momentum = 0;
    G.learning.weight_decay = 0;
    G.learning.minibatch_sz = 100;
    G.learning.lrate_anneal = 0.9;

    G.valid_min_epochs = 10;

    G.do_normalize = 1;
    G.do_normalize_std = 1;

    % stopping criterion
    % if you happen to know some other criteria, please, do add them.
    % if the criterion is zero, it won't stop unless the whole training epochs were consumed.
    G.stop.criterion = 0;
    % criterion == 1
    G.stop.recon_error.tolerate_count = 1000;

    % denoising: noise in the visible layer
    G.noise.drop = 0.1; % in the case of binary data, salt-and-pepper!
    G.noise.level = 0.1;

    % initializations
    G.W = cell(n_layers, 1);
    G.biases = cell(n_layers, 1);
    for l = 1:n_layers
        G.biases{l} = zeros(layers(l), 1);
        if l < n_layers
            G.W{l} = sqrt(6)/sqrt(layers(l)+layers(l+1)) * (rand(layers(l), layers(l+1)) - 0.5);
            %G.W{l} = 0.001 * (rand(layers(l), layers(l+1)) - 0.5);
        end
    end

    % adadelta
    G.adadelta.use = 0;
    G.adadelta.momentum = 0.995;
    G.adadelta.epsilon = 1e-8;
    G.adadelta.gW = cell(n_layers, 1);
    G.adadelta.gbiases = cell(n_layers, 1);
    G.adadelta.W = cell(n_layers, 1);
    G.adadelta.biases = cell(n_layers, 1);
    for l = 1:n_layers
        G.adadelta.gbiases{l} = zeros(layers(l), 1);
        G.adadelta.biases{l} = zeros(layers(l), 1);
        if l < n_layers
            G.adadelta.gW{l} = zeros(layers(l), layers(l+1));
            G.adadelta.W{l} = zeros(layers(l), layers(l+1));
        end
    end

    % iteration
    G.iteration.n_epochs = 100;
    G.iteration.n_updates = 0;

    % learning signals
    G.signals.recon_errors = [];
    G.signals.valid_errors = [];
    G.signals.lrates = [];
    G.signals.n_epochs = 0;

    % debug
    G.verbose = 0;

    % hook
    G.hook.per_epoch = {@save_intermediate, {'gsn.mat'}};
    G.hook.per_update = {@print_n_updates, {}};

