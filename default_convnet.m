% default_convnet - 
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
function [C] = default_convnet (size_in, channel_in, full_layers, conv_layers, poolratios, strides)
    % structure
    n_full = length(full_layers);
    n_conv = size(conv_layers, 1);
    n_layers = n_full + n_conv + 1; % +1 for input

    cin = channel_in;
    szin = size_in;
    layers = zeros(n_layers, 1);
    layers(1) = size_in * size_in * channel_in;
    for l = 1:n_conv
        cin = conv_layers(l, 2);
        szout_prepool = floor((szin - sqrt(conv_layers(l, 1))) / strides(l) + 1);
        szin = ceil(szout_prepool / poolratios(l));
        layers(l+1) = cin * szin * szin;
    end
    for l = 1:n_full
        layers(n_conv + l + 1) = full_layers(l);
    end

    C.structure.size_in = size_in;
    C.structure.full_layers = full_layers;
    C.structure.conv_layers = conv_layers;
    C.structure.layers = layers;
    C.structure.channel_in = channel_in;
    C.structure.poolratios = poolratios;
    C.structure.strides = strides;

    % output type
    C.output.binary = 1;% for classification
    %C.output.binary = 0; % for regression

    % nonlinearity: the name of the variable will change in the later revision
    % 0 - sigmoid
    % 1 - tanh
    % 2 - relu
    C.hidden.use_tanh = 1;
    C.conv.use_tanh = 1;

    % pooling
    % 0 - max
    % 1 - average
    % 2 - TODO: stochastic
    C.pooling = zeros(n_conv, 1);

    % learning parameters
    C.learning.lrate = 1e-3;
    C.learning.lrate0 = 5000;
    C.learning.momentum = 0;
    C.learning.weight_decay = 0;
    C.learning.minibatch_sz = 100;
    C.learning.lrate_anneal = 0.9;

    C.valid_min_epochs = 10;

    C.dropout.use = 0;
    % by default
    C.dropout.prob = 0.5;

    C.do_normalize = 1;
    C.do_normalize_std = 1;

    % stopping criterion
    % if you happen to know some other criteria, please, do add them.
    % if the criterion is zero, it won't stop unless the whole training epochs were consumed.
    C.stop.criterion = 0;
    % criterion == 1
    C.stop.recon_error.tolerate_count = 1000;

    % denoising
    C.noise.drop = 0.1;
    C.noise.level = 0.1;

    % local contrast normalization
    C.lcn.use = 0;
    C.lcn.neigh = 4;

    % initializations
    C.cW = cell(n_conv, 1);
    C.cbiases = cell(n_conv, 1);
    for l = 1:n_conv
        if l == 1
            cin = channel_in;
        else
            cin = conv_layers(l-1,2);
        end

        if C.conv.use_tanh == 2
            C.cbiases{l} = 0.01 * ones(conv_layers(l,2), 1);
        else
            C.cbiases{l} = zeros(conv_layers(l,2), 1);
        end
        if l < n_layers
            %C.cW{l} = 2 * sqrt(6)/sqrt(conv_layers(l,1)+conv_layers(l,2)) * (rand(conv_layers(l,1)*cin, conv_layers(l,2)) - 0.5);
            C.cW{l} = 0.0001 * (rand(conv_layers(l,1)*cin, conv_layers(l,2)) - 0.5);
        end
    end
    C.W = cell(n_full, 1);
    C.biases = cell(n_full+1, 1);
    for l = 1:(n_full+1)
        C.biases{l} = zeros(layers(n_conv+l), 1);
        if l < n_full + 1
            C.W{l} = 2 * sqrt(6)/sqrt(layers(n_conv+l)+layers(n_conv+l+1)) * (rand(layers(n_conv+l),layers(n_conv+l+1)) - 0.5);
            %C.W{l} = 0.001 * (rand(layers(n_conv+l),layers(n_conv+l+1)) - 0.5);
        end
    end

    % adadelta
    C.adadelta.use = 0;
    C.adadelta.momentum = 0.99;
    C.adadelta.epsilon = 1e-6;
    C.adadelta.gcW = cell(n_conv, 1);
    C.adadelta.gcbiases = cell(n_conv, 1);
    C.adadelta.cW = cell(n_conv, 1);
    C.adadelta.cbiases = cell(n_conv, 1);
    for l = 1:n_conv
        if l == 1
            cin = channel_in;
        else
            cin = conv_layers(l-1,2);
        end

        C.adadelta.cbiases{l} = zeros(conv_layers(l,2), 1);
        C.adadelta.gcbiases{l} = zeros(conv_layers(l,2), 1);
        if l < n_layers
            C.adadelta.cW{l} = zeros(conv_layers(l,1) * cin, conv_layers(l,2));
            C.adadelta.gcW{l} = zeros(conv_layers(l,1) * cin, conv_layers(l,2));
        end
    end
    C.adadelta.gW = cell(n_full, 1);
    C.adadelta.gbiases = cell(n_full, 1);
    C.adadelta.W = cell(n_full, 1);
    C.adadelta.biases = cell(n_full, 1);
    for l = 1:n_full+1
        C.adadelta.biases{l} = zeros(layers(n_conv+l),1);
        C.adadelta.gbiases{l} = zeros(layers(n_conv+l),1);
        if l < n_full+1
            C.adadelta.W{l} = zeros(layers(n_conv+l),layers(n_conv+l+1));
            C.adadelta.gW{l} = zeros(layers(n_conv+l),layers(n_conv+l+1));
        end
    end

    % iteration
    C.iteration.n_epochs = 100;
    C.iteration.n_updates = 0;

    % learning signals
    C.signals.recon_errors = [];
    C.signals.valid_errors = [];
    C.signals.lrates = [];
    C.signals.n_epochs = 0;

    % debug
    C.verbose = 0;
    C.debug.do_display = 0;
    C.debug.display_interval = 10;
    C.debug.display_fid = 1;
    C.debug.display_function = @visualize_dae;

    % hook
    C.hook.per_epoch = {@save_intermediate, {'convnet.mat'}};
    C.hook.per_update = {@print_n_updates, {}};

