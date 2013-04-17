% add the path of RBM code
addpath('..');
addpath('~/work/Algorithms/liblinear-1.7/matlab');

% load MNIST
load 'mnist_14x14.mat';

% shuffle the training data
perm_idx = randperm (size(X,1));

n_all = size(X, 1);
n_train = ceil(n_all * 3 / 4);
n_valid = floor(n_all /4);

X_valid = X(perm_idx(n_train+1:end), :);
X_valid_labels = X_labels(perm_idx(n_train+1:end));
X = X(perm_idx(1:n_train), :);
X_labels = X_labels(perm_idx(1:n_train));

layers = [size(X,2), 200, 100, 50, 2];
n_layers = length(layers);
blayers = [1, 1, 1, 1, 0];

use_tanh = 0;
do_pretrain = 1;

if do_pretrain
    Ds = cell(n_layers - 1, 1);
    H = X;
    H_valid = X_valid;

    for l = 1:n_layers-1
        % construct DAE and use default configurations
        D = default_dae (layers(l), layers(l+1));

        D.data.binary = blayers(l);
        D.hidden.binary = blayers(l+1);

        if use_tanh 
            if l > 1
                D.visible.use_tanh = 1;
            end
            D.hidden.use_tanh = 1;
        else
            if D.data.binary
                mH = mean(H, 1)';
                D.vbias = min(max(log(mH./(1 - mH)), -4), 4);
            else
                D.vbias = mean(H, 1)';
            end
        end

        D.learning.lrate = 1e-1;
        D.learning.lrate0 = 5000;
        D.learning.weight_decay = 0.0001;
        D.learning.minibatch_sz = 128;

        D.valid_min_epochs = 10;

        D.noise.drop = 0.2;
        D.noise.level = 0;

        %D.adagrad.use = 1;
        %D.adagrad.epsilon = 1e-8;
        D.adagrad.use = 0;
        D.adadelta.use = 1;
        D.adadelta.epsilon = 1e-8;
        D.adadelta.momentum = 0.99;

        D.iteration.n_epochs = 500;

        % save the intermediate data after every epoch
        D.hook.per_epoch = {@save_intermediate, {sprintf('dae_mnist_%d.mat', l)}};

        % print learining process
        D.verbose = 0;
        % display the progress
        D.debug.do_display = 0;

        % train RBM
        fprintf(1, 'Training DAE (%d)\n', l);
        tic;
        D = dae (D, H, H_valid, 0.1);
        fprintf(1, 'Training is done after %f seconds\n', toc);

        H = dae_get_hidden(H, D);
        H_valid = dae_get_hidden(H_valid, D);

        Ds{l} = D;
    end
end

S = default_sdae (layers);

S.data.binary = blayers(1);
S.bottleneck.binary = blayers(end);
S.hidden.use_tanh = use_tanh;

S.hook.per_epoch = {@save_intermediate, {'sdae_mnist.mat'}};

S.learning.lrate = 1e-1;
S.learning.lrate0 = 5000;
%S.learning.momentum = 0.9;
S.learning.weight_decay = 0.0001;
S.learning.minibatch_sz = 128;

%S.noise.drop = 0.2;
%S.noise.level = 0;
S.adadelta.use = 1;
S.adadelta.epsilon = 1e-8;
S.adadelta.momentum = 0.99;

%S.adagrad.use = 1;
%S.adagrad.epsilon = 1e-8;
S.valid_min_epochs = 10;

S.iteration.n_epochs = 100;

if do_pretrain
    for l = 1:n_layers-1
        S.biases{l+1} = Ds{l}.hbias;
        S.W{l} = Ds{l}.W;
    end
else
    if S.data.binary
        mH = mean(X, 1)';
        S.biases{1} = min(max(log(mH./(1 - mH)), -4), 4);
    else
        S.biases{1} = mean(X, 1)';
    end
end

fprintf(1, 'Training sDAE\n');
tic;
S = sdae (S, X, X_valid, 0.1);
fprintf(1, 'Training is done after %f seconds\n', toc);

H = sdae_get_hidden (X, S);
save 'sdae_mnist_vis.mat' H X_labels;

vis_mnist;




