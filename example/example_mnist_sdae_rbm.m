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

    for l = 1:n_layers-2
        % construct RBM and use default configurations
        R = default_rbm (size(H, 2), layers(l+1));

        R.data.binary = blayers(l);

        mH = mean(H, 1)';
        R.vbias = min(max(log(mH./(1 - mH)), -4), 4);
        R.hbias = zeros(size(R.hbias));
        %R.W = 2 / sqrt(layers(l) + layers(l+1)) * (rand(layers(l), layers(l+1)) - 0.5);
        R.W = 0.01 * (randn(layers(l), layers(l+1)));

        R.learning.lrate = 1e-2;
        R.adaptive_lrate.lrate_ub = 1e-2;

        R.learning.persistent_cd = 0;
        R.fast.use = 0;
        R.fast.lrate = R.learning.lrate;

        R.parallel_tempering.use = 0;
        R.adaptive_lrate.use = 1;
        R.enhanced_grad.use = 1;
        R.learning.minibatch_sz = 128;

        M.valid_min_epochs = 10;

        % max. 100 epochs
        R.iteration.n_epochs = 100;

        % set the stopping criterion
        R.stop.criterion = 0;
        R.stop.recon_error.tolerate_count = 1000;

        % save the intermediate data after every epoch
        R.hook.per_epoch = {@save_intermediate, {sprintf('rbm_mnist_%d.mat', l)}};
        R.hook.per_update = {};

        % print learining process
        R.verbose = 0;
        R.debug.do_display = 0;
        R.debug.display_interval = 10;
        R.debug.display_fid = 1;
        R.debug.display_function = @visualize_rbm;

        % train RBM
        fprintf(1, 'Training RBM\n');
        tic;
        R = train_rbm (R, [H; H_valid]);
        fprintf(1, 'Training is done after %f seconds\n', toc);

        Ds{l} = R;

        H = rbm_get_hidden(H, R);
        H_valid = rbm_get_hidden(H_valid, R);
    end

    l = n_layers - 1;

    % construct DAE and use default configurations
    D = default_dae (layers(l), layers(l+1));

    D.data.binary = blayers(l);
    D.hidden.binary = blayers(l+1);

    if D.data.binary
        mH = mean(H, 1)';
        D.vbias = min(max(log(mH./(1 - mH)), -4), 4);
    else
        D.vbias = mean(H, 1)';
    end

    D.learning.lrate = 1e-1;
    D.learning.lrate0 = 5000;
    D.learning.weight_decay = 0.0001;
    D.learning.minibatch_sz = 128;

    D.noise.drop = 0.2;
    D.noise.level = 0;

    D.valid_min_epochs = 10;
    D.adagrad.use = 1;
    D.adagrad.epsilon = 1e-8;

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

    Ds{l} = D;
end

S = default_sdae (layers);

S.data.binary = blayers(1);
S.bottleneck.binary = blayers(end);
S.hidden.use_tanh = use_tanh;

S.hook.per_epoch = {@save_intermediate, {'sdae_rbm_mnist.mat'}};

S.learning.lrate = 1e-3;
S.learning.lrate0 = 1000;
%S.learning.momentum = 0.5;
%S.learning.weight_decay = 0.0001;
S.learning.minibatch_sz = 256;

S.valid_min_epochs = 10;

S.adagrad.use = 1;
S.adagrad.epsilon = 1e-8;

%S.noise.drop = 0.2;
S.noise.level = 0;

S.iteration.n_epochs = 100;

if do_pretrain
    for l = 1:n_layers-1
        if l > 1
            if use_tanh
                S.biases{l+1} = Ds{l}.hbias;
                S.W{l} = Ds{l}.W;
            else
                S.biases{l+1} = Ds{l}.hbias + sum(Ds{l}.W, 1)';
                S.W{l} = Ds{l}.W / 2;
            end
        else
            S.biases{l+1} = Ds{l}.hbias;
            S.W{l} = Ds{l}.W;
        end
    end
end

fprintf(1, 'Training sDAE\n');
tic;
S = sdae (S, X, X_valid, 0.1);
fprintf(1, 'Training is done after %f seconds\n', toc);

H = sdae_get_hidden (X, S);
save 'sdae_rbm_mnist_vis.mat' H X_labels;

vis_mnist_rbm;




