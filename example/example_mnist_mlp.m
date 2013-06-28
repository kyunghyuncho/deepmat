% add the path of RBM code
addpath('..');
addpath('~/work/Algorithms/liblinear-1.7/matlab');

% load MNIST
load 'mnist_14x14.mat';

% shuffle the training data
X_labels = X_labels + 1;
X_test_labels = X_test_labels + 1;

perm_idx = randperm (size(X,1));

n_all = size(X, 1);
n_train = ceil(n_all * 3 / 4);
n_valid = floor(n_all /4);

X_valid = X(perm_idx(n_train+1:end), :);
X_valid_labels = X_labels(perm_idx(n_train+1:end));
X = X(perm_idx(1:n_train), :);
X_labels = X_labels(perm_idx(1:n_train));

layers = [size(X,2), 500, 500, 10];
n_layers = length(layers);
blayers = [1, 1, 1, 1];

use_tanh = 0;
do_pretrain = 1;

if do_pretrain
    Ds = cell(n_layers - 2, 1);
    H = X;
    H_valid = X_valid;

    for l = 1:n_layers-2
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
        D.learning.minibatch_sz = 128;

        D.noise.drop = 0.2;
        D.noise.level = 0;

        %D.adagrad.use = 1;
        %D.adagrad.epsilon = 1e-8;
        D.adadelta.use = 1;
        D.adadelta.epsilon = 1e-8;
        D.adadelta.momentum = 0.99;

        D.valid_min_epochs = 10;

        if blayers(l+1)
            D.cae.cost = 0.01;
            %D.sparsity.target = 0.1;
            %D.sparsity.cost = 0.01;
        end

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

M = default_mlp (layers);

M.output.binary = blayers(end);
M.hidden.use_tanh = use_tanh;

M.valid_min_epochs = 10;
M.dropout.use = 1;

M.hook.per_epoch = {@save_intermediate, {'mlp_mnist.mat'}};

M.learning.lrate = 1e-3;
M.learning.lrate0 = 5000;
M.learning.minibatch_sz = 128;

M.adadelta.use = 1;
M.adadelta.epsilon = 1e-8;
M.adadelta.momentum = 0.99;

M.noise.drop = 0;
M.noise.level = 0;

M.iteration.n_epochs = 100;

if do_pretrain
    for l = 1:n_layers-2
        M.biases{l+1} = Ds{l}.hbias;
        M.W{l} = Ds{l}.W;
    end
end

fprintf(1, 'Training MLP\n');
tic;
M = mlp (M, X, X_labels, X_valid, X_valid_labels, 0.1);
fprintf(1, 'Training is done after %f seconds\n', toc);

[pred] = mlp_classify (M, X_test);
n_correct = sum(X_test_labels == pred);

fprintf(2, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));




