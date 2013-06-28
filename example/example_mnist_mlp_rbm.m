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

layers = [size(X,2), 1000, 500, 10];
n_layers = length(layers);
blayers = [1, 1, 1, 1];

use_tanh = 0;
do_pretrain = 1;

if do_pretrain
    Ds = cell(n_layers - 2, 1);
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

        R.learning.lrate = 1e-3;
        R.adaptive_lrate.lrate_ub = 1e-3;

        R.learning.persistent_cd = 0;
        R.fast.use = 0;

        R.parallel_tempering.use = 0;
        R.adaptive_lrate.use = 1;
        R.enhanced_grad.use = 1;
        R.learning.minibatch_sz = 128;

        % max. 100 epochs
        R.iteration.n_epochs = 100;

        % set the stopping criterion
        R.stop.criterion = 0;
        R.stop.recon_error.tolerate_count = 1000;

        % save the intermediate data after every epoch
        R.hook.per_epoch = {@save_intermediate, {sprintf('mlp_rbm_mnist_%d.mat', l)}};
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
end

M = default_mlp (layers);

M.output.binary = blayers(end);
M.hidden.use_tanh = use_tanh;
M.dropout.use = 1;

M.hook.per_epoch = {@save_intermediate, {'mlp_mnist.mat'}};

M.learning.lrate = 1e-2;
M.learning.lrate0 = 5000;
M.learning.minibatch_sz = 128;

M.adadelta.use = 1;

M.noise.drop = 0;
M.noise.level = 0;

M.iteration.n_epochs = 100;

if do_pretrain
    for l = 1:n_layers-2
        if l > 1
            if ~use_tanh
                M.biases{l+1} = Ds{l}.hbias;
                M.W{l} = Ds{l}.W;
            else
                M.biases{l+1} = Ds{l}.hbias + sum(Ds{l}.W, 1)';
                M.W{l} = Ds{l}.W / 2;
            end
        else
            M.biases{l+1} = Ds{l}.hbias;
            M.W{l} = Ds{l}.W;
        end
    end
end

fprintf(1, 'Training MLP\n');
tic;
M = mlp (M, X, X_labels, X_valid, X_valid_labels, 0.1);
fprintf(1, 'Training is done after %f seconds\n', toc);

[pred] = mlp_classify (M, X_test);
n_correct = sum(X_test_labels == pred);

fprintf(2, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));




