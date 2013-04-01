% Cho, K., Raiko, T., Ilin, A., and Karhunen, J.
% A Two-stage Pretraining Algorithm for Deep Boltzmann Machines
% http://users.ics.aalto.fi/kcho/papers/nips12workshop.pdf

% add the path of RBM code
addpath('..');

% load MNIST
load 'mnist_14x14.mat';

% shuffle the training data
perm_idx = randperm (size(X,1));
X = X(perm_idx, :);
X_labels = X_labels(perm_idx);

X_labels = X_labels + 1;
X_test_labels = X_test_labels + 1;

% structure of DBM: 784 - 500 - 500
layers = [size(X,2), 500, 500];
n_layers = length(layers);

pretrain = 0;
centering = 1;

if pretrain
    % pretraining (stage 1)
    Xp = X;
    Rs = cell(n_layers, 1);
    Qpre = cell(n_layers, 1);
    Qpre_mask = zeros(n_layers, 1);

    for l = 1:n_layers
        if mod(l, 2) == 0
            continue;
        end

        if l+1 > n_layers
            break;
        end

        R = default_rbm (size(Xp, 2), layers(l+2));

        R.learning.lrate = 1e-2;
        R.learning.weight_decay = 1e-4;

        R.learning.persistent_cd = 0;

        R.fast.use = 0;
        R.fast.lrate = 1e-2;

        R.parallel_tempering.use = 0;

        R.adaptive_lrate.use = 1;
        R.adaptive_lrate.lrate_ub = R.learning.lrate;

        R.enhanced_grad.use = 1;
        R.learning.minibatch_sz = 128;

        % max. 100 epochs
        R.iteration.n_epochs = 100;

        if R.data.binary
            mH = mean(Xp, 1)';
            R.vbias = min(max(log(mH./(1 - mH)), -4), 4);
            R.fast.vbias = min(max(log(mH./(1 - mH)), -4), 4);
        else
            R.vbias = mean(Xp, 1)';
            R.fast.vbias = mean(Xp, 1)';
        end

        % set the stopping criterion
        R.stop.criterion = 0;
        R.stop.recon_error.tolerate_count = 1000;

        % save the intermediate data after every epoch
        R.hook.per_epoch = {@save_intermediate, {sprintf('rbm_%d.mat', l)}};
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
        R = train_rbm (R, Xp);
        fprintf(1, 'Training is done after %f seconds\n', toc);

        Rs{l} = R;

        Xp = rbm_get_hidden(Xp, R);

        Qpre{l+2} = Xp;
        Qpre_mask(l+2) = 1;
    end
end

[D] = default_dbm (layers);

if pretrain
    % pretraining (stage 2)
    D.learning.persistent_cd = 0;
    D.learning.lrate = 1e-2;
    D.learning.lrate0 = 1000;

    D.enhanced_grad.use = 0;
    D.adaptive_lrate.use = 1;
    D.adaptive_lrate.lrate_ub = D.learning.lrate;

    D.iteration.n_epochs = 200;
    D.hook.per_epoch = {@save_intermediate, {'dbm_pre.mat'}};

    fprintf(1, 'Training DBM\n');
    tic;
    [D] = dbm (D, X, 1, Qpre, Qpre_mask);
    fprintf(1, 'Training is done after %f seconds\n', toc);
end

% finetuning (stage 3)
D.learning.persistent_cd = 1;
if pretrain
    D.learning.lrate = 1e-4;
    D.learning.lrate0 = 5000;
else
    D.learning.lrate = 1e-3;
    D.learning.lrate0 = 5000;
end

D.enhanced_grad.use = 0; 
D.adaptive_lrate.use = 1;
D.adaptive_lrate.lrate_ub = D.learning.lrate;

D.iteration.n_epochs = 100;
D.hook.per_epoch = {@save_intermediate, {'dbm_mnist.mat'}};

mH = mean(X, 1)';
D.biases{1} = min(max(log(mH./(1 - mH)), -4), 4);

if centering
    D = set_dbm_centers(D);
end

fprintf(1, 'Finetuning DBM\n');
tic;
[D] = dbm (D, X);
fprintf(1, 'Finetuning is done after %f seconds\n', toc);

% classification
[Q_mf] = dbm_get_hidden(X, D, 30, 1e-6, D.mf.reg);

perm_idx = randperm (size(X,1));

n_all = size(X, 1);
n_train = ceil(n_all * 3 / 4);
n_valid = floor(n_all /4);

X_valid = X(perm_idx(n_train+1:end), :);
X_valid_labels = X_labels(perm_idx(n_train+1:end));
X = X(perm_idx(1:n_train), :);
X_labels = X_labels(perm_idx(1:n_train));

for l = 1:n_layers
    Q_train{l} = Q_mf{l}(perm_idx(1:n_train), :);
    Q_valid{l} = Q_mf{l}(perm_idx(n_train+1:end), :);
end

[Q_test] = dbm_get_hidden(X_test, D, 30, 1e-6, D.mf.reg);

M = default_mlp ([layers, 10]);
M = set_mlp_dbm (M);

M.output.binary = 1;
M.hidden.use_tanh = 0;

M.valid_min_epochs = 10;
M.dropout.use = 0;

M.hook.per_epoch = {@save_intermediate, {'mlp_dbm_mnist.mat'}};

M.learning.lrate = 1e-2;
M.learning.lrate0 = 5000;
%M.learning.momentum = 0.9;
%M.learning.weight_decay = 0.0001;
M.learning.minibatch_sz = 128; M.adagrad.use = 1;
M.adagrad.epsilon = 1e-8;

M.noise.drop = 0;
M.noise.level = 0;

M.iteration.n_epochs = 100;

for l = 1:n_layers
    M.biases{l} = D.biases{l};
    if centering
        if l > 1
            M.biases{l} = M.biases{l} - M.W{l-1}' * D.centering.centers{l-1};
        end
        if l < n_layers
            M.biases{l} = M.biases{l} - M.W{l} * D.centering.centers{l+1};
        end
    end
    if l < n_layers
        M.W{l} = D.W{l};
        M.dbm.W{l} = D.W{l};
    end
end

fprintf(1, 'Training MLP\n');
tic;
M = mlp_dbm (M, X, Q_train, X_labels, X_valid, Q_valid, X_valid_labels, 0.1);
fprintf(1, 'Training is done after %f seconds\n', toc);

[pred] = mlp_classify (M, X_test, Q_test);
n_correct = sum(X_test_labels == pred);

fprintf(2, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));





