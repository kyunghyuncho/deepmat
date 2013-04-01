% add the path of RBM code
addpath('..');
addpath('~/work/Algorithms/liblinear-1.7/matlab');

use_whitening = 1;

% load natural image patches
load 'bsds500bw_patches_8.mat';
X = (Xbw / 255);

% shuffle the training data
perm_idx = randperm (size(X,1));

n_all = size(X, 1);
n_train = ceil(n_all * 3 / 4);
n_valid = floor(n_all /4);

X_valid = X(perm_idx(n_train+1:end), :);
X = X(perm_idx(1:n_train), :);

if use_whitening
    %% ZCA
    %[Z, Wsep, Wmix, mX] = zca(X, 0.1);
    load patch8_whiten.mat;
    X = zca_whiten(X, Wsep, Wmix, mX);
    X_valid = zca_whiten(X_valid, Wsep, Wmix, mX);
end

% construct RBM and use default configurations
D = default_dae (size(X, 2), 320);

if use_whitening
    D.do_normalize = 0;
    D.do_normalize_std = 0;
end


D.data.binary = 0;
D.hidden.binary = 0;

D.learning.lrate = 1e-1;
D.learning.lrate0 = 10000;
%D.learning.momentum = 0.5;
D.learning.weight_decay = 0;

D.adagrad.use = 1;

D.rica.cost = 0.01;

D.noise.drop = 0;
D.noise.level = 0;

D.sparsity.cost = 0;
D.sparsity.target = 0;

% max. 100 epochs
D.iteration.n_epochs = 500;

% set the stopping criterion
D.stop.criterion = 0;
D.stop.recon_error.tolerate_count = 1000;

% save the intermediate data after every epoch
D.hook.per_epoch = {@save_intermediate, {'rica_patch8.mat'}};

% print learining process
D.verbose = 0;

% display the progress
D.debug.do_display = 0;

fprintf(1, 'Training rICA\n');
tic;
D = dae (D, X, X_valid, 0.1);
fprintf(1, 'Training is done after %f seconds\n', toc);



