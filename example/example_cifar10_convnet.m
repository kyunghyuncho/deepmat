% add the path of RBM code
addpath('..');

% load CIFAR-10
load cifar_10.mat;

% shuffle the training data
perm_idx = randperm (size(X,1));
X = X(perm_idx, :);
X_labels = X_labels(perm_idx);

n_all = size(X, 1);
n_train = ceil(n_all * 4 / 5);
n_valid = floor(n_all / 5);

X_valid = X(perm_idx(n_train+1:end), :);
X_valid_labels = X_labels(perm_idx(n_train+1:end));
X = X(perm_idx(1:n_train), :);
X_labels = X_labels(perm_idx(1:n_train));

% pad images
pad_k = 2;
pad_v = 0;
X = padimages(X, 32, 3, pad_k, pad_v);
X_valid = padimages(X_valid, 32, 3, pad_k, pad_v);
X_test = padimages(X_test, 32, 3, pad_k, pad_v);

use_whitening = 0;

if use_whitening
    %% ZCA
    [Z, Wsep, Wmix, mX] = zca(X, 0.1);
    X = zca_whiten(X, Wsep, Wmix, mX);
    X_valid = zca_whiten(X_valid, Wsep, Wmix, mX);
end

% structures
size_in = 32 + pad_k * 2; % supports a squre image
channel_in = 3; % grayscale image
full_layers = [2000, 2000, 10];
conv_layers = [5*5, 32, 5*5, 32]; % 32 5x5 filters x 2
poolratios = [3,3]; % 3x3 pooling x 2
pooling = [0, 1]; % max pooling + average pooling
strides = [1,1]; % every pixel

% construct convnet
C = default_convnet (size_in, channel_in, full_layers, conv_layers, poolratios, strides);

C.pooling = pooling;

C.learning.lrate = 1e-3;
C.learning.lrate0 = 5000;
C.learning.momentum = 0;
C.learning.weight_decay = 0.0005;

C.adadelta.use = 1;
C.adadelta.momentum = 0.95;
C.adadelta.epsilon = 1e-8;

C.do_normalize = 1;
C.do_normalize_std = 1;

if use_whitening
    D.do_normalize = 0;
    D.do_normalize_std = 0;
end

C.dropout.use = 1;

C.learning.minibatch_sz = 32;

C.hidden.use_tanh = 2;
C.conv.use_tanh = 2;

C.noise.drop = 0.1;
C.noise.level = 0.1;

C.lcn.use = 1;
C.lcn.neigh = 4;

% max. 100 epochs
C.iteration.n_epochs = 150;
C.valid_min_epochs = 50;

% set the stopping criterion
C.stop.criterion = 0;
C.stop.recon_error.tolerate_count = 1000;

% save the intermediate data after every epoch
C.hook.per_epoch = {@save_intermediate, {'convnet_cifar10.mat'}};

% print learining process
C.verbose = 1;

% train RBM
fprintf(1, 'Training convnet\n');
tic;
C = convnet (C, X, X_labels+1, X_valid, X_valid_labels+1, 0.1);
fprintf(1, 'Training is done after %f seconds\n', toc);

save('convnet_cifar10.mat', 'C');

if C.do_normalize
    % make it zero-mean
    Xm = mean(X, 1);
    X_test = bsxfun(@minus, X_test, Xm);
end
if C.do_normalize_std
    % make it unit-variance
    Xs = std(X, [], 1);
    X_test = bsxfun(@rdivide, X_test, Xs);
end

[pred] = convnet_classify (C, X_test);
n_correct = sum(X_test_labels+1 == pred);

fprintf(2, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));
