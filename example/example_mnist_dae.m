% add the path of RBM code
addpath('..');
%addpath('~/work/Algorithms/liblinear-1.7/matlab');

% load MNIST
load 'mnist_14x14.mat';

% shuffle the training data
perm_idx = randperm (size(X,1));
X = X(perm_idx, :);
X_labels = X_labels(perm_idx);

% construct RBM and use default configurations
D = default_dae (size(X, 2), 500);

D.data.binary = 1;
D.hidden.binary = 1;

D.learning.lrate = 1e-2;
D.learning.lrate0 = 10000;
D.learning.momentum = 0.5;
%D.learning.weight_decay = 0.0001;

D.noise.drop = 0.2;
D.noise.level = 0;

D.sparsity.cost = 0.01;
D.sparsity.target = 0.05;

% max. 100 epochs
D.iteration.n_epochs = 1 %100;

% set the stopping criterion
D.stop.criterion = 0;
D.stop.recon_error.tolerate_count = 1000;

% save the intermediate data after every epoch
D.hook.per_epoch = {@save_intermediate, {'dae_mnist.mat'}};

% print learining process
D.verbose = 0;

% display the progress
D.debug.do_display = 0;

% train RBM
fprintf(1, 'Training DAE\n');
tic;
D = dae (D, X);
fprintf(1, 'Training is done after %f seconds\n', toc);

fprintf(2, 'Training the classifier: ');
rbm_feature = 1./(1 + exp(-bsxfun(@plus, X * D.W, D.hbias')));
model = train(X_labels, sparse(double(rbm_feature)), '-s 0');
fprintf(2, 'Done\n');

fprintf(2, 'Testing the classifier: ');
rbm_feature = 1./(1 + exp(-bsxfun(@plus, X_test * D.W, D.hbias')));
[L accuracy probs] = predict(X_test_labels, sparse(double(rbm_feature)), model, '-b 1');
fprintf(2, 'Done\n');





