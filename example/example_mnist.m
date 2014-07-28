% add the path of RBM code
addpath('..');

% load MNIST
load 'mnist_14x14.mat';

% shuffle the training data
perm_idx = randperm (size(X,1));
X = X(perm_idx, :);
X_labels = X_labels(perm_idx);

% construct RBM and use default configurations
R = default_rbm (size(X, 2), 361);

R.learning.lrate = 1e-2;

R.parallel_tempering.use = 0;
R.adaptive_lrate.use = 0;
R.enhanced_grad.use = 1;
R.learning.minibatch_sz = 128;

% max. 100 epochs
R.iteration.n_epochs = 20;

% set the stopping criterion
R.stop.criterion = 0;
R.stop.recon_error.tolerate_count = 1000;

% save the intermediate data after every epoch
R.hook.per_epoch = {@save_intermediate, {'rbm_mnist.mat'}};

% print learining process
R.verbose = 1;

% display the progress
R.debug.do_display = 0;
R.debug.display_interval = 10;
R.debug.display_fid = 1;
R.debug.display_function = @visualize_rbm;

% train RBM
fprintf(1, 'Training RBM\n');
tic;
R = train_rbm (R, X);
fprintf(1, 'Training is done after %f seconds\n', toc);

fprintf(2, 'Training the classifier: ');
rbm_feature = 1./(1 + exp(bsxfun(@minus, -X * R.W, R.hbias')));
model = train(X_labels, sparse(double(rbm_feature)), '-s 0');
fprintf(2, 'Done\n');

fprintf(2, 'Testing the classifier: ');
rbm_feature = 1./(1 + exp(bsxfun(@minus, -X_test * R.W, R.hbias')));
[L accuracy probs] = predict(X_test_labels, sparse(double(rbm_feature)), model, '-b 1');
fprintf(2, 'Done\n');


% check the loglikelihood
logZ = rbm_ais (R, 100, linspace(0, 1, 10001));

F_train = rbm_energy(X, R.W, R.vbias, R.hbias);
F_test = rbm_energy(X_test, R.W, R.vbias, R.hbias);
F_rand = rbm_energy(binornd(1, 0.5, 1000, size(X,2)), R.W, R.vbias, R.hbias);

like_train = F_train - logZ;
like_test = F_test - logZ;
like_rand = F_rand - logZ;

fprintf(1, '========================');
fprintf(1, 'log P(train) = %f\n', like_train);
fprintf(1, 'log P(test) = %f\n', like_test);
fprintf(1, 'log P(rand) = %f\n', like_rand);
fprintf(1, '========================');

% grab some samples from RBM
S = rbm_sample(binornd(1, 0.5, 10, size(X,2)), R, 30, 1000);

figure;
for i=1:10
    subplot(10, 1, i);
    visualize_adv(squeeze(S(i,:,:)), 1, 1, 30, 0, 0);
end





