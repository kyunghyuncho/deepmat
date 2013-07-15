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

% structures
size_in = 32; % supports a squre image
channel_in = 3; % grayscale image
full_layers = [1000, 10];
conv_layers = [6*6, 32; 4*4, 32]; % 32 8x8 filters and 32 4x4 filters
poolratios = [2, 2]; % 4x4 pooling
strides = [2, 1]; % every second pixel

% construct convnet
C = default_convnet (size_in, channel_in, full_layers, conv_layers, poolratios, strides);

C.learning.lrate = 1e-3;
C.learning.lrate0 = 5000;
C.learning.momentum = 0;

C.adadelta.use = 1;
C.adadelta.momentum = 0.99;
C.adadelta.epsilon = 1e-8;

C.do_normalize = 0;
C.do_normalize_std = 0;

C.dropout.use = 0;

C.learning.minibatch_sz = 32;

C.hidden.use_tanh = 2;

C.noise.drop = 0;
C.noise.level = 0;

% max. 100 epochs
C.iteration.n_epochs = 150;

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
% 
% % grab some samples from GSN
% n_chains = 25;
% % inits = X(1:n_chains,:) + rand(n_chains, size(X, 2));
% inits = rand(n_chains, size(X, 2));
% S = cell(n_layers, 1);
% S{1} = inits;
% for l = 2:n_layers
%     S{l} = zeros(n_chains, layers(l));
% end
% occupied = zeros(n_layers, 1);
% occupied(1) = 1;
% 
% figure;
% hold on;
% for i=1:1000
%     [S, occupied] = gsn_sample(C, S, occupied, 0);
%     visualize_ud(S{1}', 1);
%     drawnow;
% %     pause;
%     pause(0.1);
% end

%fprintf(2, 'Training the classifier: ');
%rbm_feature = double(1./(1 + exp(bsxfun(@minus, -X * C.W, C.hbias'))));
%model = train(X_labels, sparse(rbm_feature), '-s 0');
%fprintf(2, 'Done\n');

%fprintf(2, 'Testing the classifier: ');
%rbm_feature = double(1./(1 + exp(bsxfun(@minus, -X_test * C.W, C.hbias'))));
%[L accuracy] = predict(X_test_labels, sparse(rbm_feature), model);
%fprintf(2, 'Done\n');


%% check the loglikelihood
%logZ = rbm_ais (C, 100, linspace(0, 1, 10001));

%F_train = rbm_energy(X, C.W, C.vbias, C.hbias);
%F_test = rbm_energy(X_test, C.W, C.vbias, C.hbias);
%F_rand = rbm_energy(binornd(1, 0.5, 1000, size(X,2)), C.W, C.vbias, C.hbias);

%like_train = F_train - logZ;
%like_test = F_test - logZ;
%like_rand = F_rand - logZ;

%fprintf(1, '========================');
%fprintf(1, 'log P(train) = %f\n', like_train);
%fprintf(1, 'log P(test) = %f\n', like_test);
%fprintf(1, 'log P(rand) = %f\n', like_rand);
%fprintf(1, '========================');





