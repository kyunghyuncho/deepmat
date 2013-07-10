% add the path of RBM code
addpath('..');

% load MNIST
load 'mnist_14x14.mat';
%mnist;

% shuffle the training data
perm_idx = randperm (size(X,1));
X = X(perm_idx, :);
X_labels = X_labels(perm_idx);

% construct RBM and use default configurations
layers = [size(X, 2), 1000, 1000];
n_layers = length(layers);

G = default_gsn (layers);

G.data.binary = 1;

G.learning.lrate = 1e-1;
G.learning.lrate0 = 5000;
G.learning.momentum = 0;

G.adadelta.use = 1; 
G.adadelta.momentum = 0.95;
G.adadelta.epsilon = 1e-8;

G.learning.minibatch_sz = 64;

G.hidden.use_tanh = 1;
G.hidden.noise_level = 2;

G.hidden.add_noise = zeros(n_layers, 1);
G.hidden.add_noise(end) = 1;

G.noise.drop = 0.4; % salt-and-pepper noise
G.noise.level = 0;

% max. 100 epochs
G.iteration.n_epochs = 150;

% set the stopping criterion
G.stop.criterion = 0;
G.stop.recon_error.tolerate_count = 1000;

mH = mean(X, 1)';
G.biases{1} = min(max(log(mH./(1 - mH)), -4), 4);

% save the intermediate data after every epoch
G.hook.per_epoch = {@save_intermediate, {'gsn_mnist.mat'}};

% print learining process
G.verbose = 0;

% train RBM
fprintf(1, 'Training GSN\n');
tic;
G = gsn (G, X, 2);
fprintf(1, 'Training is done after %f seconds\n', toc);

% grab some samples from GSN
n_chains = 25;
% inits = X(1:n_chains,:) + rand(n_chains, size(X, 2));
inits = rand(n_chains, size(X, 2));
S = cell(n_layers, 1);
S{1} = inits;
for l = 2:n_layers
    S{l} = zeros(n_chains, layers(l));
end
occupied = zeros(n_layers, 1);
occupied(1) = 1;

figure;
hold on;
for i=1:100
    [S, occupied] = gsn_sample(G, S, occupied, 0);
    visualize(S{1}', 1);
    drawnow;
%     pause;
    pause(0.1);
end




