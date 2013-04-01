% add the path of RBM code
addpath('..');

% load MNIST
load 'mnist_14x14.mat';

% shuffle the training data
perm_idx = randperm (size(X,1));
X = X(perm_idx, :);
X_labels = X_labels(perm_idx);

layers = [size(X, 2) 500 500 1000];
n_layers = length(layers);

Rs = cell(n_layers, 1);

% construct RBM and use default configurations
H = X;

for l=1:n_layers-1
    R = default_rbm (size(H, 2), layers(l+1));

    R.data.binary = 1;

    mH = mean(H, 1)';
    R.vbias = min(max(log(mH./(1 - mH)), -4), 4);
    %R.hbias = -4 * ones(size(R.hbias));

    R.learning.lrate = 1e-3;

    R.learning.persistent_cd = 0;
    R.parallel_tempering.use = 0;
    R.adaptive_lrate.use = 1;
    R.adaptive_lrate.lrate_ub = R.learning.lrate;
    R.enhanced_grad.use = 1;
    R.learning.minibatch_sz = 256;

    % max. 100 epochs
    R.iteration.n_epochs = 200;

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
    R = train_rbm (R, H);
    fprintf(1, 'Training is done after %f seconds\n', toc);

    Rs{l} = R;

    H = rbm_get_hidden(H, R);
end


D = default_dbn (layers);

D.hook.per_epoch = {@save_intermediate, {'dbn_mnist.mat'}};

D.learning.lrate = 1e-3;
D.learning.lrate0 = 5000;
D.learning.momentum = 0;
D.learning.weight_decay = 0.0001;
D.learning.minibatch_sz = 256;

D.learning.contrastive_step = 10;
D.learning.persistent_cd = 0;
D.learning.ffactored = 1;

D.iteration.n_epochs = 200;

for l = 1:n_layers-2
    if l > 1
        D.gen.biases{l} = (D.gen.biases{l} + Rs{l}.vbias)/2;
    else
        D.gen.biases{l} = Rs{l}.vbias;
    end
    D.gen.biases{l+1} = Rs{l}.hbias;
    D.gen.W{l} = Rs{l}.W;

    if l > 1
        D.rec.biases{l} = (D.rec.biases{l} + Rs{l}.vbias)/2;
    else
        D.rec.biases{l} = Rs{l}.vbias;
    end
    D.rec.biases{l+1} = Rs{l}.hbias;
    D.rec.W{l} = Rs{l}.W;
end

D.top.W = Rs{n_layers-1}.W;
D.top.vbias = Rs{n_layers-1}.vbias;
D.top.hbias = Rs{n_layers-1}.hbias;

fprintf(1, 'Training DBN\n');
tic;
D = dbn (D, X);
fprintf(1, 'Training is done after %f seconds\n', toc);

n_chains = 20;
n_samples = 11;
rndidx = randperm(size(X, 1));
Sall = zeros(n_samples * n_chains, size(X, 2));
for ci = 1:n_chains
    %S = dbn_sample(rand(1, size(X, 2)), D, n_samples, 1);
    S = dbn_sample(X(rndidx(ci),:), D, n_samples-1, 1);
    Sall(((ci-1) * n_samples + 1), :) = X(rndidx(ci),:);
    Sall(((ci-1) * n_samples + 2):(ci * n_samples), :) = S;
end
save 'dbn_samples.mat' Sall;




