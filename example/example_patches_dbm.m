% Cho, K., Raiko, T., Ilin, A., and Karhunen, J.
% A Two-stage Pretraining Algorithm for Deep Boltzmann Machines
% http://users.ics.aalto.fi/kcho/papers/nips12workshop.pdf

% add the path of RBM code
addpath('..');

% load natural image patches
load 'bsds500bw_patches_8.mat';
X = (Xbw / 255);

use_whitening = 0;

% shuffle the training data
perm_idx = randperm (size(X,1));
X = X(perm_idx, :);

if use_whitening
    %% ZCA
    %[Z, Wsep, Wmix, mX] = zca(X, 0.1);
    %X = zca_whiten(X, Wsep, Wmix, mX);
    %X_valid = zca_whiten(X_valid, Wsep, Wmix, mX);
    load patch8_whiten.mat;
    X = zca_whiten(X, Wsep, Wmix, mX);
end

pretrain = 0;
centering = 1;

% structure of DBM
layers = [64, 320, 320];
n_layers = length(layers);

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

        if l == 1
            R.data.binary = 0;
        end

        R.learning.lrate = 1e-3;

        R.learning.persistent_cd = 0;
        R.parallel_tempering.use = 0;

        if R.data.binary == 0
            R.grbm.learn_sigmas = ~use_whitening;
        end

        R.adaptive_lrate.use = 1;
        R.adaptive_lrate.lrate_ub = R.learning.lrate;

        mH = mean(Xp, 1)';
        if R.data.binary
            R.vbias = min(max(log(mH./(1 - mH)), -4), 4);
        else
            R.vbias = mH;
        end
        %R.hbias = -4 * ones(size(R.hbias));

        R.enhanced_grad.use = 1;
        R.learning.minibatch_sz = 128;

        if use_whitening
            R.grbm.do_normalize = 0;
            R.grbm.do_normalize_std = 0;
        end
        R.grbm.use_single_sigma = 0;

        % max. 100 epochs
        R.iteration.n_epochs = 100;

        % set the stopping criterion
        R.stop.criterion = 0;
        R.stop.recon_error.tolerate_count = 1000;

        % save the intermediate data after every epoch
        R.hook.per_epoch = {@save_intermediate, {sprintf('rbm_patch8_%d.mat', l)}};
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

D.data.binary = 0;

if pretrain
    % pretraining (stage 2)
    D.learning.persistent_cd = 0;
    D.learning.lrate = 1e-3;
    D.learning.lrate0 = 5000;

    D.enhanced_grad.use = 0;
    D.adaptive_lrate.use = 1;
    D.adaptive_lrate.lrate_ub = D.learning.lrate;

    D.iteration.n_epochs = 500;
    D.hook.per_epoch = {@save_intermediate, {'dbm_pre_patch8.mat'}};

    fprintf(1, 'Training DBM\n');
    tic;
    [D] = dbm (D, X, 1, Qpre, Qpre_mask);
    fprintf(1, 'Training is done after %f seconds\n', toc);
end

% finetuning (stage 3)
D.learning.persistent_cd = 1;
if pretrain
    D.learning.lrate = 1e-4;
else
    D.learning.lrate = 1e-4;
end
D.learning.lrate0 = 5000;

D.grbm.use_single_sigma = 1;

D.enhanced_grad.use = 0;
D.adaptive_lrate.use = 1;
D.adaptive_lrate.lrate_ub = D.learning.lrate;

D.iteration.n_epochs = 500;
D.hook.per_epoch = {@save_intermediate, {'dbm_patch8.mat'}};

if centering
    D = set_dbm_centers(D);
end

fprintf(1, 'Finetuning DBM\n');
tic;
[D] = dbm (D, X);
fprintf(1, 'Finetuning is done after %f seconds\n', toc);






