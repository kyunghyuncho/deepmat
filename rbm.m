% rbm - training restricted Boltzmann machine using Gibbs sampling
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [R] = rbm(R, patches);

if R.enhanced_grad.bit_flipping == 1
    error('NOT SUPPORTED');
end

actual_lrate = R.learning.lrate;

if R.adaptive_lrate.use == 1
    initial_lrate = R.learning.lrate;
    actual_lrate = initial_lrate;
end

if R.adaptive_lrate.alrate_use_current_batch == 1
    warning('Adaptive learning rate might diverge');
end

n_samples = size(patches, 1);
if R.structure.n_visible ~= size(patches, 2)
    error('Data is not properly aligned');
end
n_visible = R.structure.n_visible;

vbias_grad_old = zeros(size(R.vbias'));
hbias_grad_old = zeros(size(R.hbias'));
W_grad_old = zeros(size(R.W));
sigma_grad_old = zeros(size(R.vbias))';

minibatch_sz = R.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);

n_epochs = R.iteration.n_epochs;
adaptive_lrate = R.adaptive_lrate.use;
enhanced_grad = R.enhanced_grad.use;
bit_flipping = R.enhanced_grad.bit_flipping;

cd_k = R.learning.cd_k;
persistent_cd = R.learning.persistent_cd;
momentum = R.learning.momentum;
weight_decay = R.learning.weight_decay;
mf_update = R.learning.mf_update;

n_hidden = R.structure.n_hidden;
n_visible = R.structure.n_visible;

min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;

do_normalize = R.grbm.do_normalize;
do_normalize_std = R.grbm.do_normalize_std;
update_sigmas = R.grbm.learn_sigmas;
do_vsample = R.grbm.do_vsample;

lrate_lb = R.adaptive_lrate.lrate_lb;
lrate_ub = R.adaptive_lrate.lrate_ub;
exp_up = R.adaptive_lrate.exp_up;
exp_down = R.adaptive_lrate.exp_down;
max_iter_up = R.adaptive_lrate.max_iter_up;
max_iter_down = R.adaptive_lrate.max_iter_down;

if R.data.binary == 0
    if do_normalize == 1
        % make it zero-mean
        patches_mean = mean(patches, 1);
        patches = bsxfun(@minus, patches, patches_mean);
    end

    if do_normalize_std ==1
        % make it unit-variance
        patches_std = std(patches, [], 1);
        patches = bsxfun(@rdivide, patches, patches_std);
    end
end

logsigmas = log(R.sigmas.^2);

% data statistics
mu_d = mean(patches, 1)';
sigmas_d = std(patches, [], 1);
logsigmas_d = log(sigmas_d.^2);

% upper-bound.. but is there any need for it?
sigmas_ub = R.grbm.sigmas_ub;
logsigmas_ub = log(R.grbm.sigmas_ub);

epsilon_sigma = 1e-8;
epsilon_logsigma = log(epsilon_sigma^2);

anneal_counter = 0;
actual_lrate0 = actual_lrate;

if R.debug.do_display == 1
    figure(R.debug.display_fid);
end

try
    use_gpu = gpuDeviceCount;
catch errgpu
    use_gpu = false;
    disp(['Could not use CUDA. Error: ' errgpu.identifier])
end
if use_gpu
    % push
    logsigmas = gpuArray(single(logsigmas));
end

for step=1:n_epochs
    if R.verbose
        fprintf(2, 'Epoch %d/%d: ', step, n_epochs)
    end
    if use_gpu
        % push
        R.W = gpuArray(single(R.W));
        R.vbias = gpuArray(single(R.vbias));
        R.hbias = gpuArray(single(R.hbias));
        R.sigmas = gpuArray(single(R.sigmas));

        if R.fast.use
            R.fast.W = gpuArray(single(R.fast.W));
            R.fast.vbias = gpuArray(single(R.fast.vbias));
            R.fast.hbias = gpuArray(single(R.fast.hbias));
            R.fast.sigmas = gpuArray(single(R.fast.sigmas));
        end
    end

    for mb=1:n_minibatches
        R.iteration.n_updates = R.iteration.n_updates + 1;

        % p_0
        v0 = patches((mb-1) * minibatch_sz + 1:min(mb * minibatch_sz, n_samples), :);
        mb_sz = size(v0,1);

        if use_gpu > 0
            v0 = gpuArray(single(v0));
        end

        if R.data.binary
            v0 = binornd (1, v0);
        end
        
        % just for a bit of speed-up
        if R.data.binary == 0
            if persistent_cd && exist('v1') ~= 0
                fmb_sigma2s = repmat(R.sigmas, [size(v1, 1) 1]);
            else
                fmb_sigma2s = repmat(R.sigmas, [mb_sz 1]);
            end
        end
        
        if adaptive_lrate
            if mb == n_minibatches
                mb_next = 1;
            else
                mb_next = mb + 1;
            end

            v0_next = patches((mb_next-1) * minibatch_sz + 1:min(mb_next * minibatch_sz, n_samples), :);
            if R.data.binary
                v0_next = binornd (1, v0);
            end
            
            if R.data.binary == 0
                next_mb_sz = size(v0_next,1);
                
                if persistent_cd == 0
                    nmb_sigma2s = repmat(R.sigmas, [next_mb_sz 1]);
                else
                    if next_mb_sz ~= minibatch_sz
                        nmb_sigma2s = repmat(R.sigmas, [next_mb_sz 1]);
                    else
                        nmb_sigma2s = fmb_sigma2s;
                    end
                end
            end
            if use_gpu > 0
                v0_next = gpuArray(single(v0_next));
            end
        end

        if R.data.binary
            h0 = sigmoid(bsxfun(@plus, v0 * R.W, R.hbias'));
        else
            h0 = sigmoid(bsxfun(@plus, bsxfun(@rdivide, v0, R.sigmas.^2) * R.W, R.hbias'));
        end

        vbias0 = mean(v0, 1);
        hbias0 = mean(h0, 1);
        W0 = (v0' * h0) / size(v0,1);
        if R.data.binary == 0
            vbias0 = vbias0 ./ R.sigmas.^2;
            W0 = bsxfun(@rdivide, W0, R.sigmas.^2');
            sigma0 = mean((bsxfun(@minus, v0, R.vbias').^2) - ...
                v0 .* (h0 * R.W'), 1);
            sigma0 = sigma0 ./ R.sigmas.^2;
        end

        % compute reconstruction error
        hr = binornd(1, h0, size(h0,1), size(h0,2));
        if R.data.binary
            vr = sigmoid(bsxfun(@plus,hr * R.W',R.vbias'));
        else
            vr = bsxfun(@plus,hr * R.W',R.vbias');
        end

        rerr = mean(sum((v0 - vr).^2,2));
        if use_gpu > 0
            rerr = gather(rerr);
        end
        R.signals.recon_errors = [R.signals.recon_errors rerr];

        % p_1
        if (persistent_cd ~= 0 && exist('h1') == 0)
            h1 = h0;
        end

        if (persistent_cd == 0)
            h1 = h0;
        end

        for k=1:cd_k
            h1 = binornd(1, h1);
            if R.fast.use
                W = (R.W + R.fast.W);
                vbias = (R.vbias + R.fast.vbias);
                hbias = (R.hbias + R.fast.hbias);
            else
                W = R.W;
                vbias = R.vbias;
                hbias = R.hbias;
            end

            if R.data.binary
                if R.learning.T_transition
                    v1_old_0 = (v1 == 0);
                    v1_old_1 = 1 - v1_old_0;
                    v1 = bsxfun(@plus, h1 * W', vbias');
                    v1_0 = min(exp(v1), 1);
                    v1_1 = max(1 - exp(-v1), 0);
                    v1 = v1_old_0 .* v1_0 + v1_old_1 .* v1_1;

                    v1 = binornd(1, v1);

                    clear v1_old_0 v1_old_1 v1_0 v1_1;
                else
                    v1 = sigmoid(bsxfun(@plus,h1 * W', vbias'));
                    v1 = binornd(1, v1, size(v1,1), size(v1,2));
                end
            else
                v1_mean = bsxfun(@plus, h1 * W', vbias');
                if (do_vsample)
                    v1 = normrnd(v1_mean, fmb_sigma2s);
                else
                    v1 = v1_mean;
                end
            end

            if R.learning.T_transition
                h1_old_0 = (h1 == 0);
                h1_old_1 = 1 - h1_old_0;
                if R.data.binary
                    h1 = bsxfun(@plus, v1 * W, hbias');
                else
                    h1 = bsxfun(@plus, bsxfun(@rdivide, v1, R.sigmas.^2) * W, hbias');
                end
                h1_0 = min(exp(h1), 1);
                h1_1 = max(1 - exp(-h1), 0);
                h1 = h1_old_0 .* h1_0 + h1_old_1 .* h1_1;

                h1 = binornd(1, h1);

                clear h1_old_0 h1_old_1 h1_0 h1_1;
            else
                if R.data.binary
                    h1 = sigmoid(bsxfun(@plus,v1 * W,hbias'));
                else
                    h1 = sigmoid(bsxfun(@plus, bsxfun(@rdivide, v1, R.sigmas.^2) * R.W, R.hbias'));
                end
            end
        end

        vbias1 = mean(v1, 1);
        hbias1 = mean(h1, 1);
        W1 = (v1' * h1) / size(v1, 1);
        if R.data.binary == 0
            vbias1 = vbias1 ./ R.sigmas.^2;
            W1 = bsxfun(@rdivide, W1, R.sigmas.^2');
            sigma1 = mean((bsxfun(@minus, v1, R.vbias').^2) - ...
                v1 .* (h1 * R.W'), 1);
            sigma1 = sigma1 ./ R.sigmas.^2;
        end

        % get gradient
        vbias_grad = vbias0 - vbias1;
        W_grad = W0 - W1;
        if R.data.binary == 0
            sigma_grad = sigma0 - sigma1;
            if R.grbm.use_single_sigma == 1
                mean_sigma_grad = mean(sigma_grad);
                sigma_grad = mean_sigma_grad * ones(size(sigma_grad));
            end
        end
        hbias_grad = hbias0 - hbias1;

        if enhanced_grad
            vacts = (mean(v1, 1) + mean(v0, 1))/2;
            hacts = (mean(h1, 1) + mean(h0, 1))/2;

            vacts = vacts ./ R.sigmas.^2;

            W_grad = W_grad - vbias_grad' * hacts - vacts' * hbias_grad;

            vbias_grad = vbias_grad - hacts * W_grad';
            hbias_grad = hbias_grad - vacts * W_grad;
        end

        if R.learning.lrate_anneal > 0 && (step >= R.learning.lrate_anneal * n_epochs)
            anneal_counter = anneal_counter + 1;
            actual_lrate = actual_lrate0 / anneal_counter;
        else
            % now we find the optimal(?) step size
            if adaptive_lrate == 1
                base_lrate = actual_lrate;

                % we assume that the fantasy particles are truly from the model.
                vf = v1;
                if (R.adaptive_lrate.alrate_use_current_batch)
                    vd = v0;
                else
                    vd = v0_next;
                end

                candidate_lrates;

                if R.data.binary
                    [cE, cEmin, cEmax, cEs] = rbm_energy(vf, R.W, R.vbias, R.hbias);
                    [dcE, dcEmin, dcEmax, dcEs] = rbm_energy(vd, R.W, R.vbias, R.hbias);
                else
                    [cE, cEmin, cEmax, cEs] = grbm_energy(vf, R.W, R.vbias, R.hbias, R.sigmas);
                    [dcE, dcEmin, dcEmax, dcEs] = grbm_energy(vd, R.W, R.vbias, R.hbias, R.sigmas);
                end
                % current
                curr_cost = sum(-dcEs);

                % search
                costs = zeros(length(cand_lrates), 1);
                for s=1:length(cand_lrates)
                    cand_lrate = cand_lrates(s);

                    vbias_test = R.vbias + cand_lrate * (momentum * vbias_grad_old' + (1-momentum) * vbias_grad');
                    hbias_test = R.hbias + cand_lrate * (momentum * hbias_grad_old' + (1-momentum) * hbias_grad');
                    W_test = R.W + cand_lrate * (momentum * W_grad_old + (1-momentum) * W_grad);
                    if R.data.binary == 0
                        if update_sigmas == 1
                            logsigmas_test = logsigmas + cand_lrate * ((1-momentum) * sigma_grad + momentum * sigma_grad_old);
                            logsigmas_test = max(epsilon_logsigma, min(logsigmas_ub, logsigmas_test));
                            sigmas_test = sqrt(exp(logsigmas_test));
                        else
                            sigmas_test = R.sigmas;
                        end
                    end
                    

                    if R.data.binary
                        [dE, dEmin, dEmax, dEs] = rbm_energy(vd, W_test, vbias_test, hbias_test);
                        [fE, fEmin, fEmax, fEs] = rbm_energy(vf, W_test, vbias_test, hbias_test);
                    else
                        [dE, dEmin, dEmax, dEs] = grbm_energy(vd, W_test, vbias_test, hbias_test, sigmas_test);
                        [fE, fEmin, fEmax, fEs] = grbm_energy(vf, W_test, vbias_test, hbias_test, sigmas_test);
                    end

                    now_cost = sum(-dEs - logsum(double(gather(-fEs + cEs))) + log(size(vf,1)));

                    if use_gpu
                        costs(s) = gather(now_cost);
                    else
                        costs(s) = now_cost;
                    end
                end

                [chosen_cost chosen_index] = max(costs);
                actual_lrate = max(lrate_lb, min(lrate_ub, cand_lrates(chosen_index)));
            else
                if R.learning.lrate0 > 0
                    actual_lrate = R.learning.lrate / (1 + R.iterations.n_updates / R.learning.lrate0);
                else
                    actual_lrate = R.learning.lrate;
                end
            end
            actual_lrate0 = actual_lrate;
        end

        R.signals.lrates = [R.signals.lrates actual_lrate];

        % update
        vbias_grad_old = (1-momentum) * vbias_grad + momentum * vbias_grad_old;
        hbias_grad_old = (1-momentum) * hbias_grad + momentum * hbias_grad_old;
        W_grad_old = (1-momentum) * W_grad + momentum * W_grad_old;

        R.vbias = R.vbias + actual_lrate * (vbias_grad_old' - weight_decay * R.vbias);
        R.hbias = R.hbias + actual_lrate * (hbias_grad_old' - weight_decay * R.hbias);
        R.W = R.W + actual_lrate * (W_grad_old - weight_decay * R.W);
    
        if R.data.binary == 0
            if update_sigmas == 1
                sigma_grad_old = (1-momentum) * sigma_grad + momentum * sigma_grad_old;
                logsigmas = logsigmas + actual_lrate * (sigma_grad_old - weight_decay * logsigmas);
                logsigmas = max(epsilon_logsigma, min(logsigmas_ub, logsigmas));
                R.sigmas = sqrt(exp(logsigmas));
            end
        end

        if R.fast.use
            R.fast.W = (19/20) * R.fast.W + R.fast.lrate * W_grad_old;
            R.fast.vbias = (19/20) * R.fast.vbias + R.fast.lrate * vbias_grad_old';
            R.fast.hbias = (19/20) * R.fast.hbias + R.fast.lrate * hbias_grad_old';
        end

        if R.verbose == 1
            fprintf(2, '.');
        end

        if use_gpu > 0
            clear v0 h0 vr hr v0_next;
            if persistent_cd == 0
                clear v1 h1;
            end
            clear vf vd;
            clear vbias_test hbias_test W_test logsigmas_test sigmas_test;

            clear fmb_sigma2s nmb_sigma2s;

            clear vbias0 vbias1 hbias0 hbias1 W0 W1 sigma0 sigma1;
            clear W_adj
        end

        if R.stop.criterion > 0
            if R.stop.criterion == 1
                if min_recon_error > R.signals.recon_errors(end)
                    min_recon_error = R.signals.recon_errors(end);
                    min_recon_error_update_idx = R.iteration.n_updates;
                else
                    if R.iteration.n_updates > min_recon_error_update_idx + R.stop.recon_error.tolerate_count 
                        fprintf(2, '\nStopping criterion reached (recon error) %f > %f\n', ...
                            R.signals.recon_errors(end), min_recon_error);
                        stopping = 1;
                        break;
                    end
                end
            elseif R.stop.criterion == 2
                if actual_lrate < R.stop.lrate.lowrate_threshold
                    fprintf(2, '\nStopping criterion reached (lrate) %f > %f\n', ...
                        R.stop.lrate.lowlrate_threshold, actual_lrate);
                    stopping = 1;
                    break;
                end
            else
                error ('Unknown stopping criterion %d', R.stop.criterion);
            end
        end

        if length(R.hook.per_update) > 1
            err = R.hook.per_update{1}(R, R.hook.per_update{2});

            if err == -1
                stopping = 1;
                break;
            end
        end
        
        if R.debug.do_display == 1 && mod(R.iteration.n_updates, R.debug.display_interval) == 0
            R.debug.display_function (R.debug.display_fid, R, v0, v1, h0, h1, W_grad, vbias_grad, hbias_grad);
            drawnow;
        end
    end

    if use_gpu > 0
        % pull
        R.W = gather(R.W);
        R.vbias = gather(R.vbias);
        R.hbias = gather(R.hbias);
        R.sigmas = gather(R.sigmas);

        if R.fast.use
            R.fast.W = gather(R.fast.W);
            R.fast.vbias = gather(R.fast.vbias);
            R.fast.hbias = gather(R.fast.hbias);
            R.fast.sigmas = gather(R.fast.sigmas);
        end
    end

    if length(R.hook.per_epoch) > 1
        err = R.hook.per_epoch{1}(R, R.hook.per_epoch{2});

        if err == -1
            stopping = 1;
        end
    end

    if stopping == 1
        break;
    end
    
    if R.verbose == 1
        fprintf(2, '\n');
    end
        
    fprintf(2, 'Epoch %d/%d - recon_error: %f norms: %f/%f/%f\n', step, n_epochs, ...
        R.signals.recon_errors(end), ...
        R.W(:)' * R.W(:) / length(R.W(:)), ...
        R.vbias' * R.vbias / length(R.vbias), ...
        R.hbias' * R.hbias / length(R.hbias));
end

if use_gpu > 0
    % pull
    R.W = gather(R.W);
    R.vbias = gather(R.vbias);
    R.hbias = gather(R.hbias);
    R.sigmas = gather(R.sigmas);

    if R.fast.use
        R.fast.W = gather(R.fast.W);
        R.fast.vbias = gather(R.fast.vbias);
        R.fast.hbias = gather(R.fast.hbias);
        R.fast.sigmas = gather(R.fast.sigmas);
    end

    % clear
    clear h1 logsigmas;
end


