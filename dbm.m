% dbm - training DBM using Gibbs sampling
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
function [D] = dbm(D, patches, use_Qpre, Qpre, Qpre_mask);

if nargin < 3
    use_Qpre = 0;
end

actual_lrate = D.learning.lrate;

if D.adaptive_lrate.use == 1
    initial_lrate = D.learning.lrate;
    actual_lrate = initial_lrate;
end

n_samples = size(patches, 1);
if D.structure.layers(1) ~= size(patches, 2)
    error('Data is not properly aligned');
end

minibatch_sz = D.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);

n_epochs = D.iteration.n_epochs;

cd_k = D.learning.cd_k;
persistent_cd = D.learning.persistent_cd;
momentum = D.learning.momentum;
weight_decay = D.learning.weight_decay;

adaptive_lrate = D.adaptive_lrate.use;
enhanced_grad = D.enhanced_grad.use;

lrate_lb = D.adaptive_lrate.lrate_lb;
lrate_ub = D.adaptive_lrate.lrate_ub;
exp_up = D.adaptive_lrate.exp_up;
exp_down = D.adaptive_lrate.exp_down;
max_iter_up = D.adaptive_lrate.max_iter_up;
max_iter_down = D.adaptive_lrate.max_iter_down;

layers = D.structure.layers;
n_layers = length(layers);

min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;

do_normalize = D.grbm.do_normalize;
do_normalize_std = D.grbm.do_normalize_std;
update_sigmas = D.grbm.learn_sigmas;
do_vsample = D.grbm.do_vsample;

if D.data.binary == 0
    if do_normalize == 1
        % make it zero-mean
        patches_mean = mean(patches, 1);
        patches = bsxfun(@minus, patches, patches_mean);
    end

    if do_normalize_std == 1
        % make it unit-variance
        patches_std = std(patches, [], 1);
        patches = bsxfun(@rdivide, patches, patches_std);
    end
end

n_samples = size(patches, 1);

logsigmas = log(D.sigmas.^2);

% upper-bound.. but is there any need for it?
sigmas_ub = D.grbm.sigmas_ub;
logsigmas_ub = log(D.grbm.sigmas_ub);

biases_grad_old = cell(n_layers, 1);
W_grad_old = cell(n_layers, 1);
for l = 1:n_layers
    biases_grad_old{l} = zeros(size(D.biases{l}))';
    if l < n_layers
        W_grad_old{l} = zeros(size(D.W{l}));
    end
end
sigma_grad_old = zeros(size(D.biases{1}))';

n_minibatches = ceil(n_samples / minibatch_sz);
n_updates = 0;

epsilon_sigma = 1e-8;
epsilon_logsigma = log(epsilon_sigma^2);

min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;

anneal_counter = 0;
actual_lrate0 = actual_lrate;

if D.debug.do_display == 1
    figure(D.debug.display_fid);
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
    if D.verbose
        fprintf(2, 'Epoch %d/%d: ', step, n_epochs)
    end

    if use_gpu
        % push
        for l = 1:n_layers
            if l < n_layers 
                D.W{l} = gpuArray(single(D.W{l}));
            end
            D.biases{l} = gpuArray(single(D.biases{l}));
            if D.centering.use
                D.centering.centers{l} = gpuArray(single(D.centering.centers{l}));
            end
        end

        D.sigmas = gpuArray(single(D.sigmas));
    end

    for mb=1:n_minibatches
        D.iteration.n_updates = D.iteration.n_updates + 1;

        if D.verbose
            tic;
        end

        % p_0
        mb_start = (mb-1) * minibatch_sz + 1;
        mb_end = min(mb * minibatch_sz, n_samples);
        v0 = patches(mb_start:mb_end, :);

        if use_gpu > 0
            v0 = gpuArray(single(v0));
        end
        
        if D.data.binary
            v0 = binornd(1, v0);
        end
        mb_sz = size(v0,1);
        
        % just for a bit of speed-up
        if persistent_cd && exist('h1') ~= 0
            fmb_sigma2s = repmat(D.sigmas', [size(h1{1}, 1) 1]);
        else
            fmb_sigma2s = repmat(D.sigmas', [mb_sz 1]);
        end
        
        if use_Qpre
            % for pretraining
            h0 = cell(n_layers, 1);
            h0{1} = v0;
            for l = 2:n_layers
                if Qpre_mask(l)
                    h0{l} = binornd(1, Qpre{l}(mb_start:mb_end, :));
                else
                    h0{l} = zeros(mb_end - mb_start + 1, layers(l));
                end
                if use_gpu 
                    h0{l} = gpuArray(single(h0{l}));
                end
            end

            for l = 2:n_layers
                if Qpre_mask(l)
                    continue;
                end

                h0{l} = h0{l} * 0;
                if l > 1
                    if l == 2 && D.data.binary == 0
                        h0{l} = h0{l} + bsxfun(@rdivide, h0{l-1}, D.sigmas.^2') * D.W{l-1};
                    else
                        h0{l} = h0{l} + h0{l-1} * D.W{l-1};
                    end
                end

                if l < n_layers
                    h0{l} = h0{l} + h0{l+1} * D.W{l}';
                end

                h0{l} = sigmoid(bsxfun(@plus, h0{l}, D.biases{l}'));
            end

            if adaptive_lrate
                if mb == n_minibatches
                    mb_next = 1;
                else
                    mb_next = mb + 1;
                end

                nmb_start = (mb_next-1) * minibatch_sz + 1;
                nmb_end = min(mb_next * minibatch_sz, n_samples);
                if use_gpu
                    v0_next = gpuArray(single(patches(nmb_start:nmb_end, :)));
                else
                    v0_next = single(patches(nmb_start:nmb_end, :));
                end

                if D.data.binary == 0
                    next_mb_sz = size(v0_next,1);
                    
                    if persistent_cd == 0
                        nmb_sigma2s = repmat(D.sigmas, [next_mb_sz 1]);
                    else
                        if next_mb_sz ~= minibatch_sz
                            nmb_sigma2s = repmat(D.sigmas, [next_mb_sz 1]);
                        else
                            nmb_sigma2s = fmb_sigma2s;
                        end
                    end
                end
            end
        else
            % for finetuning
            h0 = dbm_get_hidden(v0, D, 10, 1e-6, D.mf.reg);
            h0{1} = v0;

            if adaptive_lrate
                if mb == n_minibatches
                    mb_next = 1;
                else
                    mb_next = mb + 1;
                end
                if use_gpu
                    v0_next = gpuArray(single(patches((mb_next-1) * minibatch_sz + 1:min(mb_next * minibatch_sz, n_samples), :)));
                else
                    v0_next = single(patches((mb_next-1) * minibatch_sz + 1:min(mb_next * minibatch_sz, n_samples), :));
                end
                if D.data.binary == 0
                    next_mb_sz = size(v0_next,1);
                    
                    if persistent_cd == 0
                        nmb_sigma2s = repmat(D.sigmas, [next_mb_sz 1]);
                    else
                        if next_mb_sz ~= minibatch_sz
                            nmb_sigma2s = repmat(D.sigmas, [next_mb_sz 1]);
                        else
                            nmb_sigma2s = fmb_sigma2s;
                        end
                    end
                end
            end
        end

        % p_1
        if (persistent_cd ~= 0 && exist('h1') == 0)
            h1 = h0;
        end
        
        if (persistent_cd == 0)
            h1 = h0;

            if use_Qpre
                for l = 2:n_layers
                    if Qpre_mask(l)
                        continue;
                    end
                    
                    h1{l} = binornd(1, h1{l});
                end
            end
        end

        
        if D.centering.use
            for l = 1:n_layers
                if D.data.binary == 0 && l == 1
                    continue;
                end
                h0{l} = bsxfun(@minus, h0{l}, D.centering.centers{l}');
            end
        end
        
        % compute reconstruction error
        if D.data.binary == 1
            vr = sigmoid(bsxfun(@plus, h0{2} * D.W{1}',D.biases{1}'));
        else
            vr = bsxfun(@plus, h0{2} * D.W{1}',D.biases{1}');
        end

        rerr = mean(sum((v0 - vr).^2,2));
        if use_gpu > 0
            rerr = gather(rerr);
        end
        D.signals.recon_errors = [D.signals.recon_errors rerr];

        for k=1:cd_k
            for oddeven = [1 0]
                for l = 1:n_layers
                    if mod(l, 2) == oddeven
                        continue;
                    end
                    h1{l} = h1{l} * 0;
                    if D.centering.use
                        if l > 1
                            if l == 2 && D.data.binary == 0
                                h1{l} = h1{l} + bsxfun(@rdivide, h1{l-1}, D.sigmas.^2') * D.W{l-1};
                            else
                                h1{l} = h1{l} + bsxfun(@minus, h1{l-1}, D.centering.centers{l-1}') * D.W{l-1};
                            end
                        end

                        if l < n_layers
                            h1{l} = h1{l} + bsxfun(@minus, h1{l+1}, D.centering.centers{l+1}') * D.W{l}';
                        end
                    else
                        if l > 1
                            if l == 2 && D.data.binary == 0
                                h1{l} = h1{l} + bsxfun(@rdivide, h1{l-1}, D.sigmas.^2') * D.W{l-1};
                            else
                                h1{l} = h1{l} + h1{l-1} * D.W{l-1};
                            end
                        end

                        if l < n_layers
                            h1{l} = h1{l} + h1{l+1} * D.W{l}';
                        end
                    end

                    h1{l} = bsxfun(@plus, h1{l}, D.biases{l}');

                    if l > 1 || D.data.binary == 1
                        h1{l} = sigmoid(h1{l});
                        h1{l} = binornd(1, h1{l});
                    else
                        if do_vsample
                            h1{l} = normrnd(h1{l}, fmb_sigma2s);
                        end
                    end
                end
                if (sum(sum(isnan(h1{1}))) > 0)
                    error('NaN found in the visual fantasy particles.\n It is advisable to adjust learning parameters.');
                end
            end
        end

        if D.centering.use
            for l = 1:n_layers
                if D.data.binary == 0 && l == 1
                    continue;
                end
                h1{l} = bsxfun(@minus, h1{l}, D.centering.centers{l}');
            end
        end
        
        % get base distribution
        base_vbias = mean(h1{1}, 1);
        base_sigma = std(h1{1}, [], 1)';
        
        % get gradient
        for l = 1:n_layers
            if D.data.binary == 0 && l == 1
                bias0 = bsxfun(@rdivide, mean(h0{l}, 1), D.sigmas.^2');
                bias1 = bsxfun(@rdivide, mean(h1{l}, 1), D.sigmas.^2');
            else
                bias0 = mean(h0{l}, 1);
                bias1 = mean(h1{l}, 1);
            end

            biases_grad{l} = bias0 - bias1;

            clear bias0 bias1;

            if l < n_layers
                if D.data.binary == 0 && l == 1
                    W0 = bsxfun(@rdivide, (h0{l}' * h0{l+1}) / mb_sz, D.sigmas.^2);
                    W1 = bsxfun(@rdivide, (h1{l}' * h1{l+1}) / size(h1{1},1), D.sigmas.^2);
                else
                    W0 = (h0{l}' * h0{l+1}) / mb_sz;
                    W1 = (h1{l}' * h1{l+1}) / size(h1{1},1);
                end

                W_grad{l} = W0 - W1;

                clear W0 W1;
            end
        end

        if D.data.binary == 0
            sigma0 = mean((bsxfun(@minus, h0{1}, D.biases{1}').^2) - ...
                h0{1} .* (h0{2} * D.W{1}'), 1);
            sigma1 = mean((bsxfun(@minus, h1{1}, D.biases{1}').^2) - ...
                h1{1} .* (h1{2} * D.W{1}'), 1);
            sigma_grad = (sigma0 - sigma1) ./ D.sigmas.^2';

            if D.grbm.use_single_sigma == 1
                mean_sigma_grad = mean(sigma_grad);
                sigma_grad = mean_sigma_grad * ones(size(sigma_grad));
            end

            clear sigma0 sigma1;
        end

        % enhanced grad
        if enhanced_grad == 1
            acts = cell(n_layers, 1);
            for l = 1:n_layers
                acts{l} = (mean(h0{l}, 1) + mean(h1{l}, 1))/2;
            end
%            if D.data.binary == 0
%                acts{1} = acts{1} ./ D.sigmas.^2';
%            end

            for l = 1:n_layers-1
                W_grad{l} = W_grad{l} - biases_grad{l}' * acts{l+1} ...
                    - acts{l}' * biases_grad{l+1};
            end

            for l = 1:n_layers
                if l > 1
                    acts1 = acts{l-1};
                    biases_grad{l} = biases_grad{l} - acts1 * W_grad{l-1};
                end
                if l < n_layers
                    acts2 = acts{l+1};
                    biases_grad{l} = biases_grad{l} - acts2 * W_grad{l}';
                end
            end

            clear acts;
        end
        
        if D.learning.lrate_anneal > 0 && (step >= D.learning.lrate_anneal * n_epochs)
            anneal_counter = anneal_counter + 1;
            actual_lrate = actual_lrate0 / anneal_counter;
        else
            if adaptive_lrate == 1
                if use_Qpre
                    h0_next = cell(n_layers, 1);
                    h0_next{1} = v0;
                    for l = 2:n_layers
                        if Qpre_mask(l)
                            h0_next{l} = Qpre{l}(nmb_start:nmb_end, :);
                        else
                            h0_next{l} = zeros(nmb_end - nmb_start + 1, layers(l));
                        end
                        if use_gpu 
                            h0_next{l} = gpuArray(single(h0_next{l}));
                        end
                    end

                    for l = 2:n_layers
                        if Qpre_mask(l)
                            continue;
                        end

                        h0_next{l} = h0_next{l} * 0;
                        if l > 1
                            if l == 2 && D.data.binary == 0
                                h0_next{l} = h0_next{l} + bsxfun(@rdivide, h0_next{l-1}, D.sigmas.^2') * D.W{l-1};
                            else
                                h0_next{l} = h0_next{l} + h0_next{l-1} * D.W{l-1};
                            end
                        end

                        if l < n_layers
                            h0_next{l} = h0_next{l} + h0_next{l+1} * D.W{l}';
                        end

                        h0_next{l} = sigmoid(bsxfun(@plus, h0_next{l}, D.biases{l}'));
                    end
                else
                    h0_next = dbm_get_hidden(v0_next, D, 10, 1e-6, D.mf.reg);
                    h0_next{1} = v0_next;
                end

                if D.centering.use
                    for l = 1:n_layers
                        if D.data.binary == 0 && l == 1
                            continue;
                        end
                        h0_next{l} = bsxfun(@minus, h0_next{l}, D.centering.centers{l}');
                    end
                end
                
                [cE, cEmin, cEmax, cEs] = dbm_energy(h1, D.W, D.biases, D.data.binary, 1., D.sigmas, base_sigma, base_vbias);

                base_lrate = actual_lrate;
                candidate_lrates;

                costs = zeros(1, length(cand_lrates));
                for s=1:length(cand_lrates)
                    W_test = cell(size(D.W));
                    biases_test = cell(size(D.biases));
                    if use_gpu
                        logsigmas_test = gpuArray(single(zeros(size(logsigmas))));
                    else
                        logsigmas_test = single(zeros(size(logsigmas)));
                    end
                    cand_lrate = cand_lrates(s);

                    for l = 1:n_layers
                        biases_test{l} = D.biases{l} + cand_lrate * (((1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l})' - weight_decay * D.biases{l});
                        if l < n_layers
                            W_test{l} = D.W{l} + cand_lrate * ((1 - momentum) * W_grad{l} + momentum * W_grad_old{l} - weight_decay * D.W{l});
                        end
                    end
                    if D.data.binary == 0
                        if update_sigmas == 1
                            logsigmas_test = logsigmas + cand_lrate * (((1-momentum) * sigma_grad + momentum * sigma_grad_old)' - weight_decay * logsigmas);
                            logsigmas_test = max(epsilon_logsigma, min(logsigmas_ub, logsigmas_test));
                            sigmas_test = sqrt(exp(logsigmas));
                        else
                            sigmas_test = sqrt(exp(logsigmas_test));
                        end
                    else
                        sigmas_test = sqrt(exp(logsigmas_test));
                    end

%                    % FIXME: Should we?
%                    h0_next = dbm_get_hidden_raw(v0_next, D.data.binary, D.structure.layers, ...
%                        W_test, biases_test, sigmas_test, 5, 1e-5);
%                    h0_next{1} = v0_next;
                    [dE, dEmin, dEmax, dEs] = dbm_energy(h0_next, W_test, biases_test, D.data.binary, 1., sigmas_test, base_sigma, base_vbias);
                    [fE, fEmin, fEmax, fEs] = dbm_energy(h1, W_test, biases_test, D.data.binary, 1., sigmas_test, base_sigma, base_vbias);

                    now_cost = sum(-double(gather(dEs)) - logsum(double(gather(-fEs + cEs))) + log(size(h1{1},1)));

                    costs(s) = now_cost;

                    clear W_test biases_test logsigmas_test sigmas_test;
                    %clear h0_next;
                end

                [chosen_cost chosen_index] = max(costs);
                actual_lrate = min(lrate_ub, max(lrate_lb, cand_lrates(chosen_index)));
            else
                actual_lrate = D.learning.lrate / (1 + D.iteration.n_updates / D.learning.lrate0);
            end
            actual_lrate0 = actual_lrate;
        end

        D.signals.lrates = [D.signals.lrates actual_lrate];
        
%        if D.debug.do_display == 1 && mod(D.iteration.n_updates, D.debug.display_interval) == 0
%            D.debug.display_function (D.debug.display_fid, D, v0, v1, W_grad, vbias_grad, hbias_grad, sigma_grad);
%            drawnow;
%        end
%        
        % update
        for l = 1:n_layers
            biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
            D.biases{l} = D.biases{l} + actual_lrate * (biases_grad_old{l}' - weight_decay * D.biases{l});
            if l < n_layers
                W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                D.W{l} = D.W{l} + actual_lrate * (W_grad_old{l} - weight_decay * D.W{l});
            end
        end
        if D.data.binary == 0
            if update_sigmas == 1
                sigma_grad_old = (1-momentum) * sigma_grad + momentum * sigma_grad_old;
                logsigmas = logsigmas + actual_lrate * (sigma_grad_old' - weight_decay * logsigmas);
                logsigmas = max(epsilon_logsigma, min(logsigmas_ub, logsigmas));
                D.sigmas = sqrt(exp(logsigmas));
            end
        end

        if D.verbose == 1
            fprintf(2, '%2.3fs.', toc);
        end

        if D.stop.criterion > 0
            if D.stop.criterion == 1
                if min_recon_error > D.signals.recon_errors(end)
                    min_recon_error = D.signals.recon_errors(end);
                    min_recon_error_update_idx = D.iteration.n_updates;
                else
                    if D.iteration.n_updates > min_recon_error_update_idx + D.stop.recon_error.tolerate_count 
                        fprintf(2, '\nStopping criterion reached (recon error) %f > %f\n', ...
                            D.signals.recon_errors(end), min_recon_error);
                        stopping = 1;
                        break;
                    end
                end
            else
                error ('Unknown stopping criterion %d', D.stop.criterion);
            end
        end

        if length(D.hook.per_update) > 1
            err = D.hook.per_update{1}(D, D.hook.per_update{2});

            if err == -1
                stopping = 1;
                break;
            end
        end

        if D.centering.use
            for l = 1:n_layers
                if D.data.binary == 0 && l == 1
                    continue;
                end
                h0{l} = bsxfun(@plus, h0{l}, D.centering.centers{l}');
                h1{l} = bsxfun(@plus, h1{l}, D.centering.centers{l}');
            end
        end
        
        
        if use_gpu > 0
            clear v0 h0;
            clear v0_next h0_next;
            if persistent_cd == 0
                clear h1;
            end

            clear fmb_sigma2s;
            clear base_sigma base_vbias;
        end

    end

    if use_gpu > 0
        % pull
        for l = 1:n_layers
            if l < n_layers
                D.W{l} = gather(D.W{l});
            end
            D.biases{l} = gather(D.biases{l});
            if D.centering.use
                D.centering.centers{l} = gather(D.centering.centers{l});
            end
        end
        D.sigmas = gather(D.sigmas);
    end

    if length(D.hook.per_epoch) > 1
        err = D.hook.per_epoch{1}(D, D.hook.per_epoch{2});

        if err == -1
            stopping = 1;
        end
    end

    if stopping == 1
        break;
    end
    
    if D.verbose == 1
        fprintf(2, '\n');
    end
        
    fprintf(2, 'Epoch %d/%d - recon_error: %f\n', step, n_epochs, ...
        D.signals.recon_errors(end));
end

if use_gpu > 0
    % pull
    for l = 1:n_layers
        if l < n_layers
            D.W{l} = gather(D.W{l});
        end
        D.biases{l} = gather(D.biases{l});
        if D.centering.use
            D.centering.centers{l} = gather(D.centering.centers{l});
        end
    end
    D.sigmas = gather(D.sigmas);

    clear h1 logsigmas;
end

