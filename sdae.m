% sdae - training a stacked DAE (finetuning)
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
function [S] = sdae(S, patches, valid_patches, valid_portion);

if nargin < 3
    early_stop = 0;
    valid_patches = [];
    valid_portion = 0;
else
    early_stop = 1;
    valid_err = -Inf;
    valid_best_err = -Inf;
end

actual_lrate = S.learning.lrate;

n_samples = size(patches, 1);

layers = S.structure.layers;
n_layers = length(layers);

if layers(1) ~= size(patches, 2)
    error('Data is not properly aligned');
end

minibatch_sz = S.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);

n_epochs = S.iteration.n_epochs;

momentum = S.learning.momentum;
weight_decay = S.learning.weight_decay;

biases_grad = cell(n_layers, 1);
W_grad = cell(n_layers, 1);
biases_grad_old = cell(n_layers, 1);
W_grad_old = cell(n_layers, 1);
for l = 1:n_layers
    biases_grad{l} = zeros(size(S.biases{l}))';
    if l < n_layers
        W_grad{l} = zeros(size(S.W{l}));
    end
    biases_grad_old{l} = zeros(size(S.biases{l}))';
    if l < n_layers
        W_grad_old{l} = zeros(size(S.W{l}));
    end
end

min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;

do_normalize = S.do_normalize;
do_normalize_std = S.do_normalize_std;

if S.data.binary == 0
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

anneal_counter = 0;
actual_lrate0 = actual_lrate;

if S.debug.do_display == 1
    figure(S.debug.display_fid);
end

try
    use_gpu = gpuDeviceCount;
catch errgpu
    use_gpu = false;
    disp(['Could not use CUDA. Error: ' errgpu.identifier])
end

for step=1:n_epochs
    if S.verbose
        fprintf(2, 'Epoch %d/%d: ', step, n_epochs)
    end
    if use_gpu
        % push
        for l = 1:n_layers
            if l < n_layers 
                S.W{l} = gpuArray(single(S.W{l}));
            end
            S.biases{l} = gpuArray(single(S.biases{l}));
        end

        if S.adagrad.use 
            for l = 1:n_layers
                if l < n_layers 
                    S.adagrad.W{l} = gpuArray(single(S.adagrad.W{l}));
                end
                S.adagrad.biases{l} = gpuArray(single(S.adagrad.biases{l}));
            end
        elseif S.adadelta.use
            for l = 1:n_layers
                if l < n_layers 
                    S.adadelta.gW{l} = gpuArray(single(S.adadelta.gW{l}));
                    S.adadelta.W{l} = gpuArray(single(S.adadelta.W{l}));
                end
                S.adadelta.gbiases{l} = gpuArray(single(S.adadelta.gbiases{l}));
                S.adadelta.biases{l} = gpuArray(single(S.adadelta.biases{l}));
            end
        end
    end

    for mb=1:n_minibatches
        S.iteration.n_updates = S.iteration.n_updates + 1;

        % p_0
        v0 = patches((mb-1) * minibatch_sz + 1:min(mb * minibatch_sz, n_samples), :);
        mb_sz = size(v0,1);

        if use_gpu > 0
            v0 = gpuArray(single(v0));
        end

        % add error
        v0_clean = v0;

        if S.data.binary == 0 && S.noise.level > 0
            v0 = v0 + S.noise.level * gpuArray(randn(size(v0)));
        end

        if S.noise.drop > 0
            mask = binornd(1, 1 - S.noise.drop, size(v0));
            v0 = v0 .* mask;
            clear mask;
        end

        h0e = cell(n_layers, 1);
        h0e{1} = v0;

        for l = 2:n_layers
            h0e{l} = bsxfun(@plus, h0e{l-1} * S.W{l-1}, S.biases{l}');

            if l < n_layers || S.bottleneck.binary
                h0e{l} = sigmoid(h0e{l}, S.hidden.use_tanh);
            end
        end

        h0d = cell(n_layers, 1);
        h0d{end} = h0e{end};

        for l = n_layers-1:-1:1
            h0d{l} = bsxfun(@plus, h0d{l+1} * S.W{l}', S.biases{l}');
            if l == 1 && S.data.binary
                h0d{l} = sigmoid(h0d{l});
            end
            if l > 1
                h0d{l} = sigmoid(h0d{l}, S.hidden.use_tanh);
            end
        end

        % compute reconstruction error
        hr = sdae_get_hidden(v0_clean, S);
        vr = sdae_get_visible(hr, S);

        if S.data.binary
            rerr = -mean(sum(v0_clean .* log(max(vr, 1e-16)) + (1 - v0_clean) .* log(max(1 - vr, 1e-16)), 2));
        else
            rerr = mean(sum((v0_clean - vr).^2,2));
        end
        if use_gpu > 0
            rerr = gather(rerr);
        end
        S.signals.recon_errors = [S.signals.recon_errors rerr];

        % reset gradients
        for l = 1:n_layers
            biases_grad{l} = 0 * biases_grad{l};
            if l < n_layers
                W_grad{l} = 0 * W_grad{l};
            end
        end

        % backprop
        deltad = cell(n_layers, 1);
        deltad{1} = h0d{1} - v0_clean;
        biases_grad{1} = mean(deltad{1}, 1);

        for l = 2:n_layers
            deltad{l} = deltad{l-1} * S.W{l-1};
            if l < n_layers || S.bottleneck.binary
                deltad{l} = deltad{l} .* dsigmoid(h0d{l}, S.hidden.use_tanh);
            end
            biases_grad{l} = mean(deltad{l}, 1);
            W_grad{l-1} = (deltad{l-1}' * h0d{l}) / (size(v0, 1));
        end

        deltae = cell(n_layers, 1);
        deltae{end} = deltad{end};

        for l = n_layers-1:-1:1
            deltae{l} = deltae{l+1} * S.W{l}';
            if l == 1 && S.data.binary
                deltae{l} = deltae{l} .* dsigmoid(h0e{l});
            end
            if l > 1
                deltae{l} = deltae{l} .* dsigmoid(h0e{l}, S.hidden.use_tanh);
                biases_grad{l} = biases_grad{l} + mean(deltae{l}, 1);
            end
            W_grad{l} = W_grad{l} + (h0e{l}' * deltae{l+1}) / (size(v0, 1));
        end

        % learning rate
        if S.adagrad.use
            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                end
            end

            for l = 1:n_layers
                if l < n_layers
                    S.adagrad.W{l} = S.adagrad.W{l} + W_grad_old{l}.^2;
                end

                S.adagrad.biases{l} = S.adagrad.biases{l} + biases_grad_old{l}.^2';
            end

            for l = 1:n_layers
                S.biases{l} = S.biases{l} - S.learning.lrate * (biases_grad_old{l}' + ...
                    weight_decay * S.biases{l}) ./ sqrt(S.adagrad.biases{l} + S.adagrad.epsilon);
                if l < n_layers
                    S.W{l} = S.W{l} - S.learning.lrate * (W_grad_old{l} + ...
                        weight_decay * S.W{l}) ./ sqrt(S.adagrad.W{l} + S.adagrad.epsilon);
                end
            end

        elseif S.adadelta.use
            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                end
            end

            if S.iteration.n_updates == 1
                adamom = 0;
            else
                adamom = S.adadelta.momentum;
            end

            for l = 1:n_layers
                if l < n_layers
                    S.adadelta.gW{l} = adamom * S.adadelta.gW{l} + (1 - adamom) * W_grad_old{l}.^2;
                end

                S.adadelta.gbiases{l} = adamom * S.adadelta.gbiases{l} + (1 - adamom) * biases_grad_old{l}.^2';
            end

            for l = 1:n_layers
                dbias = -(biases_grad_old{l}' + ...
                    weight_decay * S.biases{l}) .* (sqrt(S.adadelta.biases{l} + S.adadelta.epsilon) ./ ...
                    sqrt(S.adadelta.gbiases{l} + S.adadelta.epsilon));
                S.biases{l} = S.biases{l} + dbias;

                S.adadelta.biases{l} = adamom * S.adadelta.biases{l} + (1 - adamom) * dbias.^2;
                clear dbias;

                if l < n_layers
                    dW = -(W_grad_old{l} + ...
                        weight_decay * S.W{l}) .* (sqrt(S.adadelta.W{l} + S.adadelta.epsilon) ./ ...
                        sqrt(S.adadelta.gW{l} + S.adadelta.epsilon));
                    S.W{l} = S.W{l} + dW;

                    S.adadelta.W{l} = adamom * S.adadelta.W{l} + (1 - adamom) * dW.^2;

                    clear dW;
                end

            end
        else
            if S.learning.lrate_anneal > 0 && (step >= S.learning.lrate_anneal * n_epochs)
                anneal_counter = anneal_counter + 1;
                actual_lrate = actual_lrate0 / anneal_counter;
            else
                if S.learning.lrate0 > 0
                    actual_lrate = S.learning.lrate / (1 + S.iteration.n_updates / S.learning.lrate0);
                else
                    actual_lrate = S.learning.lrate;
                end
                actual_lrate0 = actual_lrate;
            end

            S.signals.lrates = [S.signals.lrates actual_lrate];

            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                end
            end

            for l = 1:n_layers
                S.biases{l} = S.biases{l} - actual_lrate * (biases_grad_old{l}' + weight_decay * S.biases{l});
                if l < n_layers
                    S.W{l} = S.W{l} - actual_lrate * (W_grad_old{l} + weight_decay * S.W{l});
                end
            end
        end

        if S.verbose == 1
            fprintf(2, '.');
        end

        if use_gpu > 0
            clear v0 h0d h0e v0_clean vr hr deltae deltad 
        end

        if early_stop
            n_valid = size(valid_patches, 1);
            rndidx = randperm(n_valid);
            v0valid = gpuArray(single(valid_patches(rndidx(1:round(n_valid * valid_portion)),:)));

            hr = sdae_get_hidden(v0valid, S);
            vr = sdae_get_visible(hr, S);

            if S.data.binary
                rerr = -mean(sum(v0valid .* log(max(vr, 1e-16)) + (1 - v0valid) .* log(max(1 - vr, 1e-16)), 2));
            else
                rerr = mean(sum((v0valid - vr).^2,2));
            end
            if use_gpu > 0
                rerr = gather(rerr);
            end

            S.signals.valid_errors = [S.signals.valid_errors rerr];

            if valid_err == -Inf
                valid_err = rerr;
                valid_best_err = rerr;
            else
                prev_err = valid_err;
                valid_err = 0.99 * valid_err + 0.01 * rerr;

                if step > S.valid_min_epochs && (1.1 * valid_best_err) < valid_err
                    fprintf(2, 'Early-stop! %f, %f\n', valid_err, prev_err);
                    stopping = 1;
                    break;
                end

                if valid_err < valid_best_err
                    valid_best_err = valid_err;
                end
            end
        else
            if S.stop.criterion > 0
                if S.stop.criterion == 1
                    if min_recon_error > S.signals.recon_errors(end)
                        min_recon_error = S.signals.recon_errors(end);
                        min_recon_error_update_idx = S.iteration.n_updates;
                    else
                        if S.iteration.n_updates > min_recon_error_update_idx + S.stop.recon_error.tolerate_count 
                            fprintf(2, '\nStopping criterion reached (recon error) %f > %f\n', ...
                                S.signals.recon_errors(end), min_recon_error);
                            stopping = 1;
                            break;
                        end
                    end
                else
                    error ('Unknown stopping criterion %d', S.stop.criterion);
                end
            end
        end

        if length(S.hook.per_update) > 1
            err = S.hook.per_update{1}(S, S.hook.per_update{2});

            if err == -1
                stopping = 1;
                break;
            end
        end
        
        if S.debug.do_display == 1 && mod(S.iteration.n_updates, S.debug.display_interval) == 0
            S.debug.display_function (S.debug.display_fid, S, v0, v1, h0, h1, W_grad, vbias_grad, hbias_grad);
            drawnow;
        end
    end

    if use_gpu > 0
        % pull
        for l = 1:n_layers
            if l < n_layers
                S.W{l} = gather(S.W{l});
            end
            S.biases{l} = gather(S.biases{l});
        end

        if S.adagrad.use
            for l = 1:n_layers
                if l < n_layers
                    S.adagrad.W{l} = gather(S.adagrad.W{l});
                end
                S.adagrad.biases{l} = gather(S.adagrad.biases{l});
            end
        elseif S.adadelta.use
            for l = 1:n_layers
                if l < n_layers
                    S.adadelta.W{l} = gather(S.adadelta.W{l});
                    S.adadelta.gW{l} = gather(S.adadelta.gW{l});
                end
                S.adadelta.biases{l} = gather(S.adadelta.biases{l});
                S.adadelta.gbiases{l} = gather(S.adadelta.gbiases{l});
            end
        end
    end

    if length(S.hook.per_epoch) > 1
        err = S.hook.per_epoch{1}(S, S.hook.per_epoch{2});

        if err == -1
            stopping = 1;
        end
    end

    if stopping == 1
        break;
    end
    
    if S.verbose == 1
        fprintf(2, '\n');
    end
        
    fprintf(2, 'Epoch %d/%d - recon_error: %f\n', step, n_epochs, ...
        S.signals.recon_errors(end));
end

if use_gpu > 0
    % pull
    for l = 1:n_layers
        if l < n_layers
            S.W{l} = gather(S.W{l});
        end
        S.biases{l} = gather(S.biases{l});
    end

    if S.adagrad.use
        for l = 1:n_layers
            if l < n_layers
                S.adagrad.W{l} = gather(S.adagrad.W{l});
            end
            S.adagrad.biases{l} = gather(S.adagrad.biases{l});
        end
    elseif S.adadelta.use
        for l = 1:n_layers
            if l < n_layers
                S.adadelta.W{l} = gather(S.adadelta.W{l});
                S.adadelta.gW{l} = gather(S.adadelta.gW{l});
            end
            S.adadelta.biases{l} = gather(S.adadelta.biases{l});
            S.adadelta.gbiases{l} = gather(S.adadelta.gbiases{l});
        end
    end
end


