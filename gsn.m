% gsn - training an GSN (stochastic backprop)
% Copyright (C) 2013 KyungHyun Cho
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
function [G] = gsn(G, patches, n_walkback, valid_patches, valid_portion);

if nargin < 3
    n_walkback = G.structure.n_layers * 2;
end

valid_err = 0;

if nargin < 4
    early_stop = 0;
    valid_patches = [];
    valid_portion = 0;
else
    early_stop = 1;
    valid_err = -Inf;
    valid_best_err = -Inf;
    valid_violate_cnt = 0;
end

actual_lrate = G.learning.lrate;

n_samples = size(patches, 1);

layers = G.structure.layers;
n_layers = length(layers);

if layers(1) ~= size(patches, 2)
    error('Data is not properly aligned');
end

minibatch_sz = G.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);

n_epochs = G.iteration.n_epochs;

momentum = G.learning.momentum;
weight_decay = G.learning.weight_decay;

biases_grad = cell(n_layers, 1);
W_grad = cell(n_layers, 1);
biases_grad_old = cell(n_layers, 1);
W_grad_old = cell(n_layers, 1);
for l = 1:n_layers
    biases_grad{l} = zeros(size(G.biases{l}))';
    if l < n_layers
        W_grad{l} = zeros(size(G.W{l}));
    end
    biases_grad_old{l} = zeros(size(G.biases{l}))';
    if l < n_layers
        W_grad_old{l} = zeros(size(G.W{l}));
    end
end

min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;

do_normalize = G.do_normalize;
do_normalize_std = G.do_normalize_std;

if G.data.binary == 0
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

rerr_ma = 0;

% try
%     use_gpu = gpuDeviceCount;
% catch errgpu
%     use_gpu = false;
%     disp(['Could not use CUDA. Error: ' errgpu.identifier])
% end
use_gpu = 0;

% figure;

for step=1:n_epochs
    if G.verbose
        fprintf(2, 'Epoch %d/%d: ', step, n_epochs)
    end
    if use_gpu
        % push
        G = push_to_gpu (G);
    end

    for mb=1:n_minibatches
        G.iteration.n_updates = G.iteration.n_updates + 1;

        mb_start = (mb - 1) * minibatch_sz + 1;
        mb_end = min(mb * minibatch_sz, n_samples);

        % p_0
        v0 = patches(mb_start:mb_end, :);
        mb_sz = size(v0,1);

        if use_gpu > 0
            v0 = gpuArray(single(v0));
        end

        % add error
        v0_clean = v0;

        % forward pass
        h0 = cell(n_layers, n_walkback * 2 + 1);
        v1clean = cell(1, n_walkback * 2 + 1);
        occupied = zeros(n_layers, n_walkback * 2 + 1);

        h0{1,1} = v0;
        v1clean{1,1} = v0;
        occupied(1,1) = 1;

        for wi = 1:(n_walkback * 2 + 1)
            for l = 1:min(wi,n_layers)
                if sum(occupied(1, 2:end)) == n_walkback
                    break;
                end
                
                if l == 2
                    if G.data.binary == 0 && G.noise.level > 0
                        if use_gpu
                            h0{l-1,wi-1} = h0{l-1,wi-1} + G.noise.level * parallel.gpu.GPUArray.randn(size(h0{l-1,wi-1}));
                        else
                            h0{l-1,wi-1} = h0{l-1,wi-1} + G.noise.level * randn(size(h0{l-1,wi-1}));
                        end
                    end
                    
                    if G.noise.drop > 0
                        mask = binornd(1, 1-G.noise.drop, size(h0{l-1,wi-1}));
                        if G.data.binary
                            sandp = binornd(1, 0.5, size(h0{l-1,wi-1}));
                        else
                            sandp = zeros(size(h0{l-1,wi-1}));
                        end
                        h0{l-1,wi-1} = h0{l-1,wi-1} .* mask + (1 - mask) .* sandp;
                        clear mask;
                    end
                end

                if l == 1 && wi == 1
                    continue;
                end

                prev_exist = 0;

                if l > 1 && occupied(l-1, wi-1) == 1
                    prev_exist = prev_exist + 1;
                end
                if l < n_layers && occupied(l+1, wi-1) == 1
                    prev_exist = prev_exist + 1;
                end

                if prev_exist == 0
                    continue;
                end

                occupied(l, wi) = 1;
                h0{l, wi} = zeros(mb_sz, G.structure.layers(l));
                if l > 1 && occupied(l-1, wi-1) == 1
                    h0{l, wi} = h0{l, wi} + h0{l-1, wi-1} * G.W{l-1};
                end
                if l < n_layers && occupied(l+1, wi-1) == 1
                    h0{l, wi} = h0{l, wi} + h0{l+1, wi-1} * G.W{l}';
                end
                h0{l, wi} = bsxfun(@plus, h0{l, wi}, G.biases{l}');

                % pre-sigmoid noise
                if l > 1
                    if G.hidden.add_noise(l)
                        if use_gpu
                            h0{l, wi} = h0{l, wi} + G.hidden.noise_level * parallel.gpu.GPUArray.randn(mb_sz, G.structure.layers(l));
                        else
                            h0{l, wi} = h0{l, wi} + G.hidden.noise_level * randn(mb_sz, G.structure.layers(l));
                        end
                    end
                end

                if l == 1 
                    if G.data.binary
                        h0{l, wi} = sigmoid(h0{l, wi});
                    end
                    v1clean{l, wi} = h0{l, wi};
                else
                    h0{l, wi} = sigmoid(h0{l, wi}, G.hidden.use_tanh);
                end
                
                % FIXME: unable to train a GSN with post-sigmoid noise.
%                % post-sigmoid noise
%                if l > 1
%                    if G.hidden.add_noise(l)
%                        if use_gpu
%                            h0{l, wi} = h0{l, wi} + G.hidden.noise_level * parallel.gpu.GPUArray.randn(mb_sz, G.structure.layers(l));
%                        else
%                            h0{l, wi} = h0{l, wi} + G.hidden.noise_level * randn(mb_sz, G.structure.layers(l));
%                        end
%                    end
%                end
            end

            if sum(occupied(1, 2:end)) == n_walkback
                break;
            end
        end
        
        % reset gradients
        for l = 1:n_layers
            biases_grad{l} = 0 * biases_grad{l};
            if l < n_layers
                W_grad{l} = 0 * W_grad{l};
            end
        end

        % error backprop
        delta = cell(n_layers, n_walkback * 2 + 1);

        rerr = 0;
        for wi = 2:(n_walkback * 2 + 1)
            if occupied(1, wi) 
                rerr = rerr + mean(sum((v1clean{1, wi} - v0_clean).^2,2));
            end
        end


        if use_gpu > 0
            rerr = gather(rerr);
        end

        rerr_ma = rerr * 0.1 + rerr_ma * 0.9;

        G.signals.recon_errors = [G.signals.recon_errors rerr];

        for wi = (n_walkback * 2 + 1):-1:1
            for l = 1:min(wi,n_layers)
                if occupied(l, wi) == 0
                    continue;
                end

                delta{l, wi} = zeros(mb_sz, G.structure.layers(l));

                if wi < (n_walkback * 2 + 1)
                    if l > 1 && occupied(l-1,wi+1) == 1
                        delta{l, wi} = delta{l, wi} + delta{l-1,wi+1} * G.W{l-1};
                    end
                    if l < n_layers && occupied(l+1,wi+1) == 1
                        delta{l, wi} = delta{l, wi} + delta{l+1,wi+1} * G.W{l}';
                    end
                end
                
                if l == 1 
                    if G.data.binary && wi < (n_walkback * 2 + 1)
                        delta{l, wi} = delta{l, wi} .* dsigmoid(h0{l, wi});
                    end
                    
                    delta{l, wi} = delta{l, wi} + (v1clean{l, wi} - v0_clean);
                    %delta{l, wi} = delta{l, wi} + (h0{l, wi} - v0_clean);
                else
                    delta{l, wi} = delta{l, wi} .* dsigmoid(h0{l, wi}, G.hidden.use_tanh);
                end

                if wi > 1
                    biases_grad{l} = biases_grad{l} + mean(delta{l, wi}, 1);
                    if l > 1 && occupied(l-1,wi-1) == 1
                        W_grad{l-1} = W_grad{l-1} + (h0{l-1, wi-1}' * delta{l, wi})/size(v0,1);
                    end
                    if l < n_layers && occupied(l+1,wi-1) == 1
                        W_grad{l} = W_grad{l} + (delta{l, wi}' * h0{l+1, wi-1})/size(v0,1);
                    end
                end
            end
        end

        % learning rate
        if G.adadelta.use
            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                end
            end

            if G.iteration.n_updates == 1
                adamom = 0;
            else
                adamom = G.adadelta.momentum;
            end

            for l = 1:n_layers
                if l < n_layers
                    G.adadelta.gW{l} = adamom * G.adadelta.gW{l} + (1 - adamom) * W_grad_old{l}.^2;
                end

                G.adadelta.gbiases{l} = adamom * G.adadelta.gbiases{l} + (1 - adamom) * biases_grad_old{l}.^2';
            end

            for l = 1:n_layers
                dbias = -(biases_grad_old{l}' + ...
                    weight_decay * G.biases{l}) .* (sqrt(G.adadelta.biases{l} + G.adadelta.epsilon) ./ ...
                    sqrt(G.adadelta.gbiases{l} + G.adadelta.epsilon));
                G.biases{l} = G.biases{l} + dbias;

                G.adadelta.biases{l} = adamom * G.adadelta.biases{l} + (1 - adamom) * dbias.^2;
                clear dbias;

                if l < n_layers
                    dW = -(W_grad_old{l} + ...
                        weight_decay * G.W{l}) .* (sqrt(G.adadelta.W{l} + G.adadelta.epsilon) ./ ...
                        sqrt(G.adadelta.gW{l} + G.adadelta.epsilon));
                    G.W{l} = G.W{l} + dW;

                    G.adadelta.W{l} = adamom * G.adadelta.W{l} + (1 - adamom) * dW.^2;

                    clear dW;
                end

            end
        else
            if G.learning.lrate_anneal > 0 && (step >= G.learning.lrate_anneal * n_epochs)
                anneal_counter = anneal_counter + 1;
                actual_lrate = actual_lrate0 / anneal_counter;
            else
                if G.learning.lrate0 > 0
                    actual_lrate = G.learning.lrate / (1 + G.iteration.n_updates / G.learning.lrate0);
                else
                    actual_lrate = G.learning.lrate;
                end
                actual_lrate0 = actual_lrate;
            end

            G.signals.lrates = [G.signals.lrates actual_lrate];

            % update
            for l = 1:n_layers
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                if l < n_layers
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                end
            end

            for l = 1:n_layers
                G.biases{l} = G.biases{l} - actual_lrate * (biases_grad_old{l}' + weight_decay * G.biases{l});
                if l < n_layers
                    G.W{l} = G.W{l} - actual_lrate * (W_grad_old{l} + weight_decay * G.W{l});
                end
            end
        end

        if G.verbose == 1
            fprintf(2, '.%f.', rerr);
        end

        if use_gpu > 0
            clear v0 h0 h0d h0e v0_clean vr hr deltae deltad v1clean
        end

        if early_stop
            n_valid = size(valid_patches, 1);
            rndidx = randperm(n_valid);
            v0valid = gpuArray(single(valid_patches(rndidx(1:round(n_valid * valid_portion)),:)));
            valid_sz = size(v0valid, 1);

            valid0 = cell(n_layers, 1);
            valid0{1} = v0valid;
            for l = 2:n_layers
                valid0{l} = zeros(valid_sz, G.structure.layers(l));
            end

            valid0 = gsn_sample(G, valid0, 1);

            rerr = mean(sum((v0valid - valid0{1}).^2, 1));

            if use_gpu > 0
                rerr = gather(rerr);
            end

            G.signals.valid_errors = [G.signals.valid_errors rerr];

            if valid_err == -Inf
                valid_err = rerr;
                valid_best_err = rerr;
                valid_violate_cnt = 0;

                M_best = G;
                M_best = pull_from_gpu (M_best);
            else
                prev_err = valid_err;
                valid_err = 0.99 * valid_err + 0.01 * rerr;


                if step > G.valid_min_epochs
                    if (1.1 * valid_best_err) < valid_err 
                        fprintf(2, 'Early-stop! %f, %f\n', valid_err, valid_best_err);
                        stopping = 1;
                        break;
                    end

                    if valid_best_err <= valid_err
                        valid_violate_cnt = valid_violate_cnt + 1;
                        if valid_violate_cnt > (n_minibatches * G.valid_min_epochs)
                            fprintf(2, 'Unable to improve! %f, %f\n', valid_err, valid_best_err);
                            stopping = 1;
                            break;
                        end
                    else
                        valid_violate_cnt = 0;
                    end

                end

                if valid_err < valid_best_err
                    valid_best_err = valid_err;

                    G_best = G;
                    G_best = pull_from_gpu (G_best);
                end
            end
        else
            if G.stop.criterion > 0
                if G.stop.criterion == 1
                    if min_recon_error > G.signals.recon_errors(end)
                        min_recon_error = G.signals.recon_errors(end);
                        min_recon_error_update_idx = G.iteration.n_updates;
                    else
                        if G.iteration.n_updates > min_recon_error_update_idx + G.stop.recon_error.tolerate_count 
                            fprintf(2, '\nStopping criterion reached (recon error) %f > %f\n', ...
                                G.signals.recon_errors(end), min_recon_error);
                            stopping = 1;
                            break;
                        end
                    end
                else
                    error ('Unknown stopping criterion %d', G.stop.criterion);
                end
            end
        end

        if length(G.hook.per_update) > 1
            err = G.hook.per_update{1}(G, G.hook.per_update{2});

            if err == -1
                stopping = 1;
                break;
            end
        end
        
    end

    if use_gpu > 0
        % pull
        G = pull_from_gpu (G);
    end

    if length(G.hook.per_epoch) > 1
        err = G.hook.per_epoch{1}(G, G.hook.per_epoch{2});

        if err == -1
            stopping = 1;
        end
    end

    if stopping == 1
        break;
    end
    
    if G.verbose == 1
        fprintf(2, '\n');
    end
        
    fprintf(2, 'Epoch %d/%d - recon_error: %f(%f) valid_error: %f\n', step, n_epochs, rerr_ma, rerr, valid_err);
end

if use_gpu > 0
    % pull
    G = pull_from_gpu (G);
end

if early_stop
    G = G_best;
end

end

function [G] = push_to_gpu (G)
    n_layers = length(G.structure.layers);

    % push
    for l = 1:n_layers
        if l < n_layers 
            G.W{l} = gpuArray(single(G.W{l}));
        end
        G.biases{l} = gpuArray(single(G.biases{l}));
    end

    if G.adadelta.use
        for l = 1:n_layers
            if l < n_layers 
                G.adadelta.gW{l} = gpuArray(single(G.adadelta.gW{l}));
                G.adadelta.W{l} = gpuArray(single(G.adadelta.W{l}));
            end
            G.adadelta.gbiases{l} = gpuArray(single(G.adadelta.gbiases{l}));
            G.adadelta.biases{l} = gpuArray(single(G.adadelta.biases{l}));
        end
    end
end

function [G] = pull_from_gpu (G)
    n_layers = length(G.structure.layers);


    for l = 1:n_layers
        if l < n_layers
            G.W{l} = gather(G.W{l});
        end
        G.biases{l} = gather(G.biases{l});
    end

    if G.adadelta.use
        for l = 1:n_layers
            if l < n_layers
                G.adadelta.W{l} = gather(G.adadelta.W{l});
                G.adadelta.gW{l} = gather(G.adadelta.gW{l});
            end
            G.adadelta.biases{l} = gather(G.adadelta.biases{l});
            G.adadelta.gbiases{l} = gather(G.adadelta.gbiases{l});
        end
    end

end


