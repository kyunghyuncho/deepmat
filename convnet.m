% convent - Traing a convolution neural network for images
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
function [C] = convnet(C, patches, targets, ...
    valid_patches, valid_targets, ...
    valid_portion, valid_interval, use_cvp);

if nargin < 4
    early_stop = 0;
    valid_patches = [];
    valid_targets = [];
    valid_portion = 0;
    valid_err = 0;
    valid_interval = 100;
else
    early_stop = 1;
    valid_err = -Inf;
    valid_best_err = -Inf;
    valid_violate_cnt = 0;
    valid_interval = 100;
end

if nargin < 7
    use_cvp = 1;
end

matrel = version('-release');
if str2num(matrel(1:4)) >= 2013
    % gpu support for convn() was added in 2013a
    try
        use_gpu = gpuDeviceCount;
    catch errgpu
        use_gpu = false;
        disp(['Could not use CUDA. Error: ' errgpu.identifier])
    end
else
    use_gpu = 0;
end

actual_lrate = C.learning.lrate;

n_samples = size(patches, 1);

layers = C.structure.layers;
n_layers = length(layers);

conv_layers = C.structure.conv_layers;
n_conv = size(conv_layers, 1);
full_layers = C.structure.full_layers;
n_full = length(C.structure.full_layers);
strides = C.structure.strides;
poolratios = C.structure.poolratios;

if layers(1) ~= size(patches, 2)
    error('Data is not properly aligned');
end

minibatch_sz = C.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);

if use_cvp
    cvp = crossvalind('Kfold', targets, n_minibatches);
end

if size(targets, 2) == 1 && C.output.binary
    % populate the target labels
    n_classes = max(targets(:));
    new_targets = zeros(n_samples, n_classes);

    for ti = 1:n_samples
        new_targets(ti, targets(ti)) = 1; % well?
    end
    
    targets = new_targets;
end

if size(valid_targets, 2) == 1 && C.output.binary
    % populate the target labels
    n_classes = max(valid_targets(:));
    n_valid_samples = size(valid_targets, 1);
    new_targets = zeros(n_valid_samples, n_classes);

    for ti = 1:n_valid_samples
        new_targets(ti, valid_targets(ti)) = 1; % well?
    end
    
    valid_targets = new_targets;
end

n_epochs = C.iteration.n_epochs;

momentum = C.learning.momentum;
weight_decay = C.learning.weight_decay;

% convolutional layer
cW_grad_old = cell(n_conv, 1);
cbiases_grad_old = cell(n_conv, 1);
cW_grad = cell(n_conv, 1);
cbiases_grad = cell(n_conv, 1);
for l = 1:n_conv
    if l == 1
        cin = C.structure.channel_in;
    else
        cin = conv_layers(l-1,2);
    end

    if use_gpu
        cbiases_grad{l} = gpuArray.zeros(conv_layers(l,2), 1);
        cbiases_grad_old{l} = gpuArray.zeros(conv_layers(l,2), 1);
        cW_grad{l} = gpuArray.zeros(conv_layers(l,1)*cin, conv_layers(l,2));
        cW_grad_old{l} = gpuArray.zeros(conv_layers(l,1)*cin, conv_layers(l,2));
    else
        cbiases_grad{l} = zeros(conv_layers(l,2), 1);
        cbiases_grad_old{l} = zeros(conv_layers(l,2), 1);
        cW_grad{l} = zeros(conv_layers(l,1)*cin, conv_layers(l,2));
        cW_grad_old{l} = zeros(conv_layers(l,1)*cin, conv_layers(l,2));
    end
end
% full layer
W_grad_old = cell(n_full, 1);
biases_grad_old = cell(n_full+1, 1);
W_grad = cell(n_full, 1);
biases_grad = cell(n_full+1, 1);
for l = 1:(n_full+1)
    if use_gpu
        biases_grad{l} = gpuArray.zeros(layers(n_conv+l), 1);
        biases_grad_old{l} = gpuArray.zeros(layers(n_conv+l), 1);
        if l < n_full + 1
            W_grad{l} = gpuArray.zeros(layers(n_conv+l),layers(n_conv+l+1));
            W_grad_old{l} = gpuArray.zeros(layers(n_conv+l),layers(n_conv+l+1));
        end
    else
        biases_grad{l} = zeros(layers(n_conv+l), 1);
        biases_grad_old{l} = zeros(layers(n_conv+l), 1);
        if l < n_full + 1
            W_grad{l} = zeros(layers(n_conv+l),layers(n_conv+l+1));
            W_grad_old{l} = zeros(layers(n_conv+l),layers(n_conv+l+1));
        end
    end
end

min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;

do_normalize = C.do_normalize;
do_normalize_std = C.do_normalize_std;

if do_normalize == 1
    % make it zero-mean
    patches_mean = mean(patches, 1);
    patches = bsxfun(@minus, patches, patches_mean);
    if early_stop
        valid_patches = bsxfun(@minus, valid_patches, patches_mean);
    end
end

if do_normalize_std ==1
    % make it unit-variance
    patches_std = std(patches, [], 1);
    patches = bsxfun(@rdivide, patches, patches_std);
    if early_stop
        valid_patches = bsxfun(@rdivide, valid_patches, patches_std);
    end
end

anneal_counter = 0;
actual_lrate0 = actual_lrate;

if C.debug.do_display == 1
    figure(C.debug.display_fid);
end

rerr_ma = 0;

for step=1:n_epochs
    if C.verbose
        fprintf(2, 'Epoch %d/%d: ', step, n_epochs)
    end
    if use_gpu
        % push
        C = push_to_gpu (C);
    end

    for mb=1:n_minibatches
        %tic;
        C.iteration.n_updates = C.iteration.n_updates + 1;

        if use_cvp
            v0 = patches(cvp == mb, :);
        else
            mb_start = (mb - 1) * minibatch_sz + 1;
            mb_end = min(mb * minibatch_sz, n_samples);

            % p_0
            v0 = patches(mb_start:mb_end, :);
        end
        mb_sz = size(v0,1);

        if use_gpu > 0
            v0 = gpuArray(single(v0));
        end

        % add error
        v0_clean = v0;

        if C.noise.level > 0
            v0 = v0 + C.noise.level * randn(size(v0));
        end

        if C.noise.drop > 0
            mask = binornd(1, 1-C.noise.drop, size(v0));
            v0 = v0 .* mask;
            clear mask;
        end

        % forward pass
        h0_conv = cell(n_conv, 3);
        h0_full = cell(n_full+1, 1);

        cinsz = zeros(n_conv, 1);
        szinsz = zeros(n_conv, 1);
        cin = C.structure.channel_in;
        szin = C.structure.size_in;
        repost = reshape(v0, [mb_sz, szin, szin, cin]);

        for l = 1:n_conv
            cinsz(l) = cin;
            szinsz(l) = szin;

            fsz = size(C.cW{l},1) / cin;
            n_filters = size(C.cW{l}, 2);
            szout_prepool = floor((szin - sqrt(fsz)) / strides(l) + 1);
            szout = ceil(szout_prepool / poolratios(l));
            if use_gpu
                respfull = gpuArray.zeros(mb_sz, szout, szout, n_filters);
                idxfull = gpuArray.zeros(mb_sz, szout_prepool, szout_prepool, n_filters);
                h0_conv{l, 1} = gpuArray.zeros(mb_sz, szout_prepool, szout_prepool, n_filters);
                h0_conv{l, 2} = respfull;
                h0_conv{l, 3} = idxfull;
            else
                respfull = zeros(mb_sz, szout, szout, n_filters);
                idxfull = zeros(mb_sz, szout_prepool, szout_prepool, n_filters);
                h0_conv{l, 1} = zeros(mb_sz, szout_prepool, szout_prepool, n_filters);
                h0_conv{l, 2} = respfull;
                h0_conv{l, 3} = idxfull;
            end

            % for each filter map
            for fidx = 1:n_filters
                fil = reshape(C.cW{l}(:, fidx), [1, sqrt(fsz), sqrt(fsz), cin]);
                resp = convn(repost, fil, 'valid');

                % stride
                resp = resp(:, 1:strides(l):end, :);
                resp = resp(:, :, 1:strides(l):end);

                resp = resp + C.cbiases{l}(fidx);
                
                % nonlinearity
                resp = sigmoid(resp, C.hidden.use_tanh);

                %% save before max
                h0_conv{l, 1}(:,:,:,fidx) = resp;

                % pooling
                if C.structure.poolratios(l) > 1
                    switch C.pooling
                    case 0
                        [resp, respidx] = convnet_maxpool (resp, C.structure.poolratios(l));
                    case 1
                        error('NOT SUPPORTED');
                    case 2
                        error('NOT SUPPORTED');
                    end
                else
                    if use_gpu
                        respidx = parallel.gpu.GPUArray.ones(size(resp));
                    else
                        respidx = ones(size(resp));
                    end
                end

                respfull(:,:,:,fidx) = resp;
                idxfull(:,:,:,fidx) = respidx;

                % save after max
                h0_conv{l, 2}(:,:,:,fidx) = resp;
                h0_conv{l, 3}(:,:,:,fidx) = respidx;
            end

            szin = szout;
            cin = n_filters;
            repost = respfull;
        end

        % stretch
        h0_full{1} = reshape(repost, [mb_sz C.structure.layers(n_conv+1)]);
        h0mask = cell(n_full+1, 1);

        for l = 2:n_full+1
            if C.dropout.use && l < n_full + 1
                h0mask{l} = single(rand(size(h0_full{l-1})) - C.dropout.prob < 0);
                h0_full{l-1} = h0mask{l} .* h0_full{l-1};
            end

            h0_full{l} = bsxfun(@plus, h0_full{l-1} * C.W{l-1}, C.biases{l}');

            if l < n_full + 1
                h0_full{l} = sigmoid(h0_full{l}, C.hidden.use_tanh);
            end
            if l == n_full + 1 && C.output.binary
                h0_full{l} = sigmoid(h0_full{l});
            end
        end

        % reset gradients
        for l = 1:n_conv
            cbiases_grad{l} = 0 * cbiases_grad{l};
            cW_grad{l} = 0 * cW_grad{l};
        end
        for l = 1:n_full+1
            biases_grad{l} = 0 * biases_grad{l};
            if l < n_full + 1
                W_grad{l} = 0 * W_grad{l};
            end
        end

        if C.output.binary
            vr = h0_full{end};
            if use_cvp
                xt = targets(cvp == mb, :);
            else
                xt = targets(mb_start:mb_end, :);
            end
            rerr = -mean(sum(xt .* log(max(vr, 1e-16)) + ...
                (1 - xt) .* log(max(1 - vr, 1e-16)), 2));
        else
            rerr = mean(sum(delta{end}.^2,2));
        end

        if use_gpu > 0
            rerr = gather(rerr);
        end

        rerr_ma = rerr * 0.1 + rerr_ma * 0.9;

        C.signals.recon_errors = [C.signals.recon_errors rerr];

        if C.verbose == 1
            fprintf(2, 'rerr = %f\n', C.signals.recon_errors(end));
        end

        % backprop
        % fully connected layers, first
        dfull = h0_full{end} - targets(cvp == mb, :);% - h0_full{end};
        for l = n_full+1:-1:1
            biases_grad{l} = biases_grad{l} + mean(dfull, 1)';
            if l > 1
                W_grad{l-1} = W_grad{l-1} + (h0_full{l-1}' * dfull) / mb_sz;

                dfull = dfull * C.W{l-1}';
                dfull = dfull .* dsigmoid(h0_full{l-1}, C.hidden.use_tanh);
                if C.dropout.use && l > 2
                    dfull = dfull .* h0mask{l-1};
                end
            end
        end

        clear h0mask;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % convolutional layers, next
        l = n_conv;

        n_filters = size(C.cW{l}, 2);
        %szout_prepool = (floor((szinsz(end) - 2 * strides(l)) / strides(l)) - 1);
        szout_prepool = (floor((szinsz(end) - sqrt(fsz)) / strides(l)) + 1);
        szout = ceil(szout_prepool / poolratios(l));

        dconv = reshape(dfull, [mb_sz, szout, szout, n_filters]); % we start from here

        for l = n_conv:-1:1
            if l > 1
                lower = h0_conv{l-1, 2};
                cin = size(C.cW{l-1}, 2);
            else
                cin = C.structure.channel_in;
                lower = reshape(v0, [mb_sz, C.structure.size_in, C.structure.size_in, cin]);
            end
            n_filters = size(C.cW{l}, 2);
            szout_prepool = (floor((szinsz(l) - sqrt(fsz)) / strides(l)) + 1);
            szout = ceil(szout_prepool / poolratios(l));

            if l > 1
                if use_gpu
                    dconv_next = gpuArray.zeros(size(lower));
                else
                    dconv_next = zeros(size(lower));
                end
            end

            %fprintf(2, 'conv layer %d\n', l);
            fsz = length(C.cW{l}(:,1)) / cin;
            rcW = reshape(C.cW{l}, [size(C.cW{l},1), 1, 1, 1, size(C.cW{l},2)]);
            rfilter = repmat(reshape(rcW, [1, sqrt(fsz), sqrt(fsz), cin, size(rcW, 5)]), [mb_sz, 1,1,1, 1]);

            postpool = dconv;
            postpool = postpool(:,repmat(1:size(postpool,2),poolratios(l),1), ...
                repmat(1:size(postpool,3),poolratios(l),1), :);
            postpool = postpool(:,1:size(h0_conv{l,3},2),1:size(h0_conv{l,3},3), :);
            prepool = h0_conv{l, 3} .* postpool;

            cbiases_grad{l} = mean(reshape(postpool, [prod(size(postpool))/n_filters, n_filters]), 1)';

            lowerl = repmat(lower, [1, 1, 1, 1, n_filters]);

            if use_gpu
                wgs = gpuArray.zeros(1, size(cW_grad{l},1), size(cW_grad{l},2), size(prepool,2) * size(prepool,3));
            else
                wgs = zeros(1, size(cW_grad{l},1), size(cW_grad{l},2), size(prepool,2) * size(prepool,3));
            end

            for ai = 1:size(prepool, 2)
                for aj = 1:size(prepool, 3)
                    % for each activation
                    acts = reshape(prepool(:, ai, aj, :), [size(prepool,1), 1, size(prepool, 4)]);
                    inblock = lowerl(:, ...
                        ((ai-1)*strides(l)+1):((ai-1)*strides(l)+sqrt(fsz)), ...
                        ((aj-1)*strides(l)+1):((aj-1)*strides(l)+sqrt(fsz)), ...
                        :, :);
                    inblock = bsxfun(@times, reshape(inblock, [mb_sz, fsz * cin, n_filters]), acts);
                    wgs(:, :, :, (ai-1)*size(prepool,3)+aj) = mean(inblock, 1);

                    if l > 1
                        % backprop
                        res = ones(1,length(size(rfilter)));
                        res(1) = size(acts, 1);
                        res(end) = size(acts, 3);
                        dconv_next(:,((ai-1)*strides(l)+1):((ai-1)*strides(l)+sqrt(fsz)), ...
                            ((aj-1)*strides(l)+1):((aj-1)*strides(l)+sqrt(fsz)), :) = ...
                        dconv_next(:,((ai-1)*strides(l)+1):((ai-1)*strides(l)+sqrt(fsz)), ...
                            ((aj-1)*strides(l)+1):((aj-1)*strides(l)+sqrt(fsz)), :) + ...
                            sum(bsxfun(@times, rfilter, reshape(acts, res)), 5);
                    end

                    clear inblock acts;
                end
            end

            cW_grad{l} = reshape(sum(wgs,4), size(cW_grad{l}));

            clear postpool prepool rfilter rcW lowerl dconv;

            if l > 1
                dconv = dconv_next;
                dconv = dconv .* dsigmoid(lower, C.hidden.use_tanh);
            end

            clear lower;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % learning rate
        if C.adadelta.use
            % update
            for l = 1:n_full+1
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                if l < n_full+1
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                end
            end
            for l = 1:n_conv
                cbiases_grad_old{l} = (1 - momentum) * cbiases_grad{l} + momentum * cbiases_grad_old{l};
                cW_grad_old{l} = (1 - momentum) * cW_grad{l} + momentum * cW_grad_old{l};
            end

            if C.iteration.n_updates == 1
                adamom = 0;
            else
                adamom = C.adadelta.momentum;
            end

            for l = 1:n_full+1
                if l < n_full+1
                    C.adadelta.gW{l} = adamom * C.adadelta.gW{l} + (1 - adamom) * W_grad_old{l}.^2;
                end
                C.adadelta.gbiases{l} = adamom * C.adadelta.gbiases{l} + (1 - adamom) * biases_grad_old{l}.^2;
            end
            for l = 1:n_conv
                C.adadelta.gcW{l} = adamom * C.adadelta.gcW{l} + (1 - adamom) * cW_grad_old{l}.^2;
                C.adadelta.gcbiases{l} = adamom * C.adadelta.gcbiases{l} + (1 - adamom) * cbiases_grad_old{l}.^2;
            end

            for l = 1:n_full+1
                dbias = -(biases_grad_old{l} + ...
                    weight_decay * C.biases{l}) .* (sqrt(C.adadelta.biases{l} + C.adadelta.epsilon) ./ ...
                    sqrt(C.adadelta.gbiases{l} + C.adadelta.epsilon));
                C.biases{l} = C.biases{l} + dbias;

                C.adadelta.biases{l} = adamom * C.adadelta.biases{l} + (1 - adamom) * dbias.^2;
                clear dbias;

                if l < n_full+1
                    dW = -(W_grad_old{l} + ...
                        weight_decay * C.W{l}) .* (sqrt(C.adadelta.W{l} + C.adadelta.epsilon) ./ ...
                        sqrt(C.adadelta.gW{l} + C.adadelta.epsilon));
                    C.W{l} = C.W{l} + dW;

                    C.adadelta.W{l} = adamom * C.adadelta.W{l} + (1 - adamom) * dW.^2;

                    clear dW;
                end
            end

            for l = 1:n_conv
                dbias = -(cbiases_grad_old{l} + ...
                    weight_decay * C.cbiases{l}) .* (sqrt(C.adadelta.cbiases{l} + C.adadelta.epsilon) ./ ...
                    sqrt(C.adadelta.gcbiases{l} + C.adadelta.epsilon));
                C.cbiases{l} = C.cbiases{l} + dbias;

                C.adadelta.cbiases{l} = adamom * C.adadelta.cbiases{l} + (1 - adamom) * dbias.^2;
                clear dbias;

                dW = -(cW_grad_old{l} + ...
                    weight_decay * C.cW{l}) .* (sqrt(C.adadelta.cW{l} + C.adadelta.epsilon) ./ ...
                    sqrt(C.adadelta.gcW{l} + C.adadelta.epsilon));
                C.cW{l} = C.cW{l} + dW;

                C.adadelta.cW{l} = adamom * C.adadelta.cW{l} + (1 - adamom) * dW.^2;

                clear dW;
            end
        else
            if C.learning.lrate_anneal > 0 && (step >= C.learning.lrate_anneal * n_epochs)
                anneal_counter = anneal_counter + 1;
                actual_lrate = actual_lrate0 / anneal_counter;
            else
                if C.learning.lrate0 > 0
                    actual_lrate = C.learning.lrate / (1 + C.iteration.n_updates / C.learning.lrate0);
                else
                    actual_lrate = C.learning.lrate;
                end
                actual_lrate0 = actual_lrate;
            end

            C.signals.lrates = [C.signals.lrates actual_lrate];

            % update
            for l = 1:n_conv
                cbiases_grad_old{l} = (1 - momentum) * cbiases_grad{l} + momentum * cbiases_grad_old{l};
                cW_grad_old{l} = (1 - momentum) * cW_grad{l} + momentum * cW_grad_old{l};
            end
            for l = 1:n_full+1
                biases_grad_old{l} = (1 - momentum) * biases_grad{l} + momentum * biases_grad_old{l};
                if l < n_full+1
                    W_grad_old{l} = (1 - momentum) * W_grad{l} + momentum * W_grad_old{l};
                end
            end


            for l = 1:n_conv
                C.cbiases{l} = C.cbiases{l} - actual_lrate * (cbiases_grad_old{l} + weight_decay * C.cbiases{l});
                C.cW{l} = C.cW{l} - actual_lrate * (cW_grad_old{l} + weight_decay * C.cW{l});
            end
            for l = 1:n_full+1
                C.biases{l} = C.biases{l} - actual_lrate * (biases_grad_old{l} + weight_decay * C.biases{l});
                if l < n_full+1
                    C.W{l} = C.W{l} - actual_lrate * (W_grad_old{l} + weight_decay * C.W{l});
                end
            end
        end

        if C.verbose == 1
            fprintf(2, '.');
        end

        if use_gpu > 0
            clear v0 h0_full h0_conv dfull v0_clean vr hr deltae deltad 
        end

        if early_stop && mod(C.iteration.n_updates, valid_interval) == 0
            n_valid = size(valid_patches, 1);
            rndidx = randperm(n_valid);
            if use_gpu
                v0valid = gpuArray(single(valid_patches(rndidx(1:round(n_valid * valid_portion)),:)));
            else
                v0valid = valid_patches(rndidx(1:round(n_valid * valid_portion)),:);
            end

            if C.output.binary
                vr = convnet_classify(C, v0valid, 1);
            else
                vr = convnet_classify(C, v0valid);
            end
            if use_gpu > 0
                vr = gather(vr);
            end

            if C.output.binary
                % use the classification accuracy for early-stop
                xt = valid_targets(rndidx(1:round(n_valid * valid_portion)), :);
                yt = vr;

                [mp, mi] = max(gather(yt), [], 2);
                [tp, ti] = max(gather(xt), [], 2);

                n_correct = sum(mi == ti);
                rerr = 1 - n_correct/(round(n_valid * valid_portion));
            else
                rerr = mean(sum((valid_targets(rndidx(1:round(n_valid * valid_portion), :)) - vr).^2,2));
            end
            if use_gpu > 0
                rerr = gather(rerr);
            end

            C.signals.valid_errors = [C.signals.valid_errors rerr];

            if valid_err == -Inf
                valid_err = rerr;
                valid_best_err = rerr;
                valid_violate_cnt = 0;

                M_best = C;
                M_best = pull_from_gpu (M_best);
            else
                prev_err = valid_err;
                valid_err = 0.99 * valid_err + 0.01 * rerr;

                if C.verbose == 1
                    fprintf(2, 'valid err = %f\n', valid_err);
                end


                if step > C.valid_min_epochs
                    if (1.1 * valid_best_err) < valid_err 
                        fprintf(2, 'Early-stop! %f, %f\n', valid_err, valid_best_err);
                        stopping = 1;
                        break;
                    end

                    if valid_best_err <= valid_err
                        valid_violate_cnt = valid_violate_cnt + 1;
                        if valid_violate_cnt > (n_minibatches * C.valid_min_epochs)
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

                    M_best = C;
                    M_best = pull_from_gpu (M_best);
                end
            end
        else
            if C.stop.criterion > 0
                if C.stop.criterion == 1
                    if min_recon_error > C.signals.recon_errors(end)
                        min_recon_error = C.signals.recon_errors(end);
                        min_recon_error_update_idx = C.iteration.n_updates;
                    else
                        if C.iteration.n_updates > min_recon_error_update_idx + C.stop.recon_error.tolerate_count 
                            fprintf(2, '\nStopping criterion reached (recon error) %f > %f\n', ...
                                C.signals.recon_errors(end), min_recon_error);
                            stopping = 1;
                            break;
                        end
                    end
                else
                    error ('Unknown stopping criterion %d', C.stop.criterion);
                end
            end
        end

        if length(C.hook.per_update) > 1
            err = C.hook.per_update{1}(C, C.hook.per_update{2});

            if err == -1
                stopping = 1;
                break;
            end
        end
        
        if C.debug.do_display == 1 && mod(C.iteration.n_updates, C.debug.display_interval) == 0
            C.debug.display_function (C.debug.display_fid, C, v0, v1, h0, h1, W_grad, vbias_grad, hbias_grad);
            drawnow;
        end

        %if C.verbose 
        %    fprintf(2, 'how slow? %f sec\n', toc);
        %end
    end

    if use_gpu > 0
        % pull
        C = pull_from_gpu (C);
    end

    if length(C.hook.per_epoch) > 1
        err = C.hook.per_epoch{1}(C, C.hook.per_epoch{2});

        if err == -1
            stopping = 1;
        end
    end

    if stopping == 1
        break;
    end
    
    if C.verbose == 1
        fprintf(2, '\n');
    end
        
    fprintf(2, 'Epoch %d/%d - recon_error: %f valid_error: %f\n', step, n_epochs, rerr_ma, valid_err);
end

if use_gpu > 0
    % pull
    C = pull_from_gpu (C);
end

if early_stop
    C = M_best;
end

end

function [C] = push_to_gpu (C)
    n_layers = length(C.structure.layers);
    n_conv = size(C.structure.conv_layers, 1);
    n_full = length(C.structure.full_layers);

    % push
    for l = 1:n_conv
        C.cW{l} = gpuArray(single(C.cW{l}));
        C.cbiases{l} = gpuArray(single(C.cbiases{l}));
    end

    for l = 1:(n_full+1)
        if l < n_full + 1 
            C.W{l} = gpuArray(single(C.W{l}));
        end
        C.biases{l} = gpuArray(single(C.biases{l}));
    end

    if C.adadelta.use
        for l = 1:n_conv
            C.adadelta.cbiases{l} = gpuArray(single(C.adadelta.cbiases{l}));
            C.adadelta.gcbiases{l} = gpuArray(single(C.adadelta.gcbiases{l}));
            C.adadelta.cW{l} = gpuArray(single(C.adadelta.cW{l}));
            C.adadelta.gcW{l} = gpuArray(single(C.adadelta.gcW{l}));
        end
        for l = 1:n_full
            C.adadelta.biases{l} = gpuArray(single(C.adadelta.biases{l}));
            C.adadelta.gbiases{l} = gpuArray(single(C.adadelta.gbiases{l}));
            if l < n_full
                C.adadelta.W{l} = gpuArray(single(C.adadelta.W{l}));
                C.adadelta.gW{l} = gpuArray(single(C.adadelta.gW{l}));
            end
        end
    end
end

function [C] = pull_from_gpu (C)
    n_layers = length(C.structure.layers);
    n_conv = size(C.structure.conv_layers, 1);
    n_full = length(C.structure.full_layers);

    for l = 1:n_conv
        C.cW{l} = gather(C.cW{l});
        C.cbiases{l} = gather(C.cbiases{l});
    end

    for l = 1:(n_full+1)
        if l < n_full + 1 
            C.W{l} = gather(C.W{l});
        end
        C.biases{l} = gather(C.biases{l});
    end

    if C.adadelta.use
        for l = 1:n_conv
            C.adadelta.cbiases{l} = gather(C.adadelta.cbiases{l});
            C.adadelta.gcbiases{l} = gather(C.adadelta.gcbiases{l});
            C.adadelta.cW{l} = gather(C.adadelta.cW{l});
            C.adadelta.gcW{l} = gather(C.adadelta.gcW{l});
        end
        for l = 1:n_full
            C.adadelta.biases{l} = gather(C.adadelta.biases{l});
            C.adadelta.gbiases{l} = gather(C.adadelta.gbiases{l});
            if l < n_full
                C.adadelta.W{l} = gather(C.adadelta.W{l});
                C.adadelta.gW{l} = gather(C.adadelta.gW{l});
            end
        end
    end


end


