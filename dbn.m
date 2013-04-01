% dbn - training a DBN with up-down algorithm
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
function [D] = dbn(D, patches);

actual_lrate = D.learning.lrate;

n_samples = size(patches, 1);

layers = D.structure.layers;
n_layers = length(layers);

if layers(1) ~= size(patches, 2)
    error('Data is not properly aligned');
end

minibatch_sz = D.learning.minibatch_sz;
n_minibatches = ceil(n_samples / minibatch_sz);

n_epochs = D.iteration.n_epochs;

momentum = D.learning.momentum;
weight_decay = D.learning.weight_decay;

rec.biases_grad = cell(n_layers-1, 1);
rec.W_grad = cell(n_layers-1, 1);
rec.biases_grad_old = cell(n_layers-1, 1);
rec.W_grad_old = cell(n_layers-1, 1);
for l = 1:n_layers-1
    rec.biases_grad{l} = zeros(size(D.rec.biases{l}))';
    if l < n_layers
        rec.W_grad{l} = zeros(size(D.rec.W{l}));
    end
    rec.biases_grad_old{l} = zeros(size(D.rec.biases{l}))';
    if l < n_layers
        rec.W_grad_old{l} = zeros(size(D.rec.W{l}));
    end
end

gen.biases_grad = cell(n_layers-1, 1);
gen.W_grad = cell(n_layers-1, 1);
gen.biases_grad_old = cell(n_layers-1, 1);
gen.W_grad_old = cell(n_layers-1, 1);
for l = 1:n_layers-1
    gen.biases_grad{l} = zeros(size(D.gen.biases{l}))';
    if l < n_layers
        gen.W_grad{l} = zeros(size(D.gen.W{l}));
    end
    gen.biases_grad_old{l} = zeros(size(D.gen.biases{l}))';
    if l < n_layers
        gen.W_grad_old{l} = zeros(size(D.gen.W{l}));
    end
end

top.W_grad = zeros(size(D.top.W));
top.vbias_grad = zeros(size(D.top.vbias));
top.hbias_grad = zeros(size(D.top.hbias));

top.W_grad_old = zeros(size(D.top.W));
top.vbias_grad_old = zeros(size(D.top.vbias));
top.hbias_grad_old = zeros(size(D.top.hbias));

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

for step=1:n_epochs
    if D.verbose
        fprintf(2, 'Epoch %d/%d: ', step, n_epochs)
    end
    if use_gpu
        % push
        for l = 1:n_layers-1
            if l < n_layers-1
                D.rec.W{l} = gpuArray(single(D.rec.W{l}));
                D.gen.W{l} = gpuArray(single(D.gen.W{l}));
            end
            D.rec.biases{l} = gpuArray(single(D.rec.biases{l}));
            D.gen.biases{l} = gpuArray(single(D.gen.biases{l}));
        end

        D.top.W = gpuArray(single(D.top.W));
        D.top.vbias = gpuArray(single(D.top.vbias));
        D.top.hbias = gpuArray(single(D.top.hbias));
    end

    for mb=1:n_minibatches
        D.iteration.n_updates = D.iteration.n_updates + 1;

        v0 = patches((mb-1) * minibatch_sz + 1:min(mb * minibatch_sz, n_samples), :);
        mb_sz = size(v0,1);

        if use_gpu > 0
            v0 = gpuArray(single(v0));
        end

        % learning rate
        if D.learning.lrate_anneal > 0 && (step >= D.learning.lrate_anneal * n_epochs)
            anneal_counter = anneal_counter + 1;
            actual_lrate = actual_lrate0 / anneal_counter;
        else
            if D.learning.lrate0 > 0
                actual_lrate = D.learning.lrate / (1 + D.iteration.n_updates / D.learning.lrate0);
            else
                actual_lrate = D.learning.lrate;
            end
            actual_lrate0 = actual_lrate;
        end

        D.signals.lrates = [D.signals.lrates actual_lrate];

        % recognition (wake)
        h0r = cell(n_layers-1, 1);
        h0r{1} = v0;
        if D.learning.ffactored == 0
            h0r{1} = binornd(1, h0r{1});
        end

        for l = 2:(n_layers-1)
            h0r{l} = sigmoid(bsxfun(@plus, h0r{l-1} * D.rec.W{l-1}, D.rec.biases{l}'));
            if D.learning.ffactored == 0
                h0r{l} = binornd(1, h0r{l});
            end
        end

        r0 = cell(n_layers-1, 1);
        for l = 1:(n_layers-2)
            r0{l} = sigmoid(bsxfun(@plus, h0r{l+1} * D.gen.W{l}', D.gen.biases{l}'));
        end

        % update generative weight
        for l = 1:(n_layers-2)
            gen.W_grad{l} = ((h0r{l} - r0{l})' * h0r{l+1}) / size(v0, 1);
            gen.W_grad_old{l} = (1 - momentum) * gen.W_grad{l} + momentum * gen.W_grad_old{l};
            D.gen.W{l} = D.gen.W{l} + actual_lrate * (gen.W_grad_old{l} - D.learning.weight_decay * D.gen.W{l});

            gen.biases_grad{l} = mean(h0r{l} - r0{l}, 1);
            gen.biases_grad_old{l} = (1 - momentum) * gen.biases_grad{l} + momentum * gen.biases_grad_old{l};
            D.gen.biases{l} = D.gen.biases{l} + actual_lrate * (gen.biases_grad_old{l}' - D.learning.weight_decay * D.gen.biases{l});
        end

        % contrastive steps
        if D.learning.persistent_cd == 0 || D.iteration.n_updates == 1
            vtop = h0r{end};
        end
        for cs = 1:D.learning.contrastive_step
            htop = sigmoid(bsxfun(@plus, vtop * D.top.W, D.top.hbias'));
            htop = binornd(1, htop);
            vtop = sigmoid(bsxfun(@plus, htop * D.top.W', D.top.vbias'));
            vtop = binornd(1, vtop);
        end

        tv0 = h0r{end}; 
        th0 = sigmoid(bsxfun(@plus, tv0 * D.top.W, D.top.hbias'));
        tv1 = vtop;
        th1 = sigmoid(bsxfun(@plus, tv1 * D.top.W, D.top.hbias'));

        top.W_grad = (tv0' * th0)/size(tv0,1) - (tv1' * th1)/size(tv1,1);
        top.vbias_grad = mean(tv0, 1) - mean(tv1, 1);
        top.hbias_grad = mean(th0, 1) - mean(th1, 1);

        top.W_grad_old = (1 - momentum) * top.W_grad + momentum * top.W_grad_old;
        top.vbias_grad_old = (1 - momentum) * top.vbias_grad' + momentum * top.vbias_grad_old;
        top.hbias_grad_old = (1 - momentum) * top.hbias_grad' + momentum * top.hbias_grad_old;

        D.top.W = D.top.W + actual_lrate * (top.W_grad_old - D.learning.weight_decay * D.top.W);
        D.top.vbias = D.top.vbias + actual_lrate * (top.vbias_grad_old - D.learning.weight_decay * D.top.vbias);
        D.top.hbias = D.top.hbias + actual_lrate * (top.hbias_grad_old - D.learning.weight_decay * D.top.hbias);
        
        % generation (sleep)
        h0r{n_layers-1} = tv1;
        for l = (n_layers-2):-1:1
            h0r{l} = sigmoid(bsxfun(@plus, h0r{l+1} * D.gen.W{l}', D.gen.biases{l}'));
            if D.learning.ffactored == 0
                h0r{l} = binornd(1, h0r{l});
            end
        end

        for l = 2:(n_layers - 1)
            r0{l} = sigmoid(bsxfun(@plus, h0r{l-1} * D.rec.W{l-1}, D.rec.biases{l}'));
        end

        % update recognition weight
        for l = 1:(n_layers-1)
            if l < n_layers - 1
                rec.W_grad{l} = (h0r{l}' * (h0r{l+1} - r0{l+1})) / size(h0r{1}, 1);
                rec.W_grad_old{l} = (1 - momentum) * rec.W_grad{l} + momentum * rec.W_grad_old{l};
                D.rec.W{l} = D.rec.W{l} + actual_lrate * (rec.W_grad_old{l} - D.learning.weight_decay * D.rec.W{l});
            end

            if l > 1
                rec.biases_grad{l} = mean(h0r{l} - r0{l}, 1);
                rec.biases_grad_old{l} = (1 - momentum) * rec.biases_grad{l} + momentum * rec.biases_grad_old{l};
                D.rec.biases{l} = D.rec.biases{l} + actual_lrate * (rec.biases_grad_old{l}' - D.learning.weight_decay * D.rec.biases{l});
            end
        end

        clear h0r r0 htop tv0 th0 tv1 th1; 
        if D.learning.persistent_cd  == 0
            clear vtop;
        end

        % compute reconstruction error
        hr = v0;
        for l = 2:n_layers-1
            hr = sigmoid(bsxfun(@plus, hr * D.rec.W{l-1}, D.rec.biases{l}'));
        end
        hr = sigmoid(bsxfun(@plus, hr * D.top.W, D.top.hbias'));
        hr = sigmoid(bsxfun(@plus, hr * D.top.W', D.top.vbias'));
        for l = n_layers-1:-1:2
            hr = sigmoid(bsxfun(@plus, hr * D.gen.W{l-1}', D.gen.biases{l-1}'));
        end
        
        rerr = mean(sum((v0 - hr).^2,2));
        if use_gpu > 0
            rerr = gather(rerr);
        end
        D.signals.recon_errors = [D.signals.recon_errors rerr];

        if D.verbose == 1
            fprintf(2, '.');
        end

        if use_gpu > 0
            clear v0 h0d h0e v0_clean vr hr deltae deltad 
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
        
        if D.debug.do_display == 1 && mod(D.iteration.n_updates, D.debug.display_interval) == 0
            D.debug.display_function (D.debug.display_fid, D, v0, v1, h0, h1, W_grad, vbias_grad, hbias_grad);
            drawnow;
        end
    end

    if use_gpu > 0
        % pull
        for l = 1:n_layers-1
            if l < n_layers-1
                D.rec.W{l} = gather(D.rec.W{l});
                D.gen.W{l} = gather(D.gen.W{l});
            end
            D.rec.biases{l} = gather(D.rec.biases{l});
            D.gen.biases{l} = gather(D.gen.biases{l});
        end

        D.top.W = gather(D.top.W);
        D.top.vbias = gather(D.top.vbias);
        D.top.hbias = gather(D.top.hbias);
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
    for l = 1:n_layers-1
        if l < n_layers-1
            D.rec.W{l} = gather(D.rec.W{l});
            D.gen.W{l} = gather(D.gen.W{l});
        end
        D.rec.biases{l} = gather(D.rec.biases{l});
        D.gen.biases{l} = gather(D.gen.biases{l});
    end

    D.top.W = gather(D.top.W);
    D.top.vbias = gather(D.top.vbias);
    D.top.hbias = gather(D.top.hbias);
end


