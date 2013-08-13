% convnet_classify
% Copyright (C) 2013 KyungHyun Cho
%
%This program is free software; you can redistribute it and/or
%modify it under the terms of the GNU General Public License
%as published by the Free Software Foundation; either version 2
%of the License, or (at your option) any later version.
%
%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License
%along with this program; if not, write to the Free Software
%Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [c, posterior] = convnet_classify(C, x0, raw)

if nargin < 3
    raw = 0;
end

layers = C.structure.layers;
n_layers = length(layers);

posterior = x0;

full_layers = C.structure.full_layers;
conv_layers = C.structure.conv_layers;
n_full = length(full_layers);
n_conv = size(conv_layers,1);

strides = C.structure.strides;
poolratios = C.structure.poolratios;

mb_sz = size(x0, 1);

cin = C.structure.channel_in;
szin = C.structure.size_in;

repost = reshape(posterior, [mb_sz, szin, szin, cin]);

if C.lcn.use
    subwindow = fspecial('gaussian', C.lcn.neigh);
    %subwindow_sum = ones(C.lcn.neigh);
end

if C.lcn.use 
    subsum = convn(repost, reshape(subwindow_sum, [1, C.lcn.neigh, C.lcn.neigh, 1]), 'same');
    repost = repost - subsum / C.lcn.neigh^2;
    repost2 = repost.^2;
    subsum = convn(repost2, reshape(subwindow_sum, [1, C.lcn.neigh, C.lcn.neigh, 1]), 'same');
    repost = repost ./(sqrt(subsum + 1e-12) / C.lcn.neigh);
end

for l = 1:n_conv
    fsz = length(C.cW{l}(:,1)) / cin;
    n_filters = size(C.cW{l}, 2);
    szout_prepool = (floor((szin - sqrt(fsz)) / strides(l)) + 1);
    szout = ceil(szout_prepool / poolratios(l));
    respfull = zeros(mb_sz, szout, szout, n_filters);

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
        
        if C.lcn.use 
            subsum = convn(resp, reshape(subwindow_sum, [1, C.lcn.neigh, C.lcn.neigh, 1]), 'same');
            resp = resp - subsum / C.lcn.neigh^2;
            resp2 = resp.^2;
            subsum = convn(resp2, reshape(subwindow_sum, [1, C.lcn.neigh, C.lcn.neigh, 1]), 'same');
            resp = resp ./(sqrt(subsum + 1e-12) / C.lcn.neigh);
        end

        if C.structure.poolratios(l) > 1
            % pooling
            switch C.pooling(l)
                case 0
                    resp = convnet_maxpool (resp, C.structure.poolratios(l));
                case 1
                    resp = convnet_avgpool (resp, C.structure.poolratios(l));
                case 2
                    error('NOT SUPPORTED');
            end
        end

        respfull(:,:,:,fidx) = resp;
    end

    szin = szout;
    cin = n_filters;
    repost = respfull;
end

% stretch
posterior = reshape(repost, [mb_sz C.structure.layers(n_conv+1)]);

for l = 2:n_full+1
    if C.dropout.use && l > 2
        posterior = (1 - C.dropout.prob) * posterior;
    end

    posterior = bsxfun(@plus, posterior * C.W{l-1}, C.biases{l}');

    if l < n_full + 1
        posterior = sigmoid(posterior, C.hidden.use_tanh);
    end
    if l == n_full + 1 && C.output.binary
        posterior = softmax(posterior);
    end
end

if raw
    c = posterior;
else
    [maxp, c] = max(posterior, [], 2);
end


