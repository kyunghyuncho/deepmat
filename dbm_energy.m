
% it's just -log P*(v,h)
function [E, Emin, Emax, Es] = dbm_energy (vs, Ws, biases, binary, ...
    temperature, sigmas, base_sigma, base_vbias)

if nargin < 4
    binary = 1;
end

if nargin < 5
    temperature = 1;
end

if nargin < 6
    error('GDBM');
end

n_samples = size(vs{1},1);

if length(temperature) == 1
    t = temperature * ones(n_samples, 1);
end

tvs = vs;

for l = 1:length(vs)
    tvs{l} = bsxfun(@times, vs{l}, t);
end

if binary
    Es = zeros(n_samples,1);

    for l=1:(length(vs)-1)
        Es = Es + sum((tvs{l} * Ws{l}) .* tvs{l+1},2) + tvs{l} * biases{l};
    end

    Es = Es + tvs{end} * biases{end};
    Es = -1 * Es;

    Emin = min(Es);
    Emax = max(Es);
    E = mean(Es);

    clear tvs;
else
    n_samples = size(vs{1},1);

    tsigmas = repmat(sigmas.^2', [n_samples 1]);
    tvbias = repmat(biases{1}', [n_samples 1]);
    bsigmas = repmat(base_sigma.^2', [n_samples 1]);
    bvbias = repmat(base_vbias, [n_samples 1]);

    tsigmas = bsxfun(@times, tsigmas, t);
    tvbias = bsxfun(@times, tvbias, t);
    bsigmas = bsxfun(@times, bsigmas, 1 - t);
    bvbias = bsxfun(@times, bvbias, 1 - t);

    tsigmas = tsigmas + bsigmas;
    tvbias = tvbias + bvbias;

    clear bvbias bsigmas;

    Es = zeros(n_samples,1);

    Es = Es + sum(((vs{1}./tsigmas) * Ws{1}) .* tvs{2},2) - sum(((vs{1} - tvbias).^2) ./ (2 * tsigmas), 2);

    for l=2:(length(vs)-1)
        Es = Es + sum((tvs{l} * Ws{l}) .* tvs{l+1},2) + tvs{l} * biases{l};
    end

    Es = Es + tvs{end} * biases{end};
    Es = -1 * Es;

    Emin = min(Es);
    Emax = max(Es);
    E = mean(Es);

    clear tvs tsigmas tvbias;
end









































