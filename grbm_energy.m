
function [E, Emin, Emax, Es] = grbm_energy (x, W, vbias, hbias, sigmas)

n_samples = size(x,1);

Wh = ones(n_samples,1)*hbias' + bsxfun(@rdivide, x, sigmas.^2) * W;

Es = bsxfun(@minus,x,vbias');
Es = sum(bsxfun(@rdivide, Es.^2, sigmas.^2),2)/2;

Wh_plus = max(Wh, 0);
Es = Es - sum(log(exp(-Wh_plus)+exp(Wh-Wh_plus))+Wh_plus,2);
Es = Es';

Emin = min(Es);
Emax = max(Es);
E = mean(Es);

end
