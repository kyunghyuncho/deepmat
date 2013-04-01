% rbm_energy - Free energy
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
function [E, Emin, Emax, Es] = rbm_energy (x, W, vbias, hbias)

n_samples = size(x,1);

Wh = ones(n_samples,1)*hbias' + x * W;
Wh_plus = max(Wh, 0);
Es = -x * vbias - sum(log(exp(-Wh_plus)+exp(Wh-Wh_plus))+Wh_plus,2);

Emin = min(Es);
Emax = max(Es);
E = mean(Es);

end

