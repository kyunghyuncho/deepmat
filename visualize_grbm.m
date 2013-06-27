% visualize_grbm - Visualize 
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
function [f] = visualize_grbm(f, R, v0, vf, h0, hf, W_grad, vbias_grad, hbias_grad, sigma_grad);

    n_visible = floor(sqrt(size(R.W,1))).^2;

    set(0, 'CurrentFigure', f);

    subplot(3,3,1);
    visualize(R.W(1:n_visible,:));
    title ('Weights');

    subplot(3,3,2);
    visualize(R.vbias(1:n_visible));

    if R.adaptive_lrate.use == 1
        subplot(3,3,3);
        semilogy(R.signals.lrates);
        title('Learning rates');
    end

    subplot(3,3,4);
    visualize(v0(:,1:n_visible)');
    title ('Data particles');
    
    subplot(3,3,5);
    visualize(vf(:,1:n_visible)');
    title ('Fantasy particles');

    subplot(3,3,6);
    plot(R.signals.recon_errors);
    title ('Reconstruction errors');

    subplot(3,3,7);
    visualize(W_grad(1:n_visible,:));
    title ('Gradient (Weights)');

    subplot(3,3,8);
    visualize(R.sigmas', 0, 0, 0);
