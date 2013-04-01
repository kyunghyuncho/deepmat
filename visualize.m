% visualize - Visualize 
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
% Original code written by Tapani Raiko
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
function visualize(W, horiz, do_sort, borders)
if (nargin < 2)
    horiz = 0;
end

if (nargin < 3)
    do_sort = 0;
end

if (do_sort == 1)
    Wnorms = sum(W.^2,1);
    [B,IX] = sort(Wnorms,'descend');
    W = W(:,IX);
end

% how many pixels for borders?
if nargin < 4
    borders = 1;
end
%fprintf('Visualizing patches\n');
[ndim,nunits]=size(W);
npix = floor(sqrt(ndim)+0.999);
npix2 = floor(sqrt(nunits)+0.999);
minW=min(W(:));
maxW=max(W(:));
bigpic = -(minW+maxW)/2*ones(((npix+borders)*npix2+borders));
if (nunits/npix2<=npix2-1),
    bigpic = bigpic(:,1:(npix+borders)*(npix2-1)+borders);
end;
for i=1:nunits;
    if (horiz)
    bigpic(mod(i-1,npix2)*(npix+borders)+borders+1:mod(i-1,npix2)*(npix+borders)+borders+npix,...
        floor((i-1)/npix2)*(npix+borders)+borders+1:floor((i-1)/npix2)*(npix+borders)+borders+npix)...
        = reshape(W(:,i),npix,npix)';
    else
    bigpic(mod(i-1,npix2)*(npix+borders)+borders+1:mod(i-1,npix2)*(npix+borders)+borders+npix,...
        floor((i-1)/npix2)*(npix+borders)+borders+1:floor((i-1)/npix2)*(npix+borders)+borders+npix)...
        = reshape(W(:,i),npix,npix);
    end
end;
imagesc(bigpic);
colormap(gray);
axis off;
axis equal;
%fprintf('done.\n');
