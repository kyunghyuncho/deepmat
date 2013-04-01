function visualize_adv(W, horiz, n_rows, n_cols, do_sort, borders)

if (nargin < 2)
    horiz = 0;
end

if (nargin < 5)
    do_sort = 0;
end

if (do_sort == 1)
    Wnorms = sum(W.^2,1);
    [B,IX] = sort(Wnorms,'descend');
    W = W(:,IX);
end

% how many pixels for borders?
if (nargin < 6)
    borders = 1;
end

fprintf('Visualizing patches\n');
[ndim,nunits]=size(W);
npix = floor(sqrt(ndim)+0.999);
npix2 = floor(sqrt(nunits)+0.999);
minW=min(W(:));
maxW=max(W(:));

if (nargin < 4)
    n_rows = npix2;
    n_cols = npix2;
end

%bigpic = -(minW+maxW)/2*ones(((npix+borders)*npix2+borders));
%bigpic = -(minW+maxW)/2*ones(((n_rows+borders)*npix2+borders), ((n_cols+borders)*npix2+borders));
bigpic = -(minW+maxW)/2*ones(((npix+borders)*n_rows+borders), ((npix+borders)*n_cols+borders));
%if (nunits/npix2<=npix2-1),
%    bigpic = bigpic(:,1:(npix+borders)*(npix2-1)+borders);
%end;
idx = 0;
for i=1:n_rows
    for j=1:n_cols
        idx = idx + 1;

        if idx > nunits
            break;
        end

        if (horiz)
            bigpic((i-1)*(npix+borders)+borders+1:(i-1)*(npix+borders)+borders+npix,...
                (j-1)*(npix+borders)+borders+1:(j-1)*(npix+borders)+borders+npix)...
                = reshape(W(:,idx),npix,npix)';
        else
            bigpic((i-1)*(npix+borders)+borders+1:(i-1)*(npix+borders)+borders+npix,...
                (j-1)*(npix+borders)+borders+1:(j-1)*(npix+borders)+borders+npix)...
                = reshape(W(:,idx),npix,npix);
        end
    end
end;
imagesc(bigpic);
colormap(gray);
axis off;
axis equal;
fprintf('done.\n');
