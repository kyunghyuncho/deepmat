
load 'sdae_mnist_vis.mat';

colors = colormap;

figure;
hold on;
for c = 1:10
    x = H(X_labels == (c-1), 1);
    y = H(X_labels == (c-1), 2);
    rndidx = randperm(length(x));
    x = x(rndidx(1:500));
    y = y(rndidx(1:500));

    plot(x, y, 'x', 'Color', colors(ceil(c/10 * 64), :));
end
hold off;


