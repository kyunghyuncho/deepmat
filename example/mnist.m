% load the data
load ('mnist_all_bw.mat');

% concatenate them into one matrix.
X_test = double([ test_0_bw, zeros(size(test_0_bw,1),1) ; ...
	test_1_bw, ones(size(test_1_bw,1),1); ...
	test_2_bw, 2*ones(size(test_2_bw,1),1); ...
	test_3_bw, 3*ones(size(test_3_bw,1),1); ...
	test_4_bw, 4*ones(size(test_4_bw,1),1); ...
	test_5_bw, 5*ones(size(test_5_bw,1),1); ...
	test_6_bw, 6*ones(size(test_6_bw,1),1); ...
	test_7_bw, 7*ones(size(test_7_bw,1),1); ...
	test_8_bw, 8*ones(size(test_8_bw,1),1); ...
	test_9_bw, 9*ones(size(test_9_bw,1),1) ]);

X_train = double([ train_0_bw, zeros(size(train_0_bw,1),1) ; ...
	train_1_bw, ones(size(train_1_bw,1),1); ...
	train_2_bw, 2*ones(size(train_2_bw,1),1); ...
	train_3_bw, 3*ones(size(train_3_bw,1),1); ...
	train_4_bw, 4*ones(size(train_4_bw,1),1); ...
	train_5_bw, 5*ones(size(train_5_bw,1),1); ...
	train_6_bw, 6*ones(size(train_6_bw,1),1); ...
	train_7_bw, 7*ones(size(train_7_bw,1),1); ...
	train_8_bw, 8*ones(size(train_8_bw,1),1); ...
	train_9_bw, 9*ones(size(train_9_bw,1),1) ]);

for i=0:9
    eval(sprintf('clear test_%d_bw train_%d_bw', i, i));
end

reduced_data = 0;

if (reduced_data)
	% Let's try first 10% of each digit
	X_train_reduced = [];
	X_test_reduced = [];
	for i=0:9
		X_digit = X_train(X_train(:,28*28+1)==i,:);
		X_train_reduced = [X_train_reduced; X_digit(1:ceil(size(X_digit,1)/10),:)];
		X_digit = X_test(X_test(:,28*28+1)==i,:);
		X_test_reduced = [X_test_reduced; X_digit(1:ceil(size(X_digit,1)/10),:)];
	end
	X_train = X_train_reduced;
	X_test = X_test_reduced;
	X_test_labels = X_test_reduced(:,785);
else
        X_labels = X_train(:,785);
	X_test_labels = X_test(:,785);
end

% do the preprocessing of the data so that each component takes the value {0,1}
X = X_train(:,1:784) / max(max(X_train(:,1:784)));
X_test = X_test(:,1:784) / max(max(X_test(:,1:784)));

clear X_train;

digit_sz = 28;

% # of visible neurons.
n_visible = 28 * 28;

filename_prefix='mnist';
