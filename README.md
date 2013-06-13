deepmat
====
WARNING: this is not my main code, and there is no warranty attached!

= Restricted Boltzmann Machine & Deep Belief Networks =
 - Binary/Gaussian Visible Units + Binary Hidden Units
 - Enhanced Gradient, Adaptive Learning Rate
 - Adadelta for RBM
 - Contrastive Divergence
 - (Fast) Persistent Contrastive Divergence
 - Parallel Tempering
 - DBN: Up-down Learning Algorithm

= Deep Boltzmann Machine =
 - Binary/Gaussian Visible Units + Binary Hidden Units
 - (Persistent) Contrastive Divergence
 - Enhanced Gradient, Adaptive Learning Rate
 - Two-stage Pretraining Algorithm (example)
 - Centering Trick (fixed center variables only)

= Denoising Autoencoder (Tied Weights) =
 - Binary/Gaussian Visible Units + Binary(Sigmoid)/Gaussian Hidden Units
 - tanh/sigm/relu nonlinearities
 - Shallow: sparsity, contractive, soft-sparsity (log-cosh) regularization
 - Deep: stochastic backprop
 - Adagrad, Adadelta

= Multi-layer Perceptron =
 - Stochastic Backpropagation, Dropout
 - tanh/sigm/relu nonlinearities
 - Adagrad, Adadelta
 - Balanced minibatches using crossvalind()

