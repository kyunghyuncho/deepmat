%MAXPOOLING implements a max-pooling operation for 4D tensors shaped as:
% (rows, cols, channels, samples) and returns the pooled data with the
% corresponding indices.
%
%   [m, idx] = MaxPooling(IM, [2 2])
%
% IM can also be a 2D tensor, the missing dims are set to 1.

% AUTORIGHTS
% Copyright (C) 2011 Jonathan Masci <jonathan@idsia.ch>
%
% This file is available under the terms of the
% GNU GPLv2.
