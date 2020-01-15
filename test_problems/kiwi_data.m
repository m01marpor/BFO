%   Maximize a variant of the ever-famous Rosenbrock "banana valley" function

function [ x0, xlower, xupper ] = kiwi_data

x0 = [ -1. 1.2 ];         % Rosenbrock (1)
xlower = -Inf * ones(size(x0));
xupper =  Inf * ones(size(x0));
