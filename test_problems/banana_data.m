%   Maximize the the ever-famous Rosenbrock "banana valley" function

function [ x0, xlower, xupper, xtype, xscale, max_or_min ] = banana_data( x )

x0 = [ -1.2 1 ];         % Rosenbrock (1)
xlower = -Inf * ones(size(x0));
xupper =  Inf * ones(size(x0));
xtype  = 'c';
xscale = [ 1 1 ];
max_or_min = 'min';
