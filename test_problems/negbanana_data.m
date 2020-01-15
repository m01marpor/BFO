%   Maximize the negative of the ever-famous Rosenbrock "banana valley" function with

function [ x0, xlower, xupper, xtype, xscale, max_or_min] = negbanana_data

x0 = [ -1.2 1 ];         % Rosenbrock (1)
xlower = -Inf * ones(size(x0));
xupper =  Inf * ones(size(x0));
xtype  = 'c';
xscale = 1;
max_or_min = 'max';
