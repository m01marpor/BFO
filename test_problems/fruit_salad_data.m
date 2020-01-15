function [ x0, xlower, xupper, xtype, xscale, max_or_min, xlevel ] = fruit_salad_data

x0         = [ -1 1 -1 1 -1 1 ];
xlower     = -Inf * ones(1,6);
xupper     =  Inf * ones(1,6);
xtype      = 'cccccc';
xscale     = 1;
max-or-min = 'min';
xlevel     = [ 1 1 2 2 3 3 ];
