function [ x0, xlower, xupper, xtype, xscale, max_or_min, xlevel ] = fruit_bowl_data( x )

x0         = [ -1 1 -1 1 ];
xlower     = -Inf * ones(1,4);
xupper     =  Inf * ones(1,4);
xtype      = 'cccc';
xscale     = 1;
max_or_min = 'min';
xlevel     = [ 1 1 2 2 ];
