function [ xl, xu ] = banana_vb(  x, level, xlevel, xlower, xupper )

xl = xlower;
xu = xupper;
xu(2) = x(1);
