%   Minimize the ever-famous Rosenbrock "banana valley" function with

function fx = badly_scaled_banana( x )

fx  =  100 * ( 0.000001*x(2) - (100*x(1))^2 )^2 + (1-100*x(1))^2;
