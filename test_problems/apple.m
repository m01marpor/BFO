%   Minimize a variant of the ever-famous Rosenbrock "banana valley" function

function fx = apple( x )

fx  = [ 9 * ( x(2) - x(1)^2 );  1.2 - x(1) ];
fx = fx'*fx;
