%   Minimize a quadratic function 

function fx = myobj( x, p )

fx = x' * p + 0.5 * x' * x;
