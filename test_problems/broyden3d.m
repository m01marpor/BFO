%   Minimize the Broyden tridiagonal function

function fx = broyden3d( i, x )


fx  = ( ( 3 - 2*x(2) )*x(2) - x(1) - 2*x(3) + 1 )^2;
