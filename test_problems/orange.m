%   Minimize a variant of the ever-famous Rosenbrock "banana valley" function

function fx = orange( x )

fx  = (9.5*( x(2) - x(1)^2 ))^2 + (1.3-x(1))^2;
