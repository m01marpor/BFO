%   Minimize the ever-famous Rosenbrock "banana valley" function with

function fx = banana( x, fmax, params )  
    
fx  = 100 * ( x(2) - x(1)^2 )^2 ;
if ( fx > fmax )
   return;
else
   fx = fx +  (1-x(1))^2;
end

