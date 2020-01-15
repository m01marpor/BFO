function fx = negbanana( x )
rx = [ 10*(x(2)-x(1)^2);  (1-x(1)) ];
fx = -rx'*rx;
