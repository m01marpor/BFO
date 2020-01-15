function fx = fruit_bowl( x )

fk  = norm( kiwi( x(1:2) ) )^2;
fnb = negbanana( x(3:4) );
fx = fk + fnb; 
