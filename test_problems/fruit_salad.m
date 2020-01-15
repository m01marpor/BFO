function fx = fruit_salad( x )

fk  = norm( kiwi( x(1:2) ) )^2;
fnb = negbanana( x(3:4) );
fa  = norm( apple( x(5:6) ) )^2;
fx = fk + fnb + fa;
