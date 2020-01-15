%   An auxiliary function building the complete objective function value
%   from calls to the element functions of a coordinate-partially-separable
%   problem.

%   Programming : Ph. Toint (This version 20 I 2018)

function fx = sumfi( x, eldom, fi )

nel = length( eldom );
fx  = 0;
for iel = 1:nel
   fx = fx + fi{iel}( iel, x( eldom{ iel } ) );
end


