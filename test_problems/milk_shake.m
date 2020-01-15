function taste = milk_shake( cx )

cx         = cx{1};
gstyle     = cx{1};
hstyle     = cx{2};
banan      = cx{3};
strawberry = cx{4};
mango      = cx{5};
carrot     = cx{6};
pepper     = cx{7};

switch( gstyle )
case 'fruity'
   switch( hstyle )
   case 'homely'
      taste = (0.8 + banana( [ banan, strawberry ] ) );
   case 'exotic'
      taste = (0.7 + banana( [ banan, mango ] ) );
   end
case 'mixed'
   taste = 0.5 + banana( [ banan, mango ] ) * banana( [ carrot, pepper ] );
case 'veggy'
   taste = banana( [ carrot, pepper ] );
end

return

end
