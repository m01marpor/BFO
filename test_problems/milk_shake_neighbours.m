function [ cneighbours, xtypes, xlowers, xuppers ] = ...
                                milk_shake_neighbours( cx, xtype, xlower, xupper )

%neighboursof = [ cx{1}, ' ', cx{2} ]%

cneighbours= {};
xtypes = {};
xlowers = [];
xuppers = [];

if ( ismember( xtype( 1 ), { 'w','z' } ) )
   return
end

switch ( cx{ 1 } )
case 'fruity'
   switch( cx{2} )
   
   case 'homely'

      cneighbours{1} = {{ 'fruity', 'exotic', cx{3}, cx{4}, cx{5}, cx{6}, cx{7} }};
      xtypes{1}      = 'sscrcrr';
      xlowers(1:7,1) = xlower;
      xuppers(1:7,1) = xupper;

      if ( ~ismember( xtype( 2 ), { 'w', 'z' } ) )
         cneighbours{2} = {{ 'veggy', 'exotic', cx{3}, cx{4}, cx{5}, cx{6}, cx{7} }};
         xtypes{2}      = 'skrrrcc';
         xlowers(1:7,2) = [ -Inf  -Inf   0   0   0   0   0 ];
         xuppers(1:7,2) = [  Inf,  Inf,  2,  2,  2,  2,  2 ];
      end
      
   case 'exotic'
   
      cneighbours{1} = {{ 'fruity', 'homely', cx{3}, cx{4}, cx{5}, cx{6}, cx{7} }};
      xtypes{1}      = 'ssccrrr';
      xlowers(1:7,1) = xlower;
      xuppers(1:7,1) = xupper;
      
      if ( ~ismember( xtype( 2 ), { 'w', 'z' } ) )
         cneighbours{2} = {{ 'mixed', 'exotic', cx{3}, cx{4}, cx{5}, cx{6}, cx{7} }};
         xtypes{2}      = 'skcrccc';
         xlowers(1:7,2) = [ -Inf, -Inf, -2, -2, -2, -2, -2 ];
         xuppers(1:7,2) = [  Inf,  Inf,  3,  2,  3,  3,  3 ];
      end
      
   end
   
case 'mixed'

      cneighbours{1} = {{ 'fruity', 'exotic', cx{3}, cx{4}, cx{5}, cx{6}, cx{7} }};
      xtypes{1}      = 'sscrccc';
      xlowers(1:7,1) = [ -Inf, -Inf, -2, -2, -2, -2, -2 ];
      xuppers(1:7,1) = [  Inf,  Inf,  2,  2,  2,  2,  2 ];
      
      if ( ~ismember( xtype( 2 ), { 'w', 'z' } ) )
         cneighbours{2} = {{ 'veggy', 'exotic', cx{3}, cx{4}, cx{5}, cx{6}, cx{7} }};
         xtypes{2}      = 'skrrrcc';
         xlowers(1:7,2) = [ -Inf  -Inf   0   0   0   0   0 ];
         xuppers(1:7,2) = [  Inf,  Inf,  2,  2,  2,  2,  2 ];
      end
      
case 'veggy'

      cneighbours{1} = {{ 'fruity', 'homely', cx{3}, cx{4}, cx{5}, cx{6}, cx{7} }};
      xtypes{1}      = 'ssccrrr';
      xlowers(1:7,1) = [ -Inf, -Inf, -2, -2, -2, -2, -2 ];
      xuppers(1:7,1) = [  Inf,  Inf,  2,  2,  2,  2,  2 ];
      
      if ( ~ismember( xtype( 2 ), { 'w', 'z' } ) )
         cneighbours{2} = {{ 'mixed', 'exotic', cx{3}, cx{4}, cx{5}, cx{6}, cx{7} }};
         xtypes{2}      = 'skcrccc';
         xlowers(1:7,2) = [ -Inf, -Inf, -2, -2, -2, -2, -2 ];
         xuppers(1:7,2) = [  Inf,  Inf,  3,  2,  3,  3,  3 ];
      end
end

return

end
