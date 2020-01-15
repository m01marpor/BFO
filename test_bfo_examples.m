%  BFO: test examples
%
%  Ph. Toint, M. Porcelli 15 I 2020.  
%

addpath ./test_problems

%   Set proper system utility strings for you OS
%   Default: Unix System 

COPY = 'cp';    %copy 
DELETE = 'rm';  %del
MAKE_DIRECTORY = 'mkdir';

%   Clean up remains of previous runs.

if ( exist( 'test_bfo_examples.log' ) )
   system( [DELETE, ' test_bfo_examples.log'] );
end

%   Define the name of the result file.

diary( 'test_examples.log' );

%   Set cutest = 1 if the CUTEst Matlab interface is installed
%   (HS4.SIF, KOWOSB.SIF and YFIT.SIF must be availabe in your $MASTSIF)

cutest = 0;  

%   0. Minimize the hyperbolic cosine over x (starting from 1) 

[ x, fx ] = bfo( @cosh, 1 )

%   0.1 Minimize an objective functions depending on parameters as, e.g. 
%     if the function myobj is specified by
%     function fx = myobj( x, p )
%
%      fx = x' * p + 0.5 * x' * x;
%
%     then minizing this function over x  for p = [ 1 1 1 1 1 ] starting 
%     from the origin

x0 = zeros( 5, 1 );
p  = ones( size( x0 ) );
[ x, fx, msg, wrn, neval ] = bfo( @(x)myobj( x, p ), x0 );

%   1. Minimize the ever-famous Rosenbrock "banana valley" function (see banana.m)

[ x, fx ] = bfo( @banana, [-1.2 1] )

%   2. Minimize the banana function subject to x(1) >=0 and x(2) <=2

[ x, fx ] = bfo( @banana, [-1.2,1],'xlower',[0,-Inf],'xupper',[Inf,2] )

%   3. Minimize the banana function by limiting accuracy and maximum number of 
%      objective function evaluations:

[ x, fx ] = bfo( @banana, [-1.2 1], 'epsilon', 1e-2, 'maxeval', 50 )

%   4. Minimize the banana function, assuming that x(1) is fixed to -1.2:

[ x, fx ] = bfo( @banana, [-1.2, 1], 'xtype', 'fc' )

%   5. Minimize the banana function, assuming that x(1) can only take integer
%      values:

[ x, fx ] = bfo( @banana, [-1, 1], 'xtype', 'ic' )

%   6. Minimize the banana function, assuming that x(1) and x(2) can only move
%      along unit multiples of the (1, 1) and ( -1, 1) vectors, respectively:

[ x, fx ] = bfo( @banana, [-1, 1], 'xtype', 'ii', 'lattice-basis', [ 1 -1; 1 1 ] )

%   7. Maximize the negative of the banana function:

[ x, fx ] = bfo( @negbanana, [-1.2 1], 'max-or-min','max' )

%   8. Minimize the banana function without any printout:

[ x, fx ] = bfo( @banana, [-1.2, 1], 'verbosity', 'silent' )

%   9. Minimize the banana function with checkpointing every 10 evaluations in
%      the file 'bfo.restart'

[ x, fx ] = bfo( @banana, [-1.2, 1], 'save-freq', 10, 'restart-file', 'bfo.restart' )

%   10. Restart the minimization of the banana function after a saved
%      check-pointing run using the file 'bfo.restart':

[ x, fx ] = bfo( @banana, [-1.2, 1], 'restart', 'use', 'restart-file', 'bfo.restart' )

%   11. Train bfo on the "fruit training set" and save the resulting algorithmic 
%      parameters in the file 'fruity':

[ ~, ~, msg, wrn, ~, ~, ~, trained_parameters ] = bfo(                              ...
            'training-mode', 'train',                                               ...
            'trained-bfo-parameters', 'fruity',                                     ...
            'training-problems' ,     {@banana,     @apple,     @kiwi},             ...
            'training-problems-data', {@banana_data,@apple_data,@kiwi_data} )

%   12. Train bfo on the "fruit training set" and use the resultant trained algorithm 
%      to solve the orange problem:

[ x, fx, msg, wrn, ~, ~, ~, trained_parameters ] = bfo( @orange, [-1.2, 1],          ...
            'training-mode', 'train-and-solve',                                      ...
            'trained-bfo-parameters', 'fruity',                                      ...
            'training-problems' ,     {@banana,     @apple,     @kiwi},              ...
            'training-problems-data', {@banana_data,@apple_data,@kiwi_data} )

%   13. Solve the orange problem after having trained BFO on the "fruit training set" 
%      and having saved the resulting algorithmic parameters in the file 'fruity' 
%      (for instance by previously using the call indicated in Example 11 above):

[ x, fx ] = bfo( @orange, [-1.2, 1], 'training-mode', 'solve',                  ...
            'trained-bfo-parameters', 'fruity')

% or

[ x, fx ] = bfo( @orange, [-1.2, 1], 'trained-bfo-parameters', 'fruity')

%   14. Train BFO on the "CUTEst training set" and save the resulting algorithmic 
%      parameters in the file 'cutest.parms':

if ( cutest )  

% Warning: check bfo_cutest_data in bfo.m to verify the problem data

[ x, fx, msg, wrn, ~, ~, ~,trained_parameters ] =                               ...
       bfo( 'training-mode', 'train',                                           ...
            'trained-bfo-parameters', 'cutest.parms',                           ...
            'training-problems', {'HS4', 'YFIT', 'KOWOSB'},                     ...
            'training-problems-library', 'cutest' )

system( [DELETE, ' AUTOMAT.d ELFUN.f GROUP.f RANGE.f OUTSDIF.d EXTER.f fort.6 mcutest* cutest.parms'] );

end

%   15. Solve the problem of computing the unconstrained min-max of the function 
%      fruit_bowl(x) defined as the sum of apple(x(1),x(2)) and
%      negbanana(x(3),x(4)), where the min is taken on x(1) and x(2) and
%      the max on x(3) and x(4):

[ x, fx ] = bfo( @fruit_bowl, [ -1.2 1 -1.2 1 ], 'xlevel', [ 1 1 2 2 ] )

%   16. Solve (very inefficiently) the problem of minimizing the banana function
%      subject to the constraints 
%          $$ 0 \leq x(1) \leq 2 $$
%      and
%          $$ x(2) \leq x(1) $$


[ x, fx ] = bfo( @banana, [ 0, 0 ], 'xlevel', [ 1 2 ],                 ...
            'max-or-min', ['min';'min'],                 ...
            'xlower', [ 0, -Inf ], 'xupper', [ 2, Inf ], ...
            'variable-bounds', 'banana_vb' )

%    17. Solve a fancy problem involving mixing milk-shakes of different styles, the
%      styles being represented by categorical variables:

[ x, fx ] = bfo( @milk_shake,                                          ...         
                         {{ 'fruity', 'exotic', 0.5, 0.5, 0.1, 0.25, 0 }},     ...
                         'xtype' , 'ssccccc',                                  ...
                         'xlower', [ -Inf, -Inf, -2, -2, -2, -2, -2 ],         ...
			   'xupper', [  Inf,  Inf,  2,  2,  2,  2,  2 ],         ...
                         'cat-states', {{ {'fruity', 'mixed', 'veggy' },       ...
                                          {'homely','exotic'},'', '', '', '', '' }} );
%      (see the milk_shake_*.m files for details).

%   18. Minimize the famous (coordinate partially-separable) Broyden tridiagonal function 
%      in 4 variables:

     [ x, fx ] = bfo( {{ @broyden3d, @broyden3d, @broyden3d, @broyden3d }},    ...
                      [ 0, -1, -1, -1, -1,  0 ], 'xtype', 'fccccf',            ...
                      'eldom', {{ [ 1 2 3 ], [ 2 3 4 ], [ 3 4 5 ], [ 4 5 6 ] }} )

%      (see broyden3d.m file for details)

system( [DELETE, ' bfo.restart fruity'] );

diary off



