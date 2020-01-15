%  BFO&BFOSS: examples to use interpolation-based search steps also in
%  combination with the exploitation of the partial separable structure
%
%  Ph. Toint, M. Porcelli 15 I 2020.  
%

addpath ./test_problems

%   Set proper system utility strings for you OS
%   Default: Unix System 

%COPY = 'cp';    %copy 
DELETE = 'rm';  %del
%MAKE_DIRECTORY = 'mkdir';

%   Clean up remains of previous runs.

if ( exist( 'test_bfo_examples_with_bfoss.log' ) )
   system( [DELETE, ' test_bfo_examples_with_bfoss.log'] );
end

%   Define the name of the result file.

diary( 'test_bfo_examples_with_bfoss.log' );

% Set the problem objective function (in sum form in this case) and 
% its partial separable structure (if present)

name = 'broyden3d';
n = 20;  %  problem full dimension

np2      = n + 2;
eldom    = cell( 1, n );
f        = cell( 1, n );
x0       = -ones( np2, 1 );

x0(1)    = 0;
x0(np2)  = 0;
xtype    = '';
xtype(1)     = 'f';
xtype(2:n+1) = 'c';
xtype( np2 ) = 'f';
for i = 1:n
    f{ i }     = @broyden3d;
    eldom{ i } = [ i i+1 i+2 ];
end
ef   = @(x)sumfi( x, eldom, f );

% set the bfo level of verbosity
verb_bfo = 'silent';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%             
% call (simple) bfo 

variant1 = 'bfo';

[ x, fxs1, msg, wrn, nevals1 ] =                                        ... 
          bfo( ef, x0, 'maxeval', 10000, 'verbosity', verb_bfo,         ...
               'reset-random-seed','no-reset');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% call bfo & exploit partial separable structure
variant2 = 'bfo&ps';

[ x, fxs2, msg, wrn, nevals2 ] =                                        ...
	      bfo( {f}, x0, 'maxeval', 10000, 'verbosity',  verb_bfo,      ...
           'eldom', {eldom}, 'reset-random-seed','no-reset' );
              
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set (some) bfoss parameters for the search step
model_mode = 'subbasis';
minimum_model_degree = 'minimal';
maximum_model_degree = 'quadratic';
ss_verbosity = 'silent';

% define the search-step function (without exploiting the partial separable
% structure)

ssfh = @( level, f, xbest, max_or_min, xincr, x_hist, f_hist, xtype,    ...
          xlower, xupper, lattice_basis )                               ...
          bfoss( level, f, xbest, 'min', xincr, x_hist, f_hist,         ...
                 xtype, xlower, xupper, lattice_basis,                  ...
                 'model-mode', model_mode,                              ...
                 'minimum-model-degree', minimum_model_degree,          ...
	             'maximum-model-degree', maximum_model_degree,          ...
                 'verbosity', ss_verbosity );

% call bfo with the search-step provided by bfoss
variant3 = 'bfo&bfoss';

[ x, fxs3, msg, wrn, nevals3 ] =                                        ...
          bfo( ef, x0, 'maxeval', 10000, 'verbosity', verb_bfo,         ...
                'search-step', ssfh, 'l-hist', -1,                      ...
                'reset-random-seed','no-reset');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define the search-step function (exploiting the partial separable structure)

ssfh = @( level, f, xbest, max_or_min, xincr, x_hist, f_hist, xtype,    ...
          xlower, xupper, lattice_basis, el_hist )                      ...
           bfoss( level, f, xbest, 'min', xincr, x_hist,                ...
	              f_hist, xtype, xlower, xupper, lattice_basis, el_hist,...
                  'model-mode', model_mode,                             ...
                  'minimum-model-degree', minimum_model_degree,         ...
	              'maximum-model-degree', maximum_model_degree,         ...
                  'verbosity', ss_verbosity );

% call bfo with the search-step provided by bfoss
variant4 = 'bfo&bfoss&ps';

[ x, fxs4, msg, wrn, nevals4 ] =                                        ...
	      bfo( {f}, x0, 'maxeval', 10000, 'verbosity', verb_bfo,        ...
                  'eldom', {eldom}, 'search-step', ssfh, 'l-hist', 1e4, ...
                  'reset-random-seed','no-reset' );
              
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% print out summary
fprintf( ' ------------------------------------------------------------------------ \n')
fprintf( '           variant             pbname      n  nfeval       f*  \n' );
fprintf( '  \n')
fprintf( '%18s %18s  %5d  %6d  %.8e \n', variant1, name, n, nevals1, fxs1 );
fprintf( '%18s %18s  %5d  %6d  %.8e \n', variant2, name, n, round(nevals2), fxs2 );
fprintf( '%18s %18s  %5d  %6d  %.8e \n', variant3, name, n, nevals3, fxs3 );
fprintf( '%18s %18s  %5d  %6d  %.8e \n', variant4, name, n, round(nevals4), fxs4 );
fprintf( ' ------------------------------------------------------------------------ \n')
diary off
