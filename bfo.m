%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                                                    %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                         BFO                        %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                                                    %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                A Brute Force Optimizer             %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                                                    %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%          (c) 2016, Ph. Toint and M. Porcelli       %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                                                    %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ xbest, fbest, msg, wrn, neval, f_hist, estcrit,                                 ...
           trained_parameters, training_history, s_hist, xincr ] = bfo( varargin )
	   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                              this_version = 'v 1.01'; % 26 IX 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%% BFO, a Brute-Force Optimizer
%
% A "Brute-Force Optimizer" (based on elementary refining grid search) 
% for unconstrained or bound-constrained optimization in continuous and/or 
% discrete variables. Its purpose is to search for a *local* minimizer of 
% the problem
%
% $$\min_{x}  f(x)$$
%
% where f is a function from R^n into R. The package is intended for the
% case where the number of variables (n) is small, i.e. not much larger than 10.
%
% The derivatives of f are assumed to be unavailable or inexistent (and are
% consequently never used by the algorithm). A starting point x0 must be 
% provided by the user. The components of x are allowed to vary either 
% continuously (within their bounds) or on a predefined lattice (again 
% between their bounds).  Such a lattice may for instance be given by the
% integer values. This feature allows bound-constrained mixed-integer
% nonlinear problems to be considered. However, it must be stressed that
% the algorithm merely ATTEMPTS to compute a LOCAL (not global) minimizer
% of the objective function.
%
% The algorithm proceeds by evaluating the objective function at points 
% differing from the current iterate by a positive (forward) and a 
% negative (backward) step in each variable. The corresponding stepsizes
% are computed on a grid given by varying fractions of the user-specified
% variable scales (xscale). For continuous variables, these fractions are
% decreased (yielding a finer grid) as soon as no progress can be made 
% from the current point and until the desired accuracy is reached. For
% discrete variables, the user-supplied increment may not be reduced.
%
% The value of f(x) is computed by a user-supplied function whose call is 
% given by
%
%     fx = f( x )                                   
%
% where x is the vector at which evaluation is required.  
% For example, minimizing the hyperbolic cosine over x (starting from 1)  can be 
% achieved by the call
%
%     [ x, fx ] = bfo( @cosh, 1 )
%
% Objective functions depending on parameters as in
%
%     fx = f( x, parameter1, parameter2 )
%
% may also be used. For instance, if the function myobj is specified by
% function fx = myobj( x, p )
%
%      fx = x' * p + 0.5 * x' * x;
%
% then minizing this function over x  for p = [ 1 1 1 1 1 ] starting 
% from the origin can be achieved by the statements
%
%      x0    = zeros( 5, 1 );
%      p     = ones( size( x0 ) );
%      [ x, fx, msg, wrn, neval ] = bfo( @(x,p)myobj( x, p ), x0 );
%
% See the description of the first input argument below for more details 
% on how to specify the "function_handle" for the objective function and
% the limitations/alternatives for specifying this and other parameters.
%
% The algorithm is stopped as soon as no progress can be made from the
% current iterate by taking forward and backward steps of length
% epsilon*xscale(j) for x(j) continuous (along the directions given by an 
% orthonormal basis of the continuous variables), and of length xscale(j) 
% for x(j) discrete.  This stopping criterion is strengthened by requiring 
% that no decrease can be obtained along the directions of a user-specified
% number of random orthonormal basis of the space of continuous variables
% (see the 'termination-basis' option below). Note that this may remain
% insufficient to guarantee a local minimizer if the objective function is
% not differentiable (because the set of descent directions may be 'thin').
% BFO may also be terminated at the user request.
%
% BFO also offer the possibility for the user to define his/her own function
% for searching for a better iterate, given the information (points and 
% function values) available.  This feature corresponds to the traditional
% "search-step" which is common in direct-search methods (see the 
% documentation concerning the input parameter 'search-step-function' for
% more details). Typical ways for the user to compute a better iterate within
% the search-step function may involve polynomial interpolation/regression, 
% krieging or RBF models as well as application-specific surrogate modelling
% techniques. 
%
%% Using BFO for multilevel min-max problems
%
% BFO can also be used for solving problems of the form
%
% $$\min_{S_1} \max_{S_2} \min_{S_3} f( S_1, S_2, S_3 )$$
%
% where there can be as many as 6 levels of alternating min-max 
% (or max-min or min-min or max-max), and where the S_i are disjoint sets of
% variables.  In addition, the variables in S_i may be subject to bounds,
% possibly depending themselves on the value of the variables in 
% S_1,... S_{i-1}.  Problems of this type are specified by the user by
% providing a vector xlevel whose i-th entry indicates the level of the i-th
% variable.  For instance, a value of xlevel given by
%
%    [ 1 3 2 1 3 2 ]
%
% with the default choice of minimization corresponds to the 3-levels problem
%
% $$\min_{x_1, x_4} \max_{x_3, x_6} \min_{x_2, x_5} f( x_1, ..., x_6 ).$$
%
% The sequence of minimization/maximization is either specified as alternating
% from the first level (as in our example), or by explicitly entering a vector
% of 'min' or 'max' strings for the argument 'max-or-min'. Note that it is
% assumed that the max or min problems are well-defined at every level (in that
% the objective function is bounded above or below in the relevant subspace). 
% If bounds are imposed on the set of variables S_i that depend on the value of 
% the variables S_1 to S_{i-1}, this is done by entering the ( 'keyword', value ) 
% pair ( 'variable-bounds', 'variable_bounds') and calling the associated 
% user-supplied function in the form
%
% [ xlow, xupp ] = variable_bounds( x, level, xlevel, xlower, xupper )
%
% where the new lower and upper bounds xlow and xupp are redefined, subject to
% the mentioned rule restricting dependence of bounds for variables of a set
% on those of sets at previous levels, from the current level, the level
% definition specified by the argument xlevel, the actual value of x and the
% value of the (constant) vectors xlower and xupper supplied at the call of
% BFO. In order to ensure feasibility of the returned value, BFO resets xlow 
% to the minimum of xlow and xupp, and xupp to its maximum. Note that this 
% feature allows the use BFO for solving (by *extremely* brute force!) simple
% optimization constrained problems of the type 
%
% $$\min_{x_1,x_2}  f( x_2, x_2 ).$$
%
% such that
%
% $$ g(x_1) \leq x_2 \leq h(x_1)$$
%
% where the function g(.) and h(.) are specified in the variable_bounds
% function.
%
% More detail is given in the description of the 'xlevel', 'variable-bounds'
% and 'max-or-min' arguments.
%
%% Training BFO for a specific problem class
%
% BFO is a user-trainable algorithm in the sense that it can be trained for
% improved performance on a specific class of problems.  A typical situation
% is when a user repeatedly solves problems which only differ marginally, for
% instance because they depend on a (reasonably slowly) varying data set. 
% Training consists of optimizing the internal algorithmic parameters specific
% to the BFO algorithm for best performance and comes in three "training-modes".
%
% Training BFO requires a set of training problems, that is a set of training
% objective functions and a corresponding set of associated data (starting
% point, value of bounds or variables' types). If the training set
% consists of the banana, apple and kiwi objective functions, and if
% the associated data is specified by the banana_data, apple_data and
% kiwi_data functions respectively, then training BFO on this set of functions
% is achieved by a call to BFO of the form
% 
%     [ ~, ... ,trained_parameters ] =                                          ...
%              bfo( 'training-mode',        'train',                            ...
%                   'training-problems',    {@banana,@apple,@kiwi},             ...
%                   'training-problems-data',{@banana_data,@apple_data,@kiwi_data})
%
%  (see below for a full description of the keywords related to training).
%
% Once BFO has been trained of a particular data set, its trained version
% (that is its version using the optimized internal algorithmic parameters)
% can be applied immediately to solve a further problem (in the
% 'train-and-solve' training mode)
%
%      [ xbest, ... ,trained_parameters ] = bfo( @orange, x0,                   ...
%                    'training-mode',        'train-and-solve',                 ...
%                    'training-problems' ,   {@banana,@apple,@kiwi},            ...
%                    'training-problems-data',{@banana_data,@apple_data,@kiwi_data})
%
% or applied later to one or more problems in the 'solve' training mode, with
% a call of the form
%
%      [ xbest, fbest, ... ,trained_parameters ] = bfo( @orange, x0,            ...
%                    'training-mode',          'solve',                         ...
%                    'trained-bfo-parameters' ,'trained.bfo.parameters')
%
% where 'trained.bfo.parameters' is the name of a file where the optimized
% algorithmic parameters have been saved by a previous run in modes 'train' or
% 'train-and-solve'. Again details are provided in the description of the
% keywords for training.
%
% BFO can be also trained on problems of the CUTEst library and facilities are 
% provided to use the CUTEst MATLAB interface. If the training set consists
% of the CUTEst problems HS4, YFIT and KOWOSB, then training BFO on this set of 
% functions is achieved by a call to BFO of the form
% 
%      [ ~, ... ,trained_parameters ] =                                         ...
%               bfo( 'training-mode', 'train',                                  ...
%                    'training-problems-library', 'cutest',                     ...
%                    'training-problems' , { 'HS4', 'YFIT', 'KOWOSB' } )
%
% The parameter 'training-problem-data' should not be set. Data for the CUTEst
% problems is transfered from CUTEst to BFO (and can possibly be modified by the 
% user) in the BFO function bfo_cutest_data.m (at the end of this file).

%%   Description of the INPUT parameters

%   PRELIMINARIES
%
%   Some input arguments of BFO require the specification of MATLAB functions
%   (for the objective function, the training objective functions, the training
%   data functions and the variable_bounds function, see below).  If the user
%   wishes to specify a function whose calling sequence is "fx = func( x )" 
%   and for which a file func.m exists in the MATLAB path, this can be done 
%   in two different ways:
%   1) by a function handle: in this case the user must pass the function 
%               handle "@func" or "@(x)func(x)".  If one wishes to to allow
%               func to depend on x but also on additional parameters, with 
%               the calling sequence "fx=func(x,parameter1,parameter2)", say, 
%               then the argument becomes "@(x)func(x,parameter1,parameter2)",
%               and the values of the parameters are then automatically passed
%               to func when it is called within BFO (provided they have been
%               properly assigned values in the calling program).
%   2) by a string: in this case, the user must pass the string 'func' or
%               '@(x)func(x)', or, if parameters are present, the string
%               '@(x)func(x,parameter1,parameter2)', with the same result as
%               that described for function handles.
%   Note that BFO will return an error message if the file "func.m" cannot be
%   found in the MATLAB path. User-supplied routines may not have names
%   starting with the four characters 'bfo_'.
%
%   The first two input arguments for BFO are either specified together and in
%   the given order, or are omitted entirely (when BFO is used in training mode 
%   only). All subsequent optional inputs are specified by passing a pair
%   ('keyword', value') to BFO, where the keyword defines the nature of the 
%   input and the value is the input value itself. Thus a call to BFO has the
%   typical form
%
%        [ outputs ] = bfo( objective-function, starting point, ...
%                          'keyword', value, ..., 'keyword', value)
%
%   where the sequence "'keyword', value, ..., 'keyword', value" may be empty,
%   or the objective-function handle/string and the starting point may be omitted 
%   in the 'train' training mode. The ( 'keyword', value ) pairs may be 
%   specified by the user in any order.
%

%   INPUT  (optional) :
%
%   We start by describing the first two arguments.
%
%   f         : a function handle or string specifying the function to be minimized.
%               NOTE: If the function f returns a value
%                     NaN: this is interpreted to mean that the function f is
%                          undefined at x (and is then not considered as a
%                          potential solution),
%                     above 1.0e25 (when maximizing) or below -1.0e25 (when minimizing):
%                           this is interpreted as an instruction given by the user 
%                           to terminate optimization. 
%               NOTE: Specifying f is mandatory in the 'train-and-solve' and 
%                     'solve' training modes.
%               NOTE: On restart, it is the responsibility of the user to 
%                     provide an objective function which is coherent
%                     with that used for the call where the restart
%                     information was saved.
%   x0        : a vector containing the starting point for the  minimization
%               (the dimension n of the space is derived from the length of x0). 
%               NOTE: specifying x0 is mandatory in the 'train-and-solve' 
%                     and 'solve' training modes.
%
%   We now describe the possible ( 'keyword', value ) pairs by considering each
%   such possible keyword and specifying the meaning and format of its associated 
%   value. Remember that the ( 'keyword', value ) pairs may be specified by the 
%   user in any order.
%
%   max-or-min: a string defining whether the objective function must be
%               maximized or minimized.  Possible values are:
%               'max' : maximization is requested,
%               'min' : minimization is requested.
%               If the parameter 'xlevel' is specified (requiring a multilevel
%               computation, see below), two different types of input for the
%               'max-or-min' argument are possible:
%               1) the argument contains a single string (as described above)
%                  which defines the type of optimization at level 1. The 
%                  sequence of minimization/maximization in successive levels 
%                  is defined as alternating from level 1 on.
%               2) the argument contains an array of strings with as many
%                  strings of the form 'max' or 'min' as they are levels 
%                  (defined by the 'xlevel' argument), the j-th string then
%                  specifying the type of optimization at level j.
%               Default: 'min'
%   f-call-type: the type of calling sequence used for computing the value
%               of f(x) by the user-supplied function.  The following types
%               are available:
%               'simple'     : the call for f(x) is given by fx = f( x ),
%               'with-bound' : the call for f(x) is given by fx = f( x, fbound ),
%                              where fbound is a value supplied by the algorithm
%                              such that the evaluation of f(x) will be judged
%                              unsuccessful if f(x) >= fbound (for minimization)
%                              or f(x) <= fbound (for maximization). 
%               The latter option may be useful for instance when f(x) is
%               computed by summing nonnegative terms (as in least-squares
%               calculations) because the evaluation itself may then be
%               stopped as soon as the (incomplete) sum exceeds fbound, saving
%               the computation of the remaining terms.
%               NOTE: The user my specify an initial bound of f(x) using the
%                     'f-bound' keyword, in which case the value supplied by
%                     the algorithm will never exceed this bound.
%               NOTE: f-call-type is NOT saved for subsequent restart.
%               NOTE: parameters can be used within the 'with-bound' option, 
%                     but they should be specified in function handle
%                     defining f(x) in the call to BFO if this option is used.
%               Default: 'simple'
%   f-bound   : a real number specifying a bound on the objective function value
%               above which (for minimization) or under which (for maximization)
%               any function evaluation in the course of the algorithm will be
%               considered unsuccessful (see the description of the 'f-call-type'
%               keyword above).
%               Default: +Inf (minimization), -Inf (maximization).
%   xlower    : the vector of size n containing the lower bounds on the 
%               problem's variables.
%               NOTE: the starting point is projected onto the bound-feasible
%                     set before minimization is actually started.
%               NOTE: lower bounds may be equal to -Inf, but these will
%                     be converted to -1.e25.
%               NOTE: a single value may be specified if all variables
%                      have the same lower bound ( ex. xlower = [ 0 ] indicates
%                      that all variables are nonnegative).
%               Default: -1.e25
%   xupper    : the vector of size n containing the upper bounds on the problem's 
%               variables
%               NOTE: the starting point is projected onto the bound-feasible
%                     set before minimization is actually started.
%               NOTE: upper bounds may be equal to +Inf, but these will
%                     be converted to 1.e25.
%               NOTE: a single value may be specified if all variables
%                     have the same upper bound ( ex. xupper = [ 1 ] indicates
%                     that all variables are bounded above by 1).
%               Default: +1.e25
%   xscale    : a strictly positive vector of size n giving the variables' relative
%               scaling. The scaling of a continuous variable is its typical
%               order of magnitude (so that a move of xscale( j ) is approximately of
%               of the correct order for a change in variable j). The scaling
%               of a discrete variable is the size of the smallest move
%               acceptable for this variable (for instance 1 for an integer variable). 
%               If a single number xscale is specified, then the uniform 
%               scaling xscale * ones(1,n) is used.  For well-scaled problems, 
%               use xscale = 1.
%               NOTE: the value(s) of xscale influence both the initial increments
%                     (because they are given by delta( j ) * xscale( j )) and the
%                     termination which occurs when no decrease in objective
%                     can be achieved for changes in variables of size 
%                     epsilon * xscale( j ).
%               Default: ones(1;n)
%   delta     : the scaling-independent initial (positive) stepsizes for the variables,
%               such that the initial move along the continuous variable i is 
%               delta( i ) * xscale( i ). If a single number delta is specified,
%               the value delta( i ) = delta is used for all continuous variables
%               and delta( i ) = 1 for all discrete ones.
%               Default: 2.64133
%   lattice-basis : a matrix of size nd by nd (where nd is the number of
%               discrete variables), whose columns span the lattice on 
%               which minimization on discrete variables must be carried out.
%               When this matrix is specified, minimization on the i-th
%               discrete variable is interpreted as minimization along fixed
%               multiples of the i-th column of the given matrix. 
%               Default: the identity matrix of size nd by nd.
%   xtype     : a string of length n, defining the type of the variables:
%               xtype(j) = 'c'  if variable j is continuous,
%               xtype(j) = 'i' if variable j is discrete in that it can
%                              only vary by multiples of xscale(j), possibly
%                              along a predefined lattice (see 'lattice-basis'),
%               xtype(j) = 'f'  if variable j is fixed.
%               NOTE: a string of length 1 may be specified if all variables
%                     are of the same type ( ex. xtype = 'c' indicates that
%                     all variables are continuous).
%               NOTE: a typical use of discrete variables is to specify 
%                     variables whose values must be integers.  This is 
%                     achieved by declaring the variable to be discrete 
%                     (xtype(j) = 'i'), its scale to be 1 (xscale(j)=1) 
%                     and its initial value to be an integer.
%               NOTE: see the description of the argument 'training-mode' for
%                     the use of xtype to specify the class of problems for
%                     which training is desired.
%               NOTE: additional types of variables are defined in the code
%                     but their use is restricted.
%               Default: 'c'
%   epsilon   : a real number specifying the accuracy level defining the 
%               termination rule for continuous variables. The algorithm is 
%               terminated when no objective function decrease can be obtained
%               for changes in variables of size epsilon * xscale( j ). 
%               Default: 0.0001
%   bfgs-finish : a meshsize under which BFO attempts to apply the BFGS
%               quasi-Newton formula to compute a good descent direction
%               whenever a full estimate of the gradient is available. Using
%               this option for small mesh-sizes (below 0.01 down to 
%               0.1* epsilon) usually results in significantly more accurate 
%               solutions, although at an increased cost in function evaluations.
%               For some smooth problems, this may be a useful alternative to
%               decreasing epsilon.
%               Default: 0 (turned off)
%   f-target  : the value of the user's objective function target, in the
%               sense that the optimization is terminated as soon as the
%               objective function value is less or equal than the argument given
%               (for minimization), or as soon as the objective function value
%               exceeds the argument (for maximization).  If multilevel
%               optimization is requested, this termination rule is applied 
%               to the optimization at level 1.
%               Default: +/-Inf
%   termination-basis : the number (>1) of successive random choices of an
%               orthonormal basis of the continuous variables used to
%               assess termination, in the sense that no decrease can be 
%               obtained on the finest grid for the continuous variables 
%               using any of the directions in any of these basis.
%               NOTE: If the objective function is known to be nonsmooth,
%                     a relatively high number (> 50) is recommended, because
%                     the set of improvement directions at non-optimal point  
%                     may be very "thin".   
%               Default: 5.
%   maxeval   : the maximum number of objective function's evaluations
%               for the current run (it is remembered from previous calls if
%               restart is being used).
%               Default: 5000 * n
%   verbosity : the volume of printed output produced by the algorithm:
%               'silent' : no output,
%               'minimal': warnings and error messages only,
%               'low'    : a one-line summary per iteration, 
%               'medium' : a one-line summary per iteration + a summary of x
%               'high'   : more detail
%               'debug'  : for developers'use only :-).
%               If multilevel optimization is requested, the verbosity of each
%               optimization level may be set independently by specifying 
%               a cell whose length is equal to the number of levels and whose
%               elements are strings describing verbosity levels following the
%               above convention.
%               Default: 'low'
%   random-seed : an positive integer specifying the seed for the random
%               number generator rng( seed, 'twister') which is used to
%               initialize random sequences at the beginning of execution.
%               Random numbers are used for the choice of alternative basis
%               vectors for the continuous variables, both when refining 
%               the grid and when checking termination.
%               Default: 0 (MATLAB default)
%   reset-random-seed : a string whose meaning is
%               'reset'    : the random number generator is reinitialized by BFO,
%               'no-reset' : the random number generator is not reinitialized by BFO.
%               NOTE : The random seed is not reinitialized on restart.
%               NOTE : The random seed is always reinitialized during the training 
%                      process
%               Default: 'reset'
%   search-step_function: a function handle or string specifying the name of a
%               user-supplied search-step function.  
%               If the argument is supplied, a search-step is carried on at 
%               every major iteration in BFO and the function specified must be
%               available in the Matlab path.  Its calling sequence is given by
%                  [ xsearch, fsearch, nevalss ] =                           ...
%                          search_step_function( f, x_hist, f_hist, xtype,   ...
%                                                xlower, xupper, lattice_basis )
%               where, on input :
%                  f       : is the handle to the objective function,
%                  x_hist  : is an (n x neval) array whose columns contain the
%                            points at which f(x) has been evaluated so far,
%                  f_hist  : is an array of length neval containing the function 
%                            values associated with the columns of x_hist,
%                  xtype   : is the status of the variables (see above)
%                  lattice_basis : is the lattice associated with the discrete 
%                            variables, if any (or the empty array otherwise)
%               and, on output,
%                  xsearch : is an array of length n containing the point returned
%                            by the user as a tentative improved iterate,
%                  fsearch : the associated objective function value f(xsearch),
%                  nevalss : the number of function evaluations performed internally 
%                            by the search-step function.
%               NOTE: It is the responsability of the user to ensure that the
%                     search-step function's interface conforms to the above.
%               NOTE: the point returned in xsearch must preserve the type of the 
%                     variables (as specified by xtype) and satisfy the bound and 
%                     lattice constraints, if relevant.  Moreover, if 
%                     xtype(i) = 'f', then xsearch(i) must be equal to x_hist(i,end), 
%                     the i-th component of the point given in the last column of x_hist.
%               NOTE: Search steps are disabled during in training mode
%               Default: none.
%   xlevel    : an array of positive integers of size n, where xlevel(i) is 
%               the index of the level to which the i-th variable is
%               associated. The number of levels i computed as 
%               nlevel = max( xlevel ) and may not exceed 6. Each variable 
%               must be assigned a level between 1 and nlevel, and each level
%               between 1 and nlevel must be associated with at least one
%               variable. 
%               Default= ones(size(x))
%   variable-bounds : a string or function handle specifying the user-supplied 
%               function computing bounds on variables of a level as a function
%               of the value of the variables at previous (i.e. of lower index)
%               levels. If the keyword is specified, the user must supply a function
%                  [ xlow, xupp ] = variable_bounds( x, xlevel, xlower, xupper )
%               where the subset of the vectors corresponding to level i may
%               be recomputed from the values of x, xlower and xupper (as
%               specified on calling BFO) under the constraint that they may
%               only depend of the values of the  variables associated with 
%               levels 1 to i-1.
%               Default: none. A valid function handle must be provided if the
%                        'variable-bounds' keyword is specified.
%   restart   : defines the strategy used for possibly restarting the 
%               algorithm
%               'none' means that restart is not allowed (the file
%                      bfo.restart or the option specified in restart-file 
%                      is then ignored)
%               'use'  means that the algorithm is restarted from the
%                      information saved in bfo.restart if this file
%                      exists, or, if the option 'restart-file' is used, in
%                      the file specified by this option (an error is 
%                      generated otherwise).
%               NOTE:  Restarted calls MUST specify the same objective function
%                      as that used in the call at which restart information was 
%                      saved. 
%               NOTE:  if a multilevel computation is required, restart is
%                      only available for the first level optimization.
%               NOTE:  during training, restart can be invoked at the level of the
%                      training process itself (i.e. within the 'average' or 'robust'
%                      optimization process).  Restart at a lower level (individual
%                      test problems) is not available.
%               NOTE:  if restart is used after a breakdown in training, the user
%                      is free to specify any training mode.  In the 'train' and
%                      'train-and-solve' modes,  the training process will be 
%                      restarted to completion and possibly (in the 'train-and-solve'
%                      mode) followed by the optimization of the function specified
%                      as first argument with optimized parameters. In the 'solve'
%                      mode, the unfinished training is abandoned and the default
%                      parameters are used for optimizing the function specified
%                      as first argument.
%               Default: 'none'
%   save-freq : an integer giving the frequency at which information is saved
%               in the file bfo.restart (or the file specified by the
%               restart-file option) for possibly restarting the algorithm.
%               Possible values are:
%               save-freq < 0 : no information is ever saved,
%               save-freq = 0 : information is saved at termination of the 
%                               algorithm
%               save-freq > 0 : information is saved every after every 
%                               save-freq-th evaluation of f(x) and at 
%                               termination
%               Default: -1
%   restart-file : the name of the file on which restart information must be 
%               written and/or read.
%               NOTE: This parameter is NOT saved in the restart information
%                     and must be re-specified (if different from its
%                     default value) at each restarted call.
%               NOTE: If information is saved during training, two files are 
%                     created by BFO:
%                     'restart-file' : contains the restart information related
%                                      to the last parameter optimization
%                                      iteration;
%                     'restart-file'.training : contains the restart information
%                                      related to the training process itself.
%               Default: bfo.restart
%   training-mode: a string which defines the BFO training mode:
%               'solve' : BFO solves a single problem, possibly using
%                         previously optimized algorithmic parameters,
%               'train' : BFO trains its algorithmic parameters on a set on problems 
%                         and saves the optimized parameters in a file (as well as
%                         returning them to the user in trained_parameters).
%               'train-and-solve': BFO trains its parameters on a set on problems
%                         and solves a problem with the trained parameters.
%               NOTE: training is performed by optimizing the algorithmic parameters for
%                     one of two classes of problems: mixed-integer or purely continuous.
%                     This class is specified by the type of problem to solve in the 
%                     'train-and-solve' mode.  In the 'train' mode (where no problem is
%                     specified), the class is determined by the input vector xtype.  By
%                     default, the class chosen is that of purely continuous problems.
%                     This can be changed to mixed-integer by specifying xtype = 'i'
%                     (or any vector xtype containing at least one 'i' entry). The default 
%                     choice of the class of continuous problem is equivalent to 
%                     specifying a xtype vector containing no 'i' entry.
%               NOTE: the objective function and starting point should NOT be
%                     specified (as the first two arguements) in the 'train' training mode.
%               Default: 'solve'
%   training-problems-library : a string which defines the library of BFO training
%                problems.  Possible values are:
%               'user-defined': the set of training problems is created by the user.
%               'cutest'      : training problems are chosen in the CUTEst library 
%                               and the CUTEst MATLAB interface is used.
%               NOTE: meaningless in "solve" training mode.
%               Default: 'user-defined'
%   training-problems : a cell of function handles or strings specifying the 
%               objective functions to be used to train BFO. It may have the form
%               {@f1,@f2,...} or {'f1','f2',...} with fi of the form f = fi(x) .
%               If training-problems-library = 'cutest', only the second form is
%               possible, where each 'fi' is the name of a CUTEst problem.
%               NOTE: training-problems is mandatory in 'train' and
%                     'train-and-solve' training modes.
%               NOTE: meaningless in "solve" training mode.
%   training-problems-data : a cell of function handles or strings specifying the
%               problems data function names of the form 
%               {@f1_data,@f2_data,...} or {'f1_data','f2_data',...} with each 
%               function fi_data being of the form 
%                  x0 = fi_data 
%               or
%                  [ x0, xlower, xupper] = fi_data 
%               or
%                  [ x0, xlower, xupper, xtype] = fi_data 
%               or
%                  [ x0, xlower, xupper, xtype, xscale] = fi_data 
%               or
%                  [ x0, xlower, xupper, xtype, xscale, max-or-min ] = fi_data 
%               or 
%                  [ x0, xlower, xupper, xtype, xscale, max-or-min, xlevel ] = fi_data 
%               or
%                  [ x0, xlower, xupper, xtype, xscale, max-or-min, xlevel, ...
%                         variable_bounds ] = fi_data 
%               where x0, xlower, xupper, xtype, xscale, max-or-min, xlevel
%               and the variable_bounds string or function handle are defined 
%               as above.
%               If the length of training-problems-data is equal to 1, it is
%               assumed that all training problems share the same data function.
%               NOTE: training-problems-data is mandatory in train and
%                     train-and-solve training mode, and must be of the same
%                     size as the training-problems argument.
%               NOTE: training-problems-data is ignored if 
%                     training-problems-library = 'cutest'.
%               NOTE: meaningless in "solve" training mode.
%   training-strategy: a string which defines the BFO training strategy:
%               'average': consider optimizing the average number of function
%                          evaluations per function on the training set,
%               'robust' : consider optimizing the worst average number of
%                          function evaluations in a box defined by the product 
%                          of intervals where each parameter is allowed to
%                          deviate from its nominal value by at most 5%.
%               NOTE: meaningless in "solve" training mode.
%               Default: 'average'
%   training-parameters : cell of strings containing the names of the BFO
%               parameters to be trained (in training-mode 'train' or
%               'train-and-solve'). Possible value are 'alpha', 'beta',
%               'gamma', 'delta', 'eta', zeta', 'inertia', 'search-type'
%               and 'random-seed' (see the description of these input
%               parameters for details on their nature).
%               NOTE: training on the parameter 'search-type' only makes sense
%                     for problems involving discrete variables.
%               NOTE: training on the parameter 'random-seed' may be useful
%                     for a very coherent training set, but may be 
%                     computationally intensive.
%               NOTE: meaningless in "solve" training mode.
%               Default: { 'alpha', 'beta', 'gamma', 'delta', 'eta', 'inertia' }
%   trained-bfo-parameters: a string containing the name of the file where
%               trained BFO parameters are written (in 'train' and
%               'train-and-solve' modes) or read (in 'solve' mode).
%               NOTE: BFO will read trained parameters in 'solve'
%                     training-mode as soon as this parameter is specified.
%               NOTE: meaningless in "solve" training mode.
%               Default: 'trained.bfo.parameters'
%   training-epsilon: a vector containing two real numbers giving the accuracy 
%               levels defining the termination rules for continuous parameters'
%               training (i.e. the optimization of the algorithmic parameters). 
%               The first is used for stopping the outer minimization of the
%               training procedure, and the second (only relevant for the 
%               'robust' training strategy) is used to terminate the inner
%               maximization procedure.
%               NOTE: If a single number is supplied by the user, BFO expands
%                     it to a vector with two identical entries.
%               NOTE: meaningless in "solve" training mode.
%               Default: 0.01
%   training-maxeval: a vector containing two integers defing the maximum number
%               of parameters' combinations that can be tested in training (i.e. 
%               in the optimization of the algorithmic parameters). 
%               The first is used for stopping the outer minimization of the
%               training procedure, and the second (only relevant for the 
%               'robust' training strategy) is used to terminate the inner
%               maximization procedure.
%               NOTE: If a single number is supplied by the user, BFO expands
%                     it to a vector with two identical entries.
%               NOTE: meaningless in "solve" training mode.
%               Default: [ 200, 100 ]
%   training-verbosity : a cell containing two strings specifying the verbosities
%               of the training process (i.e. in the optimization of the algorithmic
%               parameters), using the same  values as the 'verbosity' parameter 
%               described above.  When the 'average' training strategy is used,
%               only the first element is used.  When the 'robust' strategy is,
%               used, the first element describes the verbosity of the outer 
%               optimization (minimization) and the second that of the inner
%               optimization (maximization to determine the worst case in the box).
%               NOTE: If a single string is supplied by the user, BFO expands
%                     it to a cell with two identical string entries.
%               NOTE: meaningless in "solve" training mode.
%               Default : identical to 'verbosity'.
%   training-problem-epsilon: a real number giving the accuracy level defining
%               the termination rule for the optimization of each test problem
%               during training.
%               NOTE: meaningless in "solve" training mode.
%               Default: 0.0001
%   training-problem-maxeval: an integer defining  the maximum number of 
%               objective function's evaluations during the solution of each
%               test problem during training.
%               NOTE: meaningless in "solve" training mode.
%               Default: 5000
%   training-problem-verbosity : the verbosity of the optimization of each 
%               training problem during the training process, using the same 
%               values as the 'verbosity' parameter described above.
%               NOTE: meaningless in "solve" training mode.
%               Default : 'silent'
%
%
%   INPUT (optional and more esoteric: don't change unless you understand.  
%          These are internal BFO parameters that can be optimized by training) ;
%
%
%   alpha     : the grid expansion factor at successful iterations (>= 1)
%               Default: 2.377
%   beta      : a fraction ( in (0,1) ) defining the shrinking ratio between
%               successive grids for the continuous variables
%               Default: 0.08
%   gamma     : the maximum factor ( >= 1 ) by which the initial user-supplied
%               grid for continuous variables may be expanded
%               Default: 3.95
%   eta       : a fraction ( > 0 ) defining the decrease in objective function
%               deemed sufficient to stop polling the remaining variables, this
%               decrease being computed as eta times the decrease obtained at 
%               the last step where a complete polling over all variables has 
%               been performed.
%               Default: 1.e-7
%   zeta      : a factor (>=1) by which the grid size is expanded when a
%               particular level (in multilevel use) is re-explored after a
%               previous optimization.
%               Default: 3
%   inertia   : the number of iterations used for averaging the steps in the
%               continuous variables, the basis for these variables being
%               computed for the next iteration as an orthonormal basis whose
%               first element is the (normalized) average step.
%               NOTE: inertia = 0 disables the averaging process and the
%                     polling direction are the coordinate vectors for all 
%                     iterations.
%               Default: 15
%   search-type : a string defining the strategy to use for exploring the tree
%               of possible values for the discrete variables.  Possible values
%               are:
%               'breadth-first' : all subspaces corresponding to interesting 
%                                 values of the discrete variables are explored 
%                                 before grid refinement
%               'depth-first'   : grid refinement is performed as soon as
%                                 possible
%               'none'          : no recursion
%               Default: 'depth-first'
%
%
%   INPUT (reserved: definitely don't interfere) :
%
%
%   fevals_hist : the vector of computed function values 
%               NOTE: this is automatically specified during the algorithm 
%                     and should NOT be specified/altered by the user.
%   sspace_hist : a list of the best point found on explored discrete
%               subspaces, together with associated function values 
%               and grid spacings
%               NOTE: This is automatically specified during the algorithm 
%                     and should NOT be specified/altered by the user.
%   rpath     : the current list of discrete variables already fixed within 
%               the recursion
%               NOTE: this is automatically specified during the algorithm 
%                     and should NOT be specified/altered by the user.
%   nevr      : the current number of function evaluations
%               NOTE: this is automatically specified during the algorithm 
%                     and should NOT be specified/altered by the user.
%   level     : the current level in a multilevel computation.
%               NOTE: this is automatically specified during the algorithm 
%                     and should NOT be specified/altered by the user.
%   xincr     : the current (multilevel) grid spacing.
%               NOTE: this is automatically specified during the algorithm 
%                     and should NOT be specified/altered by the user.

%%  Description of the OUTPUT parameters

%
%   xbest     : the best approximation to a minimizer/maximizer found by the
%               algorithm
%               NOTE: meaningless in 'train' training mode.
%               NOTE: It may happen, when a lattice is specified by the user
%                     for the optimization od discrete variables, that no feasible
%                     solution (wrt bounds) can be found by BFO in the 
%                     neighbourhood of x0, even if the bounds are consistent
%                     (i.e. xlower <= xupper).  In that case  the message 
%                        ' BFO error: no feasible discrete point found on the 
%                        lattice near x0. Terminating.'
%                     is output and xbest is the solution found which locally 
%                     minimizes the ell-1 norm of the infeasibilities.
%   fbest     : the value of the objective function at xbest
%               NOTE: meaningless in 'train' training mode.
%               NOTE: It may happen, when a lattice is specified by the user
%                     for the optimization od discrete variables, that no feasible
%                     solution (wrt bounds) can be found by BFO in the 
%                     neighbourhood of x0, even if the bounds are consistent 
%                     (i.e. xlower <= xupper).  In that case the message 
%                        ' BFO error: no feasible discrete point found on the 
%                        lattice near x0. Terminating.'
%                     is output and fbest is local minimum of the ell-1 norm 
%                     of the infeasibilities.
%   msg       : a short informative message about the execution of BFO
%   wrn       : a short informative warning if appropriate
%   neval     : the total number of objective function evaluations required
%               by the algorithm
%   f_hist    : a vector containing the history of all function values computed 
%               in the solve phase (useful to make performance profiles).
%   estcrit   : an estimated criticality measure (for the continuous variables)
%               at xbest.  For unconstrained problems, this is the norm of g, 
%               a central difference approximation of the gradient on the last
%               grid. For bound-constrained problems, this is the norm of the 
%               vector P(x-g)-x, where P is the orthogonal projection on the 
%               feasible set.
%               NOTE: meaningless in 'train' training mode.
%   trained_parameters : a vector of 10 values corresponding to the optimized
%               BFO internal parameters resulting from training:
%               trained_parameters(1)  is the optimized value of alpha
%               trained_parameters(2)  is the optimized value of beta
%               trained_parameters(3)  is the optimized value of gamma
%               trained_parameters(4)  is the optimized value of delta
%               trained_parameters(5)  is the optimized value of eta
%               trained_parameters(6)  is the optimized value of zeta
%               trained_parameters(7)  is the optimized value of inertia
%               trained_parameters(8)  is the optimized value of search-type
%               trained_parameters(9)  is the optimized value of random-seed
%               This vector is empty on output of a call to BFO in 'solve' 
%               training mode.
%   training_history: a matrix containing the history of the training process.
%               Each row corresponds to an iteration of the underlying (average
%               or robust) optimization.  The content of each column is as
%               follows:
%               column 1 : 1 if average, 2 if robust
%               column 2 : the index of the iteration in the optimization process
%               column 3 : the total number of problem evaluations so far
%               column 4 : the best performance so far
%               column 5 : the corresponding value of alpha
%               column 6 : the corresponding value of beta
%               column 7 : the corresponding value of gamma
%               column 8 : the corresponding value of delta
%               column 9 : the corresponding value of eta
%               column 10: the corresponding value of zeta
%               column 11: the corresponding value of inertia
%               column 12: the corresponding value of search-type
%                          (1 = depth-first, 0 = breadth-first, 2 = none)
%               column 13: the corresponding value of rseed
%               This matrix is empty on output of a call to BFO in 'solve' 
%               training mode.
%               NOTE: trained_parameters = training_history( end, 5:13 )
%   s_hist    : a list of the best point found on explored discrete subspaces
%               NOTE: this is automatically specified during the algorithm 
%                     recursion. It is useless as an output for the user
%                     and may safely be ignored in the calling statement
%                     if no restart is desired (see examples below).
%   xincr     : the vector of (multilevel) grid spacings.
%               NOTE: this is automatically specified during the algorithm 
%                     recursion. It is useless as an output for the user.

%%  Information

%   SOURCE:     a personal and possibly misguided reinterpretation of a talk 
%               given by a student of D. Orban and Ch. Audet on using the 
%               NOMAD algorithm for optimizing code parameters (Optimization 
%               Days, May 2009), plus some other ideas.
%
%   REFERENCES: M. Porcelli and Ph. L. Toint,
%               "BFO, a trainable derivative-free Brute Force Optimizer for 
%               nonlinear bound-constrained optimization and equilibrium 
%               computations with continuous and discrete variables",
%               Report naXys-06-2015, University of Namur (Belgium), 2015.
%
%               N. I. M. Gould, D. Orban and Ph. L. Toint,
%               "CUTEst: a Constrained and Unconstrained Testing Environment
%               with safe threads", Computational Optimization and Applications,
%               Volume 60, Issue 3, pp. 545-557, 2015.

%   PROGRAMMING: Ph. Toint, M. Porcelli, from May 2010 on.
%
%   DEPENDENCIES (internal): bfo_save, bfo_restore, bfo_print_summary_vector, 
%               bfo_print_vector, bfo_print_x, bfo_print_cell, bfo_print_matrix,
%               bfo_histupd, bfo_next_level_objf, bfo_average_perf, bfo_robust_perf,
%               bfo_save_training, bfo_restore_training; bfo_get_verbosity,
%               bfo_exist_function, bfo_feasible_cstep, bfo_cutest_data, 
%               bfo_new_continuous_basis

%%  Examples of use

%   1. Minimize the ever-famous Rosenbrock "banana valley" function with
%        [ x, fx ] = bfo( @banana, [-1.2 1] )
%     where
%        function fx = banana( x )
%        fx  = 100 * ( x(2) - x(1)^2 ) ^2 + (1-x(1))^2;
%
%   2. Minimize the banana function subject to x(1) >=0 and x(2) <= 2:
%        [ x, fx ] =  bfo( @banana, [-1.2,1],'xlower',[0,-Inf],'xupper',[Inf,2] )
% 
%   3. Minimize the banana function by limiting accuracy and maximum number of 
%      objective function evaluations:
%        [ x, fx ] = ...
%                 bfo( @banana, [-1.2 1], 'epsilon', 1e-2, 'maxeval', 50 )
%
%   4. Minimize the banana function, assuming that x(1) is fixed to -1.2:
%        [ x, fx ] =  bfo( @banana, [-1.2, 1], 'xtype', 'fc' )
%
%   5. Minimize the banana function, assuming that x(1) can only take integer
%      values:
%        [ x, fx ] =  bfo( @banana, [-1, 1], 'xtype', 'ic' )
%
%   6. Minimize the banana function, assuming that x(1) and x(2) can only move
%      along unit multiples of the (1, 1) and ( -1, 1) vectors, respectively:
%        [ x, fx ] =  bfo( @banana, [-1, 1], 'xtype', 'ii',...
%                          'lattice-basis', [ 1 -1; 1 1 ] )
%
%   7. Maximize the negative of the banana function:
%        [ x, fx ] = bfo( @negbanana, [-1.2 1], 'max-or-min','max' )
%     where
%        function fx = negbanana( x )
%        fx  = - 100 * ( x(2) - x(1)^2 ) ^2 - (1-x(1))^2;
%
%   8. Minimize the banana function without any printout:
%        [ x, fx ] =  bfo( @banana, [-1.2, 1], 'verbosity', 'silent' )
%
%   9. Minimize the banana function with checkpointing every 10 evaluations in
%      the file 'bfo.restart'
%        [ x, fx ] =                                                             ...
%              bfo( @banana, [-1.2, 1], 'save-freq', 10, 'restart-file', 'bfo.restart')
%
%   10. Restart the minimization of the banana function after a saved
%      check-pointing run using the file 'bfo.restart':
%        [ x, fx ] = bfo( @banana, [-1.2, 1],  'restart', 'use',                 ...
%                                  'restart-file', 'bfo.restart'  )
%
%   11. Train bfo on the "fruit training set" and save the resulting algorithmic 
%      parameters in the file 'fruity':
%      [ ~, ... ,trained_parameters ] =                                          ...
%            bfo( 'training-mode', 'train', 'trained-bfo-parameters', 'fruity',  ...
%                 'training-problems' ,    {@banana,     @apple,     @kiwi},     ...
%                 'training-problems-data',{@banana_data,@apple_data,@kiwi_data} )
%                 
%
%   12. Train bfo on the "fruit training set" and use the resultant trained algorithm 
%      to solve the orange problem:
%      [ x, ... ,trained_parameters ] = bfo( @orange, [-1.2, 1],                 ...
%                 'training-mode', 'train-and-solve',                            ...
%                 'training-problems',     {@banana,     @apple,     @kiwi},     ...
%                 'training-problems-data',{@banana_data,@apple_data,@kiwi_data} )
%
%   13. Solve the orange problem after having trained BFO on the "fruit training set" 
%      and having saved the resulting algorithmic parameters in the file 'fruity' 
%      (for instance by previously using the call indicated in Example 11 above):
%      [ x, fx ] = bfo( @orange, [-1.2, 1],                                      ...
%                 'training-mode', 'solve', 'trained-bfo-parameters' ,'fruity')
%      or 
%      [ x, fx ] = bfo( @orange, [-1.2, 1], 'trained-bfo-parameters', 'fruity')
%
%   14. Train BFO on the "CUTEst training set" and save the resulting algorithmic 
%      parameters in the file 'cutest.parms':
%      [ x, ... ,trained_parameters ] = bfo( 'training-mode', 'train',           ...
%                 'trained-bfo-parameters', 'cutest.parms',                      ...
%                 'training-problems', {'HS4', 'YFIT', 'KOWOSB'},                ...
%                 'training-problems-library', 'cutest' )
%
%   15. Solve the problem of computing the unconstrained min-max of the function 
%      fruit_bowl(x) defined as the sum of apple(x(1),x(2)) and
%      negbanana(x(3),x(4)), where the min is taken on x(1) and x(2) and
%      the max on x(3) and x(4):
%       [ x, fx ] = bfo( @fruit_bowl, [ -1.2 1 -1.2 1 ], 'xlevel', [ 1 1 2 2 ] )
%
%   16. Solve (very inefficiently) the problem of minimizing the banana function
%      subject to the constraints 
%          $$ 0 \leq x(1) \leq 2 $$
%      and
%          $$ x(2) \leq x(1) $$
%      This can be done with
%         [ x, fx ] = bfo( @banana, [ 0, 0 ], 'xlevel', [ 1 2 ],                 ...
%                          'max-or-min', ['min';'min'],                          ...
%                          'xlower', [ 0, -Inf ], 'xupper', [ 2, Inf ],          ...
%                          'variable-bounds', 'banana_vb' )
%      where the user has provided the following
%         function [xlow, xupp ] = banana_vb( x, level, xlevel, xlower, xupper )
%         xlow(1) = 0;
%         xupp(2) = x(1);
%         xupp    = xupper;
%

%%  CONDITIONS OF USE

%   Use at your own risk! No guarantee of any kind given or implied.

%   Copyright (c) 2015, 2016. Ph. Toint and M. Porcelli. All rights reserved.
%
%   Redistribution and use of the BFO package in source and binary forms, with 
%   or without modification, are permitted provided that the following conditions
%   are met:
%
%    * Redistributions of source code must retain the above copyright notice, 
%      this list of conditions and the following disclaimer.
%    * Redistributions in binary form must reproduce the above copyright notice, 
%      this list of conditions and the following disclaimer in the documentation
%      and/or other materials provided with the distribution.
%
%   +-------------------------------------------------------------------------+
%   |                                                                         |
%   |                             DISCLAIMER                                  |
%   |                                                                         |
%   |  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS    |
%   |  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOR      |
%   |  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS      |
%   |  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE         |
%   |  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,    |
%   |  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,   |
%   |  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS  |
%   |  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND |
%   |  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR  |
%   |  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE |
%   |  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH       |
%   |  DAMAGE.                                                                |
%   |                                                                         |
%   +-------------------------------------------------------------------------+

%% Argument list processing 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Argument list processing  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%  Verbosity, objective function and training mode  %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Set default (empty) return values.

xbest              = NaN;
fbest              = NaN;
msg                = 'Unexpected exit.';
wrn                = '';
neval              = 0;
f_hist             = [];
estcrit            = NaN;
s_hist             = [];
xincr              = 0;
trained_parameters = [];
training_history   = [];

%  Check that a potentially coherent argument list exists and output help 
%  or version number if requested.

noptargs = size( varargin, 2 );                  % the number of optional arguments
if ( noptargs == 0 )
   wrn = ' BFO use: [ xbest, fbest, ... ] = bfo( [ @f, x0 ], [ [''keyword'',value] ] )';
   disp( wrn );
   msg = ' BFO error: no argument specified! Terminating.';
   disp( msg );
   return
elseif ( noptargs == 1 )
   if ( ischar( varargin{ 1 } ) &&                                                         ...
        ( strcmp( varargin{ 1 }, 'help' ) || strcmp( varargin{1}, 'man' ) ||               ...
        strcmp( varargin{ 1 }, '-h' )     || strcmp( varargin{1}, '?' ) )        )
      wrn = ' BFO use: [ xbest, fbest, ... ] = bfo( [ @f, x0 ], [ [''keyword'',value] ] )';
      msg = wrn;
   elseif ( ischar( varargin{ 1 } ) && strcmp( varargin{ 1 }, 'version' ) )
      msg = [ ' BFO version: ', this_version ];
      wrn = msg;
   else
      msg = ' BFO error: the argument list is ill-constructed! Terminating.';
      wrn = msg;
   end
   disp( msg )
   return
end

%   First read the verbosity level and training mode, if any.  If
%   training-mode is 'train', then the first two arguments should not be
%   interpreted as the objective function and the starting point, even if the
%   objective function is given in string form.

verbose        = 2;                     % the default verbosity of output as an integer ...
verbosity      = 'low';                 % ... but also in string form
lverbosity     = 1;                     % a priori not a multilevel cell
user_verbosity = 0;                     % no user-defined verbosity (yet)
user_training_verbosity = 0;
train          = 0;                     % not in train mode (yet)

for i = 1:noptargs
   if ( ischar( varargin{i} ) && strcmp( varargin{i}, 'verbosity' ) )
      if ( i < noptargs )

         % Multiple verbosity specification

         if ( iscell( varargin{ i+1 } ) )
            cverbosity     = varargin{ i+1 };
            lverbosity     = length( cverbosity );
            user_verbosity = 1;
            for ii = lverbosity:-1:1
               if ( ischar( cverbosity{ ii } ) )
                  verbose = bfo_get_verbosity( cverbosity{ ii } );
                  if ( verbose < 0 )
                     user_verbosity = 0;
                     wrn = ' BFO warning: unknown verbosity level! Default used.';
                     disp( wrn )
                  end
               else
                  user_verbosity = 0;
                  wrn = ' BFO warning: verbosity level is not a string! Default used.';
                  disp( wrn )
               end
            end
            if ( user_verbosity )
               verbosity = cverbosity;
            else
               lverbosity = 1;
               verbose    = 2;  % default
            end

         % Verbosity specification as a single string

         elseif ( ischar( varargin{ i+1 } ) )
            verbose = bfo_get_verbosity( varargin{ i+1 } );
            if ( verbose >= 0 )
               verbosity      = varargin{ i+1 };
               user_verbosity = 1;
            else
               wrn = ' BFO warning: unknown verbosity level! Default used.';
               disp( wrn )
            end
         end
      else
         msg = ' BFO error: the argument list is ill-constructed! Terminating.';
         wrn = msg;
         disp( msg )
         return
      end
   end

   %  Set the flag (train) if training-mode specification = 'train' is found.

   if ( ischar( varargin{i} ) && strcmp( varargin{i}, 'training-mode' ) )
      if ( i < noptargs )
         if ( ischar( varargin{ i+1 } ) && strcmp( varargin{i+1}, 'train' ) )
            train = 1;
         end
      else
         msg = ' BFO error: the argument list is ill-constructed! Terminating.';
         wrn = msg;
         disp( msg )
         return
      end
   end

end

%  Check the objective function handle and the starting point (if present).

if ( ischar( varargin{ 1 } ) )

   %  The ( keyword, argument ) list starts at argument 1.

   if ( train )  
      first_optional = 1;   % in 'train' training mode
   else
      first_optional = 3;   % in 'solve' or 'train-and-solve' mode
   end

else
   if ( ~isa( varargin{ 1 }, 'function_handle' ) || ~isnumeric( varargin{ 2 } ) )
      msg = ' BFO error: the first two arguments are misspecified! Terminating.';
      wrn = msg;
      if ( verbose )
         disp( msg )
      end
      return
   end
 
   %  The ( keyword, argument ) list starts at argument 3 if either the
   %  first two specify the objective function and the starting point
   %  or if the first is a string (presumed to be a keyword at this stage).

   first_optional = 3;  

end

%  Compute the number of arguments in the ( keyword, argument ) list, and check
%  that it is even. Set a warning if it is odd.

n_optional = noptargs-first_optional+1;
if ( mod( n_optional, 2 ) > 0 )   
   if ( n_optional > 0 )
      noptargs = noptargs - 1;
      wrn  = [ ' BFO warning: the number of variable optional arguments beyond',           ...
               ' the objective function handle and the starting point must be even!',      ...
               ' Ignoring last argument.'];
      if ( verbose )
         disp( wrn )
      end
   end
end

%   Read the training requirements, if any, as well as the mandatory data for training.
%   Verify at the same time that every odd argument in the ( keyword, argument ) list is 
%   a string.

solve                  = 1;         % default training modes 
train                  = 0;
training_set_ok        = 0;         % presence of necessary inputs for training
training_set_data_ok   = 0;
training_set_cutest    = 0;         % default library of training problems
training_problems_data = {};
user_mode              = 0;         % training mode not specified by user (yet)

%if ( verbose > 4 )
%   disp( [ 'first_optional = ', int2str( first_optional ) ] )
%end

%  Check the training problems library (if specified).

for i = first_optional:2:noptargs
   if ( ischar( varargin{ i } ) )
      if ( strcmp( varargin{ i }, 'training-problems-library' ) )
        if ( ischar( varargin{ i+1 } ) )
            training_problems_library = varargin{ i+1 };
            if ( strcmp( training_problems_library, 'cutest' ) )
               training_set_cutest  = 1;
            elseif ( strcmp( training_problems_library, 'user-defined' ) )
            else
               wrn = ' BFO error: unknown training problems library! Default used.';
               if ( verbose )
                  disp( wrn ) 
               end            
            end
         else
            msg = ' BFO error: unknown training problems library! Default used.' ;
            if ( verbose )
               disp( msg )
            end
         end
      end
   else
      msg = [' BFO error: argument ', int2str(2*i-1), ' should be a keyword! Terminating.' ];
      wrn = msg;
      if ( verbose )
         disp( msg )
      end
      return
   end
end
   
%   Check the training arguments.

for i = first_optional:2:noptargs
   if ( ischar( varargin{i} ) )

      %  First check the training mode.

      if ( strcmp( varargin{ i }, 'training-mode' ) )        % the training mode
         if ( ischar( varargin{ i+1 } ) )
            training_mode = varargin{ i+1 };
            if ( strcmp( training_mode, 'solve' ) )
               user_mode = 1;
            elseif ( strcmp( training_mode, 'train' ) )
               train     = 1; 
               solve     = 0;
               user_mode = 1;
            elseif ( strcmp( training_mode, 'train-and-solve' ) )
               train     = 1;  
               user_mode = 1;
            else
               wrn = ' BFO error: unknown training mode! Default used.';
               if ( verbose )
                  disp( wrn ) 
               end            
            end
         else
            msg = ' BFO error: unknown training mode! Default used.' ;
            if ( verbose )
               disp( msg )
            end
         end

      %  Next check the training problems.

      elseif ( strcmp( varargin{ i }, 'training-problems' ) ) % the training set
         if ( iscell( varargin{ i+1 } ) )
            tp            = varargin{ i+1 };
            training_size = length( tp );
            if ( training_size > 0 )
               for tproblem = 1 : training_size
                  tpt = tp{ tproblem };

                  %  For CUTEst problems, only the string form is acceptable for 
                  %  identifying the training problems, and this is the form kept 
                  %  in memory within BFO.

                  if  ( training_set_cutest )
                     if ( ischar( tpt ) )        
                        training_problems{ tproblem } =  tpt;
                     else
                        msg = [' BFO error: the ', int2str(tproblem),                      ...
                               '-th argument of training-problems is not a string.',       ...
                               ' This is mandatory for CUTEst problems. Terminating.'];
                        if ( verbose )                          
                           disp( msg )          
                        end         
                        return     
                     end

                  %  For user-supplied problems, both strings and function handle forms
                  %  are acceptable. BFO remembers the function-handle form in this case.

                  else

                     %  String form

                     if ( ischar( tpt ) )           
                        if ( bfo_exist_function( tpt ) )
                           training_problems{ tproblem } = str2func( tpt );
                        else
                           msg = [ ' BFO error: m-file for training function ', tpt,       ...
                                   ' not found. Terminating. '];
                           if ( verbose )
                              disp( msg )
                           end
                           return
                        end

                     %  Function handle form

                     elseif ( isa( tpt, 'function_handle' ) )
                        tptname = func2str( tpt );
                        if ( bfo_exist_function( tptname ) )
                           training_problems{ tproblem } =  tpt;
                        else
                           msg = [ ' BFO error: m-file for training function ', tptname ,  ...
                                   ' not found. Terminating. '];
                           if ( verbose )
                              disp( msg )
                           end
                           return
                        end
                     else
                        msg = [' BFO error: the ', int2str(tproblem),                      ...
                               '-th argument of training-problems is not a function',      ...
                               ' handle nor a string. Terminating.'];
                        if ( verbose )
                           disp( msg )
                        end
                        return
                     end
                  end
                  training_set_ok = training_set_ok + 1;
               end
            end
            training_set_ok = ( training_set_ok == training_size );
         else
            msg = ' BFO error: training-problems is not a cell!';
            if ( verbose )
              disp( msg )
            end
            return
         end

      %  Finally check the training problems data.

      elseif ( strcmp( varargin{ i }, 'training-problems-data' ) )

         %  Ignore if the training library is CUTEst.

         if (  training_set_cutest  )
            wrn = [ ' BFO warning: training-problems-data supplied for CUTEst problems.'   ...
                    ' Ignoring.' ];
            if ( verbose )
               disp( wrn )
            end
            training_set_data_ok = 1;

         %  For user-supplied problems, both strings and function handle forms
         %  are acceptable. BFO remembers the function-handle form in this case.

         else
            if ( iscell( varargin{ i+1 } ) )
               tpd                = varargin{ i+1 };
               training_data_size = length( tpd );  % the training data
               for tproblem = 1:training_data_size
                  tpdt = tpd{ tproblem };

                  %  String form

                  if ( ischar( tpdt ) )
                     if ( bfo_exist_function( tpdt ) )
                         training_problems_data{ tproblem } = str2func( tpdt );
                     else
                        msg = [ ' BFO error: m-file for training function data ', tpdt,    ...
                                ' not found. Terminating. '];
                        if ( verbose )
                           disp( msg )
                        end
                        return
                     end

                  %  Function handle form

                  elseif ( isa( tpdt, 'function_handle' ) )
                     tpdtname = func2str( tpdt );
                     if ( bfo_exist_function( tpdtname ) )
                        training_problems_data{ tproblem } = tpdt;
                     else
                        msg = [ ' BFO error: m-file for training function data ',          ...
                                tpdtname, ' not found. Terminating. '];
                        if ( verbose )
                           disp( msg )
                        end
                        return
                     end
                  else
                     msg = [ ' BFO error: the ', int2str( tproblem ), '-th argument of ',  ...
                              'training-problems-data is not a function handle ',          ...
                              'nor a string!'];
                     if ( verbose )
                        disp( msg )
                     end
                     break
                  end
                  training_set_data_ok = training_set_data_ok + 1;
               end
               training_set_data_ok = ( training_set_data_ok == training_data_size );
            else
               msg = ' BFO error: training-problems-data is not a cell!';
               if ( verbose )
                  disp( msg )
               end
               return
            end  
         end
      end
   end
end

%  Check the presence of the objective function when solving is required. 
%  If the objective function is found, compute its stripped name and verify 
%  that the relevant m-file exists.

if ( solve )
   objfok = 0;
   if ( isa( varargin{ 1 }, 'function_handle' ) )
      f                   = varargin{ 1 };                   
      objname             = func2str( f );
      [ objfok, shfname ] = bfo_exist_function( objname );
   elseif ( ischar( varargin{ 1 } ) )
      objname             = varargin{ 1 };
      
      %  Verify that the string supplied as first argument does not match any 
      %  other BFO keyword before interpreting it as the name of the objective
      %  function.

      if ( strcmp( objname, 'max-or-min' )    || strcmp( objname, 'fcall-type' )        || ...
           strcmp( objname, 'xlower' )        || strcmp( objname, 'xupper' )            || ...
           strcmp( objname, 'xscale' )        || strcmp( objname, 'delta' )             || ...
           strcmp( objname, 'xtype'  )        || strcmp( objname, 'lattice-basis' )     || ...
           strcmp( objname, 'epsilon'  )      || strcmp( objname, 'f-target' )          || ...
           strcmp( objname, 'maxeval'  )      || strcmp( objname, 'termination-basis' ) || ...
           strcmp( objname, 'verbosity' )     || strcmp( objname, 'variable_bounds' )   || ...
           strcmp( objname, 'xlevel' )        || strcmp( objname, 'restart' )           || ...
           strcmp( objname, 'save-freq' )     || strcmp( objname, 'restart-file' )      || ...
           strcmp( objname, 'f-call-type' )   || strcmp( objname, 'f-bound' )           || ...
           strcmp( objname, 'training-mode' ) || strcmp( objname, 'training-problems' ) || ...
           strcmp( objname, 'reset-random-seed' )                                       || ...
           strcmp( objname, 'search-step-function' )                                    || ...
           strcmp( objname, 'training-problems-library' )                               || ...
           strcmp( objname, 'training-problems-data' )                                  || ...
           strcmp( objname, 'training-parameters' )                                     || ...
           strcmp( objname, 'trained-bfo-parameters' )                                  || ...
           strcmp( objname, 'training-epsilon' )                                        || ...
           strcmp( objname, 'training-maxeval' )                                        || ...
           strcmp( objname, 'training-verbosity' )                                      || ...
           strcmp( objname, 'training-problem-epsilon' )                                || ...
           strcmp( objname, 'training-problem-maxeval' )                                || ...
           strcmp( objname, 'training-problem-verbosity' )                              || ...
           strcmp( objname, 'alpha' )         || strcmp( objname, 'beta' )              || ...
           strcmp( objname, 'gamma' )         || strcmp( objname, 'eta' )               || ...
           strcmp( objname, 'zeta' )          || strcmp( objname, 'inertia' )           || ...
           strcmp( objname, 'search-type' )   || strcmp( objname, 'fevals-hist' )       || ...
           strcmp( objname, 'sspace-hist' )   || strcmp( objname, 'rpath' )             || ...
           strcmp( objname, 'nevr' )          || strcmp( objname, 'level' )             || ...
           strcmp( objname, 'xincr' )         || strcmp( objname, 'bfgs-finish' )       )
      else
         [ objfok, shfname ] = bfo_exist_function( objname );
         if ( objfok )
            f = str2func( objname );
         end
      end
   end
   if ( ~objfok  )           
      msg = [ ' BFO error: m-file for objective function ', objname, ' not found!'         ...
              ' Terminating.' ];
      wrn = msg;
      if ( verbose )
         disp( msg )
      end
      return     
   end

   %  Check the presence, nature and form of the starting point.

   if ( isnumeric( varargin{ 2 } ) )
      x0 = varargin{ 2 };
      n  = length( x0 );                             % the dimension of the space
      if ( size( x0, 1 ) == 1 && size( x0, 2 ) > 1 ) % make sure the starting point is 
         x0 = x0';                                   % a column vector
      end 
   else
      msg = ' BFO error: second argument is not a valid starting point! Terminating.';
      wrn = msg;
      if ( verbose )
         disp( msg )
      end
      return 
   end    
else
%   objname = 'bfo_none';       % no name for the objective function
   x0      = [];
end

%   Miscellaneous initializations:
%   (i) general

myinf        = 1.0e25;                     % a numerical value for plus infinity
depth        = 0;                          % the recurrence depth
rpath        = [];                         % the list of variables fixed in the recurrence
estcrit      = -1;                         % a meaningless estimate of criticality
deltaf       = myinf;                      % the desirable function decrease
cur          = 1;                          % column index for current increments in xincr
ini          = 2;                          % column index for initial increments in xincr
uns          = 3;                          % column index for unscaling transform in xincr
nevalt       = 0;                          % the accumulated nbr of evaluations during
                                           % ... training (optim. internal perf. functions)
%   (ii) reset by restart

maxeval      = 5000;                       % the max number of objective function evaluations
user_maxeval = 0;                          % maxeval not specified by the user
use_lattice  = 0;                          % no explicit lattice by default
latbasis     = [];                         % empty lattice basis by default

%   (iii) specific to multilevel

multilevel   = 0;                          % no multilevel structure by default
level        = 1;                          % single or first level by default
nlevel       = 1;                          % only one level by default
max_nlevel   = 6;                          % the maximum number of levels
msglow       = '';                         % empty message from next level

%   (iv) specific to restart

savef        = -1;                         % no restart information saving
sfname       = 'bfo.restart';              % the name of the file where restart ...
                                           % ... information is saved (when requested)

%   (v) flags to indicate if algorithmic parameters are user-specified or inherited from
%       previous computations

user_alpha      = 0;
user_beta       = 0;
user_gamma      = 0;
user_delta      = 0;
user_eta        = 0;
user_zeta       = 0;
user_inertia    = 0;
user_searchtype = 0;
user_rseed      = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Restart, if requested  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Look for a restart argument in the calling sequence or the name of a
%   restart file.

restart = 0;                                        % no restart by default
readall = 0;                                        % no complete reading of checkpointing
for i = first_optional:2:noptargs
   if ( ischar( varargin{ i } ) )
      if ( strcmp( varargin{ i }, 'restart' ) )
         if ( ischar( varargin{ i+1 } ) )
            rtype = varargin{ i+1 };
            if ( strcmp( rtype, 'none' ) )          % no restart requested
            elseif ( strcmp( rtype, 'use' ) )       % restart requested
               restart = 1;
            elseif ( strcmp( rtype, 'training' ) )  % restart within the training optim.
               restart = 1;
               readall = 1;
            else
               wrn = ' BFO warning: unknown restart mode! Ignoring.';
               if ( verbose )
                  disp( wrn )
               end
            end
         else
            wrn = ' BFO warning: unknown restart mode! Ignoring.';
            if ( verbose )
               disp( wrn )
            end
         end
      elseif ( strcmp( varargin{ i }, 'restart-file' ) )
         if ( ischar( varargin{ i+1 } ) )
            sfname = varargin{ i+1 };
         else
            wrn = ' BFO warning: wrong name for restart-file! Ignoring.';
            if ( verbose )
               disp( wrn )
            end
         end
      end
   end
end

%  If a restart is requested, read the relevant information.

if ( restart )

   %   Restore in solve mode.  At the end of this paragraph the restok flag has the
   %   following meaning:
   %   restok = 0: restoration fails because the restoration file could not be opened;
   %   restok = 1: restoration succeeded (during the solve phase)
   %   restok = 2: restoration to be completed (during the training phase)

   if ( solve )
      nobjname              = func2str( f );
      [ nobjfok, nobjname ] = bfo_exist_function( objname );    %% Correction 23 VI 2016
%      if ( length( nobjname ) > 18 )                           %% Correction 23 VI 2016
%          nobjname = nobjname(1:18);                           %% Correction 23 VI 2016
%      end                                                      %% Correction 23 VI 2016
      overbose = verbose;
      [ objname, maximize, epsilon, ftarget, maxeval, neval, f_hist, xtype, xtry,          ...
        xscale, xlower, xupper, verbose, alpha, beta, gamma, eta, zeta, inertia,           ...
        stype, rseed, term_basis, use_trained, s_hist, use_lattice, latbasis,              ...
        bfgs_finish, training_history, nscaled, unscaled, ssfname, restok ] =              ...
        bfo_restore( sfname, readall );
      if ( user_verbosity || readall )
         verbose = overbose;
      end
      if ( restok == 0 )
         msg     = [' BFO error: could not open ', sfname, '. Terminating.'];
         restart = 0;
      elseif ( restok == 1 )
         if ( ~strcmp( objname, nobjname ) && ~readall )
            msg     = [ ' BFO error: attempt to restart with a different objective.',      ...
                        ' Terminating.' ];
	    restart = 0;
         elseif ( length( xlower ) ~= n )
            msg     = ' BFO error: wrong restart file! Terminating.';
            restart = 0;
         end
         user_scl   = 1;
         user_ftarg = 1;                          % objective target not specified by the user
         xincr      = xtry;
         if ( maximize )
            max_or_min = 'max';
         else
            max_or_min = 'min';
         end
         if ( readall && size( training_history, 1 ) > 0 )
            nevalt = training_history( end, 3 );
         end
      end
   else
      restok = 2;
   end

   %  Restore the training process itself.

   if ( restok == 2 && ~( user_mode && ~train ) )
      overbose = verbose;
      [ p0, verbose, training_strategy, training_parameters, training_problems,            ...
        training_problems_data, training_set_cutest, trained_bfo_parameters,               ...
        training_epsilon, training_maxeval, training_verbosity,                            ...
        training_problem_epsilon, training_problem_maxeval, training_problem_verbosity,    ...
        trestok ]  =  bfo_restore_training( [ sfname, '.training'] );
      if ( user_verbosity )
        verbose = overbose;
      end
      if ( ~trestok )
         msg = [ ' BFO error: cannot open file ', [ sfname, '.training' ],                 ...
                 '! Terminating.' ];
         restart = 0;
      end
      not_restarting_training  = 0;
      nbr_train_params = length( training_parameters );
      restart_training = 'training';
      train            = 1;
      use_trained      = 0;
      alpha            = p0( 1 );
      beta             = p0( 2 );
      gamma            = p0( 3 );
      delta            = p0( 4 );
      eta              = p0( 5 );
      zeta             = p0( 6 );
      inertia          = p0( 7 );
      stype            = p0( 8 );
      rseed            = p0( 9 );

   %  No restart of the training process

   else
      restart_training         = 'none';
      use_trained              = 0;
      not_restarting_training  = 1;
   end

   %  Restoration failed.

   if ( ~restart )
      if ( overbose )
         disp( msg )
      end
      return

   %  Restoration succeeded: remember that algorithmic parameters are inherited.

   else
      if ( readall )
         user_alpha      = 1;
         user_beta       = 1;
         user_gamma      = 1;
         user_eta        = 1;
         user_zeta       = 1;
         user_inertia    = 1;
         user_searchtype = 1;
         user_rseed      = 1;
      end
   end

%  No restart at all: the training data must still be verified for coherence.

else
   restart_training = 'none';
   not_restarting_training  = 1;

   %  If training is required, check that the size of the training problems set is the same 
   %  of that of the training problem data (for a user-supplied library) or
   %  that there is only one data function (which is then to be used for all
   %  problems). If not, return unless a further "solve" phase is required.
   %  In this case, training is skipped.

   if ( train && training_set_ok && training_set_data_ok && ~training_set_cutest )
      if ( ~( training_data_size == training_size ) && training_data_size ~= 1 )
         if ( solve )
            train = 0;
            wrn = [ ' BFO error: training required but training-problems and',             ...
                     ' training-problems-data have different lengths! Aborting training.'];
            if ( verbose )
               disp( wrn )
            end
         elseif ( train )
            msg = [ ' BFO error: training required but training-problems and',             ...
                     ' training-problems-data have different lengths! Terminating.'];
            if ( verbose )
               disp( msg )
            end
            return
         end
      end
   end

   %   If training is required, verify that it is possible in that the necessary
   %   data is present.

   if ( train )

      %  Check that the training problem set is present.

      if ( ~training_set_ok ) 
         if ( solve )
            train = 0;
            msg   = [ ' BFO error: training required but training-problems unspecified!',  ...
                      ' Aborting training.'];
            if ( verbose )
               disp( msg )
            end
         else
            msg = [ ' BFO error: training required but training-problems unspecified!',    ...
                    ' Terminating.' ];
            if ( verbose )
               disp( msg )
            end
            return
         end
      end

      %  Check that the complete data for training is available for a user-supplied library.

      if ( ~training_set_data_ok && ~training_set_cutest )
         if ( solve )
            train = 0;
            msg   = [' BFO error: training required but training-problems-data',           ...
                     ' unspecified!  Aborting training.' ];
            if ( verbose )
               disp( msg )
            end
         else
            msg = [ ' BFO error: training required but training-problems-data',            ...
                    ' unspecified!  Terminating.' ];
            if ( verbose )
               disp( msg )
            end
            return
         end
      end
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Set default parameters before optional arguments are allowed to modify them.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( ~restart || ~readall )
   
   %  Default training parameters (skipped because too late if during restart).

   use_trained = 0;                               % no use of trained parameters by default 

   if ( train  )
     training_strategy  = 'average';              % training strategy based on the average
                                                  % ... number of function evaluations
     training_epsilon   = [ 1e-2 1e-2 ];          % accuracy for the training phase
     training_maxeval   = [ 200 100 ];            % maximum number of training evals
     training_verbosity = { 'silent', 'silent' }; % no output from the parameters optimization
     nbr_train_params   = 6;                      % a priori choice of training parameters
     training_problem_epsilon   = 1e-4;           % accuracy for each training problem
     training_problem_maxeval   = 5000;           % max evals for each training problem
     training_problem_verbosity = 'silent';       % training each problem is silent
     training_parameters    = { 'alpha', 'beta', 'gamma', 'delta', 'eta', 'inertia' };
     trained_bfo_parameters = 'trained.bfo.parameters';  % default file name for ...
                                                  % ... trained parameters
     xtype = 'c';                                 % training for continuous problems by def.
   end

   %  The general (non trainable) BFO default values

   if ( solve  )
      epsilon     = 1e-4;                  % the default accuracy requ. in the solve phase
      bfgs_finish = 0;                     % no BFGS finish requested
      maximize    = 0;                     % minimize by default
      max_or_min  = 'min';                 % ... also in string form
      ftarget     = -0.99999999 * myinf;   % the target objective function value
      xlower      = -myinf * ones(n,1);    % no lower bound by default
      xupper      =  myinf * ones(n,1);    % no upper bound by default
      xtype(1:n)  = 'c';                   % the default variable type is continuous
      term_basis  = 5;                     % the number of random basis used for ...
                                           % ... assessing termination
      user_tbasis = 0;                     % true if the user specifies term_basis 
      use_variable_bounds = 0;             % true if variable bounds are used
      vb_name     = 'bfo_none';            % empty variable bound function
  end
  user_scl   = 0;                          % scales not specified by the user
  user_ftarg = 0;                          % objective target not specified by the user
  nscaled    = 0;                          % no scaled continuous variable
end
fcallt       = 'simple';                   % the function call type
withbound    = 0;                          % the call for f does not use fbound
fbound       = myinf;                      % default for minimization
user_fbound  = 0;                          % the unsuccessful f bound is not user specified
searchstep   = 0;                          % the use of a user-supplied search-step function
ssfname      = 'bfo_none';                 % no default for the search-step function
resetrng     = 1;                          % reset the RNG by default...
reset_random_seed = 'reset';               % ... also in string form

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Process the (remaining) variable argument list  %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = first_optional:2:noptargs-1

   %  The lower bounds

   if ( strcmp( varargin{ i }, 'xlower' ) )
      if ( solve ) 
         if ( isnumeric( varargin{ i+1 } ) )
            xtry = varargin{ i+1 };
            [ s1, s2 ] = size( xtry );
            if ( s1 == 1 && s2 == 1 )
               xlower   = xtry * ones( n, 1 );
            elseif ( min( s1, s2 ) ~= 1 || max( s1, s2 ) ~= n )
               wrn = ' BFO warning: wrong size of input for parameter xlower! Default used.';
               if ( verbose )
                  disp( wrn )
               end
            else
               if ( s2 > s1 )      % make sure the lower bound is a column vector 
                   xlower = max( [ xtry ; -myinf*ones( 1, n ) ])';      
               else
                   xlower = max( [ xtry'; -myinf*ones( 1, n ) ])';     
               end
            end
         else
            wrn = ' BFO warning: wrong type of input for parameter xlower! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
      end

   %  The upper bounds

   elseif ( strcmp( varargin{ i }, 'xupper' ) )
      if ( solve )  
         if ( isnumeric( varargin{ i+1 } ) )
            xtry = varargin{ i+1 };
            [ s1, s2 ] = size( xtry );
            if ( s1 == 1 && s2 == 1 )
               xupper   = xtry * ones( n, 1 );
            elseif ( min( s1, s2 ) ~= 1 || max( s1, s2 ) ~= n )
               wrn = ' BFO warning: wrong size of input for parameter xupper! Default used.';
               if ( verbose )
                  disp( wrn )
               end
            else
               if ( s2 > s1 )    % make sure the upper bound is a column vector 
                   xupper = min( [ xtry ; myinf*ones( 1, n ) ])';
               else
                   xupper = min( [ xtry'; myinf*ones( 1 ,n ) ])';
               end
            end
         else
            wrn = ' BFO warning: wrong type of input for parameter xupper! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
      end

   %  The expansion shrinking factor

   elseif ( strcmp( varargin{ i }, 'alpha' ) )
      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            alpha      = max( min( 1.e8, abs( varargin{ i+1 } ) ), 1 );
            user_alpha = 1;
         else
            wrn = ' BFO warning: wrong type of input for parameter alpha! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
      end

   %  The grid shrinking factor

   elseif ( strcmp( varargin{ i }, 'beta' ) )
      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            beta      = min( max( 1.e-8, abs( varargin{ i+1 } ) ), 0.999999 );
            user_beta = 1;
         else
            wrn = ' BFO warning: wrong type of input for parameter beta! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
      end

   %  The maximum grid expansion factor

   elseif ( strcmp( varargin{ i }, 'gamma' ) )
      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            gamma      = max( 1, abs( varargin{ i+1 } ) );
            user_gamma = 1;
         else
            msg = ' BFO error: wrong type of input for parameter gamma! Default used.';
            disp( msg )
         end
      end

   %  The initial increment factor

    elseif ( strcmp( varargin{ i }, 'delta' ) )
      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            delta = abs( varargin{ i+1 } );
            [ s1, s2 ] = size( delta )
            if ( s1 == 1 && s2 == 1 )
               user_delta = 1;
             elseif ( max( s1, s2 ) == n )			   
 			   user_delta = 2;                         
            elseif ( min( s1, s2 ) ~= 1 )
               wrn = ' BFO error: wrong size of input for parameter delta! Default used.';
               if ( verbose )
                  disp( wrn )
               end
            elseif ( ( max( s1, s2 ) ~= n ) && solve )
               wrn = ' BFO error: wrong size of input for parameter delta! Default used.';
               if ( verbose )
                  disp( wrn )
               end
            end
         else
            msg = ' BFO error: wrong type of input for parameter delta! Default used.';
            disp( msg )
         end
      end

   %  The sufficient decrease fraction

   elseif ( strcmp( varargin{ i }, 'eta' ) )
      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            eta      = max( 1.e-5, abs( varargin{ i+1 } ) );
            user_eta = 1;
         else
            wrn = ' BFO warning: wrong type of input for parameter eta! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
      end

   %  The multilevel grid expansion factor for re-exploration of a previously
   %  visited level.

   elseif ( strcmp( varargin{ i }, 'zeta' ) )
      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            zeta      = max( 1, abs( varargin{ i+1 } ) );
            user_zeta = 1;
         else
            wrn = ' BFO warning: wrong type of input for parameter zeta! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
      end

   %  The inertia for continuous step accumulation

   elseif ( strcmp( varargin{ i }, 'inertia' ) )
      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            inertia      = abs( round( varargin{ i+1 } ) );
            user_inertia = 1;
         else
            wrn = ' BFO warning: wrong type of input for parameter inertia! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
      end

   %  The discrete tree search strategy

   elseif ( strcmp( varargin{ i }, 'search-type' ) )
      if ( not_restarting_training )
         if ( ischar( varargin{ i+1 } ) )
            searchtype = varargin{ i+1 };
            if ( strcmp( searchtype, 'breadth-first' ) )
               stype           = 0;
               user_searchtype = 1;
            elseif ( strcmp( searchtype, 'depth-first' ) )
               stype           = 1;
               user_searchtype = 1;
            elseif ( strcmp( searchtype, 'none' ) )
               stype           = 2;
               user_searchtype = 1;
            else
               wrn = [ ' BFO warning: wrong type of input for parameter search-type!',     ...
                       ' Default used.' ];
               if ( verbose )
                  disp( wrn )
               end
            end
         else
            wrn = [ ' BFO warning: wrong type of input for parameter search-type!',        ...
                    ' Default used.' ];
            if ( verbose )
               disp( wrn )
            end
         end
      end

   %  The random number generator's seed

   elseif ( strcmp( varargin{ i }, 'random-seed' ) )
      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            rseed      = round( abs( varargin{ i+1 } ) );
            user_rseed = 1;
         else
            wrn = [ ' BFO warning: wrong type of input for parameter random-seed!',        ...
                  ' Default used.' ];
            if ( verbose )
               disp( wrn )
            end
         end
      end

   %  The variable scaling

   elseif ( strcmp( varargin{ i }, 'xscale' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         xtry = abs( varargin{ i+1 } );
         [ s1, s2 ] = size( xtry );
         if ( s1 == 1 && s2 == 1 )
            user_scl = 1;       % remember that scales are specified by the user
            if ( solve )
                xscale   = xtry * ones( n, 1 );
            end
          elseif ( min( s1, s2 ) ~= 1 )
            wrn = ' BFO warning: wrong size of input for parameter xscale! Default used.';
            if ( verbose )
               disp( wrn )
            end
         elseif ( ( max( s1, s2 ) ~= n ) && solve )
            wrn = ' BFO warning: wrong size of input for parameter xscale! Default used.';
            if ( verbose )
               disp( wrn )
            end
         else
            user_scl = 2;       % remember that scales are specified by the user
            if ( s1 < s2 )      % make sure the scaling is a column vector 
               xscale = xtry';
            else
               xscale = xtry;
            end
         end
      else
         wrn = ' BFO warning: wrong type of input for parameter xscale! Default used.';
         if ( verbose )
            disp( wrn )
         end
      end

   %  The multilevel grid spacing (should only occur when entering levels > 1 )

   elseif ( strcmp( varargin{ i }, 'xincr' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         xincr = varargin{ i+1 };
      else
         wrn = ' BFO warning: wrong type of input for parameter xincr! Default used.';
         if ( verbose )
            disp( wrn )
         end
      end

   %  The variable types

   elseif ( strcmp( varargin{ i }, 'xtype' ) )
      if ( ischar( varargin{ i+1 } ) )
         xtyp  =  varargin{ i+1 };
         xtok  = 1;
         for ii = 1:length( xtyp )
            if ( xtyp( ii ) ~= 'c' && xtyp( ii ) ~= 'i' && xtyp( ii ) ~= 'f' &&         ...
                 xtyp( ii ) ~= 'w' && xtyp( ii ) ~= 'z' )
               xtok = 0;
               break
            end
         end
         if ( xtok )
            if ( solve )  
               if ( length( xtyp ) == 1 )
                  xtype( 1:n ) = xtyp;
               elseif ( length( xtyp ) ~= n )
                  wrn = [ ' BFO warning: wrong size of input for parameter xtype!',        ...
                          ' Default used.' ];
                  if ( verbose )
                     disp( wrn )
                  end
               else
                  xtype = xtyp;
               end
            else
               xtype = xtyp;
            end
         else
            wrn = ' BFO warning: wrong type of input for parameter xtype! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
      else
         wrn = ' BFO warning: wrong type of input for parameter xtype! Default used.';
         if ( verbose )
            disp( wrn )
         end
      end

  %  The lattice basis for integer variables

   elseif ( strcmp( varargin{ i }, 'lattice-basis' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         latbasis    = varargin{ i+1 };
         use_lattice = ( length( latbasis ) > 0 );
      else
         msg = ' BFO error: wrong type of input for parameter lattice-basis! Terminating.';
         if ( verbose )
            disp( msg )
         end
         return
      end

   %  The maximum number of objective function's evaluations

   elseif ( strcmp( varargin{ i }, 'maxeval' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         maxeval     = abs( round( varargin{ i+1 } ) );
         user_maxeval = 1;
      else
         wrn = ' BFO warning: wrong type of input for parameter maxeval! Default used.';
         if ( verbose )
            disp( wrn )
         end
      end

   %  The termination accuracy

   elseif ( strcmp( varargin{ i }, 'epsilon' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         epsilon    = abs( varargin{ i+1 } );
         [ s1, s2 ] = size( epsilon );
         if ( s1 ~= 1 || s2 ~= 1 )
            wrn = ' BFO waring: wrong size of input for parameter epsilon! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
      else
         wrn = ' BFO warning: wrong type of input for parameter epsilon! Default used.';
         if ( verbose )
            disp( wrn )
         end
      end

   %  The quasi-Newton BFGS finish meshsize

   elseif ( strcmp( varargin{ i }, 'bfgs-finish' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         bfgs_finish =  varargin{ i+1 };
      else
         wrn = ' BFO warning: wrong type of input for parameter bfgs-finish! Default used.';
         if ( verbose )
            disp( wrn )
         end
      end

   %  The possible reset of the random number generator

   elseif ( strcmp( varargin{ i }, 'reset-random-seed' ) )
      if ( ischar( varargin{ i+1 } ) )
         reset_random_seed = varargin{ i+1 };
         if ( strcmp( varargin{ i+1 }, 'reset' ) )
            resetrng = 1;
         elseif ( strcmp( varargin{ i+1 }, 'no-reset' ) )
            resetrng = 0;
         else
            wrn = [ ' BFO warning: meaningless input for parameter reset-random-seed! ', ...
                      'Default used.' ];
            if ( verbose )
               disp( wrn )
            end
         end
      else
         wrn = [ ' BFO warning: wrong type of input for parameter reset-random-seed! ',    ...
                   'Default used.' ];
         if ( verbose )
            disp( wrn )
         end
      end

   %  The objective function target

   elseif ( strcmp( varargin{ i }, 'f-target' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         ftarget    = varargin{ i+1 };
         user_ftarg = 1;
      else
         wrn = ' BFO warning: wrong type of input for parameter f-target! Default used.';
         if ( verbose )
            disp( wrn )
         end
      end

   %  The saved information about discrete subspaces

   elseif ( strcmp( varargin{ i }, 'sspace-hist' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         s_hist = varargin{ i+1 };
      else
         wrn = [' BFO internal warning: wrong type of input for parameter sspace-hist!', ...
                ' Default used.'];
         if ( verbose )
            disp( wrn )
         end
      end

   %  The recurrence path and depth

   elseif ( strcmp( varargin{ i }, 'rpath' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         rpath = varargin{ i+1 };
      else
         wrn = [ ' BFO internal warning: wrong type of input for parameter rpath!',        ...
                 '  Default used.' ];
         if ( verbose )
            disp( wrn )
         end
      end
      depth = length( rpath );

   %  The number of function calls higher in the recursion

   elseif ( strcmp( varargin{ i }, 'nevr' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         neval = varargin{ i+1 };
      else
         wrn = [ ' BFO internal warning: wrong type of input for parameter nevr!',         ...
                 ' Default used.' ];
         if ( verbose )
            disp( wrn )
         end
      end
      
   %  The vector of function values higher in the recursion

   elseif ( strcmp( varargin{ i }, 'fevals-hist' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         f_hist = varargin{ i+1 };
      else
         wrn = [ ' BFO internal warning: wrong type of input for parameter fevals-hist!',  ...
               ' Default used.' ];
         if ( verbose )
            disp( wrn )
         end
      end

    %  The number of random basis used for testing final termination

   elseif ( strcmp( varargin{ i }, 'termination-basis' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         term_basis  = max( 1, round( varargin{ i+1 } ) );
         user_tbasis = 1;
      else
         wrn = [ ' BFO warning:  wrong input for the number of termination basis!',        ...
                 ' Default used.' ];
         if ( verbose )
            disp( wrn )
         end
      end

   %  The desirable function decrease on the current grid

   elseif ( strcmp( varargin{ i }, 'deltaf' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         deltaf = abs( varargin{ i+1 } );
      else
         wrn = ' BFO warning: wrong type of input for parameter deltaf! Default used.';
         if ( verbose )
            disp( wrn )
         end
      end

%  Maximize or minimize 

   elseif ( strcmp( varargin{ i }, 'max-or-min' ) )
      if ( ischar( varargin{ i+1 } ) )
         if ( restart && ~strcmp(varargin{ i+1 }, max_or_min ) ) 
            wrn = ' BFO warning: inconsistent max-or-min at restart. Using saved value.';
            if ( verbose )
               disp( wrn )
            end
         end
         max_or_min = varargin{ i+1 };
         if ( size( max_or_min, 2 ) ~= 3 )
            wrn = [ ' BFO warning:  badly specified choice of minimization or',            ...
                    ' maximization!  Default used.' ];
            if ( verbose )
               disp( wrn )
            end
            max_or_min = 'min';
         end
         if ( strcmp( max_or_min( 1,: ), 'max' ) )
            maximize = 1;
         elseif ( strcmp( max_or_min( 1,: ), 'min' ) )
            maximize = 0;
         else
            wrn = [ ' BFO warning:  unknown choice of minimization or maximization!',      ...
                    ' Default used.' ];
            if ( verbose )
               disp( wrn )
            end
         end
      else
         wrn = ' BFO error:  unknown choice of minimization or maximization! Default used.';
         if ( verbose )
            disp( wrn ) 
         end
      end

   %  The call type for f(x)

   elseif ( strcmp( varargin{ i }, 'f-call-type' ) )
       if ( ischar( varargin{ i+1 } ) )
         fcallt = varargin{ i+1 };
         if ( strcmp( fcallt, 'simple' ) )
         elseif ( strcmp( fcallt, 'with-bound' ) )
            withbound = 1;
         else
            wrn = [ ' BFO warning: wrong type of input for parameter f-call-type!',        ...
                    ' Default used.' ];
            if ( verbose )
               disp( wrn )
            end
         end
      else
         wrn = [ ' BFO warning: wrong type of input for parameter f-call-type!',           ...
                 ' Default used.' ];
         if ( verbose )
            disp( wrn )
         end
       end
  
   %  The "unsuccessful" bound for f(x)

   elseif ( strcmp( varargin{ i }, 'f-bound' ) )
       if ( isnumeric( varargin{ i+1 } ) )
         fbound      = varargin{ i+1 };
         user_fbound = 1;
      else
         wrn = [ ' BFO warning: wrong type of input for parameter f-bound!',               ...
                 ' Default used.' ];
         if ( verbose )
            disp( wrn )
         end
       end
  
   %  The save strategy

   elseif ( strcmp( varargin{ i }, 'save-freq' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         savef = varargin{ i+1 };
      else
         wrn = ' BFO warning:  wrong input for the save-freq parameter! Default used.';
         if ( verbose )
            disp( wrn )
         end
      end

   %  The name of the user-supplied search-step function

   elseif ( strcmp( varargin{ i }, 'search-step-function' ) )
      if ( ischar( varargin{ i+1 } ) )
         ssfname = varargin{ i+1 };
         if ( ~strcmp( ssfname, 'bfo_none' ) )
            if ( bfo_exist_function( ssfname ) )
               bfo_srch = str2func( ssfname );
               searchstep = 1;
            else
               wrn = [ ' BFO warning: m-file for search-step function ', ssfname,          ...
                       ' not found. No search-step used. '];
               if ( verbose )
                  disp( wrn )
               end
            end
         end
      elseif ( isa( varargin{ i+1 }, 'function_handle' ) )
         ssfname = func2str(  varargin{ i+1 } );
         if ( bfo_exist_function( ssfname ) )
            bfo_srch = varargin{ i+1 };
            searchstep = 1;
         else
            wrn = [ ' BFO warning: m-file for search-step function ', ssfname,             ...
                                ' not found. No search-step used. '];
            if ( verbose )
               disp( wrn )
            end
         end
      else
         wrn = [ ' BFO warning: wrong type of input for search-step-function!',            ...
                 ' No search-step used.' ];
         if ( verbose )
            disp( wrn )
         end
      end

   %  The multilevel structure and assignment of variables to levels
       
   elseif ( strcmp( varargin{ i }, 'xlevel' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         xlevel = abs( round( varargin{ i+1 } ) );
         if ( min( xlevel ) ~= 1 )
            msg = ' BFO error: lowest level different from 1! Terminating.';
            if ( verbose )
                disp( msg )
            end
            return
         end
         multilevel = 1;
      else
         wrn = ' BFO warning:  wrong input for xlevel parameter! Ignored.';
         if ( verbose )
            disp( wrn )
         end
      end
      
    %  The current level in a multilevel framework
      
    elseif ( strcmp( varargin{ i }, 'level' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         level  = abs( round( varargin{ i+1 } ) );
      else
         wrn = ' BFO warning:  wrong input for level parameter! Ignored.';
         if ( verbose )
            disp( wrn )
         end
      end

   %  The potential use of level dependent bounds
       
   elseif ( strcmp( varargin{ i }, 'variable-bounds' ) )
      if ( ischar( varargin{ i+1 } ) )
         vb_name = varargin{ i+1 };
         if ( ~strcmp( vb_name, 'bfo_none' ) )
            if ( bfo_exist_function( vb_name ) )
               variable_bounds     = str2func( vb_name );
               use_variable_bounds = 1;
            else
               msg = [ ' BFO error: m-file for variable bounds function ', vb_name,        ...
                       ' not found. Terminating. '];
               if ( verbose )
                  disp( msg )
               end
               return
            end
         end
      elseif ( isa( varargin{ i+1 }, 'function_handle' ) )
         vb_name = func2str(  varargin{ i+1 } );
         if ( bfo_exist_function( vb_name ) )
            variable_bounds     = varargin{ i+1 };
            use_variable_bounds = 1;
         else
            msg = [ ' BFO error: m-file for variable bounds function ', vb_name,           ...
                                ' not found. Terminating. '];
            if ( verbose )
               disp( msg )
            end
            return
         end
      else
         wrn = ' BFO warning: wrong type of input for variable-bounds! Default bounds used.';
         if ( verbose )
            disp( wrn )
         end
      end

   %  The training strategy

   elseif ( strcmp( varargin{ i }, 'training-strategy' ) )
      if ( ~strcmp( restart_training, 'use' ) )
         if ( ischar( varargin{ i+1 } ) )
            training_strategy = varargin{ i+1 };
            if ( strcmp( training_strategy, 'average' ) ) 
            elseif ( strcmp( training_strategy, 'robust' ) )
            else
               training_strategy = 'average';
               wrn = ' BFO warning: wrong type of training strategy! Default used.';
               if ( verbose )
                  disp( wrn )
               end
            end
         else
            training_strategy = 'average';
            wrn = [ ' BFO warning: wrong type of input for parameter training-strategy!',  ...
                    ' Default used.'];
            if ( verbose )
               disp( wrn )
            end
         end
      end

   %  The training parameters

   elseif ( strcmp( varargin{ i }, 'training-parameters' ) )
      if ( ~strcmp( restart_training, 'use' ) )
         if ( iscell( varargin{ i+1 } ) )
            tp  = varargin{ i+1 };
            ntp = length( tp );
            if ( ntp > 0 )
               nbr_train_params    = 0;
               training_parameters = {};
               for itp = 1:ntp
                   tpi = tp{ itp };
                   if ( ischar( tpi ) ) 
                      if ( strcmp( tpi, 'alpha'  )  || strcmp( tpi, 'beta'        ) ||     ...
                           strcmp( tpi, 'gamma'  )  || strcmp( tpi, 'delta'       ) ||     ...
                           strcmp( tpi, 'eta'    )  || strcmp( tpi, 'zeta'        ) ||     ...
                           strcmp( tpi, 'inertia' ) || strcmp( tpi, 'search-type' ) ||     ...
                           strcmp( tpi, 'random-seed' ) )
                         nbr_train_params = nbr_train_params + 1;
                         training_parameters{ nbr_train_params } = tpi;
                      else
                         wrn = [ ' BFO warning: unknown ', int2str( itp ),                 ...
                                 '-th training parameter! Ignored.' ];
                         if ( verbose )
                            disp( wrn )
                         end
                      end
                   else
                      wrn = [ ' BFO warning: unknown ', int2str( itp ),                    ...
                              '-th training parameter! Ignored.' ];
                      if ( verbose )
                         disp( wrn )
                      end
                   end
               end
            else
               nbr_train_params = 0;
            end
         else
            wrn = [ ' BFO warning: wrong type of input for training-parameters!',          ...
                    ' Default used.'];
            if ( verbose )
               disp( wrn )
            end
         end
      end

   %  The name of the file containing previously trained parameters (in 'solve'
   %  training mode) or where new trained parameters will be saved (in 'train'
   %  and 'train-and-solve' training modes)

   elseif ( strcmp( varargin{ i }, 'trained-bfo-parameters' ) )
      if ( ischar( varargin{ i+1 } ) )
         trained_bfo_parameters = varargin{ i+1 }; 
         use_trained = 1;       
      else
         wrn = [ ' BFO warning: wrong type of input for parameter trained-bfo-parameters!',...
                 ' Default name used.'];
         if ( verbose )
            disp( wrn )
         end
      end
   
   %  The accuracy threshold for the training process

   elseif ( strcmp( varargin{ i }, 'training-epsilon' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         training_epsilon = abs( varargin{ i+1 } );
         sa = length( training_epsilon );
         if ( sa == 1 )
            training_epsilon = [ training_epsilon training_epsilon ];
         end
      else
         wrn = ' BFO warning:  wrong input for the training-epsilon parameter! Default used.';
         if ( verbose )
            disp( wrn )
         end
      end

   %  The maximum number of parameter configuration allowed in the training phase

   elseif ( strcmp( varargin{ i }, 'training-maxeval' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         training_maxeval = varargin{ i+1 };
         if ( min( training_maxeval ) < 0 )
            wrn = ' BFO warning: training-maxeval is negative! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
         sa = length( training_maxeval );
         if ( sa == 1 )
            training_maxeval = [ training_maxeval training_maxeval ];
         end
      else
         wrn = [ ' BFO warning: wrong type of input for parameter training-maxeval!',      ...
                ' Default used.'];
         if ( verbose )
            disp( wrn )
         end
      end
      
   %  The verbosity of the training process

   elseif ( strcmp( varargin{ i }, 'training-verbosity' ) )
      t_verbosity = varargin{ i+1 };
      if ( iscell( t_verbosity ) )
         tverbose = bfo_get_verbosity( t_verbosity{ 1 } );
         if ( tverbose >= 0 )
            training_verbosity{ 1 } = t_verbosity{ 1 };
            user_training_verbosity = 1;
         else
            wrn = ' BFO warning: unknown training verbosity level! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
         lv = length( t_verbosity );
         if ( lv > 1 )
            tverbose = bfo_get_verbosity( t_verbosity{ 2 } );
            if ( tverbose >= 0 )
               training_verbosity{ 2 } = t_verbosity{ 2 };
               user_training_verbosity = 1;
            else
               wrn = ' BFO warning: unknown training verbosity level! Default used.';
               if ( verbose )
                  disp( wrn )
               end
               user_training_verbosity = 0;
            end
         end
      elseif ( ischar( t_verbosity ) )
         tverbose = bfo_get_verbosity( t_verbosity );
         if ( tverbose >= 0 )
            training_verbosity = { t_verbosity, t_verbosity };
            user_training_verbosity = 1;
         else
            wrn = ' BFO warning: unknown training verbosity level! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
      else
         wrn = [ ' BFO warning:  wrong input for the training-verbosity parameter!',       ...
                Default used.' ];
         if ( verbose )
            disp( wrn )
         end
      end

   %  The accuracy threshold for optimizing each test problem during training

   elseif ( strcmp( varargin{ i }, 'training-problem-epsilon' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         training_problem_epsilon = abs( varargin{ i+1 } );
      else
         wrn = [ ' BFO warning:  wrong input for the training-problem-epsilon parameter!', ...
                 ' Default used.' ];
         if ( verbose )
            disp( wrn )
         end
      end

   %  The maximum number of objective function's evaluations for each problem
   %  during training

   elseif ( strcmp( varargin{ i }, 'training-problem-maxeval' ) )
      if ( isnumeric( varargin{ i+1 } ) )
         training_problem_maxeval = abs( round( varargin{ i+1 } ) );
      else
         wrn = [ ' BFO warning: wrong type of input for parameter',                        ...
                 ' training-problem-maxeval! Default used.' ];
         if ( verbose )
            disp( wrn )
         end
      end

   %  The verbosity of the each problem training

   elseif ( strcmp( varargin{ i }, 'training-problem-verbosity' ) )
      tp_verbosity = varargin{ i+1 };
      if ( ischar( tp_verbosity ) )
         if ( bfo_get_verbosity( tp_verbosity ) >= 0 )
            training_problem_verbosity = tp_verbosity;
         else
            wrn = ' BFO warning: unknown training problem verbosity level! Default used.';
            if ( verbose )
               disp( wrn )
            end
         end
      else
         wrn = [' BFO warning:  wrong input for the training-problem-verbosity parameter!',...
                ' Default used.' ];
         if ( verbose )
            disp( wrn )
         end
      end

   %  Keywords already handled before the current parsing are just ignored 
   %  without generating any error or warning.
      
   elseif ( strcmp( varargin{ i }, 'verbosity'                 ) )
   elseif ( strcmp( varargin{ i }, 'restart'                   ) )
   elseif ( strcmp( varargin{ i }, 'restart-file'              ) )
   elseif ( strcmp( varargin{ i }, 'training-mode'             ) )
   elseif ( strcmp( varargin{ i }, 'training-problems'         ) )
   elseif ( strcmp( varargin{ i }, 'training-problems-data'    ) )
   elseif ( strcmp( varargin{ i }, 'training-problems-library' ) )

   %  Unidentified keyword: ignore it and issue a warning.

   else
      wrn = [ ' BFO warning: undefined keyword ', varargin{ i }, '! Ignoring.' ];
      if ( verbose )
         disp( wrn )
      end
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Choose so far unspecified algorithmic parameters according to the type of problem.     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   alpha        is the grid expansion factor                   
%   beta         is the grid shrinking factor                   
%   gamma        is the max grid expansion factor              
%   delta        is the initial scale for continuous variables
%   NOTE: the parameter delta is only used at *first* entry in BFO (that is at level == 1
%         and depth == 0), the exchange of variable increment information between subsequent
%         depths or levels being handled via the increment table xincr. In addition, delta
%         is not saved at checkpointing during solve, and therefore cannot be recovered by
%         restoration, even if the complete checkpointing file is read.
%   eta          is the sufficient descent fraction             
%   zeta         is the increase in grid size when reoptimizing at a level previously visited
%   inertia      is the inertia for continuous step averaging  
%   stype        is the tree search  strategy for discrete variables (in integer form)
%   searchtype   is the tree search  strategy for discrete variables (in string form)
%   rseed        is the random number generator's seed

%   "Reasonable" values:

%   alpha         = 2;                   % the grid expansion factor                   
%   beta          = 0.5;                 % the grid shrinking factor                   
%   gamma         = 4;                   % the max grid expansion factor              
%   delta         = 1;                   % the initial scale for continuous variables
%   eta           = 1e-5;                % the sufficient descent fraction             
%   zeta          = 2;                   % the increase in grid size when ...
                                         % ... reoptimizing at a level previously visited
%   inertia       = 10;                  % the inertia for continuous step averaging  
%   stype         = 0;                   % breadth-first search for discrete variables 
%   searchtype    = 'breadth-first';      % ... also in string form
%   rseed         = 0;                   % random number generator's seed

micase = length( find( xtype == 'i' ) ); % positive in the mixed-integer case

%   Define the default algorithmic parameters as resulting from 
%   the "average performance criterion" (see paper).

if ( ~user_alpha )
   if ( micase ) 
      alpha = 2;
   else
      alpha = 1.4248;
   end
end
if ( ~user_beta )
   if ( micase )
      beta = 0.3135;
   else
      beta = 0.1997;
   end
end
if ( ~user_gamma )
   if ( micase )
      gamma = 5;
   else
      gamma = 2.3599;
   end
end
if ( ~user_delta )
   if ( micase )
      delta = 3.603;
   else
      delta = 1.0368;
   end
   user_delta = 1;
end
if ( ~user_eta )
   if ( micase )
      eta = 0.4528;
   else
      eta = 0.00001;
   end
end
if ( ~user_zeta )
   if ( micase )
      zeta = 1.5;
   else
      zeta = 1.5;
   end
end
if ( ~user_inertia )
   if ( micase )
      inertia = 10;
   else
      inertia = 11;
   end
end
if ( ~user_searchtype )
   if ( micase )
      searchtype = 'depth-first';
      stype      = 1;
   else
      searchtype = 'depth-first';
      stype      = 1;
   end
end
if ( ~user_rseed )
   if ( micase ) 
      rseed = 91;
   else
      rseed = 53;
   end
end

%% Start the training phase

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%                                     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%   Algorithmic parameters' training  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%                                     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( train )

   %  Print the header and adapt the verbosity level, unless it has been specified 
   %  by the user.

   if ( verbose  )
      fprintf( '\n')
      fprintf(   '  ********************************************************\n')
      fprintf(   '  *                                                      *\n')
      fprintf(   '  *   BFO: brute-force optimization without derivatives  *\n')
      fprintf(   '  *                                                      *\n')
      fprintf( [ '  *     (c)  Ph. Toint, M. Porcelli, 2016  (',this_version, ')      *\n'] )
      fprintf(   '  *                                                      *\n')
      fprintf(   '  ********************************************************\n')
      fprintf( '\n')
   end

   %  The current default parameters, with their lower/upper bounds and their scale.

   if ( nbr_train_params > 0 )
      p0     = [  alpha  beta   gamma  delta  eta    zeta  inertia  stype rseed  ];
      plower = [    1    0.01     1    0.25   1e-5     1        5      0     0   ];
      pupper = [    2    0.95    10     10     0.5    10       30      2   100   ];
      pscale = [    1      1      1      1       1     1        1      1     1   ];
      pdelta = [        (pupper(1:6)-plower(1:6))/10            1      1     1   ]; 

      %   If the user has chosen which parameters to train, identify them in the list
      %   by releasing their type from fixed to continuous or integer, according to 
      %   their nature.

      ptype  = 'fffffffff';
      for  i = 1:nbr_train_params 
          if (     strcmp( training_parameters{ i }, 'alpha'       ) )
             ptype( 1 )  = 'c';
          elseif ( strcmp( training_parameters{ i }, 'beta'        ) )
             ptype( 2 )  = 'c';
          elseif ( strcmp( training_parameters{ i }, 'gamma'       ) )
             ptype( 3 )  = 'c';
          elseif ( strcmp( training_parameters{ i }, 'delta'       ) )
             ptype( 4 )  = 'c';
          elseif ( strcmp( training_parameters{ i }, 'eta'         ) )
             ptype( 5 )  = 'c';
          elseif ( strcmp( training_parameters{ i }, 'zeta'        ) )
             ptype( 6 )  = 'c';
          elseif ( strcmp( training_parameters{ i }, 'inertia'     ) )
             ptype( 7 )  = 'i';
          elseif ( strcmp( training_parameters{ i }, 'search-type' ) )
             ptype( 8 )  = 'i';
          elseif ( strcmp( training_parameters{ i }, 'random-seed' ) )
             ptype( 9 )  = 'i';
          end
      end

   %  The user has supplied and empty list of parameters to train.

   else
      if ( solve)
         wrn = ' BFO warning: empty set of training parameters! Aborting training.';
         if ( verbose )
            disp( wrn )
         end
         train = 0;
      else
         msg = ' BFO error: empty set of training parameters! Terminating.';
         if ( verbose )
            disp( msg )
         end
         return;
      end
   end

   %  The parameters to train are now correctly identified: proceed to training.

   if ( train )

      %  Checkpointing is required: save the initial training algorithmic parameters 
      %  and the training context in the file sfname.training.

      if ( savef > 0 )
         savok = bfo_save_training( sfname, p0, verbose ,                                  ...
                                    training_strategy, training_parameters,                ...
                                    training_problems, training_problems_data,             ...
                                    training_set_cutest, trained_bfo_parameters,           ...
                                    training_epsilon, training_maxeval, training_verbosity,...
                                    training_problem_epsilon, training_problem_maxeval,    ...
                                    training_problem_verbosity );
      end

      %  If the user has not specified the training verbosities, choose them to be identical
      %  to that of the main process.

      if ( ~user_training_verbosity )
         training_verbosity = { verbosity, verbosity };
      end

      %  Possibly print information at the beginning of the training process.

      if ( verbose )
         disp( [ ' BFO training is running ... (', training_strategy, ' training strategy)' ])
         disp( ' ' )
         if ( verbose > 2 )
            fprintf( '%s ', '    parameters: ' )
            fprintf( '%s ', training_parameters{ 1:nbr_train_params } )
            fprintf( '\n\n' )
            if ( strcmp( training_strategy, 'average' ) )
               disp( [ '    training epsilon           = ',                                ...
                                                num2str( training_epsilon( 1 ) ) ] )
               disp( [ '    training maxeval           = ',                                ...
                                                num2str( training_maxeval( 1 ) ) ] )
               disp( [ '    training verbosity         = ',                                ...
                                                training_verbosity{ 1 } ] )
            else
               disp( [ '    training epsilon           = [ ',                              ...
                                                num2str( training_epsilon( 1 ) ), ', ',    ...
                                                num2str( training_epsilon( 2 ) ), ' ]' ] )
               disp( [ '    training maxeval           = [ ',                              ...
                                                num2str( training_maxeval( 1 ) ), ', ',    ...
                                                num2str( training_maxeval( 2 ) ) , ' ]' ] )
               disp( [ '    training verbosity         = { ',                              ...
                                                training_verbosity{ 1 } , ', ',            ...
                                                training_verbosity{ 2 }, ' }'  ] )
            end
            disp( [ '    training problem epsilon   = ',                                   ...
                                                num2str( training_problem_epsilon ) ] )
            disp( [ '    training problem maxeval   = ',                                   ...
                                                num2str( training_problem_maxeval ) ] )
            disp( [ '    training problem verbosity = ', training_problem_verbosity ] )
            if ( verbose > 3 )
               disp( ' ' )
               if ( training_set_cutest )
                  bfo_print_cell( '   ', 'training problems from CUTEst',                  ...
                                  training_problems );
               else
                  bfo_print_cell( '   ', 'user_specified training problems',               ...
                                  training_problems );
               end
            end
            fprintf( '\n %30s \n\n', 'Value of the starting algorithmic parameters:' );
            fprintf( ' %.12e   %-12s\n', alpha,   'alpha'   );
            fprintf( ' %.12e   %-12s\n', beta,    'beta'    );
            fprintf( ' %.12e   %-12s\n', gamma,   'gamma'   );
            fprintf( ' %.12e   %-12s\n', del,     'delta'   );
            fprintf( ' %.12e   %-12s\n', eta,     'eta'     );
            fprintf( ' %.12e   %-12s\n', zeta,    'zeta'    );
            fprintf( ' %18d   %-12s\n',  inertia, 'inertia' );
            fprintf( ' %18d   %-12s\n',  stype,   'stype'   );
            fprintf( ' %18d   %-12s\n',  rseed,   'rseed'   );
            fprintf( '\n' );
         end
      end

      %  Train the parameters using the "average" training strategy.

      if ( strcmp( training_strategy, 'average' ) )
	 [ trained_parameters, fperf, msgt, wrnt, ~, ~, ~, ~, training_history ] =         ...
               bfo( @(p,bestperf)bfo_average_perf( p, bestperf,                            ...
                                          training_problems, training_problems_data,       ...
                                          training_set_cutest,                             ...
                                          training_verbosity{ 1 },                         ...
                                          training_problem_epsilon,                        ...
                                          training_problem_maxeval,                        ...
                                          training_problem_verbosity ),                    ...
                    p0,  'xlower', plower, 'xupper', pupper, 'xtype', ptype, 'xscale',     ...
                    pscale, 'epsilon', training_epsilon( 1 ), 'maxeval',                   ...
                    training_maxeval( 1 ), 'verbosity', training_verbosity{ 1 },           ...
                    'termination-basis', 1, 'search-type', 'none', 'save-freq', savef,     ...
                    'restart', restart_training, 'restart-file', sfname, 'f-call-type',    ...
                    'with-bound', 'delta', pdelta );

      %  Train the parameters using the "robust" training strategy.

      elseif ( strcmp( training_strategy, 'robust' ) )
	 [ trained_parameters, fperf, msgt, wrnt, ~, ~, ~, ~, training_history ] =         ...
               bfo( @(p,bestperf)bfo_robust_perf( p,  bestperf,                            ...
                                          training_parameters, training_problems,          ...
                                          training_problems_data, training_set_cutest,     ...
                                          training_epsilon( 2 ), training_maxeval( 2 ),    ...
                                          training_verbosity{ 2 },                         ...
                                          training_problem_epsilon,                        ...
                                          training_problem_maxeval,                        ...
                                          training_problem_verbosity ),                    ...
                    p0,  'xlower', plower, 'xupper', pupper, 'xtype', ptype, 'xscale',     ...
                    pscale, 'epsilon', training_epsilon( 1 ), 'maxeval',                   ...
                    training_maxeval( 1 ), 'verbosity', training_verbosity{ 1 },           ...
                    'termination-basis', 1, 'search-type', 'none', 'save-freq', savef,     ...
                    'restart', restart_training, 'restart-file', sfname, 'f-call-type',    ...
                    'with-bound', 'delta', pdelta );
      end

      if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ) )
         msg   = msgt;
         return
      end
      if ( length( wrnt ) >= 11 && strcmp( wrnt(1:11), ' BFO warning' ) )
	 wrn = wrnt;
      end

      %  Update the current algorithmic parameters for further use in train-and-solve 
      %  training mode.

      for i = 1:nbr_train_params
         if (     strcmp( training_parameters{ i }, 'alpha'       ) )
            alpha = trained_parameters( 1 );
         elseif ( strcmp( training_parameters{ i }, 'beta'        ) )
            beta  = trained_parameters( 2 );
         elseif ( strcmp( training_parameters{ i }, 'gamma'       ) )
            gamma = trained_parameters( 3 );
         elseif ( strcmp( training_parameters{ i }, 'delta'       ) )
            delta = trained_parameters( 4 );
         elseif ( strcmp( training_parameters{ i }, 'eta'         ) )
            eta   = trained_parameters( 5 );
         elseif ( strcmp( training_parameters{ i }, 'zeta'        ) )
            zeta  = trained_parameters( 6 );
         elseif ( strcmp( training_parameters{ i }, 'inertia'     ) )
            inertia = trained_parameters( 7 );
         elseif ( strcmp( training_parameters{ i }, 'search-type' ) )
            stype = trained_parameters( 8 );
         elseif ( strcmp( training_parameters{ i }, 'random-seed' ) )
            rseed = trained_parameters( 9 );
         end      
      end  

      %  Save the trained parameters for future runs.

      fid = fopen( trained_bfo_parameters, 'w' );
      if ( fid == -1 )
         msg = [ ' BFO warning: cannot open file ', trained_bfo_parameters,                ...
                 '! Not saving trained parameters.' ];
         disp( msg )
      else
         fprintf( fid, ' *** BFO trained parameters file %s\n', date );
         fprintf( fid, ' *** (c) Ph. Toint & M. Porcelli\n' );
         fprintf( fid,' %.12e   %-12s\n', alpha,   'alpha'  );
         fprintf( fid,' %.12e   %-12s\n', beta,    'beta'   );
         fprintf( fid,' %.12e   %-12s\n', gamma,   'gamma'  );
         fprintf( fid,' %.12e   %-12s\n', delta,   'delta'  );
         fprintf( fid,' %.12e   %-12s\n', eta,     'eta'    );
         fprintf( fid,' %.12e   %-12s\n', zeta,    'zeta'   );
         fprintf( fid,' %18d   %-12s\n',  inertia, 'inertia');
         fprintf( fid,' %18d   %-12s\n',  stype,   'stype'  );
         fprintf( fid,' %18d   %-12s\n',  rseed,   'rseed'  );
      end
      fclose(fid);

      msg = [ ' Training successful: trained parameters saved in file ',                   ...
              trained_bfo_parameters, '.' ];

      if ( verbose )
         fprintf( '\n%60s\n\n', msg )
         lh      = size( training_history, 1 );
         perf0   = training_history(  1, 4 );
         perfend = training_history( lh, 4 );
         disp( [ ' Performance improved by ', num2str(round( 100*(1- perfend/perf0))),     ...
                    '% in ', num2str( training_history( lh, 3 ) ),                         ...
                    ' problem function evaluations.' ] )
      end

      %  Possibly print the trained parameters and return these to the user,

      if ( verbose > 2 )
         fprintf( '\n %30s \n\n', 'Value of the trained algorithmic parameters:' );
         fprintf( ' %.12e   %-12s\n', alpha,   'alpha'   );
         fprintf( ' %.12e   %-12s\n', beta,    'beta'    );
         fprintf( ' %.12e   %-12s\n', gamma,   'gamma'   );
         fprintf( ' %.12e   %-12s\n', del,     'delta'   );
         fprintf( ' %.12e   %-12s\n', eta,     'eta'     );
         fprintf( ' %.12e   %-12s\n', zeta,    'zeta'    );
         fprintf( ' %18d   %-12s\n',  inertia, 'inertia' );
         fprintf( ' %18d   %-12s\n',  stype,   'stype'   );
         fprintf( ' %18d   %-12s\n',  rseed,   'rseed'   );
      end

      %   Define the string for the discrete search type from the optimized 
      %   value of stype.

      if ( stype == 1 )
         searchtype = 'depth-first';
      elseif ( stype == 0 )
         searchtype = 'breadth-first';
      elseif ( stype == 2 )
         searchtype = 'none';
      end

      %  Terminate if solving is not required.

      if ( ~solve )
         return
      end

   end
end

%%  Start the solve phase

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      Optimization     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                            %
%                    Phase 1: Verify the data and analyze the problem.                       %
%                                                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Verify the multilevel data. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( multilevel )

   nlevel  = max( xlevel );              %  the number of levels
   lmaxmin = size( max_or_min, 1 );      %  the length of the max-or-min specification

   %  Adapt the verbosity level, if level dependent verbosity has been specified.
   %  In this mode, the verbosity cell is used as long as possible, and extended
   %  using its last value if it is too short.

   blank_line = 1;                       %  indicates whether a blank line is printed
   if ( lverbosity > 1 )                 %  after each higher level evaluation
      if ( lverbosity >= level )
         verbose = bfo_get_verbosity( verbosity{ level } );
         if ( level < nlevel && strcmp( verbosity{ level+1 }, 'silent' ) )
            blank_line = 0;
         end
      else
         verbosity{ level } = verbosity{ level-1 };
         verbose = bfo_get_verbosity( verbosity{ level } );
      end
   end

   %  Verify that the input level is acceptable in view of the content of xlevel.

   if ( level < 1 || level > nlevel )
      msg = [ ' BFO error: the level input parameter is not between 1 and max(xlevel)!',   ...
              ' Terminating.' ];
      if ( verbose )
         disp( msg )
      end
      return

   elseif ( level == 1 )

       %  Check the size of the multilevel arguments.
    
      xlevel = abs( round( xlevel) );
      if ( length( xlevel ) ~= n )
         msg = [ ' BFO error: the xlevel input parameter has a size different from n!',    ...
                 ' Terminating.' ];
         if ( verbose )
            disp( msg )
         end
         return
      end
   
      %  Determine if the number of levels is acceptable.
   
      if ( nlevel > max_nlevel )
         msg = [ ' BFO error: too many levels! Terminating.' ];
         if ( verbose )
            disp( msg )
         end
      end
   
      %  Check that every level has at least one active variable.

      levsize  = zeros( 1, nlevel );
      for i = 1:n
         ilevel            = xlevel( i );
         levsize( ilevel ) = levsize( ilevel ) + 1;
      end
      if ( min( levsize ) == 0 )
         msg = [ ' BFO error: not every level is assigned a variable! Terminating.' ];
         if ( verbose )
            disp( msg )
         end
         return
      end
   
      %  Check the max-min specification size and content.
   
      if ( lmaxmin > 1 )
         if ( lmaxmin ~= nlevel )
            msg = [ ' BFO error: the multilevel max-min specification has the wrong size!',...
                    ' Terminating.'];
            if ( verbose )
               disp( msg )
            end
            return
         end
         for lev = 1:nlevel
            if ( strcmp( max_or_min( lev ,: ), 'max' ) == 0 &&                             ...
                 strcmp( max_or_min( lev, : ), 'min' ) == 0  )
               msg = [ ' BFO error: incorrect multilevel max-min specification!',          ...
                       ' Terminating.' ];
               if ( verbose )
                  disp( msg )
               end
               return
            end
         end
      end
    end

   %  Save the original status of the variables and construct the distribution 
   %  of variables within levels.

   xtsave  = xtype;
   for lev = 1:nlevel
      vlevel{lev} = [];
   end
   for i = 1:n
       vlevel{xlevel(i)} = [ vlevel{xlevel(i)} i ];
   end

   %  Decide which of maximization or minimization applies at the current level.

   if ( lmaxmin > 1 )
      if ( strcmp( max_or_min( level, : ), 'max' ) )
         maximize = 1;
      else
         maximize = 0;
      end
      if ( verbose >= 10 )
         disp( [ ' maximize = ', int2str(maximize),' for level ', int2str( level ) ] ) 
      end
   end

   %  Reset the default termination basis number to 1, unless user specified.

   if ( ~user_tbasis )
      term_basis = 1;
   end

   %  Increase the default maxeval, unless specified by the user.

   if ( ~user_maxeval )
      maxeval = 20000;
   end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%   Analyze the various types of variables and the starting point.  %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Find the indices of each type of variable and project the initial
%   point onto the feasible set.  More specifically,
%      icont   contains the indices of the active continuous variables,
%      idisc   contains the indices of the active discrete variables,
%      ifixd   contains the indices of the variables fixed by the user,
%      ifroz   contains the indices of the variables whose value is fixed 
%              in the course of multilevel optimization,
%      iwait   contains the indices of the variables whose value is fixed
%              by a recursive call in the presence of discrete variables.
%   For each type of variable assign the correct scale to the multilevel increments.

nactive    = 0;   % to contain the number of active variables (at current level)
nactivd    = 0;   % to contain the number of discrete active variables (at current level)
icont      = [];  % to contain the indices of continuous variables (at current level)
idisc      = [];  % to contain the indices of discrete variables (at current level)
ifixd      = [];  % to contain the indices of fixed variables (at current level)
ifroz      = [];  % to contain the indices of frozen variables (at current level)
iwait      = [];  % to contain the indices of waiting variables (at current level)
ncbound    = 0;   % to contain the number of continuous variables...
                  %         ... with a finite lower/upper bound (at current level)
ndbound    = 0;   % idem for discrete variables
idprb      = [];  % to contain the indices of the discrete variables in the ...
                  %         ... problem (for all levels)

%  Note that xincr is already set when restart occurs, unless this restart occurs during
%  training and the complete checkpointing file has not been read. Set a flag if it needs
%  to be recomputed.

need_xincr = ~restart || ~readall;

%  Loop on the variables.

%  Find the type of the i-the variable, assign its associated increment to the 
%  multilevel increment structure xincr, freeze it if it does not belong to the 
%  current level and determine the subsets of variables of each type
%  at the current level. Finally project the non-fixed components of the 
%  starting point onto their feasible interval and compute the increment table
%  xincr if needed. Note also that the second column of xincr is set for later
%  reference to the values of the initial increments defined from delta and xscale. 

incb  = [];                                     % no inconsistent bounds so far
for i = 1:n

   %  Continuous variables

   if ( xtype( i ) == 'c' )

      % Assign the multilevel increments at first entry in BFO.

      if ( level == 1 && depth == 0  )
         if ( ~user_scl )             
            xscale( i ) = 1; 
         end
         if ( need_xincr )
            if ( length( delta ) == 1  )
               xincr( i, cur:ini ) = delta;
            else
               xincr( i, cur:ini ) = delta(i);
            end
            xincr( i, uns ) = xscale( i );
         end
         if ( xscale( i ) ~= 1 && ~restart )
            nscaled       = nscaled + 1;
            x0( i )       =     x0( i ) / xscale( i );
            xlower( i )   = xlower( i ) / xscale( i );
            xupper( i )   = xupper( i ) / xscale( i );
            xscale( i )   = 1;
         end
      end

      % Freeze variable i if it is not assigned to the current level.

      if ( multilevel &&  xlevel( i ) ~= level )
          xtype( i ) = 'z';
          ifroz      = [ ifroz i ];

      %   Variable i is possibly active at the current level. Check if it is truly 
      %   active or fixed because its lower and upper bounds are the same (using
      %   the unscaled bounds, if relevant).

      else
         if ( xlower( i ) > xupper( i ) )
            xlower( i ) = -Inf;
            xupper( i ) =  Inf;
            incb        = [ incb i ];
         elseif ( xlower( i ) < xupper( i ) * ( 1 - eps ) )
            icont       = [ icont i ];
            nactive     = nactive + 1;
            x0( i )     = max( xlower( i ), min( x0( i ), xupper( i ) ) );
            if ( xlower( i ) > - myinf || xupper( i ) < myinf )
                ncbound = ncbound + 1;
            end
         else 
            ifixd      = [ ifixd i ];
            x0( i )    = 0.5 *( xlower( i ) + xupper( i ) );
            xtype( i ) = 'f';
         end
      end

   %  Discrete variables

   elseif ( xtype( i ) == 'i' )

      idprb = [ idprb i ];

      % Assign the multilevel increments at first entry in BFO.
      
      if ( level == 1 && depth == 0  && need_xincr )
         if ( ~user_scl )
            xscale( i ) = 1;
         end
         if ( user_delta <= 1 )    
%        if ( user_delta < 1 )
            xincr( i, cur:ini ) = xscale( i );
         else 
            xincr( i, cur:ini ) = delta( i ) * xscale( i );
			
         end
         xincr( i, uns ) = 1;
      end 

      % Freeze variable i if it is not assigned to the current level.

      if ( multilevel &&  xlevel( i ) ~= level )
         xtype( i ) = 'z';
         ifroz      = [ ifroz i ];

      %   Variable i is possibly active at the current level. Check if it is truly 
      %   active or fixed because its lower and upper bounds are the same.

      else
         if ( xlower( i ) > xupper( i ) )
            xlower( i ) = -Inf;
            xupper( i ) =  Inf;
            incb        = [ incb i ];
         elseif ( xlower( i ) < xupper( i ) * ( 1 - eps ) )
            idisc      = [ idisc i ];
            nactive    = nactive + 1;
            nactivd    = nactivd + 1;
            if ( xlower( i ) > - myinf || xupper( i ) < myinf )
                ndbound = ndbound + 1;
            end
            
            %  The variable is truly active: make sure it lies between its lower and
            %  upper bounds.  This can be done one variable at a time if the discrete
            %  variables vary on the (default) canonical lattice.

            if ( ~use_lattice )

               %  No lattice: perform a simple 1D search to find a point 
               %  between the bounds.
               %  (i) the i-th component needs to be increased as it is below its
               %      lower bound.

               if ( x0( i ) < xlower ( i ) )

                  if ( verbose >= 10 )
                     disp( [ ' Increasing the value of the ', num2str( i ),                ...
                             '-th component of x0 for feasibility.'] )
                  end

                  xi = x0( i ) + xincr( i, cur );
                  while ( xi < xlower( i ) )
                     if ( verbose >= 10 )
                        disp( [ '     x(', num2str( i ), ') = ', num2str( xi ) ] )
                     end
                     xi = xi + xincr( i, cur );
                  end
                  if ( xi > xupper( i ) )
                     msg = [ 'BFO error: inconsistent discrete bounds for variable ',      ...
                              num2str(i), '. Terminating.' ];
                     if ( verbose )
                        disp( msg )
                     end
                     return
                  end 
                  x0( i ) = xi;

               %  (ii) the i-th component needs to be decreased as it is above its
               %      upper bound.

               elseif ( x0( i ) > xupper ( i ) )

                  if ( verbose >= 10 )
                     disp( [ ' Decreasing the value of the  ', num2str( i ),               ...
                             '-th component of x0 for feasibility.'] )
                  end

                  xi = x0( i ) - xincr( i, cur );
                  while ( xi > xupper( i ) )
                     if ( verbose >= 10 )
                        disp( [ '     x(', num2str( i ), ') = ', num2str( xi ) ] )
                     end
                     xi = xi - xincr( i, cur );
                  end
                  if ( xi < xlower( i ) )
                     msg = [ 'BFO error: inconsistent discrete bounds for variable ',      ...
                              num2str(i), '. Terminating.' ];
                     if ( verbose )
                        disp( msg )
                     end
                     return
                  end 
                  x0( i ) = xi;
               end
            end
         else 

            %  The lower and upper bound are the same: fix the i-th variable.

            ifixd      = [ ifixd i ];
            x0( i )    = 0.5 * ( xlower( i ) + xupper( i ) );
            xtype( i ) = 'f';
         end
      end

   %  Fixed variables

   elseif ( xtype( i ) == 'f' && need_xincr )
      ifixd = [ ifixd i ];
      if ( level == 1 )
         if ( ~user_scl )
            xscale( i ) = 1;
         end
         if ( depth == 0 )
            xincr( i, cur:ini ) = 0;
            xincr( i, uns )     = 1;
         end
      end

   %  Waiting variables (i.e. temporarily fixed discrete variables within subspace recursion)

   elseif ( xtype( i ) == 'w' && need_xincr )
      iwait = [ iwait i ];
      idprb = [ idprb i ];
      if ( level == 1 && ~user_scl )
         xscale( i )    = 1;
         xincr(  i, 1 ) = 1;
      end

   %  Frozen variables (i.e. variables inactive at the current level)

   elseif ( xtype( i ) == 'z' && need_xincr )
      ifroz = [ ifroz i ];
      if ( level == 1 && ~user_scl )
         xscale( i )    = 1;
         xincr(  i, 1 ) = 1;
      end

   elseif ( need_xincr )
      msg = [ ' BFO error: (internal) impossible variable type: ', xtype( i ),             ...
              ' Terminating.' ];
      disp( msg )
      return
   end

end

%  Build the summary vector of variable types for use within the search-step, if relevant.

if ( searchstep )
   xtss = xtype;
   for i =1:n
      if ( xtss( i ) == 'z' || xtss( i ) == 'w' )
         xtss( i ) = 'f';
      end
   end
end

%  Output warning if inconsistent bounds were found.

lincb = length( incb );
if ( lincb > 0 )
   if ( lincb == 1 )
      wrn = [ ' BFO warning: inconsistent bounds for variable ', num2str( incb ),          ...
              '. Making it unconstrained.' ];
   else
      wrn = [ ' BFO warning: inconsistent bounds for ', int2str( lincb ),                  ...
              ' variables. Making them unconstrained.' ];
   end
   if ( verbose )
      if ( lincb == 1 )
         disp( wrn )
      else
         disp( ' BFO warning: inconsistent bounds for variables' );
         is = 1;
         for iii = 1:ceil( lincb/10 )
            it = min( is + 9, lincb );
            fprintf( '  %5d   %5d   %5d   %5d   %5d   %5d   %5d   %5d   %5d   %5d',        ...
                     incb( is:it ) );
            fprintf( '\n' );
            is = is + 10;
         end
         disp( ' Making these variables unconstrained.' )
      end
   end
end

%  If the starting point is infeasible for its discrete bounds and a lattice is used,
%  apply BFO for finding the (locally) minimal infeasibility on these variables,
%  and exit if no discrete feasible point can be found on the lattice.

if ( nactivd > 0 && norm( x0( idisc )' - max( [ xlower( idisc )';                          ...
                          min( [ x0( idisc )'; xupper( idisc )'] ) ] ), 1 ) )

    %  Adapt the verbosity for the search for a feasible x0.

   if ( verbose >= 4 )
      disp( ' Recomputing a feasible discrete part of x0 on the lattice.' );
      if ( verbose >= 10 )
         fverb = 'medium';
      else
         fverb = 'minimal';
      end
   else
      fverb = 'silent';
   end

   %  Fix the continuous variables and remember the evaluation count.

   xts          = xtype;
   xts( icont ) = 'f';
   neval0       = neval;
   zz           = zeros( size( x0 ) );

   %  Minimize the ell-1 norm of the discrete infeasibilities on 
   %  the user-supplied lattice.

   [ xtry, infeasible, msg0, wrn0, neval, f_hist ] =                                       ...
        bfo( @(x)norm(max([xlower'-x';zz'])+max([x'-xupper';zz']),1), x0,                  ...
             'xscale', xscale, 'xtype', xts, 'maxeval', maxeval, 'verbosity', fverb,       ...
             'nevr', neval, 'fevals-hist', f_hist, 'alpha', alpha, 'beta', beta,           ...
             'gamma', gamma, 'eta', eta, 'zeta', zeta, 'inertia', inertia, 'search-type',  ...
             searchtype, 'random-seed', rseed, 'lattice-basis', latbasis, 'xincr', xincr,  ...
             'reset-random-seed', reset_random_seed, 'search-step-function', ssfname );

   if ( ( length( msg0 ) >= 10 && strcmp( msg0(1:10), 'BFO error' ) ) ||                   ...
        isnan( infeasible ) )
      msg = [ msg0( 1:12 ), ' search for lattice feasible x0 returned the message: ',      ...
              msg0( 13:length( msg0 ) ) ];
      return
   end
   if ( length( wrn0 ) >= 11 && strcmp( wrn0(1:11), 'BFO warning' ) )
      msg = [ wrn0( 1:13 ), ' search for lattice feasible x0 returned the warning ',       ...
              wrn0( 14:length( wrn0 ) ) ];
      wrn = wrna;
   end

   if ( verbose >= 2 )
      disp( [ ' discrete infeasibility = ', num2str( infeasible ), ' computed in ',        ...
              num2str( neval - neval0 ), ' BFO evaluations.' ] )
   end

   %  Terminate if locally infeasible.

   if ( infeasible )
      msg = [ ' BFO error: no feasible discrete point found on the lattice near x0.',      ...
              ' Terminating.' ];
      if ( verbose )
         disp( msg )
      end
      fbest = infeasible;
      return
   end

   %  Use the feasible starting point.

   bfo_print_x( '', 'feasible x0', x0, 0, [], verbose )
   x0 = xtry;

end

ncont = length( icont );   % the number of continuous variables
ndisc = length( idisc );   % the number of discrete variables
nfixd = length( ifixd );   % the number of fixed variables
nfroz = length( ifroz );   % the number of frozen variables
nwait = length( iwait );   % the number of waiting variables
ndprb = length( idprb );   % the total number of discrete variables across levels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  Verify the use of optimized algorithmic parameters %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Check if the file trained_bfo_parameters exists if specified by the user
%  for the solve training mode. If the file exists, read the trained parameters and
%  update the current BFO algorithmic parameters.

if ( use_trained && ~train )
   fid = fopen( trained_bfo_parameters, 'r' );
   if ( fid == -1 )
      wrn = [ ' BFO warning: cannot open file ', trained_bfo_parameters,                   ...
              '! Using default values for BFO parameters.' ];
      if ( verbose )
         disp( wrn )
      end
   else
      filetitle     = fscanf( fid, '%s', 13 );
      alpha         = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      beta          = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      gamma         = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      delta         = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      eta           = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      zeta          = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      inertia       = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
      stype         = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
      rseed         = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
      fclose( fid );
      if ( verbose > 1 )
         disp( [ ' Trained parameters read from file: ', trained_bfo_parameters, '.' ] )
      end
      xincr( icont, cur:ini ) = delta;
      xincr( idisc, cur:ini ) = xscale( i );
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Miscellaneous initializations  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Adapt the unsuccessful bound for maximization.

if ( maximize )
   if ( withbound && ~user_fbound )
      fbound = -Inf;
   end

%   Adapt the default target objective function value if maximization
%   is requested.

   if ( ~user_ftarg )
      ftarget = - ftarget;
   end
end

%   Set the indentation for printout.

indent = '';
if ( verbose > 1 )
   for k = 1:depth+level-1
       indent = [ '     ' indent ];
   end
end

%   Initialize the termination checks counter.

term_loops = 0;

%   Possibly reset the random number generator.

if ( resetrng & ~restart )
   rng( rseed, 'twister' )
end

%  Check size and sufficient independence of the lattice basis vectors.

if ( use_lattice && ndprb > 0 && level == 1 && depth == 0 )
   [ nrlatb, nclatb ] = size( latbasis );

   %  Check that the lattice basis matrix is square.

   if ( nrlatb ~= ndprb && nclatb ~= ndprb )
      msg = ' BFO error: wrong dimension for the lattice-basis! Terminating.';
      if ( verbose )
          disp( msg )
      end
      return
   end

   %  Check that the lattice basis is sufficiently linearly independent.

   if ( min ( abs( eig( latbasis ) ) ) < 10^(-12) )
      msg = ' BFO error: lattice-basis is not numerically linearly independent! Terminating.';
      if ( verbose )
         disp( msg )
      end
      return
   end

   %  Verify the structure of the lattice basis, if multilevel optimization is
   %  requested. (The lattice basis must be separable between levels).
   %  Only perform this verification once (at level 1).

   if ( multilevel && level == 1 )
      for i = 1:ndprb
         id = idprb( i );
         for j = 1:ndprb
            jd = idprb( j );
            if ( xlevel( id ) ~= xlevel( jd ) && latbasis( i, j ) ~= 0 )
               msg = [ ' BFO error: lattice-basis is not coherent',                        ...
                       ' with levels'' definition! Terminating.' ];
               if ( verbose )
                  disp( msg )
               end
               return
            end
         end
      end 
   end

   %  Blow up the lattice basis to dimension n by padding with zero rows and columns.

   if ( ndprb < n )
      newlatb = zeros( n, n );
      for i = 1: n
         for j = 1: ndprb
            if ( idprb( j ) == i )
                newlatb( idprb, i ) = latbasis( :, j );
                break;
             end
         end
      end
      latbasis = newlatb;
   end 
end

%  Check if the lattice is canonical (i.e. aligned with the canonical basis).

if ( verbose >= 10 && ndisc > 0 )
   if ( use_lattice )
      canonical_d = ( norm( latbasis, 'fro' ) == norm( diag( latbasis ), 'fro' ) );
   else
      canonical_d = 1;
   end
end

%   Define the initial mesh size for the current level.

if ( ndisc > 0 )
   cmesh = max( xincr( idisc, cur ) );
end
if ( ncont > 0 )
   cmesh = max( xincr( icont, cur ) );
end
if ( nactive == 0 )
   cmesh = 0;
end

%   If there are scaled continuous variables, store the transformation from scaled 
%   to unscaled variables as a vector. 

if ( nscaled )
   unscaled = xincr( :, uns );
else
   unscaled = [];          % this avoids multiple tests when restarting
end

%   Set up the initial basis of continuous directions.

if ( ncont > 0 )
   sacc        = [];
   Q           = eye( ncont, ncont );
   canonical_c = 1;                  % the current continuous basis is the canonical one.
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%   Possibly print the initial data.  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( verbose >= 10 )
   if ( depth == 0 && level == 1 )
      fprintf( '\n%18s   %-12s\n\n', ' Value', 'Parameter'         );
      fprintf( '%18d   %-12s\n',  n,            'n'                );
      fprintf( '%18d   %-12s\n',  ncont,        'ncont'            );
      fprintf( '%18d   %-12s\n',  ndisc,        'ndisc'            );
      fprintf( '%18d   %-12s\n',  nfixd,        'nfixd'            );
      fprintf( '%18d   %-12s\n',  nfroz,        'nfroz'            );
      fprintf( '%18d   %-12s\n',  nwait,        'nwait'            );
      fprintf( '%18d   %-12s\n',  ncbound,      'ncbound'          );
      fprintf( '%18d   %-12s\n',  nscaled,      'nscaled'          );
      fprintf( '%18d   %-12s\n',  ndprb,        'ndprb'            );
      fprintf( '%18s   %-12s\n',  xtype,        'variables'' type' );
      fprintf( '%18d   %-12s\n',  nactive,      'nactive'          ); 
      fprintf( '%18s   %-12s\n',  fcallt,       'f-call-type'      );
      if ( user_fbound )
         fprintf( '%+.11e   %-12s\n',fbound,    'f-bound'          );
      end
      fprintf( '%18d   %-12s\n',  maxeval,      'maxeval'          );
      fprintf( '%18d   %-12s\n',  searchstep,   'searchstep'       );
      if ( searchstep )
         fprintf( '%18s   %-12s\n',ssfname,     'ssfname'          );
      end
      fprintf( '%18d   %-12s\n',  resetrng,     'resetrng'         );
      fprintf( '%18d   %-12s\n',  verbose,      'verbose'          );
      fprintf( '%18d   %-12s\n',  nlevel,       'nlevel'           );
      if ( use_variable_bounds )
         fprintf( '%18s   %-12s\n', vb_name,     'vb_name'         );
      end
      fprintf( '%+.11e   %-12s\n', ftarget,     'ftarget'          ); 
      fprintf( '%.12e   %-12s\n', epsilon,      'epsilon'          ); 
      fprintf( '%.12e   %-12s\n', bfgs_finish,  'bfgs_finish'      ); 
      fprintf( '%.12e   %-12s\n', alpha,        'alpha'            ); 
      fprintf( '%.12e   %-12s\n', beta,         'beta'             ); 
      fprintf( '%.12e   %-12s\n', gamma,        'gamma'            ); 
      fprintf( '%.12e   %-12s\n', delta,        'delta'            );
      fprintf( '%.12e   %-12s\n', eta ,         'eta'              );
      fprintf( '%.12e   %-12s\n', zeta ,        'zeta'             );
      fprintf( '%18d   %-12s\n',  inertia,      'inertia'          );
      fprintf( '%18d   %-12s\n',  rseed,        'rseed'            );
   end
   if ( ndisc > 0 )
      fprintf( '%18s   %-12s\n',  searchtype,   'search-type'      );
      fprintf( '%18d   %-12s\n\n', use_lattice, 'use_lattice'      );
      if ( use_lattice )
         bfo_print_matrix( indent', 'latbasis', latbasis           );
      end
   end   
   fprintf( '%18d   %-12s\n',  term_basis,      'term_basis'       );
   fprintf( '%18d   %-12s\n',  restart,         'restart'          );
   fprintf( '%18d   %-12s\n\n',  savef,         'save-freq'        );
   if ( ncbound )
      if ( nscaled )
         bfo_print_vector( indent, 'lower bounds', unscaled.*xlower);
      else
         bfo_print_vector( indent, 'lower bounds', xlower          );
      end
   end
   if ( nscaled )
      bfo_print_vector( indent, 'starting point',  unscaled.*x0    );
   else
      bfo_print_vector( indent, 'starting point',  x0              );
   end
   if ( ncbound )
      if ( nscaled )
         bfo_print_vector( indent, 'upper bounds', unscaled.*xupper);
      else
         bfo_print_vector( indent, 'upper bounds', xupper          );
      end
   end
   if ( nscaled )
      if ( ncbound )
         bfo_print_vector( indent, 'scaled lower bounds', xlower   );
      end
      bfo_print_vector( indent, 'scaled starting point',  x0       );
      if ( ncbound )
         bfo_print_vector( indent, 'scaled upper bounds', xupper   );
      end
   end
   bfo_print_vector( indent, 'variables'' (scaled) increments', xincr(:,cur) );
   if ( multilevel )
      bfo_print_vector( indent, 'xlevel', xlevel                   );
   end
   fprintf( '\n \n')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%  Search for a better starting point, if possible.  %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   See if the subspace defined by integer variables values has already been explored.
%   If this is the case, start from the best point stored in s_hist for that subspace.

fxok   = 0;
fbloop = 1;
if ( ( ndisc > 0 && depth > 0 ) || restart )
   [ nh, n1n ] = size( s_hist );
   n1 = n + 1;
   if ( depth > 0 )
      for j = 1:nh
         if ( abs( s_hist( j, rpath ) - x0( rpath )' ) < eps )
            x0    = s_hist( j, 1:n )';
            fbest = s_hist( j, n1 );
            fxok = 1;
            s_xincr = s_hist( j, n1+1:n1n );
            break
         end
      end

      %  If the subspace has already been optimized and there is no continuous variable,
      %  then there is no need to reexplore it because the grid cannot be refined in this
      %  case. The algorithm thus returns the known function value to the previous 
      %  recursion level. Note that there is no need to save the information before 
      %  returning. The situation is the same if the subspace has already been optimized
      %  with continuous variables, but with a grid at least as fine as the current one.

      if ( fxok ) 
         if ( ncont == 0 || max( xincr( icont, cur ) - s_xincr( icont )' ) >= 0 )
            msg   = 'This subspace has been explored already.';
            if ( verbose > 3 )
               disp( msg)
            end
            if ( nscaled )
               xbest = unscaled.*x0;
            else
               xbest = x0;
            end
            return
         end
      end

%  On restart, start the iteration with the best value saved so far.

   elseif ( nh > 0 )
      fbest = Inf;
      for j = 1:nh
         if ( s_hist( j, n1 ) <= fbest )
            xbest = s_hist( j, 1:n )';
            fbest = s_hist( j, n1 );
            fxok  = 1; 
            xincr( :, cur ) = s_hist( j, n1+1:n1n )';
         end
      end

      %  Compute the associated grid size.

      if ( ndisc > 0 )
         cmesh = max( xincr( idisc, cur ) );
      end
      if ( ncont > 0 )
         cmesh = max( xincr( icont, cur ) );
      end

      %  Disable the forward/backward loop if all continuous variables have 
      %  converged already.

      if ( ncont > 0 && max( xincr( icont, cur ) - epsilon * ones( ncont, 1 ) ) <= 0   )
         fbloop = 0;
      end

   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                            %
%                Phase 2: Apply the BFO algorithm to the verified problem.                   %
%                                                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Print header, optimization direction, objective function's name and starting point value. %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Set flags if the current minimization uses one of the training performance functions.

within_average_training = strcmp( shfname, 'bfo_average_perf' );
within_robust_training  = strcmp( shfname, 'bfo_robust_perf'  );

if ( verbose > 1 && ~within_average_training && ~within_robust_training  )
   if ( depth == 0 ) 
      if ( level == 1 && ~train )
         fprintf( '\n')
         fprintf(   '  ********************************************************\n')
         fprintf(   '  *                                                      *\n')
         fprintf(   '  *   BFO: brute-force optimization without derivatives  *\n')
         fprintf(   '  *                                                      *\n')
         fprintf( [ '  *     (c)  Ph. Toint, M. Porcelli, 2015  (',this_version,           ...
                    ')      *\n'] )
         fprintf(   '  *                                                      *\n')
         fprintf(   '  ********************************************************\n')
      else
         fprintf( '\n')
         fprintf( '%s  ********************************************************\n', indent)
      end
      fprintf( '\n')
      if ( multilevel )
         if ( maximize )
            fprintf( '%s  Maximizing %s', indent, shfname )
         else
            fprintf( '%s  Minimizing %s', indent, shfname )
         end
         if ( length( vlevel{level} ) == 1 )
            disp( [ ' on variable ', int2str( vlevel{level} ),                             ...
                    ' (level ',int2str(level),').' ] )
         else
            disp( [ ' on variables ', int2str( vlevel{level} ),                            ...
                    ' (level ',int2str(level),').' ] )
         end
      else
         if ( maximize )
            fprintf( '%s  Maximizing %s ...\n', indent, shfname )
         else
            fprintf( '%s  Minimizing %s ...\n', indent, shfname )
         end
      end
      fprintf( '\n')
   end
   
   if ( fxok && verbose >= 3 )         % the starting point is not 
                                       % that specified on input
      if ( depth > 0 )
         disp( [ indent, '  Switching to a better starting point.' ] )
      else
         disp( [ indent, '  Restarting from a previous run.' ] )
      end
      fprintf( '\n')
   end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Evaluate the objective function at the initial point (which is the best
%   point so far).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xbest = x0;
if ( ~fxok )

   %  Evaluate the objective function by (recursively) performing optimization 
   %  on the next level.

   if ( multilevel && level < nlevel )
      checking = level > 1 && term_basis > 1;
      [ xbest, fbest, msglow, ~, neval, f_hist, xincr ] =                                  ...
          bfo_next_level_objf( level, nlevel, xlevel, neval, f, xbest, checking,           ...
                               f_hist, xtsave, xincr, xscale, xlower, xupper,              ...
                               max_or_min, vb_name, epsilon, bfgs_finish, maxeval,         ...
                               verbosity, fcallt, alpha, beta, gamma, eta, zeta, inertia,  ...
                               searchtype, rseed, term_basis, latbasis, reset_random_seed, ...
                               ssfname );
      if ( verbose >= 2 && blank_line )
         disp( ' ' )
      end
                     
      %  Return if an error occurred down in the recursion.

      if ( length( msglow ) >= 10 && strcmp( msglow(1:10), ' BFO error' ) )
         msg = msglow;
         if ( nscaled )
            xbest = unscaled.*xbest;
         end
         return;
      end

   %  Evaluate the (single level) objective function.
       
   else

      %  The objective function is the (internal) function used for "average" training.

      if ( within_average_training )
         if ( nscaled )
            [ fbest, msgt, wrnt] = f( unscaled.*xbest, fbound );
         else
            [ fbest, msgt, wrnt] = f( xbest, fbound );
         end
         if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ))
            if ( nscaled )
              xbest = unscaled.*xbest;
            end
            return
         end
         nevalt = fbest;
         training_history = [ training_history; [ 1, 0, nevalt, fbest, xbest' ] ];

      %  The objective function is the (internal) function used for "robust" training.

      elseif ( within_robust_training )
         if ( nscaled )
            [ fbest, msgt, wrnt, t_neval ] = f( unscaled.*xbest, fbound );
         else
            [ fbest, msgt, wrnt, t_neval ] = f( xbest, fbound );
         end
         if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ))
            if ( nscaled )
              xbest = unscaled.*xbest;
            end
            return
         end
         nevalt           = t_neval;
         training_history = [ training_history; [ 2, 0, nevalt, fbest, xbest' ] ];

      %  The objective function is that specified by the user.

      elseif ( withbound )
         if ( nscaled )
            fbest = f( unscaled.*xbest, fbound );
         else
            fbest = f( xbest, fbound );
         end
      else
         if ( nscaled )
            fbest = f( unscaled.*xbest );
         else
            fbest = f( xbest );
         end
      end
      neval  = neval + 1;
      f_hist( neval )   = fbest;
      if ( searchstep )
         if ( nscaled )
            x_hist( 1:n, neval ) = unscaled.*xbest;
         else
            x_hist( 1:n, neval ) = xbest;
         end
      end
   end

   %  Check for possible termination induced by the objective function value,
   %  or for undefined function value.

   if ( isnan( fbest ) )
      fbest = Inf;
   elseif ( fbest < - myinf || strcmp( msglow, 'Optimization terminated by the user.' ) )
      msg = 'Optimization terminated by the user.';
      if ( verbose )
         disp( msg )
      end
      if ( nscaled )
         xbest = unscaled.*xbest;
      end
      return
   end

   if ( (  maximize && fbest >= ftarget ) || ( ~maximize && fbest <= ftarget ) )
      msg = ' The objective function target value has been reached. Terminating.';
      if ( verbose )
         disp( msg )
      end
      if ( nscaled )
         xbest = unscaled.*xbest;
      end
      return
   end
end

if ( nactive == 0 )
   if ( multilevel && level > 1 )
      msg = [ indent, '  No active variable at level ', int2str(level),                    ...
              '. Returning to level ', int2str( level - 1 ), '.'];
   else
      msg = [' No active variable. Terminating.'];
   end
   if ( verbose > 1 )
      disp( msg )
   end
   if ( nscaled )
      xbest = unscaled.*xbest;
   end
   return;
end

%  Print zero-th iteration summary.

if ( verbose > 1 )
   if ( within_average_training || within_robust_training )
      if ( verbose > 2 )
         fprintf( '\n' );
      end
      fprintf( '%s train     prob.obj.   training\n', indent );
      fprintf( '%s neval      neval     performance       cmesh     status\n', indent );
      fprintf( '%s%5d  %11d   %+.6e  %4e\n', indent, neval, nevalt, fbest, cmesh );
      if ( verbose > 2 )
         fprintf( '\n' );
      end
   else
      if ( verbose > 3 )
         fprintf( '\n' );
      end
      fprintf( '%s neval        fx        est.crit       cmesh       status\n', indent );
      fprintf( '%s%5d  %+.6e                %4e\n', indent, neval, fbest, cmesh );
      if ( verbose > 3 )
         fprintf( '\n' );
      end
   end
   bfo_print_x( indent, 'x0', xbest, nscaled, unscaled, verbose )
end

%  Save information for possible restart, if requested.

if ( savef > 0 && mod( neval, savef ) == 0 && ~restart )
   s_hist  = bfo_histupd( s_hist, idisc, xbest, fbest, xincr( :, cur ) );
   savok = bfo_save( sfname, shfname, maximize, epsilon, ftarget, maxeval, neval, f_hist,  ...
                     xtype, xincr, xscale, xlower, xupper, verbose, alpha, beta, gamma,    ...
                     eta, zeta, inertia, stype, rseed, term_basis, use_trained, s_hist,    ...
                     latbasis, bfgs_finish, training_history, nscaled, unscaled, ssfname );
   if ( ~savok )
      msg = [ ' BFO error: checkpointing file ', sfname',                                  ...
              ' could not be opened. Skipping checkpointing at ',                          ...
              int2str(neval), ' evaluations.' ];
      if ( verbose )
         disp( msg )
      end
   end
end

%  Terminate if the maximum number of function evaluations has been reached.

if ( neval >= maxeval )
   msg = [ ' Maximum number of ', int2str( maxeval ), ' evaluations of ', shfname,         ...
           ' reached.' ];
   if ( verbose && level == 1 )
      disp( msg )
   end
   if ( nscaled )
      xbest = unscaled.*xbest;
   end
   return;
end

%  flow is meant to contain the current lowest value of the objective 
%  function being minimized, while fbest contains the best original objective 
%  function as defined by the problem.  They differ in sign when BFO is used 
%  for maximization ('min-or-max' = 'max').

if ( maximize )
   flow = - fbest;
else
   flow =   fbest;
end

%   Initialize the current and previous iterates, and the number of successive
%   refinement steps.

x       = xbest;                       % the iterate
fx      = flow;
xp      = x;                           % the previous iterate
fxp     = flow;                        % the function value at the previous iterate
nrefine = 0;                           % the nbr of successive refinements
iqn     = 0;                           % no quasi-Newton matrix yet

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Minimization proper %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   The main iteration's loop

for itg = 1:maxeval

    %%%%%%%%%%%%%%%%%%%%%%%%%%%  POLL STEP  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% The poll step

    %  Initialize the direction of "hopeful descent" to nothing, as it may be 
    %  required even if the forward/backward loop is bypassed after a restart.

    ddir = [];

    %  Verify that the forward/backward iteration loop is necessary, as this may 
    %  not be the case after a restart.

    if ( fbloop )

      %  Initialize the index of the best move in the cycle

      ibest = 0;

      %  Initialize the estimate of the gradient and the full loop indicator
      %  (a full loop is necessary for the estimate of the gradient to be
      %  meaningful, which is detected by setting the full_loop flag.)

      gcdiff    = zeros( ncont, 1 );
      full_loop = 0;

      %  Set a flag if the algorithm is in the convergence checking mode.

      checking = ( level == 1 && term_basis > 1 );

      %  Compute the unknown values of the objective function at the
      %  neighbouring points on the current grid.

      ic    = 0;                    % initialize the continuous variable counter
      for i = 1:n                   % loop on the variables (or basis vectors)

         %  Debug printout

         if ( verbose >= 10 )
            if ( xtype( i ) == 'c' ) 
               disp( [indent, ' considering column ', int2str( i ),                        ...
                     ' of the basis with increment ', num2str( xincr( i, cur ) ) ] )
            elseif ( xtype( i ) == 'i' )
               disp( [indent, ' considering variable ', int2str( i ),                      ...
                     ' with increment ', num2str( xincr( i, cur ) ) ] )
            end
         end

         %  Initialize the trial point.

         xtry = x;

         %  Evaluate the forward point.

         if ( xtype( i ) == 'c' )

            ic = ic + 1;   % the index of variable i in the set of continuous variables
            
            step = xincr( icont, cur ) .* Q( 1:ncont,ic );
            if ( ncbound )
               [ xtry( icont ), alphaf ]  = bfo_feasible_cstep( x( icont ), step,          ...
                                                           xlower( icont ), xupper( icont ) );
            else
               xtry( icont ) = x( icont ) + step;
               alphaf        = norm( xtry( icont ) - x( icont ) );
            end
            xifwd = xtry( i );

         elseif ( xtype( i ) == 'i' )

            %   Avoid reevaluation at points already explored higher in the recursion.

            if ( depth  == 0 || i > rpath( depth ) )
               if ( use_lattice )
                  xtry      = x + xincr( i, cur ) * latbasis( :, i );
               else
                  xtry( i ) = x( i ) + xincr( i, cur );
               end

               % Enforce feasibility.

               if ( ndbound )
                  for ii = 1:n
                     if ( xtry( ii ) < xlower( ii ) || xtry( ii ) > xupper( ii ) )
                        xtry = x;
                        break
                     end
                  end
               end

            end            
            xifwd = xtry( i );  
         end

         %  Compute the  distance between the trial point and the previous iterate.

         dxxp = norm( xtry - xp );

         %  Verify that the trial point is different from the previous iterate.

         if ( dxxp > eps ) 

            %  Verify that the forward point is different from the current iterate. 

            if (  norm( xtry - x ) > eps  )

               %  Evaluate the objective function at the forward point.

               %  1) Evaluate the objective function by (recursively) performing optimization 
               %  on the next level.

               if ( multilevel && level < nlevel ) 
                 
                  [ xtry, ffwd, msglow, ~, neval, f_hist, xincr  ] =                       ...
                       bfo_next_level_objf( level, nlevel, xlevel, neval, f, xtry,         ...
                          checking, f_hist, xtsave,  xincr, xscale, xlower, xupper,        ...
                          max_or_min, vb_name, epsilon, bfgs_finish, maxeval, verbosity,   ...
                          fcallt, alpha, beta, gamma, eta, zeta, inertia, searchtype,      ...
                          rseed, term_basis, latbasis, reset_random_seed, ssfname );
                   if ( verbose >= 2 && blank_line )
                      disp( ' ' )
                   end

                   %  Return if an error occured down in the recursion.

                  if ( length( msglow ) >= 10 && strcmp( msglow(1:10), ' BFO error' ) )
                     msg   = msglow;
                     if ( nscaled )
                        xbest = unscaled.*xbest;
                     end
                     return;
                  end
       
               %  2) Evaluate the (single level) objective function.
       
               else

                  %  The objective function is the (internal) function used for 
                  %  "average" training.

                  if ( within_average_training )
                     if ( nscaled )
                        [ ffwd, msgt, wrnt ] = f( unscaled.*xtry, min( fbound, fbest ) );
                     else
                        [ ffwd, msgt, wrnt ] = f( xtry, min( fbound, fbest ) );
                     end
                     if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ))
                        if ( nscaled )
                           xbest = unscaled.*xbest;
                        end
                        return
                     end
                     nevalt = nevalt + ffwd;

                  %  The objective function is the (internal) function used for 
                  %  "robust" training.

                  elseif ( within_robust_training )
                     if ( nscaled )
                        [ ffwd, msgt, wrnt, t_neval ] = f( unscaled.*xtry, min(fbound,fbest));
                     else
                        [ ffwd, msgt, wrnt, t_neval ] = f( xtry, min( fbound, fbest ) );
                     end
                     if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ))
                        if ( nscaled )
                           xbest = unscaled.*xbest;
                        end
                        return
                     end
                     nevalt = nevalt + t_neval;
 
                  %  The objective function is that specified by the user.

                  elseif ( withbound )
                     if ( maximize )
                        if ( nscaled )
                           ffwd = f( unscaled.*xtry, max( fbound, fbest ) );
                        else
                           ffwd = f( xtry, max( fbound, fbest ) );
                        end
                     else
                        if ( nscaled )
                           ffwd = f( unscaled.*xtry, min( fbound, fbest ) );
                        else
                           ffwd = f( xtry, min( fbound, fbest ) );
                        end
                     end
                  else
                     if ( nscaled )
                        ffwd = f( unscaled.*xtry );
                     else
                        ffwd = f( xtry );
                     end
                  end
                  neval = neval + 1;
                  f_hist( neval ) = ffwd;
                  if ( searchstep )
                     if ( nscaled )
                        x_hist( 1:n, neval ) = unscaled.*xtry;
                     else
                        x_hist( 1:n, neval ) = xtry;
                     end
                  end
               end
               
               %  Print the result, if in debug mode.

               if ( verbose >= 10 )
                  if ( ( xtype( i ) == 'c' && canonical_c ) ||                             ...
                       ( xtype( i ) == 'i' && canonical_d ) )
                     fprintf( '%s coordinate - %3d xifwd = %.12e ffwd = %.12e\n',          ...
                              indent, i, xifwd, ffwd )
                  else
                     fprintf( '%s coordinate - %3d ffwd = %.12e\n', indent, i, ffwd )
                  end
               end

               %  Take maximization into account, if relevant.

               if ( maximize )
                 ffwd = - ffwd;
               end

               %  Possibly terminate at the user's request.

               if ( isnan( ffwd ) )
                    ffwd = Inf;
               elseif ( ffwd < - myinf ||                                                  ...
                    strcmp( msglow, 'Optimization terminated by the user. Terminating.' ) )
                  msg = 'Optimization terminated by the user.';
                  if ( verbose )
                     disp( msg )
                  end
                  if ( nscaled )
                     xbest = unscaled.*xbest;
                  end
                  return
               end

               %  Update the best value so far.

               if ( ffwd < flow ) 
                  xbest = xtry;
                  flow  = ffwd;
                  if ( maximize )
                     fbest = -ffwd;
                  else
                     fbest =  ffwd;
                  end
                  ibest = i;
               end

               if ( (  maximize && fbest >= ftarget ) || ( ~maximize && fbest <= ftarget ) )
                  msg = [ ' The objective function target value has been reached.',        ...
                          ' Terminating.' ];
                  if ( verbose )
                    disp( msg )
                  end
                  if ( nscaled )
                     xbest = unscaled.*xbest;
                  end
                  return
               end

            else
               ffwd = fx;
            end

            %  Terminate if the maximum number of function evaluations
            %  has been reached.

            if ( neval >= maxeval )
               msg = [' Maximum number of ', int2str( maxeval ),                           ...
                      ' evaluations of ', shfname, ' reached.' ];
               if ( verbose && level == 1 )
                  disp( msg )
               end
               if ( savef >= 0 )
                  s_hist  = bfo_histupd( s_hist, idisc, xbest, fbest, xincr( :, cur ) );
                  savok   = bfo_save( sfname, shfname, maximize,  epsilon, ftarget,        ...
                                      maxeval, neval, f_hist, xtype,  xincr,               ...
                                      xscale, xlower, xupper,  verbose, alpha, beta,       ...
                                      gamma, eta, zeta, inertia, stype, rseed, term_basis, ...
                                      use_trained, s_hist, latbasis, bfgs_finish,          ...
                                      training_history, nscaled, unscaled, ssfname );
                  if ( ~savok )
                     msg = [ ' BFO error: checkpointing file ', sfname',                   ...
                             ' could not be opened. Skipping checkpointing at ',           ...
                              int2str(neval), ' evaluations.' ];
                     if ( verbose )
                        disp( msg )
                     end
                  end
               end
               if ( nscaled )
                  xbest = unscaled.*xbest;
               end
               return;
            end

            %  Save information for possible restart, if requested.

            if ( savef > 0 && mod( neval, savef ) == 0 )
               s_hist  = bfo_histupd( s_hist, idisc, xbest, fbest, xincr( :, cur ) );
               savok   = bfo_save( sfname, shfname, maximize, epsilon, ftarget,            ...
                                   maxeval, neval, f_hist, xtype, xincr, xscale, xlower,   ...
                                   xupper, verbose, alpha, beta, gamma, eta, zeta,         ...
                                   inertia, stype, rseed, term_basis, use_trained, s_hist, ...
                                   latbasis, bfgs_finish, training_history, nscaled,       ...
                                   unscaled, ssfname );
               if ( ~savok )
                  msg = [ ' BFO error: checkpointing file ', sfname',                      ...
                          ' could not be opened. Skipping checkpointing at ',              ...
                           int2str(neval), ' evaluations.' ];
                  if ( verbose )
                     disp( msg )
                  end
               end
            end
 
            %  Terminate the loop on the variables if sufficient 
            %  decrease has already been found.

            if ( deltaf > 0 && fx - flow >= eta * deltaf )
               break;
            end
         else   
            ffwd = fxp;
         end

         %  Evaluate the backward point.

         if ( xtype( i ) == 'c' )

            step = - xincr( icont, cur ) .* Q( 1:ncont, ic );
            if ( ncbound )
               [ xtry( icont ), alphab ]  = bfo_feasible_cstep( x( icont ), step,          ...
                                                        xlower( icont ), xupper( icont ) );
            else
               xtry( icont ) = x( icont) + step;
               alphab        = norm( xtry( icont ) - x( icont ) );
            end
            xibwd = xtry( i );

         elseif ( xtype( i ) == 'i' )

            %   Avoid reevaluation at points already explored higher in 
            %   the recursion.

            if ( depth  == 0 || i > rpath( depth ) )
               if ( use_lattice )
                  xtry      = x - xincr( i, cur ) * latbasis( 1:n, i );
               else
                  xtry( i ) = x( i ) - xincr( i, cur );
               end

               % Enforce feasibility.

               if ( ndbound )
                  for ii = 1:n
                     if ( xtry( ii ) < xlower( ii ) || xtry( ii ) > xupper( ii ) )
                        xtry = x;
                        break
                     end
                  end
               end

            end
            xibwd = xtry( i );

         end

         %  Compute the  distance between the trial point and the previous iterate.

         dxxp = norm( xtry - xp );

         %  Verify that the trial point is different from the previous iterate.

         if ( dxxp > eps ) 

            %  Verify that the backward point is different from the current 
            %  iterate. 

            if (  norm( xtry - x ) > 0 )

               %  Evaluate the objective function at the backward point.

               %  1) Evaluate the objective function by (recursively) performing optimization 
               %  on the next level.

               if ( multilevel && level < nlevel )

                  %  Optimize at next level to evaluate the next level function value.

                  [ xtry, fbwd, msglow, ~, neval, f_hist, xincr ] =                        ...
                       bfo_next_level_objf( level, nlevel, xlevel, neval, f, xtry,         ...
                          checking, f_hist, xtsave, xincr, xscale,  xlower, xupper,        ...
                          max_or_min, vb_name, epsilon, bfgs_finish, maxeval, verbosity,   ...
                          fcallt, alpha,  beta, gamma, eta, zeta, inertia, searchtype,     ...
                          rseed, term_basis, latbasis, reset_random_seed, ssfname );

                  if ( verbose >= 2 && blank_line )
                     disp( ' ' )
                  end

                  %  Return if an error occurred down in the recursion.

                  if ( length( msglow ) >= 10 && strcmp( msglow( 1:10 ), ' BFO error' ) )
                     msg = msglow;
                     if ( nscaled )
                        xbest = unscaled.*xbest;
                     end
                     return;
                  end
       

               %  2) Evaluate the (single level) objective function.
       
               else                                         

                  %  The objective function is the (internal) function used for 
                  %  "average" training.

                  if ( within_average_training )
                     if ( nscaled )
                        [ fbwd, msgt, wrnt ] = f( unscaled.*xtry, min( fbound, fbest ) );
                     else
                        [ fbwd, msgt, wrnt ] = f( xtry, min( fbound, fbest ) );
                     end
                     if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ))
                        if ( nscaled )
                           xbest = unscaled.*xbest;
                        end
                        return
                     end
                     nevalt = nevalt + fbwd;

                  %  The objective function is the (internal) function used for 
                  %  "robust" training.

                  elseif ( within_robust_training )
                     if ( nscaled )
                        [ fbwd, msgt, wrnt, t_neval ] = f( unscaled.*xtry, min(fbound,fbest));
                     else
                        [ fbwd, msgt, wrnt, t_neval ] = f( xtry, min( fbound, fbest ) );
                     end
                     if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ))
                        if ( nscaled )
                           xbest = unscaled.*xbest;
                        end
                        return
                     end
                     nevalt = nevalt + t_neval;

                  %  The objective function is that specified by the user.

                  elseif ( withbound  )
                     if ( maximize )
                        if ( nscaled )
                           fbwd = f( unscaled.*xtry, max( fbound, fbest ) );
                        else
                           fbwd = f( xtry, max( fbound, fbest ) );
                        end
                     else
                        if ( nscaled )
                           fbwd = f( unscaled.*xtry, min( fbound, fbest ) );
                        else
                           fbwd = f( xtry, min( fbound, fbest ) );
                        end
                     end
                  else
                     if ( nscaled )
                        fbwd = f( unscaled.*xtry );
                     else
                        fbwd = f( xtry );
                     end
                  end
                  neval = neval + 1;
                  f_hist( neval ) = fbwd;
                  if ( searchstep )
                     if ( nscaled )
                        x_hist( 1:n, neval ) = unscaled.*xtry;
                     else
                        x_hist( 1:n, neval ) = xtry;
                     end
                  end
               end
               
               %  Possibly terminate at the user's request.

               if ( isnan( fbwd ) )
                  fbwd = Inf;
               elseif ( abs( fbwd ) > myinf ||                                             ...
                    strcmp( msglow, ' Optimization terminated by the user. Terminating.' ) )
                  msg = ' Optimization terminated by the user.';
                  if ( verbose )
                     disp( msg )
                  end
                  if ( nscaled )
                     xbest = unscaled.*xbest;
                  end
                  return
               end

               %  Print the result, if in debug mode.

               if ( verbose >= 10 )
                  if ( ( xtype( i ) == 'c' && canonical_c ) ||                             ...
                       ( xtype( i ) == 'i' && canonical_d ) )
                     fprintf( '%s coordinate - %3d xibwd = %.12e fbwd = %.12e\n',          ...
                              indent, i, xibwd, fbwd )
                  else
                     fprintf( '%s coordinate - %3d fbwd = %.12e\n', indent, i, fbwd )
                  end
               end

               %  Take maximization into account, if relevant.

               if ( maximize )
                  fbwd = - fbwd;
               end

               %  Update the best value so far.

               if ( fbwd < flow ) 
                  xbest = xtry;
                  flow  = fbwd;
                  if ( maximize )
                     fbest = -fbwd;
                  else
                     fbest =  fbwd;
                  end
                  ibest = -i;
               end

               if ( (  maximize && fbest >= ftarget ) || ( ~maximize && fbest <= ftarget ) )
                  msg = ' The objective function target value has been reached. Terminating.';
                  if ( verbose )
                     disp( msg )
                  end
                  if ( nscaled )
                     xbest = unscaled.*xbest;
                  end
                  return
               end

            else
               fbwd = fx;
            end

            %  Terminate if the maximum number of function evaluations
            %  has been reached.

            if ( neval >= maxeval )
               msg = [ ' Maximum number of ', int2str( maxeval ), ' evaluations of ',      ...
                       shfname, ' reached.' ];
               if ( verbose && level == 1 )
                  disp( msg )
               end
               if ( savef >= 0 )
                  s_hist = bfo_histupd( s_hist, idisc, xbest, fbest, xincr( :, cur ) );
                  savok  = bfo_save( sfname, shfname, maximize, epsilon,  ftarget,         ...
                                     maxeval, neval, f_hist, xtype,  xincr,                ...
                                     xscale, xlower, xupper, verbose, alpha, beta,         ...
                                     gamma, eta, zeta, inertia, stype, rseed, term_basis,  ...
                                     use_trained, s_hist, latbasis, bfgs_finish,           ...
                                     training_history, nscaled, unscaled, ssfname );
                  if ( ~savok )
                     msg = [ ' BFO error: checkpointing file ', sfname',                   ...
                             ' could not be opened. Skipping checkpointing at ',           ...
                             int2str(neval), ' evaluations.' ];
                     if ( verbose )
                        disp( msg )
                     end
                  end
               end
               if ( nscaled )
                  xbest = unscaled.*xbest;
               end
               return;
            end

            %  Save information for possible restart, if requested.

            if ( savef > 0 && mod( neval, savef ) == 0 )
               s_hist = bfo_histupd( s_hist, idisc, xbest, flow, xincr( :, cur ) );
               savok  = bfo_save( sfname, shfname, maximize, epsilon, ftarget, maxeval,    ...
                                  neval, f_hist, xtype, xincr, xscale, xlower, xupper,     ...
                                  verbose, alpha, beta,  gamma, eta, zeta,  inertia,       ...
                                  stype, rseed, term_basis, use_trained, s_hist,           ...
                                  latbasis, bfgs_finish, training_history, nscaled,        ...
                                  unscaled, ssfname );
               if ( ~savok )
                  msg = [ ' BFO error: checkpointing file ', sfname',                      ...
                          ' could not be opened. Skipping checkpointing at ',              ...
                          int2str(neval), ' evaluations.' ];
                  if ( verbose )
                     disp( msg )
                  end
               end
            end

            %  Terminate the loop on the variables if sufficient 
            %  decrease has already been found.

            if ( deltaf > 0 && fx - flow >= eta * deltaf )
               if ( verbose >= 10 )
                  fprintf( '%s sufficient descent -> breaking poll loop\n', indent )
               end
               break;
            end
         else
            fbwd = fxp;
         end

         %  Update the estimate of the gradient, avoiding the use of
         %  meaningless function values.

         if ( xtype( i ) == 'c' )
            if ( ffwd < Inf & fbwd < Inf )
               if ( abs( xifwd - xibwd ) > eps )
                  gcdiff( ic ) = ( ffwd - fbwd ) / ( alphaf + alphab );
               end
            elseif ( ffwd < Inf )      % avoid using meaningless function values
               if ( abs( xifwd ) > eps )
                  gcdiff( ic ) = ( ffwd - fx ) / alphaf;
               end 
            elseif ( fbwd < Inf )      % avoid using meaningless function values
               if ( abs( xibwd ) > eps )
                  gcdiff( ic ) = ( fx - fbwd ) / alphab;
               end 
            end
         end

         %  Reset the acceptable decrease.

         if ( i == n )
            full_loop = 1;
            if ( fx > flow )
               deltaf = fx - flow;
            end
         end

      end  %  End of the loop on the dimensions

      %  The forward-backward loop on continuous variables has been performed 
      %  completely. As a consequence, an central difference approximation of the
      %  gradient (if it exists) can be computed and possibly used within a
      %  quasi-Newton update.

      if ( ncont > 0 && full_loop )

         %  Compute the approximate gradient and estimate for the criticality measure.

         grad = Q' * gcdiff;
         estcrit = norm( x( icont ) - max( [ xlower( icont )';                             ...
                                             min( [ ( x( icont ) - grad )';                ...
                                                     xupper( icont )' ] ) ] )' ) / ncont;

         %  Compute a very approximate quasi-Newton (BFGS) step.

         ddir = - grad;                      % set the hopeful descent direction to the
                                             % negative gradient
         if ( max( xincr( icont, cur ) - bfgs_finish * ones( ncont, 1 ) ) <= 0 )
                                             % the current continuous mesh is small enough
            if ( iqn  == 0 )                 % set the first 'full grad' iterate
               iqn   = 1;
               gradp = grad;                 % remember the gradient
               xqnp  = x;                    % remember the iterate
               H = eye( ncont, ncont );
            else                                 % no longer the first 'full grad' iterate
               s  = x( icont ) - xqnp( icont );  % the difference in 'full grad' iterates
               if ( norm( s ) > eps )
                  y   = grad - gradp;            % the corresponding gradient difference
                  yts = y' * s;                  % curvature along the diff in iterates
                  yn2 = y' * y;
                  if ( yts > eps )               % the curvature is positive
                     if ( iqn < 0 )
                        H = ( yn2 / yts ) * eye( ncont, ncont );
                     end
                     Hs    = H * s;
                     H     = H - ( Hs * Hs' ) / (s' * Hs) + (y * y') / yts;    % BFGS
                  end
                  gradp = grad;
                  xqnp  = x;
               end
               ddir = - H \ grad;                % the BFGS direction (sort of)
               iqn = itg;
            end
         end

      end

   %  The forward/backward loop was bypassed after a restart or during a termination 
   %  loop with a single variable: define the criticality estimate as unavailable 
   %  and require the loop to be executed from now on.

   else

      full_loop = 0;
      fbloop    = 1; 

   end  %  end of the forward/backward loop

   %  No progress was made during the last loop on polling directions.

   if ( flow >= fx  )
 
      %  Store this best value for possible restart.

      if ( ndisc > 0 && nactive > 0 )
         s_hist = bfo_histupd( s_hist, idisc, xbest, fbest, xincr( :, cur ) );
         if ( verbose >= 10 )
            s_history = s_hist
         end

         %  If integer variables are present and no progress was made,
         %  consider the neighbouring continuous or discrete subproblems.
         %  This is no longer necessary for termination checking loops 
         %  beyond the first, or (obviously) if recursion is not wanted.
         %  Note that recursion is not entered before the first checking loop
         %  in depth-first search mode.

         if ( nactivd > 0  &&                                                              ...
            ( ( stype == 0 && term_loops < 2 ) || ( stype == 1 && term_loops == 1  ) )     ...
              && stype ~= 2 )
          
            %   Print the result of the current iteration.

            if ( verbose > 1 && ndisc > 0 )
               if ( ~full_loop || ndisc == n )
                   fprintf( '%s%5d  %+.6e  ------------  %4e    %+3d\n',                   ...
                            indent, neval, fbest, cmesh, ibest );
               else
                   fprintf( '%s%5d  %+.6e  %4e  %4e    %+3d\n',                            ...
                           indent, neval, fbest, estcrit, cmesh, ibest  );
               end
               bfo_print_x( indent, 'x', xbest, nscaled, unscaled, verbose )
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%  RECURSIVE STEP %%%%%%%%%%%%%%%%%%%%%%%%%%
            %% The recursive step

            %  Define the parameters for the recursive calls.

            xis = xincr;
            if ( stype == 1 )
               xis( icont ) = xincr( icont, ini );
            end

            %  Only perform the checking loops on neighbouring discrete
            %  subspaces during the first top level checking loop.

            if ( term_loops == 1 )
               tbs = term_basis;
            else
               tbs = 1;
            end

            %  Recursively call the minimizer for the forward and backward 
            %  subspaces corresponding to active discrete variables.

            for j = 1:n  

               if ( xtype( j ) == 'i'                     &&                               ...
                    ( depth == 0 || j >  rpath( depth ) ) &&                               ...
                    xupper( j ) - xlower( j ) > eps        )

                  %  Reset the scale and variable types.

                  xs  = x;
                  xts = xtype;

                  %  Declare the j-th variable to be fixed for minimization in
                  %  the associated subspaces (make it a "waiting variable") and
                  %  augment the recursion path accordingly.

                  xts( j ) = 'w';
                  rpj      = [ rpath j ];

                  %  Forward recursive minimization for (discrete) variable j.

                  if ( verbose > 1 )
                     disp( [indent,'      ====  recursive call for the ',                  ...
                      int2str( j ), 'th discrete direction  =====' ] )
                  end

                  if ( use_lattice )
                     xs      = x + xincr( j, cur ) * latbasis( 1:n, j );
                  else
                     xs( j ) = x( j ) + xincr( j, cur );
                  end

                  % Enforce feasibility.

                  if ( ndbound )
                     for ii = 1:n 
                        if ( xs( ii ) < xlower( ii ) || xs( ii ) > xupper( ii ) )
                           xs = x;
                        break
                        end
                      end
                  end

                  %  Call BFO recursively if the trial point differs from 
                  %  the current iterate.

                  if ( norm ( xs - x ) > 0 ) 
                     [ xtry, fxtry, msgs, ~, neval, f_hist, ~, ~, th, s_hist ] =           ...
                           bfo( f, xs, 'xscale', xscale, 'xtype', xts, 'epsilon', 2*cmesh, ...
                                'maxeval', maxeval, 'verbosity', verbosity, 'xlower',      ...
                                xlower, 'xupper', xupper, 'termination-basis', tbs,        ...
                                'sspace-hist', s_hist, 'rpath', rpj, 'deltaf', deltaf,     ...
                                'nevr', neval, 'save-freq', savef, 'f-call-type', fcallt,  ...
                                'max-or-min', max_or_min, 'fevals-hist', f_hist, 'alpha',  ...
                                alpha, 'beta', beta, 'gamma', gamma, 'eta', eta, 'zeta',   ...
                                zeta, 'inertia', inertia, 'search-type', searchtype,       ...
                                'random-seed', rseed, 'lattice-basis', latbasis,           ...
                                'xincr', xis, 'bfgs-finish', bfgs_finish,                  ...
                                'reset-random-seed', reset_random_seed,                    ...
                                'search-step-function', ssfname );

                     %  Return if an error occured down in the recursion.

                     if ( length( msgs ) >= 10 && strcmp( msgs(1:10), ' BFO error' ) )
                        msg = msgs;
                        if ( nscaled )
                           xbest = unscaled.*xbest;
                        end
                        return;
                     end

                     %  Update the training history for what happened in the recursion.

                     if ( within_average_training || within_robust_training )
                        nevalt = nevalt + th( end, 3 );
                     end

                     %  See if a better point has been found. If yes, define it as the 
                     %  new iterate. 

                     if ( maximize )
                        if ( -fxtry < flow ) 
                           xbest = xtry;
                           fbest = fxtry;
                           flow  = -fxtry;
                           ibest = j;
                        end
                     else
                        if ( fxtry < flow ) 
                           xbest = xtry;
                           fbest = fxtry;
                           flow  = fxtry;
                           ibest = j;
                        end
                     end

                  else
                     if ( verbose >= 10 )
                         disp( ['  forward recursive call skipped because ||x - xs|| = ',  ...
                                num2str( norm( x - xs ) ) ]);
                     end                     
                  end

                  %  Terminate if the maximum number of function evaluations
                  %  has been reached.

                  if ( neval >= maxeval )
                     msg   = [ ' Maximum number of ', int2str( maxeval ),                  ...
                               ' evaluations of ', shfname, ' reached.' ];
                     if ( verbose && level == 1 )
                        disp( msg )
                     end
                     if ( savef >= 0 )
                        s_hist = bfo_s_histupd( s_hist, idisc, xbest, fbest, xincr(:, cur) );
                        savok  = bfo_save( sfname, shfname, maximize, epsilon, ftarget,    ...
                                           maxeval, neval, f_hist, xtype, xincr, xscale,   ...
                                           xlower, xupper, verbose, alpha, beta, gamma,    ...
                                           eta, zeta, inertia, stype, rseed, term_basis,   ...
                                           use_trained, s_hist, latbasis, bfgs_finish,     ...
                                           training_history, nscaled, unscaled, ssfname );
                        if ( ~savok )
                           msg = [ ' BFO error: checkpointing file ', sfname',             ...
                                   ' could not be opened. Skipping checkpointing at ',     ...
                                   int2str(neval), ' evaluations.' ];
                           if ( verbose )
                              disp( msg )
                           end
                        end
                     end
                     if ( nscaled )
                        xbest = unscaled.*xbest;
                     end
                     return;
                  end

                  %  Backward recursive minimization for (discrete) variable j.

                  if ( verbose > 1 )
                     disp( [indent, ...
                         '      =========================================================='] )
                  end

                  if ( use_lattice )
                     xs      = x - xincr( j, cur ) * latbasis( 1:n, j );
                  else
                     xs( j ) = x( j ) - xincr( j, cur );
                  end

                  % Enforce feasibility.

                  if ( ndbound )
                     for ii = 1:n
                        if ( xs( ii ) < xlower( ii ) || xs( ii ) > xupper( ii ) )
                           xs = x;
                           break
                        end
                     end
                  end

                  %  Call BFO recursively if the trial point differs from
                  %  the current iterate.

                  if ( norm( xs - x ) > 0 )
                     [ xtry, fxtry, msgs, ~, neval, f_hist, ~, ~, th, s_hist ] =        ...
                           bfo( f, xs, 'xscale', xscale, 'xtype', xts, 'epsilon', 2*cmesh, ...
                                'maxeval', maxeval, 'verbosity', verbosity, 'xlower',      ...
                                xlower, 'xupper', xupper, 'termination-basis', tbs,        ...
                                'sspace-hist', s_hist, 'rpath', rpj, 'deltaf', deltaf,     ...
                                'nevr', neval, 'save-freq', savef, 'f-call-type', fcallt,  ...
                                'max-or-min', max_or_min, 'fevals-hist', f_hist, 'alpha',  ...
                                alpha, 'beta', beta, 'gamma', gamma, 'eta', eta, 'zeta',   ...
                                zeta, 'inertia', inertia, 'search-type', searchtype,       ...
                                'random-seed', rseed, 'lattice-basis', latbasis,           ...
                                'xincr', xis, 'bfgs-finish', bfgs_finish,                  ...
                                'reset-random-seed', reset_random_seed,                    ...
                                'search-step-function', ssfname );

                     %  Return if an error occured down in the recursion.

                     if ( length( msgs ) >= 10 && strcmp( msgs(1:10), ' BFO error' ) )
                        msg = msgs;
                        if ( nscaled )
                           xbest = unscaled.*xbest;
                        end
                        return;
                     end

                     %  Update the training history for what happened in the recursion.

                     if ( within_average_training || within_robust_training )
                        nevalt = nevalt + th( end, 3 );
                     end

                     %  See if a better point has been found. If yes, define it as the 
                     %  new iterate. 

                     if ( maximize )
                        if ( - fxtry < flow ) 
                           xbest = xtry;
                           flow  = -fxtry;
                           ibest = -j;
                        end
                     else
                        if ( fxtry < flow ) 
                           xbest = xtry;
                           flow  = fxtry;
                           ibest = -j;
                        end
                     end

                  else
                     if ( verbose >= 10 )
                         disp( ['  backward recursive call skipped because ||x - xs|| = ', ...
                                num2str( norm( x - xs ) ) ]);
                     end                     
                  end
  
                  %  Terminate if the maximum number of function evaluations
                  %  has been reached.

                  if ( neval >= maxeval )
                     msg = [ ' Maximum number of ', int2str( maxeval ),                    ...
                             ' evaluations of ', shfname, ' reached.' ];
                     if ( verbose && level == 1 )
                        disp( msg )
                     end
                     if ( savef >= 0 )
                        s_hist = bfo_histupd( s_hist, idisc, xbest, fbest, xincr( :, cur ));
                        savok  = bfo_save( sfname, shfname, maximize, epsilon, ftarget,    ...
                                           maxeval, neval, f_hist, xtype, xincr, xscale,   ...
                                           xlower, xupper, verbose, alpha, beta, gamma,    ...
                                           eta, zeta, inertia, stype, rseed, term_basis,   ...
                                           use_trained, s_hist, latbasis, bfgs_finish,     ...
                                           training_history, nscaled, unscaled, ssfname );
                        if ( ~savok )
                           msg = [ ' BFO error: checkpointing file ', sfname',             ...
                                   ' could not be opened. Skipping checkpointing at ',     ...
                                   int2str(neval), ' evaluations.' ];
                           if ( verbose )
                              disp( msg )
                           end
                        end
                     end
                     if ( nscaled )
                        xbest = unscaled.*xbest;
                     end
                     return;
                  end

                  if ( verbose > 1 )
                     disp( [ indent, '      ===  end of recursive call for the ',          ...
                           int2str( j ),'th discrete direction'] )
                  end
               end
            end
         end
      end
   end

   %%%%%%%%%%%%%%%%%%%%%%%%%  USER-DEFINED SEARCH STEP  %%%%%%%%%%%%%%%%%%%%%%
   %% The search step

   successful_search_step = 0; 
   if ( searchstep )

      if ( verbose >= 4 )
         disp( ' ' )
         disp( [ ' Calling the user-supplied search-step function ', ssfname ] )
         if ( verbose >= 10 )
            x_hist
            f_hist
            xtss
            xlower
            xupper
            latbasis
         end
      end

      [ xsearch, fsearch, nevalss ] = bfo_srch( f, x_hist, f_hist, xtss, xlower, xupper,   ...
                                               latbasis );

      % Make sure the returned vector is in column format.

      if ( size( xsearch, 2 ) > 1 )
         xsearch = xsearch';
      end

      % Include the associated evaluation in the evaluation history.

      x_hist = [ x_hist, xsearch ];
      f_hist = [ f_hist, fsearch ];
      neval  = neval + nevalss;

      if ( maximize )
         fsearch = - fsearch;
      end

      %  Move to ( xsearch, fsearch ) if best.

      if ( fsearch < fbest )
         xbest = xsearch;
         fbest = fsearch;
         successful_search_step = 1;
         if ( verbose >= 4 )
            disp( [ ' Successful return from ', ssfname ] )
         end
      else
         if ( verbose >= 4 )
            disp( [ ' Unsuccessful return from ', ssfname ] )
         end
      end

      %  Terminate if the maximum number of function evaluations has been reached.

      if ( neval >= maxeval )
         msg = [' Maximum number of ', int2str( maxeval ), ' evaluations of ', shfname,    ...
                ' reached.' ];
         if ( verbose && level == 1 )
            disp( msg )
         end
         if ( savef >= 0 )
            s_hist  = bfo_histupd( s_hist, idisc, xbest, fbest, xincr( :, cur ) );
            savok   = bfo_save( sfname, shfname, maximize,  epsilon, ftarget,              ...
                                maxeval, neval, f_hist, xtype,  xincr,                     ...
                                xscale, xlower, xupper,  verbose, alpha, beta,             ...
                                gamma, eta, zeta, inertia, stype, rseed, term_basis,       ...
                                use_trained, s_hist, latbasis, bfgs_finish,                ...
                                training_history, nscaled, unscaled, ssfname );
            if ( ~savok )
               msg = [ ' BFO error: checkpointing file ', sfname',                         ...
                       ' could not be opened. Skipping checkpointing at ',                 ...
                        int2str(neval), ' evaluations.' ];
               if ( verbose )
                  disp( msg )
               end
            end
         end
         if ( nscaled )
            xbest = unscaled.*xbest;
         end
         return;
      end

   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  TERMINATION STEP %%%%%%%%%%%%%%%%%%%%%%%%%
   %% The termination step

   %  Record the training history at current iteration, if relevant.
   
   if ( within_average_training )
      if ( depth == 0 && ~maximize )
         ittrain = training_history( end, 2 ) + 1;
         training_history = [ training_history; [ 1, ittrain, nevalt, fbest, xbest' ] ];
      else
         ittrain = training_history( end, 2 ) + 1;
         training_history = [ 1, ittrain, nevalt, fbest, xbest' ];
      end
   elseif ( within_robust_training )
      ittrain = training_history( end, 2 ) + 1;
      training_history = [ training_history; [ 2, ittrain, nevalt, fbest, xbest' ] ];
   end

   %  Determine which bounds on continuous variables are nearly saturated at 
   %  xbest, if any, as well as the matrix of normal to these bounds. (A bound 
   %  is nearly saturated when its distance from the current iterate is not 
   %  larger than the current grid spacing).

   isat = [];              % to be the set of nearly saturated continuous variables
   nsat = 0;               % the number of nearly saturated bounds
   N    = [];              % the normals to nearly saturated bounds
   if ( ncbound > 0 && ncont > 0 )
      ic = 0;
      for i = icont
         ic = ic + 1;      % the index of i in the set of continuous variables

         %  Check if nearly saturated.

         if ( xbest( i )  - xlower( i ) <= xincr( i, cur ) ||                              ...
              xupper( i ) - xbest( i )  <= xincr( i, cur ) )
            if ( verbose >= 10 )
               disp( [' variable ', int2str(i), ' has a (nearly) saturated bound'] )
            end
            isat               = [ isat ic ];
            nsat               = nsat + 1;
            N( 1:ncont, nsat ) = zeros( ncont, 1 );
            N( ic, nsat )      = 1;
         end
      end

      if ( verbose >= 10 )
         disp( [' a total of ', int2str( nsat ), ' continuous variables (out of ',         ...
                  int2str( ncont ),') are nearly saturated'] )
         if ( nsat > 0 )
            corresponding_N = N( 1:ncont,1:nsat )
         end
      end

   end

   %  Progress has been made.

   if ( flow < fx )

      %  Reset the termination loop counter.

      term_loops = 0;

      %  Print the one-line iteration summary.

      if ( verbose > 1  )
         if ( within_average_training || within_robust_training )
            if ( verbose > 2 )
               fprintf( '\n' );
               fprintf( '%s train     prob.obj.   training\n', indent );
               fprintf( '%s neval       neval    performance       cmesh     status\n',    ...
                        indent );
            end
            fprintf( '%s%5d  %11d   %+.6e  %4e    %+3d\n',                                 ...
                     indent, neval, nevalt, fbest, cmesh, ibest );
            if ( verbose > 2 )
               fprintf( '\n' );
            end
         else
            if ( verbose > 3 )
               fprintf( '\n' );
               fprintf( '%s neval        fx        est.crit       cmesh       status\n',   ...
                        indent );
            end
            if ( ~full_loop || ndisc == n )
               fprintf( '%s%5d  %+.6e  ------------  %4e    %+3d\n',                       ...
                        indent, neval, fbest, cmesh, ibest );
            else
               fprintf( '%s%5d  %+.6e  %4e  %4e    %+3d\n',                                ...
                        indent, neval, fbest, estcrit, cmesh, ibest  );
            end
            if ( verbose > 3 )
               fprintf( '\n' );
            end
         end
         bfo_print_x( indent, 'x', xbest, nscaled, unscaled, verbose )
      end

      %  Accumulate the average direction of descent over the last 
      %  inertia iterations.

      if ( ncont > 0 ) 
         ns = size( sacc, 2 );
         s  = xbest - x;
         if ( inertia > 0 )
            if ( ns == 0 )
               sacc = [ s( icont ) ];
            elseif ( ns < inertia )
               sacc = [ sacc s( icont ) ];
            else
               sacc = [ sacc( 1:ncont,2:inertia ) s( icont ) ];
            end
            avs = sum( sacc, 2 );

            %  Use the average direction if an approximate gradient is not available.

            if ( ~full_loop )
               ddir = avs;
            end
         end

         %  Project the average direction onto the nullspace of the 
         %  nearly saturated bounds and ensure the normals of the nearly 
         %  saturated constraints belong to the new basis. 

         Q =  bfo_new_continuous_basis( ncont, N, ddir );
         if ( verbose >= 10 )
            newQ1 = Q
            canonical_c =  nsat >= ncont - 1;
         end
      end

      %  Expand the grid for continuous variables after a successful iteration, except
      %  if success is due to the user search-step, in which case grid size is irrelevant.
      %  Note that the actual (expanded) increments for discrete variables 
      %  are never used.

      if ( ncont > 0 && ~successful_search_step )
         xincr( icont, 1 ) = min( [ xupper( icont )' - xlower( icont )';                   ...
                                    alpha * xincr( icont, cur )'       ;                   ...
                                    gamma * ones( ncont, 1 )'            ] )';
         cmesh   = max( xincr( icont, cur ) );
         nrefine = -1;              % reset the nbr of successive refinements to 3 
                                    % (= 2 -(-1)) before acceleration occurs again.
      end

      %  Move to the best point.

      xp  = x;
      fxp = fx;
      x   = xbest;
      fx  = flow;

   %  Test for termination on the current grid and overall.

   else

      %  Further grid refinement is possible, possibly leading 
      %  to further minimization.

      if ( ncont > 0 && max( xincr( icont, cur ) - epsilon * ones( ncont, 1 ) ) > 0 ) 

         %  Print the one-line iteration summary.

         if ( verbose > 1 )
            if ( within_average_training || within_robust_training )
               if ( verbose > 2 )
                  fprintf( '\n' );
                  fprintf( '%s train     prob.obj.   training\n', indent );
                  fprintf( '%s neval       neval    performance       cmesh     status\n', ...
                           indent );
               end
               fprintf( '%s%5d  %11d   %+.6e  %4e  refine\n',                              ...
                        indent, neval, nevalt, fbest, cmesh, ibest );
               if ( verbose > 2 )
                  fprintf( '\n' );
               end
            else
               if ( verbose > 3 )
                  fprintf( '\n' );
                  fprintf( '%s neval        fx        est.crit       cmesh       status\n',...
                         indent );
               end
               fprintf( '%s%5d  %+.6e  %4e  %4e  refine\n',                                ...
                        indent, neval, fbest, estcrit, cmesh );
               if ( verbose > 3 )
                  fprintf( '\n' );
               end
            end
            bfo_print_x( indent, 'x', x, nscaled, unscaled, verbose )
         end
         
         %  Refine the grid for continuous variables, accelerating when more  
         %  than 2 successive refinements have taken place. Note that the actual
         %  (reduced) increments for discrete variables are never used.

         nrefine = nrefine + 1;
         if ( nrefine > 2 )
            shrink = beta * beta;
         else
            shrink = beta;
         end
         xincr( icont, cur ) = max( [ 0.5 * epsilon * ones( ncont, 1 )';
                                    shrink * xincr( icont, cur )' ] );
         deltaf = shrink * deltaf;
         cmesh  = max( xincr( icont, cur ) );

         %   Choose a new random basis for the continuous variables.

         Q = bfo_new_continuous_basis( ncont, N, ddir );
         if ( verbose >= 10 )
             canonical_c = nsat >= ncont - 1;
             newQ2 = Q
         end

      %  Convergence is achieved on the finest grid.

      else

         %  Increment the number of termination loops performed so far.

         term_loops = term_loops + 1;

         %  Termination has been verified for the required term_basis random 
         %  basis for the continuous variables.
         %  Note that another termination loop is useless if there are no continuous 
         %  variables.  If there is only one continuous variable, an additional termination 
         %  loop is only potentially useful for exploring neighbouring discrete 
         %  subspaces (if any) for depth-first search.

         if  ( ( term_loops >= term_basis || ncont == 0 ) ||                               ...
               ( ncont == 1 && ( ndisc == 0 || stype ~= 1 ) ) )

            if ( verbose > 1 )
               if ( within_average_training || within_robust_training )
                  if ( verbose > 2 )
                     fprintf( '\n' );
                     fprintf( '%s train     prob.obj.   training\n', indent );
                     fprintf( [ '%s neval       neval    performance',                     ...
                                '       cmesh     status\n'], indent );
                  end
                  fprintf( '%s%5d  %11d   %+.6e  %4e  converged\n',                        ...
                           indent, neval, nevalt, fbest, cmesh, ibest );
                  if ( verbose > 2 )
                     fprintf( '\n' );
                  end
               else
                  if ( verbose > 3 )
                     fprintf( '\n' );
                     fprintf( [ '%s neval        fx        est.crit',                      ...
                                '       cmesh       status\n'],  indent );
                  end
                  fprintf( '%s%5d  %+.6e  %4e  %4e  converged\n',                          ...
                           indent, neval, fbest, estcrit, cmesh  );
                  if ( verbose > 3 )
                     fprintf( '\n' );
                  end
               end
               bfo_print_x( indent, 'xbest', xbest, nscaled, unscaled, verbose )
            end

            if ( depth == 0 )
               if ( multilevel )
                  msg = [ indent, ' Convergence at level ', int2str( level ), ' in ',      ...
                          int2str( neval ), ' evaluations of ', shfname, '.'];
               else
                  msg = [ indent, ' Convergence in ', int2str( neval ),                    ...
                       ' evaluations of ', shfname, '.'];
               end
               if ( verbose > 1 )
                  disp( msg )
               end
            end

            if ( savef >= 0 )
               s_hist = bfo_histupd( s_hist, idisc, xbest, fbest, xincr( :, cur ) )
               savok  = bfo_save( sfname, shfname, maximize, epsilon, ftarget,             ...
                                  maxeval, neval, f_hist, xtype, xincr, xscale,            ...
                                  xlower, xupper, verbose, alpha, beta, gamma,  eta,       ...
                                  zeta, inertia, stype, rseed, term_basis, use_trained,    ...
                                  s_hist, latbasis, bfgs_finish, training_history, nscaled,...
                                  unscaled, ssfname );
               if ( ~savok )
                  msg = [ ' BFO error: checkpointing file ', sfname',                      ...
                          ' could not be opened. Skipping final checkpointing.' ];
                  if ( verbose )
                     disp( msg )
                  end
               end
            end
            if ( nscaled )
               xbest = unscaled.*xbest;
            end
            return;

         %  More loops on random basis are required to assert termination.

         else
            
            if ( verbose > 1 )
               if ( within_average_training || within_robust_training )
                  if ( verbose > 2 )
                     fprintf( '\n' );
                     fprintf( '%s train     prob.obj.   training\n', indent );
                     fprintf( [ '%s neval       neval    performance',                     ...
                                '       cmesh     status\n' ], indent );
                  end
                  fprintf( '%s%5d  %11d   %+.6e  %4e  checking\n',                         ...
                           indent, neval, nevalt, fbest, cmesh, ibest );
                  if ( verbose > 2 )
                     fprintf( '\n' );
                  end
               else
                  if ( estcrit >= 0 )
                     if ( verbose > 3 )
                        fprintf( '\n' );
                        fprintf( [ '%s neval        fx        est.crit',                   ...
                                   '       cmesh       status\n'],  indent );
                     end
                     fprintf( '%s%5d  %+.6e  %4e  %4e  checking\n',                        ...
                              indent, neval, fbest, estcrit, cmesh  );

                  %  estcrit may (exceptionally) be -1 when BFO has been
                  %  restarted after convergence has occurred in the continuous
                  %  variables : the forward-backward loop on these variables
                  %  is then skipped at the first iteration of the restarted
                  %  algorithm and estcrit is not assigned a meaningful
                  %  value. Moreover printing a new function value is only
                  %  informative if there are discrete variables.

                  elseif ( ndisc > 0 ) 
                     if ( verbose > 3 )
                        fprintf( '\n' );
                        fprintf( [ '%s neval        fx                ',                   ...
                                   '       cmesh       status\n'],  indent );
                     end
                     fprintf( '%s%5d  %+.6e                %4e  checking\n',               ...
                              indent, neval, fbest, cmesh  );
                  end
                  if ( verbose > 3 )
                     fprintf( '\n' );
                  end
               end
               bfo_print_x( indent, 'xbest', xbest, nscaled, unscaled, verbose )
            end

            % Reset the number of successive refinements to 3 (= 2 -(-1)) before 
            % acceleration occurs again.

            nrefine = -1;

            %   Again, if there is only one continuous variable, the new termination 
            %   loop is only potentially useful for exploring neighbouring discrete 
            %   subspaces (if any) for depth-first search.  If this the case, unset the
            %   fbloop flag to skip the continuous variables in the next termination 
            %   loop. 

            fbloop = ( ncont > 1  || ndisc == 0 || stype ~= 1 );

            %   Choose a new random basis for the continuous variables.

            if ( fbloop  && ncont > 1 )
               Q = bfo_new_continuous_basis( ncont, N, ddir );
               if ( verbose >= 10 ) 
                  canonical_c = nsat >= ncont - 1;
                  newQ3 = Q
               end
            end

         end
      end
   end

end %  end of the main optimization loop

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Algorithm's termination  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Exit because the maximum number of iterations was reached and save restart
%   information if requested.

msg = [' Maximum number of ', int2str( maxeval ),                                          ...
       ' evaluations of ', shfname, ' reached.' ];
if ( verbose && level == 1 )
   disp( msg )
end
if ( savef >= 0 )
   s_hist = bfo_histupd( s_hist, idisc, xbest, fbest, xincr( :, cur ) );
   savok  = bfo_save( sfname, shfname, maximize, epsilon, ftarget, maxeval,  neval,        ...
                      f_hist, xtype, xincr, xscale, xlower, xupper, verbose,               ...
                      alpha, beta, gamma, eta, zeta, inertia, stype, rseed, term_basis,    ...
                      use_trained, s_hist, latbasis, bfgs_finish, training_history,        ...
                      nscaled, unscaled, ssfname );
   if ( ~savok )
      msg = [ ' BFO error: checkpointing file ', sfname',                                  ...
              ' could not be opened. Skipping final checkpointing.' ];
      if ( verbose )
         disp( msg )
      end
   end
end

if ( nscaled )
   xbest = unscaled.*xbest;
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function Q = bfo_new_continuous_basis( ncont, N, ddir )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Compute a new random basis Q for the continuous variables, taking care to include 
%  normals to the (nearly) saturated bound constraints (N) and possibly including a hopeful 
%  descent direction (ddir).

%  INPUT:

%  ncont : the number of continuous variables
%  N     : an array whose columns contain the normalized normals to the (nearly) saturated 
%          constraints, [] is no constraint is nearly saturated
%  ddir  : a hopeful direction of descent in the continuous variables, [] if none available

%  OUTPUT:

%  Q     : the new basis

%  DEPENDENCIES : -

%  PROGRAMMING: Ph. Toint and M. Porcelli, May 2010. (This version 22 IX 2015)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

use_ddir = ( length( ddir ) > 0 );
nsat     = size( N, 2 );

if ( nsat > 0 )
   if ( nsat < ncont )
      if ( use_ddir && nsat < ncont-1 )
         [ Q, ~ ] = qr( [ N( 1:ncont, 1:nsat ) ddir rand( ncont, ncont-nsat-1 ) ] );
      else
         [ Q, ~ ] = qr( [ N( 1:ncont, 1:nsat ) rand( ncont, ncont-nsat ) ] );
      end
   else
      Q = N( 1:ncont, 1:ncont );
   end
else
   if ( use_ddir & ncont > 1 )
      [ Q, ~ ] = qr( [ ddir rand( ncont, ncont-1 ) ] );
   elseif ( ncont > 1 )
      [ Q, ~ ] = qr( rand( ncont, ncont ) );
   else
      Q = 1;
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ totalnf, msg, wrn ] = bfo_average_perf( p0, bestperf,                           ...
                                                   training_problems,                      ...
                                                   training_problems_data,                 ...
                                                   training_set_cutest,                    ...
                                                   training_verbosity,                     ...
                                                   training_problem_epsilon,               ...
                                                   training_problem_maxeval,               ...
                                                   training_problem_verbosity )
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Optimize BFO parameters using the Audet-Orban formulation, that is
%  minimizing the total number of function evaluations (or, equivalently, its
%  average per problem).

%  INPUT:

%  p0                     : the initial value of the parameters to be trained
%  betsperf               : the value of the best performance seen so far
%  training_problems      : the list of objective functions of problems to be used 
%                           for training
%  training_problems_data : the associated list of data functions
%  training_set_cutest    : the associated problems library 
%  training_maxeval       : the maximum numbers of training function evaluations
%  training_verbosity     : the verbosity of the training process
%  training_problem_verbosity : the verbosities for the solution of the training problems
%  training_problem_epsilon : the accuracy at which each of the training
%                           problems must be solved
%  training_problem_maxeval  : the maximum number of objective evaluations in the
%                           solution of each training problem

%  OUTPUT:

%  totalnf : the total number of function evaluation to solve all problems in
%            the training set
%  msg     : a message returned from the training process
%  wrn     : a warning returned from the training process

%  DEPENDENCIES : bfo, training_problems{i}, training_problems_data{i},
%                 bfo_cutest_data

%  PROGRAMMING: Ph. Toint and M. Porcelli, May 2010. (This version 2 I 2015)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Set default output.

msg     = '';
wrn     = '';
totalnf = 0;

%  Decode the meaning of each parameter

alpha    = p0(1);
beta     = p0(2);
gamma    = p0(3);
delta    = p0(4);
eta      = p0(5);
zeta     = p0(6);
inertia  = p0(7);
rseed    = p0(9);
tverbose = bfo_get_verbosity( training_verbosity );

%  Choose the associated search-strategy (in string form).

if ( p0(8) == 0 )
   stype   = 'breadth-first';
elseif ( p0(8) == 1 )
   stype   = 'depth-first';  % default
elseif ( p0(8) == 2 )
   stype   = 'none';
end

%  Loop over all problems in the training set, accumulating function evaluations.

nproblems      = length( training_problems );

for i = 1:nproblems

   %  The training library is a set of CUTEst problems: get the problem dependent
   %  data from CUTEst.
   
   if ( training_set_cutest )  

      [ x0, xlower, xupper, xtype ] =  bfo_cutest_data( training_problems{ i } );
      max_or_min = 'min';    
      fun        = @cutest_obj;                        % the CUTEst training function 
      nout       = 4;
      xscale     = ones( 1, length( x0 ) );

   %  The training library is a user-supplied set of problems whose handles are 
   %  available in training_problems.

   else       

      fun  = training_problems{ i };                   % the particular training function
      if ( length( training_problems_data ) == 1 )
         fun_data = training_problems_data{ 1 };
      else
         fun_data = training_problems_data{ i };
      end

      %  Read the data associated with the current problem, depending on how
      %  much information is provided by the user-supplied data function.
      
      nout = nargout( fun_data );   % the number of outputs in its associated data function

      if ( nout == 1 )
         x0         = feval( fun_data );
         n          = length( x0 );
         xlower     = -Inf * ones( 1, n );
         xupper     =  Inf * ones( 1, n );
         xtype      = 'c';
         max_or_min = 'min';
         xscale     = ones( 1, length( x0 ) );
      elseif ( nout == 3 )
         [x0, xlower, xupper ] = feval( fun_data );
         xtype      = 'c';
         max_or_min = 'min';
         xscale     = ones( 1, length( x0 ) );
      elseif ( nout == 4 )
         [x0, xlower, xupper, xtype ] = feval( fun_data );
         max_or_min = 'min';
         xscale     = ones( 1, length( x0 ) );
      elseif ( nout == 5 )
         [x0, xlower, xupper, xtype, xscale ] = feval( fun_data );
         max_or_min = 'min';
      elseif ( nout == 6 )
         [x0, xlower, xupper, xtype, xscale, max_or_min ] = feval( fun_data );
      elseif ( nout == 7 )
         [x0, xlower, xupper, xtype, xscale, max_or_min, xlevel ] = feval( fun_data );
      elseif ( nout == 8 )
         [x0, xlower, xupper, xtype, xscale, max_or_min, xlevel, vb_name ] =               ...
            feval( fun_data );
      else
         disp( [' BFO warning: error in output sequence of the ', int2str( i ),            ...
                '-th training_problem_data function ! Results unreliable !'] );
      end
   end
  
   %  Solve the current test problem with the current set of algorithmic parameters.

   if ( tverbose > 2 )
      if ( training_set_cutest ) 
         shpbname = training_problems{ i };
      else
         [ ~, shpbname ] = bfo_exist_function( func2str( fun ) );
      end
      fprintf( '%-40s', [' BFO training: running ', shpbname, ' ...' ] );
   end

   if ( nout < 7 )          % single level optimization

      [ ~, ~, msgp, wrnp, neval ] = bfo( fun , x0, 'xscale', xscale, 'xlower', xlower,     ...
             'xupper', xupper, 'epsilon', training_problem_epsilon,                        ...
             'maxeval', training_problem_maxeval, 'verbosity', training_problem_verbosity, ...
             'alpha', alpha, 'beta', beta, 'gamma', gamma, 'delta', delta, 'eta', eta,     ...
             'zeta', zeta, 'inertia', inertia, 'search-type', stype, 'random-seed', rseed, ...
              'xtype', xtype, 'max-or-min', max_or_min );

   elseif ( nout < 8 )      % multilevel optimization without variable bounds

      [ ~, ~, msgp, wrnp, neval ] = bfo( fun , x0, 'xscale', xscale, 'xlower', xlower,     ...
             'xupper', xupper, 'epsilon', training_problem_epsilon,                        ...
             'maxeval', training_problem_maxeval, 'verbosity', training_problem_verbosity, ...
             'alpha', alpha, 'beta', beta, 'gamma', gamma, 'delta', delta,'eta', eta,      ...
             'zeta', zeta, 'inertia', inertia, 'search-type', stype, 'random-seed', rseed, ...
             'xtype', xtype, 'max-or-min', max_or_min, 'xlevel', xlevel );

   else                     % multilevel optimization with variable bounds

      [ ~, ~, msgp, wrnp, neval ] = bfo( fun , x0, 'xscale', xscale, 'xlower', xlower,     ...
             'xupper', xupper, 'epsilon', training_problem_epsilon,                        ...
             'maxeval', training_problem_maxeval, 'verbosity', training_problem_verbosity, ...
             'alpha', alpha, 'beta', beta, 'gamma', gamma, 'delta', delta, 'eta', eta,     ...
             'zeta', zeta, 'inertia', inertia, 'search-type', stype, 'random-seed', rseed, ...
             'xtype', xtype, 'max-or-min', max_or_min, 'xlevel', xlevel, 'variable-bounds',...
              vb_name );

   end

   %  Verify the result of this optimization.

   if ( length( msgp ) >= 10 && strcmp( msgp(1:10), ' BFO error' ) )
      msg = [ ' BFO error: training on problem ', func2str(fun), ' returned the message:', ...
                msgp(12,:) ];
      if ( tverbose )
         disp( msg )
      end
      return
   end

   if ( length( wrnp ) >= 11 && strcmp( wrnp(1:11), ' BFO warning' ) )
      msg = [ ' BFO warning: training on problem ', func2str( fun ),                       ...
              ' issued the warning:', wrnp(13,:) ];   
      if ( verbose )
         disp( wrn )
      end
   end

   if ( tverbose > 2 )
      fprintf( '%15s %5d\n', ' Done: neval = ', neval );
   end

   %  Cleanly terminate the call to CUTEst.

   if ( training_set_cutest )  
      cutest_terminate()
   end

   %  Update the function evaluation count.

   totalnf = totalnf + neval;

   %  Exit if the current performance is already beyond the best.

   if ( totalnf > bestperf )
      break;
   end
end

return 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ fworst, msg, wrn, nevalt ] = bfo_robust_perf( p0, bestperf,                     ...
              training_parameters, training_problems, training_problems_data,              ...
              training_set_cutest, training_epsilon, training_maxeval, training_verbosity, ...
              training_problem_epsilon, training_problem_maxeval, training_problem_verbosity )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Optimize BFO parameters using the robust optimization formulation, that is
%  minimizing the worst behaviour of the algorithm on the training problems for a set
%  of algorithmic parameters differing from at most 5% from its nominal value 
%  (i.e. min max (total number of fevals))).

%  INPUT:

%  p0                     : the initial value of the parameters to be trained
%  best_perf              : the best "worst performance" so far
%  training_parameters    : a cell containing the names of the parameters to be trained
%  training_problems      : the list of objective functions of problems to be used 
%                           for training
%  training_problems_data : the associated list of data functions
%  training_set_cutest    : the associated problems library 
%  training_epsilon       : the accuracy to be used for training optimization
%  training_maxeval       : the maximum number of training function evaluations
%  training_verbosity     : the verbosity of the training problem itself 
%  training_problem_epsilon : the accuracy at which each of the training
%                           problems must be solved
%  training_problem_maxeval : the maximum number of objective evaluations in the
%                           solution of each training problem
%  training_problem_verbosity : the verbosity of the solution of each test problem

%  OUTPUT:

%  fworst                 : the maximum number of function evaluations for the
%                           solution of the problems in the training set
%                           within a box centered at the supplied parameter
%                           set p0 and varying those of +/- 5% for continuous
%                           variables
%  msg                    : a message returned from the training process
%  wrn                    : a warning returned from the training process
%  nevalt                 : the number of training problem's function
%                           evaluation during the call to bfo_robust_perf

%  DEPENDENCIES : bfo, bfo_average_perf

%  PROGRAMMING: Ph. Toint,and M. Porcelli, May 2010. (This version 1 I 2015)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fup = 1.05;                 %  the relative upper limit of the box (1 + 5%)
fdo = 0.95;                 %  the relative lower limit of the box (1 - 5%)

tverbose = bfo_get_verbosity( training_verbosity );  % the local verbosity

%  Define the box (for the continuous algorithmic parameters to be trained) in which 
%  the worst performance is sought.

ptyp  = 'fffffffff';                         % all are fixed by default
xlow  = p0;
xupp  = p0;
xsca  = ones( size( p0 ) );
for  i = 1:length( training_parameters )      % loop on the selected training parameters

   if (     strcmp( training_parameters{ i }, 'alpha'       ) )
      ptyp( 1 ) = 'c';
      xlow( 1 ) = fdo * p0( 1 );
      xupp( 1 ) = fup * p0( 1 );
   elseif ( strcmp( training_parameters{ i }, 'beta'        ) )
      ptyp( 2 ) = 'c';
      xlow( 2 ) = fdo * p0( 2 );
      xupp( 2 ) = fup * p0( 2 );
   elseif ( strcmp( training_parameters{ i }, 'gamma'       ) )
      ptyp( 3 ) = 'c';
      xlow( 3 ) = fdo * p0( 3 );
      xupp( 3 ) = fup * p0( 3 );
   elseif ( strcmp( training_parameters{ i }, 'delta'       ) )
      ptyp( 4 ) = 'c';
      xlow( 4 ) = fdo * p0( 4 );
      xupp( 4 ) = fup * p0( 4 );
   elseif ( strcmp( training_parameters{ i }, 'eta'         ) )
      ptyp( 5 ) = 'c';
      xlow( 5 ) = fdo * p0( 5 );
      xupp( 5 ) = fup * p0( 5 );
   elseif ( strcmp( training_parameters{ i }, 'zeta'        ) )
      ptyp( 6 )  = 'c';
      xlow( 6 )  = fdo * p0( 6 );
      xupp( 6 )  = fup * p0( 6 );
   elseif ( strcmp( training_parameters{ i }, 'inertia'     ) )
   elseif ( strcmp( training_parameters{ i }, 'search-type' ) )
   elseif ( strcmp( training_parameters{ i }, 'random-seed' ) )
   end
end
xsca = ones( 9, 1 );
delt = [ 0.025*ones(1,6) 1 1 1 ];

%  Find the worst performance in this box by maximizing the total number of
%  function evaluations.

if ( tverbose > 1 )
   disp ( ' ---------- Evaluating worst case in the box ----------------' )
end

[ ~, fworst, msg, wrn, ~, ~, ~, ~, th ] =                                                  ...
     bfo( @(x,bestperf)bfo_average_perf( x, bestperf, training_problems,                   ...
                                training_problems_data, training_set_cutest,               ...
                                training_verbosity, training_problem_epsilon,              ...
                                training_problem_maxeval, training_problem_verbosity ),    ...
          p0, 'xscale', xsca, 'xtype', ptyp, 'xupper', xupp, 'xlower', xlow, 'epsilon',    ...
          training_epsilon, 'termination-basis', 1, 'verbosity', training_verbosity,       ...
          'search-type', 'none', 'max-or-min', 'max', 'maxeval', training_maxeval,         ...
          'f-target', bestperf, 'f-call-type', 'with-bound', 'f-bound', bestperf,          ...
          'delta', delt );

if ( tverbose > 1 )
   disp ( ' ------------------------------------------------------------' )
end

nevalt = th( end, 3 );

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [ xbest, fbest, msg, wrn, neval, fevals_hist, xincr ] =                           ...
         bfo_next_level_objf( level, nlevel, xlevel, neval, objf, x, checking,             ...
                              fevals_hist, xtype, xincr, xscale, xlower, xupper,           ...
                              max_or_min, vb_name, epsilon, bfgs_finish,  maxeval,         ...
                              verbosity, fcallt, alpha, beta, gamma, eta, zeta, inertia,   ...
                              stype, rseed, term_basis, latbasis, reset_random_seed, ssfname )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  This function provides a call to BFO for optimizing variables at a level beyond that
%  of the calling process.

%  INPUT:

%  level                    : the level at which optimization must be performed
%  nlevel                   : the total number of levels
%  xlevel                   : the assignment of variables to levels
%  obj                      : the objective function handle
%  checking                 : true is the calling process is close to convergence
%  fevals_hist              : the history of objective function evaluations
%  xtype                    : the variables types
%  xincr                    : the current set of (level dependent) stepsizes
%  xscale                   : the variables' scalings
%  xlower                   : the lower bounds on the variables
%  xupper                   : the upper bounds on the variables
%  max-or-min               : the max/min program for the various levels
%  vb_name                  : the name of the user-supplied variable bounds function
%  epsilon                  : the increment accuracy
%  bfgs_finish              : the meshsize under which BFGS is attempted
%  maxeval                  : the maximum number of function evaluations
%  verbosity                : the desired verbosity level at level level and beyond
%  fcallt                   : the type of objective function call
%  alpha                    : the grid expansion factor
%  beta                     : the grid expansion/reduction factor
%  gamma                    : the maximum grid expansion factor for continuous variables
%  delta                    : the single mesh parameter
%  eta                      : the sufficient decrease factor
%  zeta                     : the multilevel re-expansion factor
%  inertia                  : the number of iterations use for continuous step averaging
%  stype                    : the discrete variables search type
%  rseed                    : the random number generator's seed
%  term_basis               : the number of random basis used for assessing termination
%  latbasis                 : the lattice basis ([] if none)
%  reset_random_seed        : the request for resetting the random seed
%  sssfname                 : the name of the search step is requested

%  OUTPUT:

%  xbest                    : the best point resulting from optimization at level 
%                             level and beyond
%  fbest                    : the objective function value at xbest
%  msg                      : a termination message
%  wrn                      : a termination warning
%  neval                    : the cumulated number of function evaluations
%  fevals_hist              : the cumulated history of objective function evaluations
%  xincr                    : the (possibly updated) set of (level dependent) stepsizes

%  DEPENDENCIES : bfo, vb_name

%  PROGRAMMING: Ph. Toint,and M. Porcelli, November 2014. (This version 4 I 2015)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  If needed, recompute the bounds as a function of the variables in the
%  levels up to the current one.

xlnxt = xlower;
xunxt = xupper;
if ( ~strcmp( vb_name, 'bfo_none' ) )
   vbname        = str2func( vb_name );
   [ xl, xu ]    = vbname( x, level, xlevel, xlower, xupper );
   inxt          = find( xlevel > level );                   % indices of next levels vars
   xlnxt( inxt ) = min( [ xl( inxt )'; xu( inxt )' ] )';     % only update those bounds...
   xunxt( inxt ) = max( [ xl( inxt )'; xu( inxt )' ] )';
   xnxt          = min( [ max( [ xlnxt'; x' ]); xunxt' ] );  % ... and variables
else
   xnxt          = x;
end

%  Extract the correct min-max specification, if specified alternating by default.

if ( size( max_or_min, 1 ) == 1 )
   if ( strcmp( max_or_min, 'max' ) )
       max_or_min = 'min';
   else
       max_or_min = 'max';
   end
end

%  Impose more than one termination basis for continuous variables only if the
%  previous level is already in termination phase.

if ( checking )
   nexttermbasis = term_basis;
else
   nexttermbasis = 1;
end

%  Expand the next level grid somewhat if the level has already been explored.  
%  This is where the parameter zeta is used.

for i = 1:length( xtype )
   if ( xtype( i ) == 'c' && xlevel( i ) == level + 1 )
      xincr( i, 1 ) = min ( zeta * xincr( i, 1 ), xincr( i, 2 ) );
   end
end

%  Optimize the objective function corresponding to the variables of the next level.

[ xbest, fbest, msg, wrn, neval, fevals_hist, ~, ~, ~, ssh, xincr ] =                      ...
         bfo( objf, xnxt, 'level', level+1, 'xlevel', xlevel, 'variable-bounds', vb_name,  ...
              'xtype', xtype, 'xincr', xincr, 'xscale', xscale, 'xupper', xunxt,           ...
              'xlower', xlnxt, 'max-or-min', max_or_min, 'epsilon', epsilon,               ...
              'maxeval', maxeval, 'verbosity', verbosity, 'f-call-type', fcallt,           ...
              'alpha', alpha, 'beta', beta, 'gamma', gamma, 'eta', eta, 'zeta', zeta,      ...
              'inertia', inertia, 'search-type', stype, 'random-seed', rseed, 'nevr',      ...
              neval, 'fevals-hist', fevals_hist, 'lattice-basis', latbasis,                ...
              'termination-basis', nexttermbasis, 'bfgs-finish', bfgs_finish,              ...
              'reset-random-seed', reset_random_seed, 'search-step-function', ssfname );

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function savok = bfo_save( filename, objname, maximize, epsilon, ftarget,  maxeval, neval, ...
                           fevals_hist, xtype, xincr, xscale, xlower, xupper, verbose,     ...
                           alpha, beta, gamma, eta, zeta, inertia, stype, rseed,           ...
                           term_basis, used_trained, hist, latbasis, bfgs_finish,          ...
                           training_history, nscaled, unscaled, ssfname )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Save the algorithmic parameters to the filename file, in order to allow for
%  future restart.

%  INPUT:

%  filename     : the name of the file for saving
%  objname      : the name of the objective function
%  maximize     : the maximization/minimization flag
%  filename     : the name of the file on which to write the restart information
%  epsilon      : the termination accuracy
%  ftarget      : the objective function target
%  maxeval      : the maximum number of evaluations
%  neval        : the current number of evaluations
%  fevals_hist  : the current vector of all computed function values 
%  xtype        : the variables' types
%  xincr        : the current increments
%  xscale       : the variables' scaling
%  xlower       : the current lower bounds on the variables
%  xupper       : the current upper bounds on the variables
%  verbose      : the printout quantity flag
%  alpha        : the grid expansion factor
%  beta         : the grid expansion/reduction factor
%  gamma        : the maximum grid expansion factor for continuous variables
%  eta          : the sufficient decrease factor
%  zeta         : the multilevel re-expansion factor
%  inertia      : the number of iterations use for continuous step averaging
%  stype        : the discrete variables search type
%  rseed        : the random number generator's seed
%  term_basis   : the number of random basis used for assessing termination
%  use_trained  : the flag indicating use of trained BFO parameters
%  hist         : the saved information of the explored subspaces
%  latbasis     : the lattice basis ([] if none)
%  bfgs_finish  : the meshsize under which BFGS is attempted
%  training_history: the history of training so far
%  nscaled      : the number of nontrivially scaled continuous variables
%  unscaled     : the transformation from scaled to unscaled variables
%  ssfname      : the name of the user-defined search step

%  OUTPUT:

%  savok        : 1 if the parameters could be saved, 0 otherwise

%  DEPENDENCIES: -

%  PROGRAMMING: Ph. Toint, May 2009. (This version 22 II 2015)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Save data.

fid = fopen( filename, 'w' );
if ( fid == -1 )
   savok = 0;
else
   use_lattice = ( length( latbasis ) > 0 );
   fprintf( fid, ' *** BFO optimization checkpointing file %s\n', date );
   fprintf( fid, ' *** (c) Ph. Toint & M. Porcelli\n' );
   n = length( xlower);
   [ nh, np2 ] = size( hist );
   fprintf( fid,' %18s   %-12s\n',  objname,             'objname'     );
   fprintf( fid,' %18d   %-12s\n',  maximize,            'maximize'    );
   fprintf( fid,' %18d   %-12s\n',  verbose,             'verbose'     );
   fprintf( fid,' %.11e   %-12s\n', ftarget,             'ftarget'     );
   fprintf( fid,' %.12e   %-12s\n', epsilon,             'epsilon'     );
   fprintf( fid,' %.12e   %-12s\n', bfgs_finish,         'bfgs_finish' );
   fprintf( fid,' %18d   %-12s\n',  term_basis,          'term_basis'  );
   fprintf( fid, '\n' );
   fprintf( fid,' %18d   %-12s\n',  maxeval,             'maxeval'     );
   fprintf( fid,' %18d   %-12s\n',  neval,               'neval'       );
   fprintf( fid,' %.12e   %-12s\n', alpha,               'alpha'       );
   fprintf( fid,' %.12e   %-12s\n', beta,                'beta'        );
   fprintf( fid,' %.12e   %-12s\n', gamma,               'gamma'       );
   fprintf( fid,' %.12e   %-12s\n', eta,                 'eta'         );
   fprintf( fid,' %.12e   %-12s\n', zeta,                'zeta'        );
   fprintf( fid,' %18d   %-12s\n',  inertia,             'inertia'     );
   fprintf( fid,' %18d   %-12s\n',  stype,               'stype'       );
   fprintf( fid,' %18d   %-12s\n',  rseed,               'rseed'       );
   fprintf( fid,' %18d   %-12s\n',  used_trained,        'used_trained');
   fprintf( fid,' %18d   %-12s\n',  n,                   'n'           );
   fprintf( fid,' %18d   %-12s\n',  nscaled,             'nscaled'     );
   fprintf( fid,' %18s   %-12s\n',  xtype,               'xtype'       );
   fprintf( fid,' %18d   %-12s\n',  use_lattice,         'use_lattice' );
   fprintf( fid,' %18s   %-12s\n',  ssfname,             'ssfname'  );
   fprintf( fid,' %+.12e   ', fevals_hist( 1:neval ) );
   fprintf( fid, '\n' );
   fprintf( fid, '%+.12e   ', xincr( 1:n, 1 ) );
   fprintf( fid, '\n' );
   fprintf( fid, '%+.12e   ', xincr( 1:n, 2 ) );
   fprintf( fid, '\n' );
   fprintf( fid, '%+.12e   ', xincr( 1:n, 3 ) );
   fprintf( fid, '\n' );
   fprintf( fid, '%+.12e   ', xscale( 1:n ) );
   fprintf( fid, '\n' );
   if ( nscaled )
      fprintf( fid, '%+.12e   ', unscaled( 1:n ) );
      fprintf( fid, '\n' );
   end
   fprintf( fid, '%+.12e   ', xlower( 1:n ) );
   fprintf( fid, '\n' );
   fprintf( fid, '%+.12e   ', xupper( 1:n ) );
   fprintf( fid, '\n' );
   fprintf( fid,' %18d   %-12s\n',  nh,           'nh');
   for j = 1:nh
      fprintf( fid, '%+.12e   ', hist( j, 1:np2 ) );
      fprintf( fid, '\n' );
   end
   if ( use_lattice )
      for j = 1:n
         fprintf( fid, '%+.12e   ', latbasis( j, 1:n ) );
         fprintf( fid, '\n' );
      end
   end
   lh = size( training_history, 1 );
   fprintf( fid,' %18d   %-12s\n',  lh,                  'training history length');
   if ( lh > 0 )
      for j = 1:lh
         fprintf(fid,'%1d %3d %10d %10d %.12e %.12e %.12e %.12e %.12e %.12e %2d %1d %3d\n',...
                  training_history( j, 1:13 ) );
      end
   end
   savok = 1;
end

fclose( fid );

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function savok = bfo_save_training( filename, p, verbose,                                  ...
                                    training_strategy, training_parameters,                ...
                                    training_problems, training_problems_data,             ...
                                    training_set_cutest, trained_bfo_parameters,           ...
                                    training_epsilon, training_maxeval, training_verbosity,...
                                    training_problem_epsilon, training_problem_maxeval,    ...
                                    training_problem_verbosity )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Save the training algorithmic parameters to the filename.training file, in order 
%  to allow for future restart.

%  INPUT:

%  filename     : the name of the file for saving
%  p            : the current value of the BFO algorithmic parameters
%  verbose      : the verbosity of the top level BFO routine
%  training_strategy : the strategy applied for training
%  training_problems : a cell whose entries specify (in string or function-handle form)
%                 the test problems on which training is performed 
%  training_problems_data : a cell whose entries specify the data (bounds, etc) associated
%                 with the training problems (irrelevant for CUTEst)
%  training_set_cutest : true if the training problems are extracted from CUTEst
%  trained_bfo_parameters : the name of the file where optimized parameters are being saved
%  training_epsilon : the accuracy requirement for the training problem itself
%  training_maxeval : the maximum number of optimization iterations for the training 
%                 process itself
%  training_verbosity : the verbosity of the training process itself
%  training_problem_epsilon: the accuracy requirement for the solution of each test
%                 problem during training
%  training_problem_maxeval: the maximum number of evaluations for the solution of 
%                 each test problem during training
%  training_problem_verbosity: the verbosity for the solution of each test
%                 problem during training

%  OUTPUT:

%  savok        : 1 if the parameters could be saved, 0 otherwise

%  DEPENDENCIES: -

%  PROGRAMMING: Ph. Toint, December 2014. (This version 22 XII 2014)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Save data.

fid = fopen( [ filename, '.training' ], 'w' );
if ( fid == -1 )
   savok = 0;
else
   fprintf( fid, ' *** BFO training checkpointing file %s\n', date );
   fprintf( fid, ' *** (c) Ph. Toint & M. Porcelli\n' );
   fprintf( fid,' %18s   %-12s\n',  training_strategy,   'training_strategy');
   fprintf( fid,' %18d   %-12s\n',  training_set_cutest, 'training_set_cutest');
   np = length( training_parameters );
   fprintf( fid,' %18d   %-12s\n',  np,                  'nbr_training_params');
   fprintf( fid, '%s  ', training_parameters{ 1:np } );
   fprintf( fid, '\n' );
   np = length( p );
   fprintf( fid, '%+.5e   ', p( 1:np ) );
   fprintf( fid, '\n' );
   np = length( training_problems );
   fprintf( fid,' %18d   %-12s\n',  np,                  'nbr_training_problems');
   if ( training_set_cutest )
      fprintf( fid, '%s  ', training_problems{ 1:np } );
      fprintf( fid, '\n' );
   else
      for j = 1:np
         fprintf( fid, '%s  ', func2str( training_problems{ j } ) );
      end
      fprintf( fid, '\n' );
      for j = 1:np
         fprintf( fid, '%s  ', func2str( training_problems_data{ j } ) );
      end
      fprintf( fid, '\n' );
   end
   fprintf( fid,' %18s   %-12s\n',  trained_bfo_parameters,    'trained_bfo_parameters');
   fprintf( fid,' %18d   %-12s\n',  verbose,                   'verbose');
   fprintf( fid,' %.12e   %-12s\n', training_epsilon(1),       'training_epsilon');
   fprintf( fid,' %.12e   %-12s\n', training_epsilon(2),       'training_epsilon');
   fprintf( fid,' %18d   %-12s\n',  training_maxeval(1),       'training_maxeval(1)');
   fprintf( fid,' %18d   %-12s\n',  training_maxeval(2),       'training_maxeval(2)');
   fprintf( fid,' %18s   %-12s\n',  training_verbosity{1},     'training_verbosity{2}');
   fprintf( fid,' %18s   %-12s\n',  training_verbosity{2},     'training_verbosity{2}');
   fprintf( fid,' %.12e   %-12s\n', training_problem_epsilon,  'training_problem_epsilon');
   fprintf( fid,' %18d   %-12s\n',  training_problem_maxeval,  'training_problem_maxeval');
   fprintf( fid,' %18s   %-12s\n',  training_problem_verbosity,'training_problem_verbosity');
   savok = 1;
end

fclose( fid );

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function  [ objname, maximize, epsilon, ftarget, maxeval, neval, fevals_hist, xtype,       ...
            xincr, xscale, xlower, xupper, verbose, alpha, beta, gamma, eta, zeta, inertia,...
            stype, rseed, term_basis, use_trained, hist, use_lattice, latbasis,            ...
            bfgs_finish, training_history, nscaled, unscaled, ssfname, restok ] =          ...
            bfo_restore( filename, readall )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Restores the algorithmic parameters from the saved 'filename' file.


%   INPUT :


%   filename     : the name of the file on which to read the restart information
%   readall      : 1 if the name of the objective function is irrelevant (used when
%                  restarting training)

%   OUTPUT :

%   filename     : the name of the file for saving
%   objname      : the name of the objective function
%   maximize     : the maximization/minimization flag
%   epsilon      : the gradient termination accuracy
%   ftarget      : the objective function target
%   maxeval      : the maximum number of evaluations
%   neval        : the current number of evaluations
%   fevals_hist  : the current vector of all computed function values 
%   xtype        : the variables' types
%   xincr        : the variables' increments
%   xscale       : the variables' scalings
%   xlower       : the current lower bounds on the variables
%   xupper       : the current upper bounds on the variables
%   verbose      : the printout quantity flag
%   alpha        : the grid expansion factor
%   beta         : the grid expansion/reduction factor
%   gamma        : the maximum grid expansion factor for continuous variables
%   eta          : the sufficient decrease factor
%   zeta         : the multilevel grid re-expansion factor
%   inertia      : the number of iterations use for continuous step averaging
%   stype        : the discrete variable search type
%   rseed        : the random number generator's seed
%   term_basis   : the number of random basis used for assessing termination
%   use_trained  : the flag indicating use of trained BFO parameters
%   hist         : the saved information of the explored subspaces
%   use_lattice  : the flag indicating lattice-basis use
%   latbasis     : the lattice basis ([] if none )
%   bfgs_finish  : the meshsize under which BFGS is attempted
%   training_history : the training history so far
%   nscaled      : the number of nontrivially scaled continuous variables
%   unscaled     : the transformation from scaled to unscaled variables
%   ssfname      : the name of the user-defined search step function
%   restok       : 1 if restore successful, 0 if unsuccessful, 2 if successful in training


%   PROGRAMMING: Ph. Toint,and M. Porcelli, May 2010. (This version 22 II 2015)

%   DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Attempt to open the restart file.

fid = fopen( filename, 'r' );

%  The file can't be opened: return meangless parameters.

if ( fid == -1 )
   objname       = '';
   maximize      = NaN;
   epsilon       = NaN;
   ftarget       = NaN;
   maxeval       = NaN;
   neval         = NaN;
   fevals_hist   = [];
   xtype         = '';
   xincr         = [];
   xscale        = [];
   xlower        = [];
   xupper        = [];
   verbose       = NaN;
   alpha         = NaN;
   beta          = NaN;
   gamma         = NaN;
   eta           = NaN;
   zeta          = NaN;
   inertia       = NaN;
   stype         = NaN;
   rseed         = NaN;
   hist          = [];
   use_trained   = NaN;
   term_basis    = NaN;
   use_lattice   = 0;
   latbasis      = NaN;
   restok        = 0;
   bfgs_finish   = NaN;
   training_history = [];
   nscaled       = 0;
   unscaled      = [];
   ssfname       = '';
   return

%  The restart file opened fine: read the restart parameters.

else
   filetitle     = fscanf( fid, '%s', 13 );
   objname       = fscanf( fid, '%s', 1 ); name = fscanf( fid, '%s\n', 1 );
   if ( ~readall &&                                                                       ...
        (strcmp( objname, 'bfo_average_perf' ) || strcmp( objname, 'bfo_robust_pertf' ) ) )
      maximize      = 0;
      verbose       = 0;
      epsilon       = NaN;
      ftarget       = NaN;
      maxeval       = NaN;
      neval         = 0;
      fevals_hist   = [];
      xtype         = '';
      xincr         = [];
      xscale        = [];
      xlower        = [];
      xupper        = [];
      alpha         = NaN;
      beta          = NaN;
      gamma         = NaN;
      eta           = NaN;
      zeta          = NaN;
      inertia       = NaN;
      stype         = NaN;
      rseed         = NaN;
      hist          = [];
      use_trained   = NaN;
      term_basis    = NaN;
      use_lattice   = 0;
      latbasis      = NaN;
      bfgs_finish   = NaN;
      training_history = [];
      nscaled       = 0;
      unscaled      = [];
      ssfname       = '';
      restok        = 2;
      return
   end
   maximize      = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   verbose       = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   ftarget       = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   epsilon       = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   bfgs_finish   = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   term_basis    = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   maxeval       = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   neval         = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   alpha         = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   beta          = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   gamma         = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   eta           = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   zeta          = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   inertia       = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   stype         = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   rseed         = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   use_trained   = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   n             = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   nscaled       = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   xtype         = fscanf( fid, '%s', 1 ); name = fscanf( fid, '%s\n', 1 );
   use_lattice   = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   ssfname       = fscanf( fid, '%s', 1 ); name = fscanf( fid, '%s\n', 1 );
   fevals_hist(1:neval) = fscanf( fid, '%e', neval );
   xincr(1:n,1)  = fscanf( fid, '%e', n );
   xincr(1:n,2)  = fscanf( fid, '%e', n );
   xincr(1:n,3)  = fscanf( fid, '%e', n );
   xscale(1:n,1) = fscanf( fid, '%e', n );
   if ( nscaled )
      unscaled(1:n,1) = fscanf( fid, '%e', n );
   else
      unscaled = [];
   end
   xlower(1:n,1) = fscanf( fid, '%e', n );
   xupper(1:n,1) = fscanf( fid, '%e', n );
   nh            = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   if ( nh > 0 )
      for j = 1:nh
         hist(j,1:2*n+1) = fscanf( fid, '%e', 2*n+1 );
      end
   else
      hist = [];
   end
   if ( use_lattice )
      for j = 1:nh
         latbasis(j,1:n) = fscanf( fid, '%e', n );
      end
   else
      latbasis = [];
   end
   lh = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 3 );
   if ( lh > 0 )
      training_history = zeros( lh, 13 );
      for j = 1:lh
         training_history( j, 1:13 ) = fscanf( fid, '%e', 13 );
      end
   else
      training_history = [];
   end
   restok = 1;
   fclose( fid );
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ p, verbose, training_strategy, training_parameters, training_problems,          ...
           training_problems_data, training_set_cutest, trained_bfo_parameters,            ...
           training_epsilon, training_maxeval, training_verbosity,                         ...
           training_problem_epsilon, training_problem_maxeval, training_problem_verbosity, ...
           restok ] = bfo_restore_training( filename )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  If training restart is desired, read the training algorithmic parameters and 
%  the environment of the whole training procedure.

%  INPUT :

%  filename   : the name of the file to be read

%  OUTPUT :

%  p            : the current value of the BFO algorithmic parameters
%  verbose      : the verbosity of the top level BFO routine
%  training_strategy : the strategy applied for training
%  training_problems : a cell whose entries specify (in string or function-handle form)
%                 the test problems on which training is performed 
%  training_problems_data : a cell whose entries specify the data (bounds, etc) associated
%                 with the training problems (irrelevant for CUTEst)
%  training_set_cutest : true if the training problems are extracted from CUTEst
%  trained_bfo_parameters : the name of the file where optimized parameters are being saved
%  training_epsilon : the accuracy requirement for the training problem itself
%  training_maxeval : the maximum number of optimization iterations for the training 
%                 process itself
%  training_verbosity : the verbosity of the training process itself
%  training_problem_epsilon: the accuracy requirement for the solution of each test
%                 problem during training
%  training_problem_maxeval: the maximum number of evaluations for the solution of 
%                 each test problem during training
%  training_problem_verbosity: the verbosity for the solution of each test
%                 problem during training

%  PROGRAMMING: Ph. Toint, December 2014. (This version 22 XII 2014)

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

restok = 0;
fid    = fopen( filename, 'r');
if ( fid == -1 )
   p                      = [];
   verbose                = 1;
   training_strategy      = '';
   training_parameters    = {};
   trained_bfo_parameters = '';
   training_problems      = {};
   training_problems_data = {};
   training_set_cutest    = 0;
   training_epsilon       = NaN;
   training_maxeval       = NaN;
   training_verbosity     = NaN;
   training_problem_epsilon   = NaN;
   training_problem_maxeval   = NaN;
   training_problem_verbosity = NaN;
   return
else
   filetitle           = fscanf( fid, '%s', 13 );
   training_strategy   = fscanf( fid, '%s', 1 );   name = fscanf( fid, '%s\n', 1 );
   training_set_cutest = fscanf( fid, '%d', 1 );   name = fscanf( fid, '%s\n', 1 );
   nbr_train_params    = fscanf( fid, '%d', 1 );   name = fscanf( fid, '%s\n', 1 );
   for j = 1:nbr_train_params
      training_parameters{ j } = fscanf( fid, '%s', 1 );
   end
   p                   = fscanf( fid, '%e', 9 );
   nbr_train_probs     = fscanf( fid, '%d', 1 );   name = fscanf( fid, '%s\n', 1 );
   if ( training_set_cutest )
      for j = 1:nbr_train_probs
         training_problems{ j } = fscanf( fid, '%s', 1 );
      end
      training_problems_data = {};
   else
      for j = 1:nbr_train_probs
         training_problems{ j } = str2func( fscanf( fid, '%s', 1 ) );
      end
      for j = 1:nbr_train_probs
         training_problems_data{ j } = str2func( fscanf( fid, '%s', 1 ) );
      end
   end
   trained_bfo_parameters     = fscanf( fid, '%s', 1 ); name = fscanf( fid, '%s\n', 1 );
   verbose                    = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   training_epsilon(1)        = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   training_epsilon(2)        = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   training_maxeval(1)        = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   training_maxeval(2)        = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   training_verbosity{1}      = fscanf( fid, '%s', 1 ); name = fscanf( fid, '%s\n', 1 );
   training_verbosity{2}      = fscanf( fid, '%s', 1 ); name = fscanf( fid, '%s\n', 1 );
   training_problem_epsilon   = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   training_problem_maxeval   = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   training_problem_verbosity = fscanf( fid, '%s', 1 ); name = fscanf( fid, '%s\n', 1 );
end
restok = 1;

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function bfo_print_vector( indent, name, x )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Prints a vector after its name and the string indent.

%  INPUT:

%  indent : a string (typically containing indentation blanks) to be
%           printed at the beginning of the line
%  name   : the name of the vector to be printed
%  x      : the vector to be printed

%  PROGRAMMING: Ph. Toint, April 2009. (This version 27 V 2009)

%  DEPENDENCIES: -

%  TEST
%  bfo_print_vector( '   ', 'vec', [1:n] )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length(x);

disp( [ indent, ' ', name, ' = '] )
is = 1;
for i = 1:ceil( n/10 )
   it = min( is + 9, n );
   fprintf( '%s  %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e',             ...
            indent, x( is:it ) );
   fprintf( '\n' );
   is = is + 10;
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function bfo_print_cell( indent, name, c )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Prints a cell containing function specifications in string or function-handle form.

%  INPUT :

%  indent : a string (typically containing indentation blanks) to be
%           printed at the beginning of the line
%  name   : the name of the vector to be printed
%  c      : the cell to be printed

%  PROGRAMMING: Ph. Toint, December 2014. (This version 19 XII 2014)

%  DEPENDENCIES: -

%  TEST
%  c = { 'alpha', 'beta', 'search-type', @a_rather_long_name, ...
%         'an_even_longer_name_than_before' };
%  bfo_print_vector( '   ', 'cell', c )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp( [ indent, ' ', name, ' = '] )
lcell = length( c );
for i = 1:lcell
   if ( ischar( c{ i } ) )
      fprintf( '%s  ', c{ i } )
   else
      fprintf( '%s  ', func2str( c{ i } ) )
   end
end
fprintf( '\n' );

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function bfo_print_matrix( indent, name, x )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Prints a matrix after its name and the string indent.

%  INPUT :

%  indent : a string (typically containing indentation blanks) to be
%           printed at the beginning of the line
%  name   : the name of the vector to be printed
%  x      : the matrix to be printed

%  PROGRAMMING: Ph. Toint, December 2014. (This version 14 XII 2014)

%  DEPENDENCIES: -

%  TEST
%  for n=1:9
%     bfo_print_matrix( '   ', 'mat', [1:n;1:n] )
%  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[ m, n ] = size(x);

disp( [ indent, ' ', name, ' = '] )
for i = 1: m
   is = 1;
   for j = 1:ceil( n/10 )
      it = min( is + 9, n );
      fprintf( '%s row %3d:  %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e', ...
               indent, i, x( is:it ) );
      fprintf( '\n' );
      is = is + 10;
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function bfo_print_summary_vector( indent, name, x )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Prints (on a line) the first and last few (4) components of a vector after 
%  its name and the string indent.

%  INPUT:

%  indent : a string (typically containing indentation blanks) to be
%           printed at the beginning of the line
%  name   : the name of the vector to be 'summary-printed'
%  x      : the vector to be 'summary-printed' 

%  PROGRAMMING: Ph. Toint, April 2009. (This version 27 V 2009)

%  DEPENDENCIES: -

%  TEST
%  for n=1:9
%     bfo_print_summary_vector( '   ', 'vec', [1:n] )
%  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length(x);

if ( n == 1 )
   disp( [ indent, ' ', name, ' = ( ', num2str(x(1)), ' )' ] )
elseif ( n == 2 )
   disp( [ indent, ' ', name, ' = ( ', num2str(x(1)), ' ', num2str(x(n)), ' )' ] )
elseif ( n == 3 )
   disp( [ indent, ' ', name, ' = ( ', num2str(x(1)), ' ', num2str(x(2)), ' ',             ...
         num2str(x(n)), ' )' ] )
elseif ( n == 4 )
   disp( [ indent, ' ', name, ' = ( ', num2str(x(1)), ' ', num2str(x(2)), ' ',             ...
         num2str(x(n-1)),' ', num2str(x(n)), ' )' ] )
elseif ( n == 5 )
   disp( [ indent, ' ', name, ' = ( ', num2str(x(1)), ' ', num2str(x(2)), ' ',             ...
         num2str(x(3)), ' ', num2str(x(4)), ' ', num2str(x(5)), ' )' ] )
elseif ( n == 6 )
   disp( [ indent, ' ', name, ' = ( ', num2str(x(1)), ' ', num2str(x(2)), ' ',             ...
         num2str(x(3)), ' ', num2str(x(4)), ' ', num2str(x(5)),' ', num2str(x(6)), ' )' ] )
elseif ( n == 7 )
   disp( [ indent, ' ', name, ' = ( ', num2str(x(1)), ' ', num2str(x(2)), ' ',             ...
         num2str(x(3)), ' ', num2str(x(4)), ' ', num2str(x(5)), ' ',                       ...
         num2str(x(6)), ' ', num2str(x(7)), ' )' ] )
elseif ( n == 8 )
   disp( [ indent, ' ', name, ' = ( ', num2str(x(1)), ' ', num2str(x(2)), ' ',             ...
         num2str(x(3)), ' ', num2str(x(4)), ' ', num2str(x(5)), ' ',                       ...
         num2str(x(6)), ' ', num2str(x(7)), ' ', num2str(x(8)), ' )' ] )
else
   disp( [ indent, ' ', name, ' = ( ', num2str(x(1)), ' ', num2str(x(2)), ' ',             ...
         num2str(x(3)), ' ', num2str(x(4)), ' ... ', num2str(x(n-3)), ' ',                 ...
         num2str(x(n-2)),' ', num2str(x(n-1)), ' ', num2str(x(n)), ' )' ] )
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function bfo_print_x( indent, name, x, nscaled, unscaled, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Prints a vector of variables, taking into account the possible non-trivial scale 
%  of continuous variables and the verbosity index.


%  INPUT:

%  indent   : a string (typically containing indentation blanks) to be
%           printed at the beginning of the line
%  name     : the name of the vector to be 'summary-printed'
%  x        : the vector to be 'summary-printed' 
%  nscaled  : the number of nontrivially scaled continuous variables
%  unscaled : the transformation from scaled to unscaled continuous variables
%  verbose  : the current verbosity index

%  PROGRAMMING: Ph. Toint, February 2015. (This version 22 II 2015)

%  DEPENDENCIES: -

%  TEST
%  bfo_print_x( '   ', 'vec', [1:n], 1, [1:n], 5 )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( verbose == 3 )
   if ( nscaled )
      bfo_print_summary_vector( indent, name, unscaled.*x )
   else
      bfo_print_summary_vector( indent, name, x )
   end
elseif ( verbose > 3 )
   if ( nscaled )
      bfo_print_vector( indent, name, unscaled.*x )
      if ( verbose >= 10 )
         bfo_print_vector( indent, [ 'scaled ', name ], x )
      end
   else
      bfo_print_vector( indent, name, x )
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [ xfeas, alpha ] = bfo_feasible_cstep( x, step, xlower, xupper )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Obtain a feasible point on the segment [ x, x+step ] in continuous variables.


%   INPUT:

%   x      :  the base point
%   step   :  the potential full step
%   xlower :  the lower bounds on the variables
%   xupper :  the upper bounds on the variables


%   OUTPUT:

%   xfeas  : the feasible point of the form x + alpha * step closest to x + step
%   alpha  : the corresponding alpha


%   PROGRAMMING: Ph. Toint, December 2014 (this version: 1 I 2015).

%   DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
actu  = [];                    %  the list of variable set at their upper bound
actl  = [];                    %  the list of variable set at their lower bound
alpha = 1;                     %  the initial stepsize

for i  = 1:length( x )

   %  The situation for variable i.

   xi  = x( i );
   si  = step( i );
   xli = xlower( i );
   xui = xupper( i );
   xti = xi + alpha * si;

   %  Violated lower bound

   if ( xti < xli )

      alphaj = ( xli - xi ) / si;
      if ( alphaj < ( 1 - eps ) * alpha )
         alpha = alphaj;
         actl  = [ i ];
      elseif ( alphaj < ( 1 + eps ) * alpha )
         actl = [ actl i ];
      end
      
      if ( alpha * abs( si ) <= eps * abs( xi ) )
         xfeas = x;
         alpha = 0;
        return
      end

   %  Violated upper bound

   elseif ( xti > xui )

      alphaj = ( xui - xi ) / si;
      if ( alphaj < ( 1 - eps ) * alpha )
         alpha = alphaj;
         actu  = [ i ];
      elseif ( alphaj < ( 1 + eps ) * alpha )
         actu = [ actu i ];
      end

      if ( alpha * abs( si ) <= eps * abs( xi ) )
         xfeas = x;
         alpha = 0;
         return
      end
 
   end

end

%  Build the final step.

xfeas = x + alpha * step;
if ( length( actl ) > 0 )
   xfeas( actl ) = xlower( actl );
end
if ( length( actu ) > 0 )
   xfeas( actl ) = xupper( actl );
end

alpha = norm( xfeas - x );
%alpha = alpha * norm( step );

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function hist = bfo_histupd( hist, idisc, xbest, fbest, xincr )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Updates the history of the minimization by avoiding repetitive information
%   for identical subspaces.


%   INPUT :


%   hist         : the current history information
%   idisc        : the indices of the discrete variables
%   xbest        : the current best values of the variables
%   fbest        : the best objective value
%   xincr        : the current grid increments

%   OUTPUT :

%   hist         : the updated history

%   PROGRAMMING: Ph. Toint, January 2010 (this version: 15 XII 2014).

%   DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Get the length of the subspace list and the number of variables.

nh  = size( hist, 1 );
n   = length( xbest );
np2 = 2*n+1;

%  There are no discrete variables, and there is thus only a single subspace to 
%  consider (in the first component of the history list).

if ( isempty( idisc ) )

   hist = [ xbest' fbest xincr' ];

%  If there are discrete variables, search the list of subspaces defined 
%  by their values.

else

   %  Search for the corresponding subspace in the current list.

   found = 0;
   for j = 1:nh        
      if ( abs( hist( j, idisc ) - xbest( idisc )' ) < eps )
         found = 1;
         if ( fbest <= hist( j, n+1 ) )
            hist( j, 1:np2 ) = [ xbest' fbest xincr' ];
         end
      end
   end
   if ( ~found )        % unexplored subspace: store the result
      hist = [ hist; [ xbest' fbest xincr' ] ];
   end

end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ found, fname ] = bfo_exist_function( fname )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Check if a m-file corresponding to the function specified by fname exists.

%   The routine first strips characters which should not be present in the
%   name of the corresponding m-file, and then checks it can be found in the
%   MATLAB path.


%   INPUT :

%   fname : a string corresponding to a function, for which a matching m-file
%           must be sought

%   OUTPUT :

%   found : true if the corresponding file was found
%   fname : the shortened name (without the stripped characters).

%   PROGRAMMING: Ph. Toint, December 2014 (this version: 9 XII 2014).

%   DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
found    = 0;                                  % not found yet

%  Strip unwanted characters.

fname( find( fname == ' ' ) ) = [];            % remove all blanks
fname( find( fname == '@' ) ) = [];            % remove (initial) @
lfname   = length( fname );
pos_opar = find( fname == '(' );               % find positions of opening parenthesis
n_opar   = length( pos_opar );
if ( n_opar > 0 )
   pos_cpar = find( fname == ')' );            % find positions of closing parenthesis
   fname( pos_opar( 1 ):pos_cpar( 1 ) ) = [];
   pos_opar = find( fname == '(' );            % find positions of opening parenthesis
   n_opar   = length( pos_opar );
   if ( n_opar > 0 )
      pos_cpar = find( fname == ')' );         % find positions of closing parenthesis
      fname( pos_opar( 1 ):pos_cpar( n_opar ) ) = [];
   end
end

%  Search the MATLAB path, except for the internal BFO functions.

found  =  ( exist( [ fname, '.m' ] ) == 2 || ...
            ( length( fname ) > 3 && strcmp( fname(1:4), 'bfo_' ) ) );

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function verbint = bfo_get_verbosity( verbosity )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Obtain the integer version of a string defining a verbosity level. 
%   If no match is found, a negative number is returned.


%   INPUT :

%   verbosity : a string (hopefully) containing the verbosity level.

%   OUTPUT :

%   verbint   : the corresponding integer (if matching), or -1 (no match).


%   PROGRAMMING: Ph. Toint, December 2014 (this version: 22 XII 2014).

%   DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (     strcmp( verbosity, 'silent'  ) )
   verbint = 0;
elseif ( strcmp( verbosity, 'minimal' ) )
   verbint = 1;
elseif ( strcmp( verbosity, 'low'     ) )
   verbint = 2;
elseif ( strcmp( verbosity, 'medium'  ) )
   verbint = 3;
elseif ( strcmp( verbosity, 'high'    ) )
   verbint = 4;
elseif ( strcmp( verbosity, 'debug'   ) )
   verbint = 10;
else
   verbint = -1;
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function  [x0, xlower, xupper, xtype ] = bfo_cutest_data( cutest_name )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Auxiliary function to retrieve data of a CUTEst problem where the variables with even
%  index is constrained to be integer and lower/upper bounds are rounded accordingly.

%  WARNING: a CUTEst MATLAB interface is assumed to be currently installed! 
%  The *.SIF problem files are assumed to be located in $MASTSIF 

%  INPUT:

%  cutest_name            : the name of the CUTEst problem

%  OUTPUT:

%  x0                     : the starting point
%  xlower                 : the lower bound
%  xupper                 : the upper bound
%  xtype                  : the variable type

%  DEPENDENCIES : cutest_setup

%  PROGRAMMING: M. Porcelli, May 2010. (This version 19 VI 2015)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Decode the CUTEST problem

st=(['!runcutest -A $MYARCH -p matlab -D $MASTSIF/' cutest_name '> /dev/null']);
eval( st )

%  Setup the problem data by calling the appropriate CUTEst tool.

prob = cutest_setup();

%  The starting point

x0 = prob.x;
n  = length( x0 );

%  The variable type

xtype = 'c'; 
for i = 2:n
   if ( mod( i, 2 ) == 0 ) % all variable with even indices are integer 
      xtype = ( [ xtype, 'i' ] );
   else
      xtype = ( [ xtype, 'c' ] );
   end
end

%  The lower and upper bound

prob.bl( find( prob.bl == -1e20 ) ) = -Inf;
prob.bu( find( prob.bu ==  1e20 ) ) =  Inf;

xlower = prob.bl; 
xupper = prob.bu;

%  Round to integer the components of the starting point and lower/upper bound 
%  corresponding to integer variables.

for i = 1:n
   if ( xtype( i ) == 'i' )
      x0( i ) = round( x0( i ) );
      xlower( i ) = ceil( xlower( i ) );
      xupper( i ) = floor( xupper( i ) );
   end
   x0( i ) = max( xlower( i ), min( x0( i ), xupper( i ) ) );
end

%  Look for fixed variables

for i = 1:n
   if ( xlower( i ) == xupper( i ) )
      xtype( i ) = 'f'; 
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



