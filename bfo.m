%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                                                    %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                         BFO                        %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                                                    %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                A Brute Force Optimizer             %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                                                    %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%               Ph. Toint and M. Porcelli            %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                                                    %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                    (c) 2015, 2018, 2020            %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%                                                    %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ xbest, fbest, msg, wrn, neval, f_hist, estcrit,                                 ...
           trained_parameters, training_history, s_hist,                                   ...
	   xincr, opt_context, el_hist, ev_hist ]          = bfo( varargin )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                              this_version = 'v 2.0'; % 15 I 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%% 1. BFO, a Brute-Force Optimizer
%
%
%  A "Brute-Force Optimizer" (originally based on simple refining random grid search) 
%  for unconstrained or bound-constrained optimization in continuous and/or 
%  discrete and/or categorical variables. Its purpose is to search for
%  a *LOCAL* minimizer of the problem
%
%  $$\min_{x}  f(x)= f(x)$$
%
%  or the problem
%
%  $$\min_{x}  f(x)= f_1(x) + f_2(x) + ... + f_p(x)$$                (1)
%
%  where f / each f_i is a function from R^n into R. The package is intended
%  for the case where the number of variables (n) is small (i.e. not much 
%  larger than 10), unless the problem has a sparse / coordinate-partially-separable
%  structure (see Section 2 below).  If this very common structure is present,
%  considerably larger problems can be successfully solved. 
%
%  The derivatives of f are assumed to be unavailable or inexistent (and are
%  consequently never used by the algorithm). A starting point x0 must be 
%  provided by the user. The components of x are allowed to vary either 
%  continuously (within their bounds) or on a predefined lattice (again 
%  between their bounds) or in a set of 'states' defined by a collection
%  of strings.  The lattice may for instance be given by the integer values.
%  This feature allows bound-constrained mixed-integer nonlinear problems to be
%  considered. However, it must be stressed that the algorithm merely ATTEMPTS
%  to compute a LOCAL (not global) minimizer of the objective function.
%
%  The algorithm proceeds by evaluating the objective function at points 
%  differing from the current iterate by a positive (forward) and a 
%  negative (backward) step along the vectors of a basis in the continuous
%  variables, or in each of the integer variables, or at points in a user-defined
%  neighboorhood of categorical variables. The corresponding stepsizes are computed
%  by varying fractions of the user-specified variable increments (delta). For
%  continuous variables, these fractions are decreased (yielding a shorter step)
%  as soon as no progress can be made from the current point and until the desired
%  accuracy is reached. For discrete variables, the user-supplied increment may not
%  be reduced, while categorical variables are by nature restricted to their
%  allowed 'states'.
%
%  In the standard case where the objective function is not given in sum form,
%  the value f(x) is computed by a user-supplied function whose call is 
%  given by
%
%      fx = f( x )                                   
%
%  where x is the vector at which evaluation is required.  If the problem
%  does not involve categorical variables, x is a standard numerical vector.
%  However, if categorical variables are used, then x is a 'vector state'
%  specified by a Matlab 'value cell array' (i.e. a cell array of size 1
%  containing itself a cell array, or, more simply stated, a cell array
%  with double braces {{...}}), whose i-th component is a numerical value if
%  the i-th variable is not categorical, and is a string ('state') is the i-th
%  variable is categorical. (See below for more on using categorical
%  variables.) If the problem is in sum form (1), the value of f(x) is
%  automatically assembled by BFO from formula (1) above,  the call to
%  each f_i  then being of the form
%
%      fix = f( i, x ),
%
%  where again x is either a numerical vector or a vector state.
%
%  An optimization problem can be presented to BFO in two different forms:
%  the 'extensive form' and the 'condensed form'.  In the extensive form,
%  the problem and its parameters (starting point, bounds, scalings, etc.)
%  are specified as individual arguments to BFO.  The relevant arguments
%  (whose detailed description is detailed below) are
%      f, x0, xlower, xupper, xtype, xscale, min-or-max, xlevel,
%      variable-bounds, cat-states, cat-neighbours, lattice-basis,
%      eldom
%  (only the first two are mandatory). In the condensed form, the problem
%  is specified as a MATLAB struct whose corresponding fields are
%       objf, x0, xlower, xupper, xtype, xscale, min_or_max, xlevel,
%       variable_bounds, cat_states, cat_neighbours, lattice_basis,
%       eldom
%  (note the _ (underscores) in the field names, instead of the - (hypens) 
%  used for extensive-form keywords, see below). An additional "name"
%  field may be added to the condensed form.
%
%  For example, minimizing the hyperbolic cosine over x (starting from 1)
%  can be achieved by the (extensive form) call
%
%      [ x, fx ] = bfo( @cosh, 1 )
%
%  or, if one whishes to restrict the minimization interval to [-1, 1],
%  by the call
%
%      [ x, fx ] = bfo( @cosh, 1 , 'xlower', -1, 'xupper', 1 )
%
%  The same calls, using the condensed form, is given by
%
%      [ x, fx ] = bfo( prob )
%
%  where either
%
%      prob = struc( 'objf', @cosh, 'x0', 1 )
%
%  or
%
%      prob = struc( 'objf', @cosh, 'x0', 1, 'xlower', -1, 'xupper', 1 ),
%
%  depending on the absence/presence of bounds.
%
%  In extensive form, an objective function depending on parameters as in
%
%      fx = f( x, parameter1, parameter2 )
%
%  may also be used if specified as a function handle . For instance, if
%  the function myobj is specified by function fx = myobj( x, p )
%
%       fx = x' * p + 0.5 * x' * x;
%
%  then minizing this function over x  for p = [ 1 1 1 1 1 ] starting 
%  from the origin can be achieved by the statements
%
%       x0    = zeros( 5, 1 );
%       p     = ones( size( x0 ) );
%       [ x, fx, msg, wrn, neval ] = bfo( @(x)myobj( x, p ), x0 );
%
%  See the description of the first input argument below for more details 
%  on how to specify the "function_handle" for the objective function and
%  the limitations/alternatives for specifying this and other parameters.
%
%  The algorithm is stopped as soon as no progress can be made from the
%  current iterate by taking forward and backward steps of length
%  epsilon*xscale(j) for x(j) continuous variables (along the directions given by an 
%  orthonormal basis of space of these variables), or of length xscale(j) 
%  for x(j) discrete, or in the local neighbourhood of categorical variables.
%  This stopping criterion is strengthened by requiring that no
%  decrease can be obtained along the directions of a user-specified
%  number of random orthonormal basis of the space of continuous variables
%  (see the 'termination-basis' option below). Note that a finite number
%  of such basis remains insufficient to guarantee a local minimizer (because
%  the set of descent directions may be 'thin'), but that convergence to such
%  a minimizer is ensured by theory when the number of random basis grows and
%  the objective function is smooth. BFO may also be terminated at the user
%  request.
%
%  BFO also offer the possibility for the user to define his/her own function
%  for searching for a better iterate, given the information (points and 
%  function values) available.  This feature corresponds to the traditional
%  "search-step" which is common in direct-search methods (see the 
%  documentation concerning the input parameter 'search-step' for  more
%  details). Typical ways for the user to compute a better iterate within
%  the search-step function may involve polynomial interpolation/regression, 
%  krieging or RBF models as well as application-specific surrogate modelling
%  techniques.
%
%  Finally, BFO also allows the user to specify the BFO options in a user-supplied
%  "options file" (see the bfo.options.example file for an example).
%
%%  2. Using BFO for structured/sparse/coordinate-partially-separable problems.
%
%  A large number of (often relatively large) optimization problems have some
%  underlying structure, and BFO can be made to exploit this structure to
%  considerable advantage.
%
%  The first case is when the objective function can be expressed as a sum
%  of the form
%
%  $$f(x) = \sum_{i=1}^p f_i(x)$$,                                       (1)
%
%  in which case we say that the objective function is in "sum form".  The
%  functions f_i(x) are the called "element functions". Each element function
%  f_i(.) has two arguments:
%    - i, the index of the element function, and
%    - x, the vector at which f_i(.) must be evaluated.
%  (This slighly redundant specification of element functions allows maximal
%  flexibility.) A first advantage of problems in sum form lies in the possibilities
%  for the user to define a structure exploiting search step function (for
%  instance by building individual models for each of the element functions).
%  The user specifies that an objective function is in sum form by passing
%  in the argument prob (in extensive form) or in prob.objf (in condensed form)
%  a value cell array containing the element function handles, instead of the single
%  function handle for an objective function not in sum form.  For example, the
%  (extensive form) call
%
%      [ x, fx ] = bfo( {{ @(i,x)cosh(x), @(i,x)norm(x) }} , 1 )
%
%  will apply BFO on the problem of minimizing f(x)=cosh(x)+norm(x) starting
%  from the initial value 1. If the objective function is defined in sum form
%  (i.e. if it is passed as cell array of element function handles or a cell
%  array of element function names, see below), BFO will automatically maintain
%  an evaluation history for each individual element function, and will pass this
%  information to the user when calling the user-defined search step. This is
%  achieved by adding one input and one output arguments to the call
%  to the user-defined search-step function call (see the full details
%  on the 'search-step' argument, and, in particular, that of el_hist and level).
%  A slight complication when using the sum form is that number of function
%  evaluations may no longer be integers, because not all element functions
%  necessarily need to be computed for obtaining a function value.
%
%  The main advantage of problems whose objective function is in sum form is
%  the specialization of this structure to the very important case where the
%  problem is "coordinate partially separable" (CPS) or "sparse". In this extremely
%  frequent case (e.g. when the objective function is related to a problems with
%  distinct interconnected blocks, or is resulting from the discretization of a
%  continuous problem), the objective function has the form
%
%  $$f(x) = \sum_{i=1}^p f_i(x_i)$$,
%
%  where now x_i only contains a subset of the problem's variables, defining
%  the "domains" of the element functions, or "element domains". For our present
%  purposes, a sum-form function is considered coordinate partially separable when
%  the maximal dimension of an element domain is (often much) smaller than the
%  total problem dimension. For example, the function
%
%  f( [ x(1) x(2), x(3) ]) = norm( [ x(1) x(2) ], 2 )^2 + norm( [ x(2) x(3) ], 'inf' )
%
%  is coordinate partially separable with 2 elements of domains defined by the two vectors
%  of variable indices [ 1 2 ] and [ 2 3 ].  This problem may then be passed
%  (with its structure) to BFO by the call
%
%     [ x, fx ] = bfo( {{@(i,x)norm(x,2), @(i,x)norm(x,'inf')}}, [ 1 1 1 ], ...
%                      'eldom', {{ [ 1 2 ], [ 2 3 ] }} )
%
%  where the argument 'eldom' is a cell array containing the element domain
%  definitions, and whose length is equal to that of the first argument (see below
%  for more detail). Most significantly, not every f_i(x+s) need to be evaluated
%  to compute f(x+s) if f(x) is known and s is a structured step. This results in
%  substantial gains in the number of evaluations of f(x) as BFO uses such
%  structured steps as much as possible: the optimization method used in this
%  case exploits the underlying structure by isolating independent parts of
%  the problem (see [Price and Toint, 2006] for details).
%
%  Even if the use of this feature may not be critical for small problems,
%  its use for problems of moderate or large size is often essential,
%  especially if the cost of evaluating the element functions f_i(x) is high.
%  The gains in efficiency (evaluation counts) may typically be of several orders
%  of magnitude (the amount of computation and storage internal to the
%  algorithm therefore increases relatively to the number of evaluations).
%  The evaluation gains are of course cumulable with those resulting from a
%  intelligent search-step step strategy due to the sum-form of the problem.
%
%  NOTE: Coordinate partially-separable functions as defined above are often
%        called "sparse" because their Hessians (when they exist) are sparse matrices
%        whose maximal dense principal submatrices are defined by the indices occuring
%        in the definition of the element domains.
%
%
%%  3. Using BFO with categorical variables
%
%
%  BFO also supports the use of categorical variables. Categorical variables are
%  unconstrained non-numeric variables whose possible 'states' are defined
%  by strings (such as 'blue') which may not contain blanks or start with the
%  character '+' or the character '-'. These states are not implicitly ordered,
%  as would be the case for integer or continuous variables. As a consequence,
%  the notion of neighbourhood of a categorical variable is entirely
%  application-dependent, and has to be supplied, in one form or another,
%  by the user. Moreover, the 'vector of variables' is no longer a standard
%  numerical vector when categorical variables are present, but is itself a
%  'vector state' defined by a value cell array of size n (the problem's
%  total dimension, see below), whose i-th  component is either a  number
%  when variable i is not categorical, or a string defining the current state
%  of the i-th (categorical) variable.  For example, such a 4-dimensional vector
%  state can be given by the value cell array
%
%      {{ 'blue', 3.1416, 'green', 2 }}.
%
%  Variable i is declared to be categorical by specifying xtype(i) = 's' (see
%  the xtype argument below). If a problem contains at least one categorical
%  variable, it is called a categorical problem and optimization is carried
%  on vector states (instead of vectors of numerical variables).
%  As a consequence:
%   1) the starting point x0 must be specified as a vector state,
%   2) the returned best minimizing point is a vector state,
%   3) the objective function's value is computed at vector states (meaning that
%      the argument of the function f is a vector state).
%  Moreover, the user must specify the application-dependent neighbours (also
%  called categorical neighbourhoods) of each given vector state with respect
%  to its categorical variables. This can be done in two mutually exclusive ways.
%   1) The first is to specify 'static neighbourhoods'. This is done by
%      specifying, for each categorical variable, the complete list of its
%      possible states (see the cat-states argument description below).  The
%      neighbourhood of a given vector state vs wrt to categorical variable j
%      (the hinge variable) then consists of all vector states that differ
%      from vs only in the state of the j-th variable, which takes all possible
%      values different from vs(j). In this case, all variables of the problem
%      retain their (initial) types and lower and upper bounds.
%   2) The second is to specify 'dynamical neighbourhoods'.  This more flexible
%      technique is used by specifying a user-supplied function whose purpose 
%      is to compute the neighbours of the vector state vs 'on the fly', when
%      needed by BFO (see argument cat-neighbours below). At variance with the
%      static neighbouhood case, the variable types, lower and upper bounds of the
%      neighbouring vector states (collectively called the 'context') may be
%      redefined within the following framework:
%      (i)    the total dimension of the problem (i.e. the size of the vector
%             states) must remain unchanged,
%      (ii)   variables whose current type is 'r' can be redefined to be
%             continuous ('c')
%      (iii)  variables whose current type is 'd' can be redefined to be integer
%             ('i'),
%      (iv)   variables whose current type is 'k' can be redefined to be
%             categorical ('s'),
%      (v)    categorical variables must be unconstrained,
%      (vi)   variables whose current type is 'c' can be deactivated by setting
%             their type to 'r',
%      (vii)  variables whose current type is 'i' can be deactivated by
%             setting their type to 'd'
%             remain categorical.
%      (viii) variables whose current type is 's' can be deactivated by
%             setting their type to 'k'
%      (ix)   at least one variable must remain categorical.
%      (x)    variables whose current type is 'w', 'x', 'y' or 'z' (these
%             variables are temporarily fixed to their current values by BFO)
%             may not change type or value.
%      (xi)   in the multilevel case, only variables in the current level and
%             above may have their type or bounds modified.

%  In effect, this amounts to specifying the neighbouring nodes in a (possibly
%  directed) graph whose nodes are identified by the list, types, bounds and
%  values of the variables. As a consequence, the user-supplied definition of
%  the neighbour(s) of one such node may need to take the values of all variables
%  into account. Of course, for the problem to make sense, it is still required
%  that the objective function can be computed for the new neighbouring vector
%  states and that its value is meaningfully comparable to that at vs.  Variables
%  with status  'r', 'd' or 'k' should have no impact on the objective function
%  value (they are deactivated). The use of 'r', 'd' and 'k' variables allows
%  for the effective dimension of the problem to vary depending on the states
%  of the categorical variables.  However, restriction (i) above imposes that
%  the dimension of the starting vector state x0 must encompass all possible
%  combinations of active/inactive variables (their activity and types depending
%  on categorical variables' states).  In particular, x0 may have inactive
%  continuous/integer/categorical variables with xtype 'r', 'd' or 'k', which
%  can then be activated later on depending on the evolution of the active
%  categorical variables. The number of active or inactive components of x0
%  is called the problem's 'total dimension'. The very substantial flexibility
%  allowed by this mechanism of course comes at the price of the user's full
%  responsability for overall coherence.
%
%  Additional detail is given in the descriptions of the arguments 'xtype',
%  'cat-states' and 'cat-neighbours'.
%
%  NOTE: the use of categorical variables is EXPENSIVE and prone to errors.
%        The use of such variables must therefore be avoided whenever possible,
%        typically by using integer variables or (preferably) relaxed reformulations
%        of the original problem.
%
%  NOTE: the simultaneous use of categorical variables with dynamic neighbourhood
%        and coordinate partial separability is not supported.  The
%        coordinate-partially-separable structure is then ignored. 
%
%
%%  4. Using BFO for multilevel min-max problems
%
%
%  BFO can also be used for solving problems of the form
%
%  $$\min_{S_1} \max_{S_2} \min_{S_3} f( S_1, S_2, S_3 )$$
%
%  where there can be as many as 6 levels of alternating min-max 
%  (or max-min or min-min or max-max), and where the S_i are disjoint sets of
%  variables.  In addition, the variables in S_i may be subject to bounds,
%  possibly depending themselves on the value of the variables in 
%  S_1,... S_{i-1}.  Problems of this type are specified by the user by
%  providing a vector xlevel whose i-th entry indicates the level of the i-th
%  variable.  For instance, a value of xlevel given by
%
%     [ 1 3 2 1 3 2 ]
%
%  with the default choice of minimization corresponds to the 3-levels problem
%
%  $$\min_{x_1, x_4} \max_{x_3, x_6} \min_{x_2, x_5} f( x_1, ..., x_6 ).$$
%
%  The sequence of minimization/maximization is either specified as alternating
%  from the first level (as in our example), or by explicitly entering a vector
%  of 'min' or 'max' strings for the argument 'max-or-min'. Note that it is
%  assumed that the max or min problems are well-defined at every level (in that
%  the objective function is bounded above or below in the relevant subspace). 
%  If bounds are imposed on the set of variables S_i that depend on the value of 
%  the variables S_1 to S_{i-1}, this is done by entering the ( 'keyword', value ) 
%  pair ( 'variable-bounds', 'variable_bounds') and calling the associated 
%  user-supplied function in the form
%
%  [ xlow, xupp ] = variable_bounds( x, level, xlevel, xlower, xupper )
%
%  where the new lower and upper bounds xlow and xupp are redefined, subject to
%  the mentioned rule restricting dependence of bounds for variables of a set
%  on those of sets at previous levels, from the current level, the level
%  definition specified by the argument xlevel, the actual value of x and the
%  value of the (constant) vectors xlower and xupper supplied at the call of
%  BFO. In order to ensure feasibility of the returned value, BFO resets xlow 
%  to the minimum of xlow and xupp, and xupp to its maximum. Note that this 
%  feature allows the use BFO for solving (by *extremely* brute force!) simple
%  optimization constrained problems of the type 
%
%  $$\min_{x_1,x_2}  f( x_2, x_2 ).$$
%
%  such that
%
%  $$ g(x_1) \leq x_2 \leq h(x_1)$$
%
%  where the function g(.) and h(.) are specified in the variable_bounds
%  function.
%
%  More detail is given in the description of the 'xlevel', 'variable-bounds'
%  and 'max-or-min' arguments.
%
%  NOTE: the use of multiple levels is computationnaly EXPENSIVE, and should be
%        avoided whenever possible.
%
%
%%  5. Training BFO for a specific problem class
%
%
%  BFO is a user-trainable algorithm in the sense that it can be trained for
%  improved performance on a specific class of problems.  A typical situation
%  is when a user repeatedly solves problems which only differ marginally, for
%  instance because they depend on a (reasonably slowly) varying data set. 
%  Training consists of optimizing the internal algorithmic parameters specific
%  to the BFO algorithm for best performance.
%
%  There are three possible ways to define the measure of performance on a set
%  of problems when training BFO.
%  1) The simplest performance measure is the sum of the number of objective
%     function evaluations needed by BFO to (hopefully) solve all problems in
%     the problem set. This measure corresponds to improving the average
%     performance on a specific problem.  Minimizing it is refered to as the
%     "average" training strategy.
%  2) A second measure is inspired by robust optimization and is defined as
%     the sum (on all problems) of the largest number of objective function
%     evaluations for algorithmic parameters which differ from their nominal
%     value by at most 5%.  Minimizing this largest number wrt to the algoritmic
%     parameter's nominal values is refered to as the "robust" training strategy.
%  3) A third measure is the area below the curve defined by a performance profile,
%     which is the proportion of successfully solved problems per variant for a
%     given accuracy as a function of the performance of the best variant.
%     Maximizing this area is refered to as the "perfprofile" training strategy.
%  4) A fourth measure is the area below the curve defined by a data profile,
%     which is proportion of successfully solved problems for a given accuracy
%     as a function of the computational budget (in terms of evaluations).
%     Maximizing this area is refered to as the "dataprofile" training strategy.
%
%  Training BFO requires a set of training problems, that is a set of training
%  objective functions and a corresponding set of associated data (starting
%  point, value of bounds or variables' types). If the training set
%  consists of the banana, apple and kiwi objective functions, then training BFO
%  on this set of functions is achieved by a call to BFO of the (condensed) form
%  
%      [ ~, ... ,trained_parameters ] =                                        ...
%               bfo( 'training-mode',        'train',                          ...
%                    'training-problems',    { banana_struct ,apple_struct } )
%
%  where banana_struct and apple_struct are MATLAB structures (struct) describing
%  the training problem objective function and associated data (starting point,
%  bounds, ...). It is also possible to use the  extensive form of the training
%  problems by using a call to BFO of the form
%      [ ~, ... ,trained_parameters ] =                                        ...
%               bfo( 'training-mode',         'train',                         ...
%                    'training-problems',     {@banana,@apple,@kiwi},          ...
%                    'training-problems-data',{@banana_data,@apple_data,@kiwi_data})
%
%  if the associated data is specified by the banana_data, apple_data and kiwi_data
%  functions respectively, but this form does not support the use of categorical
%  nor lattice-based variables. (see below for a full description of the keywords
%  related to training).
%
%  Once BFO has been trained of a particular data set, its trained version
%  (that is its version using the optimized internal algorithmic parameters)
%  can be applied immediately to solve a further problem (in the
%  'train-and-solve' training mode)
%
%       [ xbest, ... ,trained_parameters ] = bfo( orange_s,                    ...
%                     'training-mode',      'train-and-solve',                 ...
%                     'training-problems' , { banana_struct, apple_struct } )
%
%  or
%
%       [ xbest, ... ,trained_parameters ] = bfo( @orange, x0,                 ...
%                     'training-mode',         'train-and-solve',              ...
%                     'training-problems' ,    {@banana,@apple,@kiwi},         ...
%                     'training-problems-data',{@banana_data,@apple_data,@kiwi_data})
%
%  or applied later to one or more problems in the 'solve' training mode, with
%  a call of the form
%
%       [ xbest, fbest, ... ,trained_parameters ] = bfo( @orange, x0,          ...
%                     'training-mode',          'solve',                       ...
%                     'trained-bfo-parameters' ,'trained.bfo.parameters')
%
%  where 'trained.bfo.parameters' is the name of a file where the optimized
%  algorithmic parameters have been saved by a previous run in modes 'train' or
%  'train-and-solve'. Again details are provided in the description of the
%  keywords for training.
%
%  BFO can be also trained on problems of the CUTEst library with the extensive
%  problem formulation and facilities are provided to use the CUTEst MATLAB
%  interface. If the training set consists of the CUTEst problems HS4, YFIT
%  and KOWOSB, then training BFO on this set of functions is achieved by a call
%  to BFO of the form
%  
%       [ ~, ... ,trained_parameters ] =                                       ...
%                bfo( 'training-mode', 'train',                                ...
%                     'training-problems-library', 'cutest',                   ...
%                     'training-problems' , { 'HS4', 'YFIT', 'KOWOSB' } )
%
%  The parameter 'training-problem-data' should not be set. Data for the CUTEst
%  problems is transfered from CUTEst to BFO (and can possibly be modified by the 
%  user) in the BFO function bfo_cutest_data.m (at the end of this file).
%
%
%%  6. Restrictions
%
%  Some combinations of options are not supported.  These are
%  1) training BFO with training problems expressed in extensive form and containing
%     categorical and/or lattice-based integer variables,
%  2) the use of parameters in the function handle defining the objective function
%     of a problem, when the problem is expressed in condensed form,
%  3) the exploitation of coordinate partially-separable structure when the problem features
%     categorical variables with dynamic neighbourhoods. 
%
%%  7. Description of the INPUT parameters
%
%
%   PRELIMINARIES
%
%   Some input arguments of BFO require the specification of MATLAB functions
%   (for the objective function, the training objective functions, the search-step
%   function, the tracker function, the training data functions and the
%   variable-bounds function, see below).  If the user wishes to specify a function
%   whose calling sequence is "fx = func( x )" and for which a file func.m exists
%   in the MATLAB path, this can be done in two different ways:
%   1) by a function handle: in this case the user must pass the function 
%               handle "@func" or "@(x)func(x)".  If one wishes to to allow
%               func to depend on x but also on additional parameters, with 
%               the calling sequence "fx=func(x,parameter1,parameter2)", say, 
%               then the argument becomes "@(x)func(x,parameter1,parameter2)",
%               and the values of the parameters are then automatically passed
%               to func when it is called within BFO (provided they have been
%               properly assigned values in the calling program).
%   2) by a string: in this case, the user must pass the string 'func' or
%               '@(x)func(x)'.  The string form does NOT support the
%               passing of parameters as in the function handle form, unless the
%               parameters are simple numbers or strings.  Thus the string
%               '@(x)func(x,10.7,''a_string'')' may be used, with the same result as
%               that described for function handles. But specifying the string
%               '@(x)func(x,parameter1,parameter2)' will result in an error if
%               parameter1 or parameter2 are variables whose value has been 
%               assigned by the user! (This is because the function handle
%               str2func('@(x)func(x,parameter1,parameter2)' created within BFO
%               does not remember the value of parameter1 and parameter2.)
%   Note that BFO will return an error message if the file "func.m" cannot be
%   found in the MATLAB path. Also note that none of the user-supplied functions
%   may have a name starting with the characters 'bfo_' and that all strings such
%   as filenames, function names, state names etc may not contain blanks, quotes
%   or comas, or any of the characters '{', '}', '[',  ']' or '%'.
%
%   The first argument of BFO specifies the optimization problem to solve and
%   the second (optionnally) the relevant starting point. The latter is unecessary
%   when the problem is specified in condensed form and may be omitted in this case.
%   These two arguments are omitted entirely when BFO is used in training mode 
%   only. All subsequent optional inputs are specified by passing a pair
%   ('keyword', value') to BFO, where the keyword defines the nature of the 
%   input and the value is the input value itself. Thus a call to BFO has the
%   typical form
%
%        [ outputs ] = bfo( objective-function, starting point,                ...
%                           'keyword', value, ..., 'keyword', value )
%
%   in extensive form, or
%
%        [ outputs ] = bfo( prob-struct, ...
%                           'keyword', value, ..., 'keyword', value )
%
%   in condensed form, where the sequence "'keyword', value, ..., 'keyword', value"
%   may be empty, or the objective-function handle/string and the starting point
%   may be omitted  in the 'train' training mode. The ( 'keyword', value ) pairs
%   may be specified by the user in any order, except that, when static categorical
%   variables are present and the problem is in extensive form, the specification
%   of xtype must precede that of cat-states.


%   INPUT  (optional) :
%
%   We start by describing the first two arguments.
%
%   prob      : depending on the problem form used (condensed or extensive), prob
%               is either a struct containing the condensed problem formulation, or
%               a function handle or string (or a value cell array of those)
%               specifying the function to be minimized.
%                  (i)  Condensed formulation
%               In this formulation, the optimization problem is given by
%               a MATLAB struct whose fields are
%                  'objf', 'x0', 'xlower', 'xupper', 'xtype', 'xscale', 'max_or_min',
%                  'xlevel', 'variable_bounds', 'cat_states', 'cat_neighbours',
%                  'lattice_basis', 'eldom'
%               where x0, xlower, xupper, xtype, xscale, max_or_min, xlevel,
%               lattice_basis, the cat_states structure and the variable_bounds
%               and cat_neighbours strings or function handles are the arguments of
%               the corresponding keywords (with hyphens) defined as above.
%               The 'objf' fields contains a string or a function handle specifying
%               the objective function. The 'objf' and 'x0' fields are mandatory,
%               the others are optional. The problem is considered to be in sum form if
%               the "objf' field contains an array cell of function handles or strings.
%               If the condensed formulation is used, the BFO arguments corresponding
%               to the struct's fields are ignored and need not be specified when 
%               calling BFO. 
%                  (ii) Extensive formulation
%               In the extensive formulation, prob is a function handle or string
%               (or a value cell array of those for sum-form objective functions) 
%               specifying the function to be minimized.
%               NOTE: If the objective function returns a NaN value, this is interpreted
%                     to mean that the objective function is undefined at x (and is then
%                     not considered as a potential solution),
%               NOTE: Specifying prob is mandatory in the 'train-and-solve' and
%                     'solve' training modes.
%               NOTE: On restart, it is the responsibility of the user to 
%                     provide an objective function which is coherent
%                     with that used for the call where the restart
%                     information was saved.
%               NOTE: the string  defining the objective function (or the
%                     transformation of the objective function handle into a string)
%                     must not conflict with any of the reserved names in BFO. These
%                     are 'bfo_'*.
%   x0        : a vector (when no categorical variables are present) or (in the
%               presence of categorical variables) a value cell array (such as
%               {{ 'blue', 3.1416, 'green', 2 }} ) containing the starting point
%               for the minimization.
%               NOTE: when the condensed form is used, the specification of
%                     this argument causes its value to overwrite that specified
%                     in the prob struct (argument 1)
%               NOTE: the dimension n of the space is derived from the length of x0.
%               NOTE: specifying x0 is mandatory in the 'train-and-solve' 
%                     and 'solve' training modes.
%
%   We now describe the possible ( 'keyword', value ) pairs by considering each
%   such possible keyword and specifying the meaning and format of its associated 
%   value. Remember that the ( 'keyword', value ) pairs may be specified by the 
%   user in any order except that, when static categorical variables are present
%   and the problem is in extensive form, the specification of xtype  must precede that 
%   of cat-states.
%
%   fx0       : when specified, contains the value of the objective function at x0.
%               Default: none.
%               NOTE: It is the responsability of the user to ensure that the
%                     supplied value is indeed the value of the objective function
%                     at x0.  In particular, a supplied value which is (erroneoiusly)
%                     to small may result in the algorithm stalling at x0.
%               NOTE: a supplied value equal to Inf in absolute value will cause
%                     recomputation within BFO.
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
%               2) the argument contains a column array of strings with as many
%                  strings of the form 'max' or 'min' as they are levels 
%                  (defined by the 'xlevel' argument), the j-th string then
%                  specifying the type of optimization at level j.
%               NOTE: when the condensed form is used, the specification of
%                     this argument causes its value to overwrite that specified
%                     in the prob struct (argument 1)
%               Default: 'min'
%   f-call-type: the type of calling sequence used for computing the value
%               of the objective function f(x) by the user-supplied function.
%               The following types are available:
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
%               NOTE: if this option is used for a sum-form objective function,
%                     it is assumed that each element function only has non-negative
%                     values.
%               Default: 'simple'
%   f-bound   : a real number specifying a bound on the objective function value
%               above which (for minimization) or under which (for maximization)
%               any function evaluation in the course of the algorithm will be
%               considered unsuccessful (see the description of the 'f-call-type'
%               keyword above).
%               Default: +Inf (minimization), -Inf (maximization).
%   tracker   : if supplied, the name of a user's "tracker" or "tracker function"
%               which is a function called by BFO before each major iteration,
%               giving the user access to the current approximate solution and allowing
%               the user to terminate BFO if desired.  This function is defined as
%                    stop_optimization = trackfun( x, fx, xlower, xupper, neval )
%               where trackfun is the value of the supplied argument and
%                    x     : is the current best iterate,
%                    fx    : is the objective function's value at x,
%                    xlower: is the vector of lower bounds on x,
%                    xupper: is the vector of upper bounds on x,
%                    neval : is the current number of full objective function
%                            evaluations.
%               and
%                    stop_optimization: is nonzero iff the user wishes to
%                            terminate optimization at the current point.
%               The use of a tracker function allows a very flexible user-defined
%               termination criterion, as well as a follow-up of the iterates
%               history (for instance by plotting relevant quantities associated
%               with the current iterate).
%               Default: none.
%   xlower    : the vector of size n containing the lower bounds on the 
%               problem's variables.
%               NOTE: when the condensed form is used, the specification of
%                     this argument causes its value to overwrite that specified
%                     in the prob struct (argument 1)
%               NOTE: the starting point is projected onto the bound-feasible
%                     set before minimization is actually started.
%               NOTE: lower bounds may be equal to -Inf, but these will
%                     be converted to -1.e25.
%               NOTE: a single value may be specified if all variables
%                      have the same lower bound (ex. xlower = [ 0 ] indicates
%                      that all variables are nonnegative).
%               Default: -1.e25
%   xupper    : the vector of size n containing the upper bounds on the problem's 
%               variables.
%               NOTE: the starting point is projected onto the bound-feasible
%                     set before minimization is actually started.
%               NOTE: upper bounds may be equal to +Inf, but these will
%                     be converted to 1.e25.
%               NOTE: a single value may be specified if all variables
%                     have the same upper bound (ex. xupper = [ 1 ] indicates
%                     that all variables are bounded above by 1).
%               Default: +1.e25
%   xscale    : a vector of size n giving the variables 'scalings'.  This vector serves
%               two distinct purposes.
%               - If variable j is continuous, xscale( j ) is a strictly positive
%                  scalar indicating its typical order of magnitude. 
%               - If variable j is discrete, xscale( j ) is the distance separating two
%                 neigbouring values of variable j (ex: xscale( j ) = 1 if variable j is 
%                 integer).
%               The value of xscale( j ) is ignored if variable j is categorical.
%               It is also ignored for discrete variables if an explict lattice
%               is specified by the user with the keyword 'lattice-basis'.
%               If a single number xscale is specified, then the uniform scaling xscale 
%               is used for all continuous variables and the value 1 is used for all
%               discrete ones. For well-scaled problems, use xscale = 1 or do not use 
%               the keyword at all.
%               NOTE: when the condensed form is used, the specification of
%                     this argument causes its value to overwrite that specified
%                     in the prob struct (argument 1)
%               NOTE: the value(s) of xscale influence both the initial increments
%                     (because they are given by delta( j ) * xscale( j )) and the
%                     termination criterion, as BFO does not terminate as long as
%                     significant decrease can be achieved for changes of size 
%                     epsilon * xscale( j ) in continuous variables.
%               NOTE: specifying xscale is most useful when the scalings vary significantly
%                     between continuous variables.
%               Default: ones(1;n)
%   delta     : the scaling-independent initial (positive) increments for the
%               continuous variables, such that the initial move along variable  i is 
%               +/- delta( i ) * xscale( i ) for such variables.  The value of delta( i ) 
%               is ignored if variable i is discrete or categorical. If a single 
%               number delta is specified, the value delta( i ) = delta is used for 
%               all continuous variables.
%               Default: 2.64133
%   delta     : the scaling-independent initial (positive) increments for the
%               variables, such that the initial move along variable  i is 
%               delta( i ) * xscale( i ) If a single number delta is specified,
%               the value delta( i ) = delta is used for all continuous variables
%               and delta( i ) = 1 for all discrete ones.  It is meaningless for
%               categorical variables.
%               Default: 2.64133
%   xtype     : a string of length n, defining the type of the variables.
%               Its allowed values are:
%               xtype(j) = 'c' if variable j is continuous,
%               xtype(j) = 'i' if variable j is discrete in that it can
%                              only vary by multiples of xscale(j), possibly
%                              along a predefined lattice (see 'lattice-basis'),
%               xtype(j) = 's' if variable j is categorical,
%               xtype(j) = 'f' if variable j is fixed,
%               xtype(j) = 'r' if variable j is a continuous variable which is
%                              inactive for the current state of the categorical
%                              variables,
%               xtype(j) = 'd' if variable j is a discrete variable which is
%                              inactive for the current state of the categorical
%                              variables,
%               xtype(j) = 'k' if variable j is a categorical variable which is
%                              inactive for the current state of the categorical
%                              variables,
%               NOTE: when the condensed form is used, the specification of
%                     this argument causes its value to overwrite that specified
%                     in the prob struct (argument 1)
%               NOTE: a string of length 1 may be specified if all variables
%                     are of the same type ( ex. xtype = 'c' indicates that
%                     all variables are continuous).
%               NOTE: a typical use of discrete variables is to specify 
%                     variables whose values must be integers (hence the 'i').
%                     This is  achieved by declaring the variable to be discrete 
%                     (xtype(j) = 'i'), its scale to be 1 (xscale(j)=1, the default) 
%                     and its initial value to be an integer.
%               NOTE: variables with type 'r', 'd' and 'k' are only allowed for
%                     problems involving categorical variables.
%               NOTE: see the description of the argument 'training-mode' for
%                     the use of xtype to specify the class of problems for
%                     which training is desired.
%               NOTE: additional types of variables are defined in the code
%                     but their use is restricted.
%               Default: 'c'
%   eldom     : a value cell array of length equal to that defining the domains
%               of the objective function's elements (length(prob) in extensive formulation,
%               length(prob.objf) in condensed formulation), whose i-th
%               element is a integer vector containing the indices of the 
%               variables occurring in the i-th element function of coordinate
%               partially-separable problems. Meaningless for unstructured 
%               problems.
%               NOTE: Not specifying eldom for a sum-form objective function
%                     is equivalent to specifying that all element domains
%                     involve all variables.  Specifying eldom for an objective
%                     function which is not in sum form is an error.
%               NOTE: a variable whose index does not occur anywhere in eldom is
%                     considered fixed.
%               Default: none (the objective function is assumed not to be 
%                        coordinate partially separable by default).
%   use-cps   : an integer specifying whether exploiting the coordinate partially
%               separable structure must be attempted (when eldom is nonempty).
%               Values that may be specified by the user are:
%               0 : no exploitation of the CPS structure shoule be attempted,
%               1 : attempt exploiting the CPS structure, if present.
%               NOTE: negative values are also specified by BFO in the course of
%                     a CPS run (they specify the opposite of the index of 
%                     the independent subset of variables in coordinate 
%                     partially-separable problems).  Such values are
%                     automatically specified during the algorithm 
%                     and should NOT be specified/altered by the user.
%   cps-subspace-optimizer: a string defining the strategy used for solving the
%               subspace optimization arising in the solution of coordinate-
%               partially-separable problems.  For such problems, the problem
%               analysis determines independent subspaces on which optimization
%               may be conducted independently. BFO provides three strategies for
%               this optimization:
%               'full' : when this option is specified, the full recursive BFO
%                        is used for subspace optimization,
%               'core' : when this option is specified, a simplified version of
%                        BFO is used for the (frequent) case where
%                        * there are no active discrete or categorical variables
%                        * the (continuous) variables are unscaled
%                        * the element function calls are simple 
%                        * the optimization is not multilevel
%                        The 'full' BFO is called when any of the above four 
%                        conditions fails.
%               'min1d': when this option is specified, the above four conditions
%                        hold and the subspace is unidimensional, an even more
%                        specialized unidimensional optimizer is called.  If the four
%                        conditions hold but the subspace has dimension larger than
%                        one, the 'core' optimizer is used. The 'full' BFO is called 
%                        when any of the four conditions fails.
%               NOTE:
%               Default: 'min1d'
%   cat-states: a value cell array), whose i-th component is either '' when
%               variable i is not categorical, or a cell array of non-empty
%               strings defining all possible states of variable i
%               (ex: {{ {'blue', 'black'}, '', {'blue', 'green', 'yellow'} }} ).
%               The specification of cat-states must occur AFTER that of xtype
%               in the call to BFO.
%               NOTE: when the condensed form is used, the specification of
%                     this argument causes its value to overwrite that specified
%                     in the prob struct (argument 1)
%               NOTE: each string in the argument of cat-states must be
%                     at most 12 characters long and should not contain blanks.
%                     The number of states for a single categorical variable
%                     must not exceed 99.
%               Default: none.
%   cat-neighbours : a string or function handle specifying the user-supplied
%               function defining the dynamical categorical neighbourhoods.
%               If the keyword is specified, the user must supply a function
%               whose purpose is to compute the neighbours of the vector state
%               cx and its associated variable types xtype and lower and upper
%               bounds xlower and xupper.
%               The calling sequence of that user-supplied function is
%                   [ cneighbours, xtypes, xlowers, xuppers ] = ...
%                                cat_neighbours( vsx, xtype, xlower, xupper )
%               where
%                   cneighbours is a cell array whose elements are the neighbours
%                               of the vector state vsx,
%                   xtypes      is a cell array whose elements are the variable
%                               type associated with each of these neighbours
%                               (possibly different from xtype under the rules
%                               (i)-(viii) above),
%                   xlowers     is a matrix whose columns are the lower bounds
%                               on the variables associated with each of these
%                               neighbours (possibly different from xlower),
%                   xuppers     is a matrix whose columns are the upper bounds
%                               on the variables associated with each of these
%                               neighbours (possibly different from xlower).
%               The number of elements of cneighbours and xtypes and the number
%               of columns of xlowers and xuppers must be indentical and >= 0,
%               and defines the size of the neighbourhood of the vector state
%               vsx relative to categorical variable j. 
%               NOTE: when the condensed form is used, the specification of
%                     this argument causes its value to overwrite that specified
%                     in the prob struct (argument 1)
%               NOTE: if cat-neighbours is specified, the possible simultaneous
%                     specification of cat-states is ignored.
%               NOTE: a value of '' is interpreted as if the keyword had not been
%                     specified.
%               Default: none. A valid function handle must be provided if the
%                        'cat-neighbours' keyword is specified.
%   lattice-basis : a matrix of size nd by nd (where nd is the number
%               of discrete variables), whose columns span the lattice on 
%               which minimization on discrete variables must be carried out.
%               When this matrix is specified, minimization on the i-th
%               discrete variable is interpreted as minimization along fixed
%               multiples of the i-th column of the given matrix. 
%               NOTE: when the condensed form is used, the specification of
%                     this argument causes its value to overwrite that specified
%                     in the prob struct (argument 1)
%               NOTE: any value of the parameter xscale for discrete variables
%                     is ignored if lattice-basis is specified.
%               Default: the identity matrix of size nd by nd.
%   npoll     : the maximum number of polling directions for continuous variables
%               Default: the smallest of 50 and the number of continuous variables
%               NOTE: a larger value of npoll often leads to better performance
%                     if there are many continuous variables, but there is no
%                     point to specify a number larger than the the number
%                     of such variables.
%   epsilon   : a real number specifyig the accuracy level defining the 
%               termination rule for continuous variables. The algorithm is 
%               terminated when no objective function decrease can be obtained
%               for changes in continuous variables of size epsilon * xscale( j ). 
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
%   l-hist    : the length of the memorized evaluation history: care is taken to
%               avoid re-evaluation of the objective function at points whose
%               evaluation precedes the current one for at most l-hist.  The value
%               of l-hist must be an integer at least equal to 1.  If the user
%               specifies a negative number, the complete evaluation history is
%               remembered (definitely avoids recomputation, but more costly
%               in memory and time).
%               Default: 1 (no search-step), 250 (with search-step)
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
%               'low'    : a one-line summary per iteration,  containing
%                          - the number of objective function evaluations
%                          - the current best objective function value
%                          - (possibly) the size of projected gradient
%                          - the current maximum increment in cont. variables
%                          - the index of the successful search direction 
%                            (+ if forward, - if backward), or the current
%                            iteration status.
%                          For coordinate partially separable functions, the projected
%                          gradient size is replaced by the number of
%                          successful subset optimizations and the total 
%                          number such optimizations, and the last column
%                          contains the group and subset index of the best
%                          successful subset optimization.
%               'medium' : a one-line summary per iteration + a summary of x,
%                          the current best iterate
%               'high'   : more detail
%               'debug'  : for developers'use only :-).
%               If multilevel optimization is requested, the verbosity of each
%               optimization level may be set independently by specifying 
%               a cell whose length is equal to the number of levels and whose
%               elements are strings describing verbosity levels following the
%               above convention.
%               NOTE: numbers of objective function evaluations are always 
%                     printed as integer. They are rounded above (for printing)
%                     if this is not the case as may happen if the objective 
%                     function is given in sum form.
%               Default: 'low'
%   ps-subset-verbosity: the volume of output produced by the algorithm
%               when solving the independent subset optimizations within a
%               coordinate partially-separable problem (see the 'verbosity' 
%               keyword for possible values).
%               Default: 'silent'
%   random-seed : a positive integer specifying the seed for the random
%               number generator rng( seed, 'twister') which is used to
%               initialize random sequences at the beginning of execution.
%               Random numbers are used for the choice of alternative basis
%               vectors for the continuous variables, both when refining 
%               the grid and when checking termination.
%               Default: 0 (MATLAB default)
%   reset-random-seed : a string whose meaning is
%               'reset'    : the random number generator is reinitialized by BFO,
%               'no-reset' : the random number generator is not reinitialized
%                            by BFO.
%               NOTE : The random seed is not reinitialized on restart.
%               NOTE : The random seed is always reinitialized during the 
%                      training process
%               Default: 'reset'
%   search-step: a function handle or string specifying the name of a
%               user-supplied search-step function. If the argument is supplied, 
%               a search-step is carried on at every major iteration in BFO and 
%               the function specified must be available in the Matlab path.  
%               Its calling sequence is given by
%                  [ xsearch, fsearch, nevalss, exc, x_hist, f_hist ] =          ...
%                          search_step_function( level, f, xbest, max_or_min,    ...
%                                                xinc, x_hist, f_hist, xtype,    ...
%                                                xlower, xupper, lattice_basis )
%               if the problem involves no categorical variables and its objective
%               function is not in sum form, or by
%                  [ xsearch, fsearch, nevalss, exc, x_hist, f_hist, el_hist ] = ...
%                          search_step_function( level, f, xbest, max_or_min,    ...
%                                                xinc, x_hist, f_hist, xtype,    ...
%                                                xlower, xupper, lattice_basis, el_hist )
%               if the problem involves no categorical variables and its objective
%               function is in sum form, or by
%                  [ xsearch, fsearch, nevalss, exc, x_hist, f_hist,             ...
%                     xtype, xlower, xupper ] =                                  ...
%                          search_step_function( level, f, xbest, max_or_min,    ...
%                                                xinc, x_hist, f_hist, xtype,    ...
%                                                xlower, xupper, lattice_basis,  ...
%                                                cat_states, cat_neighbours )
%               if there are categorical variables and the objective function is
%               not in sum form, or (finally) by
%                  [ xsearch, fsearch, nevalss, exc, x_hist, f_hist,             ...
%                     xtype, xlower, xupper, el_hist ] =                         ...
%                          search_step_function( level, f, xbest, max_or_min,    ...
%                                                xinc, x_hist, f_hist, xtype,    ...
%                                                xlower, xupper, lattice_basis,  ...
%                                                cat_states, cat_neighbours, el_hist )
%               if there are categorical variables and the objective function is in
%               sum form.  In these calling sequences,
%               on input :
%                  level   : is the current level (for the multilevel case).  It is
%                            also used to indicate termination, allowing cleanup
%                            internal to the search-step function (see below)
%                  f       : is the handle to the objective function, or, if in sum form,
%                            a value cell array containing the handle of the element 
%                            functions,
%                  xbest   : is the vector of current best values of the problem's 
%                            variables,
%                  max_or_min: a string which is either 'min' or 'max', depending
%                            whether minimization or maximization is considered,
%                  xinc    : is a vector whose i-th component contains the current 
%                            increment (mesh sizes) for the i-th variable,
%                  x_hist  : if there are no categorical variables, x_hist is an
%                            (n x min(nfcalls,l_hist)) array whose columns contain 
%                            the min(nfcalls,l_hist)) points at which f(x) has
%                            been evaluated last,
%                            if the problem is categorical, x_hist is a cell
%                            of length p=length min(nfcalls,l_hist), whose elements contain
%                            the min(nfcalls,l_hist) points (vector states) at
%                            which f(x) has been evaluated so last.  It is thus of the
%                            form { vs_1, vs_2, ..., vs_p }, where vs_i are vector states.
%                  f_hist  : is an array of length min(nfcalls,l_hist) containing the 
%                            function values associated with the columns of x_hist,
%                  xtype   : is the status of the variables (see above)
%                  lattice_basis : is the lattice associated with the discrete 
%                            variables, if any (or the empty array otherwise)
%                  cat_states: the possible states of the categorical variables
%                            (see the cat-states keyword),
%                  cat_neighbours: the user supplied function defining dynamical
%                            categorical neighbourhoods if relevant, the empty string
%                            otherwise.
%                  el_hist : a cell array of length equal to length( prob ) 
%                            (in extensive formulation) or to length( prob.objf ) (in 
%                            (in condensed formulation), whose i-th entry is a struct
%                            with fields
%                            eldom: the indices of the variables occuring in the
%                                   the i-th element function
%                            xel  : a matrix/cell of vector states, whose 
%                                   columns/entries contain the last l_hist
%                                   points (in the domain of the i-th element 
%                                   function) at which this element function has
%                                   been evaluated
%                            fel  : the corresponding values of the i-th element function
%                            fbest: the corresponding value (the i-th element
%                                       function evaluated at xbest).
%               and, on output,
%                  xsearch : is an array or a vector state of length n containing
%                            the point returned by the user as a tentative
%                            improved iterate,
%                  fsearch : in not in sum form, the associated objective function 
%                            value f(xsearch), or, if in sum form, a vector containing
%                            the values of the element functions at xsearch, such that
%                            the value of the complete objective function at xsearch 
%                            is sum(fsearch)
%                  nevalss : is the number of function evaluations performed internally 
%                            by the search-step function (can be fractional is the
%                            objective function is in sum form)
%                  exc     : is the search-step exit condition.  Its values are interpreted
%                            by BFO as follows:
%                            exc =  3  seemingly at optimum: the search-step function
%                                      has determined that the base point appears to be
%                                      optimal (xsearch and fsearch are to be ignored
%                                      in this case);
%                            exc =  2: normal successful return: a search-step xsearch 
%                                      and associated value fsearch has been found, which
%                                      produces significant objective function decrease;
%                            exc =  1:  normal unusuccessful return: a  search-step xsearch
%                                      and associated fsearch have been calculated, but do
%                                      not produce significant objective function decrease;
%                            exc  = 0:  the search step function has found a point xsearch
%                                      and a valu fsearch, but leaves to BFO the decision
%                                      to use it or not;
%                            exc = -1: the search-step function could not reliably compute
%                                      a search step, and the returned xsearch and fsearch
%                                      should be ignored;
%                            exc = -2: the objective function returned a NaN or infinite
%                                      value: the returned xsearch and fsearch should be 
%                                      ignored;
%                            exc = -3: the user has requested termination:  the returned 
%                                      xsearch and fsearch should be ignored and BFO stopped;
%                  x_hist  : is the updated array or cell to which is now
%                            appended the set of vectors (or vector states) at which
%                            the objective function has been evaluated
%                            (xsearch = x_hist( 1:n,end) or x_hist{ end }).
%                  f_hist  : is the updated array of function values 
%                            (f_search = f_hist(end)).
%                  xtype   : is the (possibly updated) types of the components of
%                            the vector state xsearch (need not be set
%                            if cat_neighbours is the empty string),
%                  xlower  : is the (possibly updated) lower bounds on the
%                            components of the vector state xsearch (need not 
%                            be set if cat_neighbours is the empty string),
%                  xupper  : is the (possibly updated) upper bounds on the
%                            components of the vector state xsearch (need not
%                            be set if cat_neighbours is the empty string).
%                  el_hist : is the cell array containing the updated structs
%                            (for all element function evaluations during the 
%                            search-step computation).
%               Before BFO terminates and handles back the control to the user,
%               the search-step function is called a last time with a value of
%               level = -1, which is meant to indicate termination and allows the
%               search-step function to perform internal cleanup if necessary.
%               At this last call, all input arguments except level are meaningless
%               (most of them are empty). No output from this last call is expected
%               either.
%               NOTE: It is the responsability of the user to ensure that the
%                     search-step function's interface conforms to the above.
%               NOTE: the point returned in xsearch must preserve the type of the 
%                     variables (as specified by xtype) and satisfy the bound and 
%                     lattice constraints, if relevant.  Moreover, if 
%                     xtype(i) is different from 'c', 'i' or 's', then xsearch(i) 
%                     equal to x_hist(i,end),  the i-th component of the point
%                     must be given in the last column of x_hist.
%               NOTE: If a search-step is required, the value of l_hist should
%                     not be too small (in order to provide some data to the
%                     search-step function). A value of twice the problem
%                     total dimension is a minimum.
%               NOTE: Search steps are disabled in training mode
%               NOTE: a value of '' is interpreted as if the keyword had not been
%                     specified.
%               Default: none.
%   xlevel    : an array of positive integers of size n, where xlevel(i) is 
%               the index of the level to which the i-th variable is
%               associated. The number of levels i computed as nlevel = max( xlevel )
%               and may not exceed 6. Each variable  must be assigned a level between
%               1 and nlevel, and each level between 1 and nlevel must be
%               associated with at least one variable. 
%               NOTE: when the condensed form is used, the specification of
%                     this argument causes its value to overwrite that specified
%                     in the prob struct (argument 1)
%               Default= ones(size(x))
%   variable-bounds : a string or function handle specifying the user-supplied
%               function computing bounds on variables of a level as a
%               function of the value of the variables at previous (i.e. of
%               lower index) levels. If the keyword is specified, the user must
%               supply a function
%                  [ xlow, xupp ] = variable_bounds( x, xlevel, xlower, xupper )
%               where the subset of the vectors corresponding to level i may
%               be recomputed from the values of x, xlower and xupper (as
%               specified on calling BFO) under the constraint that they may
%               only depend of the values of the  variables associated with 
%               levels 1 to i-1. 
%               NOTE: the use of user-defined parameters in a function
%                     handle form of the variable-bounds function is not
%                     allowed.
%               NOTE: when the condensed form is used, the specification of
%                     this argument causes its value to overwrite that specified
%                     in the prob struct (argument 1)
%               NOTE: a value of '' is interpreted as if the keyword had not been
%                     specified.
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
%               'training': similar to 'use', but allows restarting BFO from a 
%                      checkpointing file saved when running in 'train' 
%                      training mode (without an objective function specified).
%                      Note that this requires the training mode 'train' to be 
%                      specified again when restarting.
%               NOTE:  Restarted calls MUST specify the same objective function
%                      as that used in the call at which restart information was 
%                      saved.
%               NOTE:  The effect of restarting the computation after a save within
%                      a recursive call (when integer or categorical variables are
%                      present) is sometimes hard to predict :-).
%               NOTE:  if a multilevel computation is required, restart is
%                      only available for the first level optimization.
%               NOTE:  during training, restart can be invoked at the level of the
%                      training process itself (i.e. within the 'average', 'robust'
%                      'perfprofile' or 'dataprofile' optimization process).  Restart 
%                      at a lower level (solution of individual test problems) is 
%                      not available.
%               NOTE:  if restart is used after a breakdown in training, the user
%                      is free to specify any training mode.  In the 'train' and
%                      'train-and-solve' modes, the training process will be 
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
%                     specified (as the first two arguments) in the 'train' training mode.
%               Default: 'solve'
%   training-problems-library : a string which defines the library of BFO training
%                problems.  Possible values are:
%               'user-defined': the set of training problems is created by the user.
%               'cutest'      : training problems are chosen in the CUTEst library 
%                               and the CUTEst MATLAB interface is used.
%               NOTE: meaningless in "solve" training mode.
%               Default: 'user-defined'
%   training-problems : a cell array containing either MATLAB structs describing the
%               training problems in condensed form (see above for theier definition
%               in the paragraph  describing the prob argument),  or function handles
%               or strings specifying the objective functions of these problems (in
%               extensive form).  In this latter case, the argument of training-problems
%               may have the form {@f1,@f2,...} or {'f1','f2',...} with fi of
%               the form f = fi(x). If training-problems-library = 'cutest',
%               only the second form is possible, where each 'fi' is the name
%               of a CUTEst problem.
%               NOTE: training-problems is mandatory in 'train' and
%                     'train-and-solve' training modes.
%               NOTE: this arguement is meaningless in "solve" training mode.
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
%                  [ x0, xlower, xupper, xtype, xscale, max-or-min, xlevel,      ...
%                         variable_bounds ] = fi_data,
%               where x0, xlower, xupper, xtype, xscale, max-or-min, xlevel and the
%               variable_bounds s string or function handle are the arguments of the
%               corresponding keywords defined as above. If the length of
%               training-problems-data is equal to 1, it is assumed that all training
%               problems share the same data function.
%               NOTE: when the 'extensive formulation' is used to specify the training
%                     problems, training-problems-data is mandatory in train and
%                     train-and-solve training mode, and must be of the same
%                     size as the training-problems argument. This argument must NOT
%                     be specified when the 'condensed formulation' is used.
%               NOTE: lattice-based integer variables and categorical variables are
%                     not supported in training mode using the extensive formulation.
%               NOTE: training-problems-data is ignored if 
%                     training-problems-library = 'cutest'.
%               NOTE: the cat_states structure is ignored (and can be set to {})
%                     when cat_neighbours is specified.
%               NOTE: this argument is meaningless in "solve" training mode.
%   training-strategy: a string which defines the BFO training strategy:
%               'average': minimize the average number of function
%                          evaluations per function on the training set,
%               'robust' : minimize the worst average number of
%                          function evaluations in a box defined by the product 
%                          of intervals where each parameter is allowed to
%                          deviate from its nominal value by at most 5%.
%               'perfprofile': maximize the area below a performance profile curve
%                          for a specified window on the performance ratios.
%               'dataprofile': maximize the area below a data profile curve
%                          for a specified window on the computational budget.
%               NOTE: meaningless in "solve" training mode.
%               Default: 'average'
%   training-parameters : cell of strings containing the names of the BFO
%               parameters to be trained (in training-mode 'train' or
%               'train-and-solve'). Possible value are 'alpha', 'beta',
%               'gamma', 'delta', 'eta', zeta', 'inertia', 'search-type',
%               'random-seed', 'iota', 'kappa', 'lambda', and 'mu' (see the 
%               definition of these input parameters for details on their nature).
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
%   training-profile-window: a vector of with two componenst specifying the lower 
%               and upper bounds of the interval of performance ratios (for 
%               perfprofile training) or of computational budget expressed as 
%               a theoretical number of simplex gradients (for dataprofile training).
%               Default: [1,20] (for perfprofile), [0,2000] (for dataprofile)
%   training-profile-cutoff-fraction: a number in [1e-10,1-1e-10] specifying the
%               fraction of the reference improvement in objective function value
%               for each training problem which is necessary to declare the
%               optimization of the training problem successful.
%               Default: 1e-4.
%   options-file: a string specifying the name of a user-supplied options file.
%               If supplied, the file is read before the other optional BFO arguments
%               and therefore the options supplied in the file are superseded by those
%               which are (possibly) provided in the BFO call itself.  The syntax
%               of the file is as follows:
%               1) each line of the file is either blank or contains a comment, or a 
%                  keyword and a (possibly) incomplete value, or the completion of a 
%                  value started at the previous line(s);
%               2) comments start with the % character (any character from the 
%                  first % in a line is ignored) and blank lines are allowed;
%               3) incomplete lines are terminated with '...';
%               3) keyword are specified as unquoted strings;
%               4) values are specified as they would be in BFO call; in particular
%                    (i) strings must be quoted,
%                   (ii) values in lines of a numerical arrays must be separated by
%                        comas and lines by semi-colons,
%                  (iii) enries within a cell must be separated by comas.
%               5) the file may not specify the 'options-file' keyword (options
%                  files are not recursive).
%               NOTE: the first two arguments of BFO (without keywords) cannot be read
%                     from the options file.
%
%   INPUT (optional and more esoteric: don't change unless you understand).  
%          These are internal BFO parameters which can be optimized by training.
%
%
%   alpha     : the grid expansion factor at successful iterations (>= 1)
%               Default: 1.42
%   beta      : a fraction ( in (0,1) ) defining the shrinking ratio between
%               successive grids for the continuous variables
%               Default: 0.2
%   gamma     : the maximum factor ( >= 1 ) by which the initial user-supplied
%               grid for continuous variables may be expanded
%               Default: 2.36
%   eta       : a fraction ( > 0 ) defining the improvement in objective function
%               deemed sufficient to stop polling the remaining variables, this
%               decrease being computed as eta times the squared mesh-size.
%               Default: 1.e-4
%   zeta      : a factor (>=1) by which the grid size is expanded when a
%               particular level (in multilevel use) is re-explored after a
%               previous optimization.
%               Default: 1.5
%   iota      : a power at least equal to 1 to which the stepsize shrinking factor
%               is raised after unsuccessful coordinate partially separable search
%               Default: 1.2550
%   kappa     : the bracket expansion factor in min1d without quadratic interpolation
%   lambda    : the min bracket expansion factor in min1d with quadratic interpolation
%   mu        : the max bracket expansion factor in min1d with quadratic interpolation
%   inertia   : the number of iterations used for averaging the steps in the
%               continuous variables, the basis for these variables being
%               computed for the next iteration as an orthonormal basis whose
%               first element is the (normalized) average step.
%               NOTE: inertia = 0 disables the averaging process.
%               Default: 11
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
%   INPUT (reserved: definitely DON'T interfere) :
%
%
%   topmost   : true iff BFO is beeing called directly by the user, by opposition
%               to internal recursive calls or save-restart occurences.
%               NOTE: this is automatically specified during the algorithm 
%                     and should NOT be specified/altered by the user.
%   f-hist    : the vector of computed function values 
%               NOTE: this is automatically specified during the algorithm 
%                     and should NOT be specified/altered by the user.
%   sspace-hist: a struct containing a list of the best point found on
%               explored discrete or categorical subspaces, together with
%               associated function values and grid spacings
%               NOTE: This is automatically specified during the algorithm 
%                     and should NOT be specified/altered by the user.
%   element-hist: a cell array of struct containing the elementwise evaluation
%               history.
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
%   elset     : the indices of the elements present if the current objective
%               (ignored if the objective function is not in sum form).
%               NOTE: this is automatically specified during the algorithm 
%                     and should NOT be specified/altered by the user.
%
%%  8. Description of the OUTPUT parameters
%
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
%   msg       : a short informative message about the execution of BFO.  If BFO
%               is terminated because of an error, this message indicates the
%               type of error causing it.
%   wrn       : a cell containing characters strings, each of them (if any)
%               containing a short informative warning if appropriate.
%   neval     : the total number of objective function (full) evaluations required
%               by the algorithm
%   f_hist    : a vector containing the history of all function values computed 
%               in the solve phase (useful to make performance profiles).
%               NOTE: when a CPS objective function is specified, this includes
%                     evaluations at the end of each set of each group during 
%                     the poll loop. The evolution of the objective function value
%                     should therefore be rescaled to span a number of full function
%                     evaluations reported by the output parameter neval.
%   estcrit   : an estimated criticality measure (for the continuous variables)
%               at xbest.  For unconstrained problems, this is the infinity
%               norm of g, a central difference approximation of the gradient
%               on the last grid. For bound-constrained problems, this is the
%               infinity norm of the vector P(x-g)-x, where P is the orthogonal
%               projection on the feasible set.
%               NOTE: meaningless in 'train' training mode or if the problem
%                     has no continuous variable.
%   trained_parameters : a vector of 13 values corresponding to the optimized
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
%               trained_parameters(10) is the optimized value of iota
%               trained_parameters(11) is the optimized value of kappa
%               trained_parameters(12) is the optimized value of lambda
%               trained_parameters(13) is the optimized value of mu
%               This vector is empty on output of a call to BFO in 'solve' 
%               training mode.
%   training_history: a matrix containing the history of the training process.
%               Each row corresponds to an iteration of the underlying (average
%               robust, perfprofile or dataprofile) optimization.  The content
%               of each column is as follows:
%               column 1 : 1 if average, 2 if robust, 3 if perfprofile, 4 if data profile
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
%               column 14: the corresponding value of iota
%               column 15: the corresponding value of kappa
%               column 16: the corresponding value of lambda
%               column 17: the corresponding value of mu
%               This matrix is empty on output of a call to BFO in 'solve' 
%               training mode.
%               NOTE: trained_parameters = training_history( end, 5:14 )
%   s_hist    : a struct containing a list of the best point (in numerical or
%               vector-state form) found on explored discrete subspaces
%               NOTE: this is automatically specified during the algorithm 
%                     recursion. It is useless as an output for the user
%                     and may safely be ignored in the calling statement
%                     if no restart is desired (see examples below).
%   xincr     : the vector of (multilevel) grid spacings.
%               NOTE: this is automatically specified during the algorithm 
%                     recursion. It is useless as an output for the user.
%   idall     : the indeced of discrete variables across levels
%   opt_context: a struct containing the optimization context at xbest, that is:
%               opt.context.xtype : the variable's types
%               opt_context.xlower: the variable's lower bounds
%               opt_context.xupper: the variable's upper bounds
%               These quantities may differ from those specified by the user
%               at x0 if categorical variables with dynamical neighbourhoods are used.
%   el_hist   : a cell array of length equal at most l_hist, whose i-th entry is
%               a struct with fields
%               eldom: a vector containing the indices of the variables occuring
%                      in the i-th element function
%               xel  : a matrix/cell array of vector states,  whose columns/entries
%                      contain the last l_hist points at which the i-th element
%                      function has been evaluated
%               fel  : the  corresponding values of the i-th element function
%               fbest: the corresponding value (the -th term in the sum
%                      defining the current best objective function value).
%               NOTE: this is only meaningful (and different from {}) if the 
%                     objective function is specified in sum form.
%               NOTE: el_hist is not saved/restored at restarts: it therefore
%                     only covers evaluation since the beginning of the calculation
%                     or the last restart.
%   ev_hist   : For objective functions in sum form, a vector of length equal 
%               to that of f_hist whose i-th entry  ev_hist(i) gives the 
%               computational effort measured in complete objective evaluations 
%               which has been necessary for obtaining f_hist(i). It is empty [] 
%               when the objective function is not in sum form.

%%  9. Information

%   SOURCE:     a personal and possibly misguided reinterpretation of a talk 
%               given by a student of D. Orban and Ch. Audet on using the 
%               NOMAD algorithm for optimizing code parameters (Optimization 
%               Days, May 2009), plus some other ideas.
%
%   REFERENCES: M. Porcelli and Ph. L. Toint,
%               "BFO, a trainable derivative-free Brute Force Optimizer for 
%               nonlinear bound-constrained optimization and equilibrium 
%               computations with continuous and discrete variables",
%               ACM Transactions on Mathematical Software, 44:1 (2017), Article 6.
%
%               M. Porcelli and Ph. L. Toint,
%               "A note on using performance and data profiles for training algorithms, 
%               ACM Transactions on Mathematical Software, 45:2 (2019), Article 20.
%
%               M. Porcelli and Ph. L. Toint,
%               "Global and local information in structured derivative free
%               optimization with BFO",
%               arXiv:2001.04801, 2020.
%
%               C. Price and Ph.L. Toint,
%               "Exploiting problem structure in pattern-search methods for 
%               unconstrained optimization",
%               Optimization Methods and Software, vol. 21(3), pp. 479-491, 2006.
%
%               N. I. M. Gould, D. Orban and Ph. L. Toint,
%               "CUTEst: a Constrained and Unconstrained Testing Environment
%               with safe threads", Computational Optimization and Applications,
%               Volume 60, Issue 3, pp. 545-557, 2015.

%   PROGRAMMING: Ph. Toint, M. Porcelli, from May 2010 on.
%
%   DEPENDENCIES (internal): bfo_save, bfo_restore, bfo_print_summary_vector, 
%               bfo_print_vector, bfo_print_x, bfo_print_cell, bfo_print_matrix,
%               bfo_shistupd, bfo_ehistup, bfo_next_level_objf, bfo_average_perf,
%               bfo_robust_perf, bfo_save_training, bfo_restore_training; 
%               bfo_get_verbosity, bfo_exist_function, bfo_feasible_cstep, 
%               bfo_cutest_data, bfo_new_continuous_basis, bfo_cellify, bfo_numerify,
%               bfo_print_banner, bfo_build_neighbours, bfo_pack_x,
%               bfo_switch_context, bfo_verify_objf, bfo_verify_ps_structure,
%               bfo_verify_x0, bfo_verify_xlower, bfo_verify_xupper, bfo_verify_xscale,
%               bfo_verify_xtype, bfo_verify_cat_states, bfo_verify_cat_neighbours,
%               bfo_verify_lattice_basis, bfo_verify_maxmin, bfo_verify_xlevel,
%               bfo_verify_variable_bounds, bfo_verify_prob, bfo_sum_objf,
%               bfo_dpprofile_perf, bfo_read_options_file, bfo_search_step_cleanup,
%               bfo_core, bfo_min1d, bfo_default_algorithmic_parameters

%   DEPENDENCIES (external): f, variable_bounds (optional), search-step (optional),
%               cat_neighbours (optional), tracker (optional)

%%  10. Examples of use

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
%   17. Solve a fancy problem involving mixing milk-shakes of different styles, the
%      styles being represented by categorical variables:
%         [ x, fx ] = bfo( @milk_shake,                                          ...         
%                          {{ 'fruity', 'exotic', 0.5, 0.5, 0.1, 0.25, 0 }},     ...
%                          'xtype' , 'ssccccc',                                  ...
%                          'xlower', [ -Inf, -Inf, -2, -2, -2, -2, -2 ],         ...
%			   'xupper', [  Inf,  Inf,  2,  2,  2,  2,  2 ],         ...
%                          'cat-states', {{ {'fruity', 'mixed', 'veggy' },       ...
%                                           {'homely','exotic'},'', '', '', '', '' }} );
%      (see the milk_shake_*.m files for details).
%
%   18. Minimize the famous (coordinate partially-separable) Broyden tridiagonal function 
%      in 4 variables:
%      [ x, fx ] = bfo( {{ @broyden3d, @broyden3d, @broyden3d, @broyden3d }},    ...
%                       [ 0, -1, -1, -1, -1,  0 ], 'xtype', 'fccccf',            ...
%                       'eldom', {{ [ 1 2 3 ], [ 2 3 4 ], [ 3 4 5 ], [ 4 5 6 ] }} )
%      where the user has provided the following
%         function fx = broyden3d( i, x )
%         fx  = ( ( 3 - 2*x(2) )*x(2) - x(1) - 2*x(3) + 1 )^2;
%

%%  CONDITIONS OF USE

%   *** Use at your own risk! No guarantee of any kind given or implied. ***

%   Copyright (c) 2016, 2018, 2020 Ph. Toint and M. Porcelli. All rights reserved.
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
%   |                                DISCLAIMER                               |
%   |                                                                         |
%   |  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS    |
%   |  "AS IS" AND ANY EXPRESSED  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT   |
%   |  LIMITED TO,THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       |
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
wrn                = {};
neval              = 0;
f_hist             = [];
s_hist             = struct([]);
el_hist            = {};
ev_hist            = [];
estcrit            = NaN;
xincr              = 0;
trained_parameters = [];
training_history   = [];
opt_context        = struct([]);

%  Check that a potentially coherent argument list exists and output help 
%  or version number if requested.

noptargs = size( varargin, 2 );                  % the number of optional arguments
if ( noptargs == 0 )
   wrn{ 1 } = ' BFO use: [ xbest, fbest, ... ] = bfo( [ @f, x0 ], [ [''keyword'',value] ] )';
   disp( wrn{ end} );
   msg = ' BFO error: no argument specified. Terminating.';
   disp( msg );
   return
elseif ( noptargs == 1 )

   %  In the case where the first argument is the condensed problem, ignore it (for now).
   
   if ( isstruct( varargin{ 1 } ) &&                                                       ...
        isfield(  varargin{ 1 }, 'objf' ) && isfield( varargin{ 1 }, 'x0' ) )

%  Handle the other single-argument cases.
   
   else
      if ( ischar( varargin{1} ) && ismember( varargin{1}, { 'help', 'man', '-h', '?' } ) )
         wrn{ end+1 } = [ ' BFO use: [ xbest, fbest, ... ] =',                             ...
	                  ' bfo( [ @f, x0 ], [ [''keyword'',value] ] )' ];
         msg = wrn{ end };
      elseif ( ischar( varargin{ 1 } ) && strcmp( varargin{ 1 }, 'version' ) )
         msg = [ ' BFO version: ', this_version ];
         wrn{ end+1 } = msg;
      else 
         msg = ' BFO error: the argument list is ill-constructed. Terminating.';
         wrn{ end+1 } = msg;
      end
      disp( msg )
      return
   end
end

%  Various initializations
%  (i) verbosity linked

verbosity               = 'low';         % default verbosity
verbose                 = 2;
user_verbosity          = 0;
user_training_verbosity = 0;             %  a priori not a training verbosity

%  (ii) problem dependent:  problem dependent quantities which are derived from
%       the values of the prob struct fields have to be initialized here, in order
%       not to overwrite the values they are given when verifying the prob struct.

condensed_main_prob = 0;                 % problem form is extensive by default
n_elements          = 1;                 % single element objective function by default
eldom               = {};                % no CPS structure by default
elset               = [];                % the default list of elements for sum-form...
                                         % ... objectives, interpreted as [ 1:n_elements]
n                   = 1;                 % this allows the verification of xtype in train
                                         % mode (when no x0 is specified), as in
					 % case this vectors has a single entry
maximize            = 0;                 % minimize by default
max_or_min          = 'min';             % ... also in string form
ssfname             = '';                % no default for the search-step function
user_scl            = 0;                 % scales not specified by the user (yet)
user_xincr          = 0;                 % increments not specified (yet)
cat_dictionnary     = {};                % the initial set of categorical states is empty
multilevel          = 0;                 % no multilevel structure by default
sum_form            = 0;                 % objective not in sum form

%  (iii) preparing the possible specification of BFO options using an options file

options             = {};                % no options read from a file yet

%  (iv) others (needed in the prob struct verification)

train               = 0;                 % not in train mode (yet)
training            = 0;                 % no training data expected  for now
myinf               = 1.0e25;            % a numerical value for plus infinity
depth               = 0;                 % the recurrence depth
level               = 1;                 % single or first level by default

%   First check if an options file should be read.  If yes, read it.

for i = 1:noptargs
   if ( ischar( varargin{i} ) && strcmp( varargin{i}, 'options-file' ) )
      if ( i < noptargs )
         optfname = varargin{ i+1 };
         if ( ischar( optfname ) )
            if ( verbose >= 10 )
               disp( [ ' Reading options file ', optfname ] );
            end
            [ options, wrnr, optverb, training, train ] = bfo_read_options_file( optfname );
            if ( ~isempty( wrnr ) )
               for iw=1:length( wrnr )
                  wrn{ end+1 } = wrnr{ iw };
                  if ( verbose )
                     disp( wrnr{ iw } )
                  end
               end
            end
            if ( ~isempty( optverb ) )
               [ verbosity, verbose, user_verbosity, wrnv ] = bfo_handle_verbosity( optverb );
               if ( ~isempty( wrnv ) )
                  wrn{ end+1 } = wrnv;
               end
            end
         else
            wrn{ end+1 } =[ ' BFO warning: the name of the options file is not a string.',...
                         ' Ignoring it.' ];
            disp( wrn{ end } )
         end
      else
         msg = ' BFO error: the argument list is ill-constructed. Terminating.';
         wrn{ end+1 } = msg;
         disp( msg )
         return
      end
   end
end

%   Then read the explicit verbosity level and training mode, if any.  If
%   training-mode is 'train', then the first two arguments should not be
%   interpreted as the objective function and the starting point, even if the
%   objective function is given in string form.

for i = 1:noptargs
   if ( ischar( varargin{i} ) && strcmp( varargin{i}, 'verbosity' ) )
      if ( i < noptargs )
         [ verbosity, verbose, user_verbosity, wrnv ] =                                    ...
                                                   bfo_handle_verbosity( varargin{ i+1 } );
         if ( ~isempty( wrnv ) )
            wrn{ end+1 } = wrnv;
         end
      else
         msg = ' BFO error: the argument list is ill-constructed. Terminating.';
         wrn{ end+1 } = msg;
         disp( msg )
         return
      end
   end

   %  Set the flag (train) if training-mode specification = 'train' is found.

   if ( ischar( varargin{i} ) && strcmp( varargin{i}, 'training-mode' ) )
      training = 1;                     %  training data is expected
      if ( i < noptargs )
         if ( ischar( varargin{ i+1 } ) && strcmp( varargin{i+1}, 'train' ) )
            train = 1;
         end
      else
         msg = ' BFO error: the argument list is ill-constructed. Terminating.';
         wrn{ end+1 } = msg;
         disp( msg )
         return
      end
   end

end

%  Check the objective function handle and the starting point (if present).
%  The ( keyword, argument ) list starts beyond argument 1 if the first
%  two arguments specify the objective function (in struct or by objective
%  function handle or name) and (possibly) the starting point

if ( ischar( varargin{ 1 } ) )             % argument 1 is presumed to be either a string
                                           % specifying the objective function, or a keyword

   %  The ( keyword, argument ) list starts at argument 1.

   if ( train )  
      first_optional = 1;   % in 'train' training mode
   else
      first_optional = 3;   % in 'solve' or 'train-and-solve' mode
   end

elseif ( isstruct( varargin{ 1 } ) )       % argument 1 is presumed to be a prob struct
   condensed_main_prob = 1;

   if ( verbose >= 4 )
      disp( ' Verifying the optimization problem in condensed form; ' )
   end
   [ prob, n, shfname, user_scl, maximize, multilevel,                                     ...
           cat_dictionnary, msg, wrnf, not_in_eldom ] =                                    ...
         bfo_verify_prob( varargin{1}, 0, verbose, myinf, '', cat_dictionnary );
   f = varargin{ 1 }.objf;
   if ( ~isempty( wrnf ) )
      wrn{ end+1 } = wrnf;
   end
   if ( ~isempty( msg ) )
      return
   end
   if ( verbose >= 4 )
       disp( ' Done.' )
   end

   if ( noptargs > 1 )
      if (  isnumeric( varargin{ 2 } ) ) % argument 2 is presumed to specify x0
         first_optional = 3;             % prob struct and x0 occur before optional args
      elseif ( iscell( varargin{ 2 } ) ) % argument 2 is presumed to specify x0
         first_optional = 3;             % prob struct and x0 occur before optional args
      elseif ( ischar( varargin{ 2 } ) ) % argument 2 is presumed to be keyword
         first_optional = 2;             % only prob struct occurs before optional args
      else
         msg = ' BFO error: the first two arguments are misspecified. Terminating.';
         wrn{ end+1 } = msg;
         if ( verbose )
            disp( msg )
         end
         return
      end
   else
      first_optional    = 2;             % no optional argument
   end

elseif ( isa( varargin{ 1 }, 'function_handle' ) )  % argument 1 should be a function handle

   if( ~isnumeric( varargin{ 2 } ) && ~iscell( varargin{ 2 } ) )
      msg = ' BFO error: the first two arguments are misspecified. Terminating.';
      wrn{ end+1 } = msg;
      if ( verbose )
         disp( msg )
      end
      return
   end
   first_optional = 3;                   % the objf handle and x0 occur before optional args

elseif ( iscell( varargin{ 1 } ) )       % the objf is in sum form
   if ( ~iscell( varargin{ 1 }{ 1 } ) )  % not a value cell array
       msg = ' BFO error: the first argument is misspecified. Terminating.';
       wrn{ end+1 } = msg;
       if ( verbose )
          disp( msg )
       end
       return
   end
   varargin{ 1 } = varargin{ 1 }{ 1 };
   for i = 1:length( varargin{ 1 } )
      if ( ~isa( varargin{ 1 }{ i }, 'function_handle' ) )
         msg = ' BFO error: the first argument is misspecified. Terminating.';
         wrn{ end+1 } = msg;
         if ( verbose )
            disp( msg )
         end
         return
      end
   end
   sum_form   = 1;
   n_elements = length( varargin{ 1 } );
   
   if( ~isnumeric( varargin{ 2 } ) && ~iscell( varargin{ 2 } ) )
      msg = ' BFO error: the first two arguments are misspecified. Terminating.';
      wrn{ end+1 } = msg;
      if ( verbose )
         disp( msg )
      end
      return
   end
   first_optional = 3;                   % the objf handle and x0 occur before optional args

else
   msg = ' BFO error: the first two arguments are misspecified. Terminating.';
   wrn{ end+1 } = msg;
   if ( verbose )
      disp( msg )
   end
   return
end

%  If options were read in an options file, place them in front of the list
%  of optional arguments.

if ( ~isempty( options ) )
   varargin = {varargin{ 1:first_optional-1},options{ 1:end },varargin{ first_optional:end }};
   noptargs = length( varargin );
end

if ( verbose >= 10 )
   combined_arg_list = varargin
end

%  Compute the number of arguments in the ( keyword, argument ) list, and check
%  that it is even. Set a warning if it is odd.

n_optional = noptargs-first_optional+1;
if ( mod( n_optional, 2 ) > 0 )   
   if ( n_optional > 0 )
      noptargs = noptargs - 1;
      wrn{ end +1 }  = [ ' BFO warning: the number of variable optional arguments beyond', ...
               ' the objective function handle and the starting point must be even.',      ...
               ' Ignoring last argument.'];
      if ( verbose )
         disp( wrn{ end } )
      end
   end
end

%   Read the training requirements, if any, as well as the mandatory data for training.
%   Verify at the same time that every odd argument in the ( keyword, argument ) list is 
%   a string.
 
solve     = 1;                       % default training modes 
user_mode = 0;                       % training mode not specified by user (yet)

%   Training data is expected: check it out.

if ( training )
   train                      = 0;
   training_size              = 0;   % number of problems for training
   training_set_data_ok       = 0;
   training_set_cutest        = 0;   % default library of training problems
   training_problems_data     = {};
   train_probs_with_data      = [];  % the indices of the problems in training_problems
                                     % for which a problem_data toutine muus be checked
   condensed_train_probs      = 0;   % true if training problems are specified as structs

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
                  msg = ' BFO error: unknown training problems library. Terminating.';
                  if ( verbose )
                     disp( msg ) 
                  end
                  return
               end
            else
               msg = ' BFO error: unknown training problems library. Terminating.' ;
               if ( verbose )
                  disp( msg )
               end
	       return
            end
         end
      else
         msg = [' BFO error: argument ', int2str(2*i-1),' should be a keyword. Terminating.'];
         wrn{ end+1 } = msg;
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
               t_mode = varargin{ i+1 };
               switch ( t_mode )
               case 'solve'
                  user_mode     = 1;
               case 'train'
                  train         = 1; 
                  solve         = 0;
                  user_mode     = 1;
               case 'train-and-solve'
                  train         = 1;
                  user_mode     = 1;
               otherwise
                  wrn{ end+1 } = ' BFO error: unknown training mode. Default used.';
                  if ( verbose )
                     disp( wrn{ end } ) 
                  end            
               end
            else
               msg = ' BFO error: unknown training mode. Default used.' ;
               if ( verbose )
                  disp( msg )
               end
            end

         %  Next check the training problems.

         elseif ( strcmp( varargin{ i }, 'training-problems' ) ) % the training set
            if ( iscell( varargin{ i+1 } ) )
               tp                = varargin{ i+1 };
               training_size     = length( tp );
	       training_problems = {};
               if ( training_size > 0 )
	          itp = 0;
                  for tproblem = 1:training_size
                     tpt = tp{ tproblem };

                     %  For CUTEst problems, only the string form is acceptable for 
                     %  identifying the training problems.

                     if  ( training_set_cutest )
                        if ( ischar( tpt ) )        
                           itp = itp + 1;
                           training_problems{ itp } = tpt;
                        else
                           wrn{ end+1 } = [' BFO error: the ', int2str(tproblem),          ...
                               '-th argument of training-problems is not a string',        ...
                               ' (mandatory for CUTEst problems).',                        ...
			       ' Ignoring this training problem.'];
                           if ( verbose )                          
                              disp( wrn{ end } )
                           end         
                        end

                     %  For user-supplied problems, both strings and function handle forms
                     %  are acceptable. BFO remembers the function-handle form in this case.

                     else

                        %  Struct form

                        if ( isstruct( tpt ) )

                           %  Ok if this is the first problem, otherwise there is a 
		  	   %  (forbidden) mix of test problem formulations

                           if ( tproblem == 1 )
                              condensed_train_probs = 1;
			   elseif ( tproblem > 1 && condensed_train_probs == 0 )
                              msg = [ ' BFO error: forbidden mix of struct test problems', ...
				      ' and other types. Terminating. '];
                              if ( verbose )
                                 disp( msg )
			         return
                              end
		           end

                           %  Test coherence of a test problem in struct form

                           if ( verbose >= 4 )
                              disp( ' Verifying training problem ', int2str( tproblem),    ...
			            ' in condensed form: ' )
                           end
                           [ tpt, ~, ~, ~, ~, ~, ~, msg, wrnf ] =                          ...
                                  bfo_verify_prob( tpt, tproblem, verbose, myinf, '', {} );
                           if ( ~isempty( wrnf ) )
                              wrn{ end+1 } = wrnf;
                           end
                           if (  ~isempty( msg )  || ~isempty( wrnf ) )
                              if ( verbose )
			         disp( [' Training problem ', int2str( tproblem),          ...
                                        ' ignored.' ] )
                              end
			   
                           %  Build the training problem structure
			
                           else
                              itp = itp + 1;
                              training_problems{ itp } = tpt;
                           end
			
                        %  String form

                        elseif ( ischar( tpt ) )
		     
		           if ( condensed_train_probs )
                              msg = [ ' BFO error: forbidden mix of struct test problems', ...
			  	      ' and other types. Terminating. '];
                              if ( verbose )
                                 disp( msg )
                              end
                              return
			   end
                           if ( length( tpt ) > 3 && strcmp( tpt( 1:4 ), 'bfo_' ) )
                              msg = [ ' BFO error: the name of the training problem ',     ...
			              int2str( tproblem ), ' starts with ''bfo_''.',       ...
				      ' Terminating.' ];
                              if ( verbose )
                                 disp( msg )
                              end
                              return
                           end
                           if ( bfo_exist_function( tpt ) )
			      itp = itp + 1;
                              training_problems{ itp } = str2func( tpt );
                           else
                              wrn{ end+1 } = [ ' BFO error: m-file for training function ',...
			                       tpt, ' not found. Ignoring this training',  ...
					       ' problem.'];
                              if ( verbose )
                                 disp( wrn{ end } )
                              end
                           end

                           train_probs_with_data = [ train_probs_with_data, tproblem ];

                        %  Function handle form

                        elseif ( isa( tpt, 'function_handle' ) )

		           if ( condensed_train_probs )
                              msg = [ ' BFO error: forbidden mix of struct test problems', ...
				      ' and other types. Terminating. '];
                              if ( verbose )
                                 disp( msg )
                              end
                              return
			   end
                           tptname = func2str( tpt );
                           if ( length( tptname ) > 3 && strcmp( tptname( 1:4 ), 'bfo_' ) )
                              msg = [ ' BFO error: the name of the training problem ',     ...
			              int2str( tproblem ), ' starts with ''bfo_''.',       ...
				      ' Terminating.' ];
                              if ( verbose )
                                 disp( msg )
                              end
                              return
                           end
                           if ( bfo_exist_function( tptname ) )
			      itp = itp + 1;
                              training_problems{ itp } =  tpt;
                           else
                              wrn{ end+1 } = [ ' BFO error: m-file for training function ',...
			                    tptname , ' not found. Ignoring this training',...
					    ' problem. '];
                              if ( verbose )
                                 disp( wrn{ end } )
                              end
                              return
                           end
                           train_probs_with_data( end+1 ) =  itp;
                        else
                           msg = [' BFO error: the ', int2str(tproblem),                   ...
                                  '-th argument of training-problems is not a function',   ...
                                  ' handle nor a string. Ignoring.'];
                           if ( verbose )
                              disp( msg )
                           end
                        end
                     end
                  end
               end
            else
               msg = ' BFO error: training-problems is not a cell.';
               if ( verbose )
                 disp( msg )
               end
               return
            end

         %  Finally check the training problems data.

         elseif ( strcmp( varargin{ i }, 'training-problems-data' ) )

            %  Ignore if the training library is CUTEst.

            if ( training_set_cutest )
               wrn{ end+1 } = [ ' BFO warning: training-problems-data supplied for',       ...
	                        ' CUTEst problems. Ignoring.' ];
               if ( verbose )
                  disp( wrn{ end } )
               end

            %  Also ignore if training problems are given in condensed form.

           elseif( condensed_train_probs )
	
               wrn{ end+1 } = [ ' BFO warning: training-problems-data supplied for',       ...
	                        ' training problems in condensed form. Ignoring.' ];
               if ( verbose )
                  disp( wrn{ end } )
               end

            %  For user-supplied problems, both strings and function handle forms
            %  are acceptable. BFO remembers the function-handle form in this case.

            else
	 
	       if ( ~isempty( train_probs_with_data ) )   % check there are problems to verify

                  if ( iscell( varargin{ i+1 } ) )
                     tpd                = varargin{ i+1 };
                     training_data_size = length( tpd );  % the training data
                     verif_data         = [];             % problems whose training data ok
                     for tproblem = 1:training_data_size
                        tpdt = tpd{ tproblem };
                        if ( any( train_probs_with_data == tproblem ) )

                           %  String form

                           if ( ischar( tpdt ) )
                              if ( bfo_exist_function( tpdt ) )
                                   training_problems_data{ tproblem } = str2func( tpdt );
                              else
                                 msg = [ ' BFO error: m-file for training function data ', ...
			                 tpdt, ' not found. Terminating. '];
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
                                 msg = [ ' BFO error: m-file for training function data ', ...
                                         tpdtname, ' not found. Terminating. '];
                                 if ( verbose )
                                    disp( msg )
                                 end
                                 return
                              end
                           else
                              msg = [ ' BFO error: the ', int2str( tproblem ),             ...
			              '-th argument of training-problems-data is not a',   ...
			      	      ' function handle nor a string.'];
                              if ( verbose )
                                 disp( msg )
                              end
                              break
                           end
		        end
		     verif_data = [ verif_data tproblem ];
                     end
 	             train_probs_with_data( verif_data ) = [];
                     training_set_data_ok = ( isempty( train_probs_with_data )  )    ||    ...
		                            ( length( training_problems_data ) == 1 );
                  else
                     msg = ' BFO error: training-problems-data is not a cell.';
                     if ( verbose )
                        disp( msg )
                     end
                     return
	          end
               else
	          training_set_data_ok = 1;  %  Ok when no problem to verify
               end  
            end
         end
      end
   end
end

%  Check the presence of the objective function when solving is required. 
%  If the objective function is found, compute its stripped name and verify 
%  that the relevant m-file exists. Note that these verifications are only necessary
%  if the problem is in extensive form, because they have been performed already for
%  the condensed form.

if ( solve )

   if ( ~condensed_main_prob )
      [ f, ~, shfname, msg ] = bfo_verify_objf( varargin{ 1 }, verbose );
      if ( ~isempty( msg ) )
         return
      end
   end

   %  Check the presence and form of the starting point, and define
   %  the problem's dimension.  This occurs in the extensive form (in which case the
   %  first two arguments are the objective function handle and x0) or in the
   %  condensed form whenever x0 is specified after prob (argument 1).  In both
   %  cases, the first-optional argument is in position 3. 

   if ( first_optional == 3 && noptargs > 1 )

      if ( iscell( varargin{ 2 } ) )
         [ x0, n, msg ] = bfo_verify_x0( varargin{ 2 }{ 1 }, verbose );
      else
         [ x0, n, msg ] = bfo_verify_x0( varargin{ 2 }, verbose );
      end

      if ( ~isempty( msg ) )
         return
      end
      if ( condensed_main_prob )
         prob.x0 = x0;            %  Overwrite the struct x0 if specified explicitly.
      end
   end
else
   x0 = [];
end

%   Miscellaneous initializations (which can be delayed after the prob_struct verification):
%   (i) general

rpath        = [];                         % the list of variables fixed in the recurrence
estcrit      = -1;                         % a meaningless estimate of criticality
cur          = 1;                          % column index for current increments in xincr
ini          = 1;                          % column index for initial increments in xincr
                                           % (for now)
nevalt       = 0;                          % the accumulated nbr of evaluations during
                                           % ... training (optim. internal perf. functions)
%   (ii) reset by restart

maxeval      = 5000;                       % the max number of objective function evaluations
user_maxeval = 0;                          % maxeval not specified by the user
latbasis     = [];                         % empty lattice basis by default

%   (iii) specific to multilevel

nlevel       = 1;                          % only one level by default
max_nlevel   = 6;                          % the maximum number of levels
msglow       = '';                         % empty message from next level
idall        = [];                         % the list of discrete variables across levels

%   (iv) specific to restart

savef        = -1;                         % no restart information saving
sfname       = 'bfo.restart';              % the name of the file where restart ...
                                           % ... information is saved (when requested)

%   (v) flags to indicate if algorithmic parameters are user-specified or inherited from
%        previous computations

user_alpha       = 0;
user_beta        = 0;
user_gamma       = 0;
user_delta       = 0;
user_eta         = 0;
user_zeta        = 0;
user_inertia     = 0;
user_searchtype  = 0;
user_rseed       = 0;
user_iota        = 0;
user_kappa       = 0;
user_lambda      = 0;
user_mu          = 0;
user_cps         = 0;

%   (vi) default values for the profile training strategy.  
%        They need to be specified for all cases because they are assumed 
%        to have a value when calling the saving function.

fstar    = [];
tpval    = {};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Restart, if requested  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Look for a restart argument in the calling sequence or for the name of a
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
               wrn{ end+1 } = ' BFO warning: unknown restart mode. Ignoring.';
               if ( verbose )
                  disp( wrn{ end } )
               end
            end
         else
            wrn{ end+1 } = ' BFO warning: unknown restart mode. Ignoring.';
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      elseif ( strcmp( varargin{ i }, 'restart-file' ) )
         if ( ischar( varargin{ i+1 } ) )
            sfname = varargin{ i+1 };
         else
            wrn{ end+1 } = ' BFO warning: wrong name for restart-file. Ignoring.';
            if ( verbose )
               disp( wrn{ end } )
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

      overbose = verbose;
      [ rshfname, maximize, epsilon, ftarget, maxeval, neval, f_hist, xtype, xtry,         ...
        xscale, xlower, xupper, verbose, alpha, beta, gamma, eta, zeta, inertia,           ...
        stype, rseed, iota, kappa, lambda, mu, term_basis, ~, s_hist, latbasis,            ...
        bfgs_finish, training_history, fstar, tpval, ssfname, ~, ~, cat_dictionnary,       ...
        restok ] = bfo_restore( sfname, readall );

      topmost = 0;                   %  A restart is not a direct call.

      if ( user_verbosity  || readall )
         verbose = overbose;
      end
      if ( restok == 0 )
         msg     = [' BFO error: could not open ', sfname, '. Terminating.'];
         restart = 0;
      elseif ( restok == 1 )
         if ( ~strcmp( rshfname, shfname ) && ~readall )
            msg     = [ ' BFO error: attempt to restart with a different objective.',      ...
                        ' Terminating.' ];
            restart = 0;
         elseif ( length( xlower ) ~= n )
            msg     = ' BFO error: wrong restart file. Terminating.';
            restart = 0;
         end
         user_ftarg = 1;                     % objective target not specified by the user
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
      overbose           = verbose;
      training_save_file = [ sfname, '.training'];
      [ p0, verbose, training_strategy, training_parameters, training_problems,            ...
        training_set_cutest, trained_bfo_parameters, training_epsilon, training_maxeval,   ...
	training_verbosity, training_problem_epsilon, training_problem_maxeval,            ...
	training_problem_verbosity, trestok ]  = bfo_restore_training( training_save_file );
      if ( user_verbosity )
        verbose = overbose;
      end
      if ( ~trestok )
         msg = [ ' BFO error: cannot open file ', training_save_file, '. Terminating.' ];
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
      iota             = p0( 10 );
      kappa            = p0( 11 );
      lambda           = p0( 12 );
      mu               = p0( 13 );

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
         user_alpha       = 1;
         user_beta        = 1;
         user_gamma       = 1;
         user_eta         = 1;
         user_zeta        = 1;
         user_inertia     = 1;
         user_searchtype  = 1;
         user_rseed       = 1;
         user_iota        = 1;
         user_kappa       = 1;
         user_lambda      = 1;
         user_mu          = 1;
      end
   end

%  No restart at all: the training data must still be verified for coherence.

else
   restart_training         = 'none';
   not_restarting_training  = 1;

   %  If training is required, check that the size of the training problems set is the same 
   %  as that of the training problem data (for a user-supplied library) or
   %  that there is only one data function (which is then to be used for all
   %  problems). If not, return unless a further "solve" phase is required.
   %  In this case, training is skipped.

   if ( train && training_set_data_ok && ~training_set_cutest )
      if ( ~training_set_data_ok && training_data_size ~= 1 )
         if ( solve )
            train = 0;
            wrn{ end+1 } = [ ' BFO error: training required but training-problems and',    ...
                             ' training-problems-data are incompatible. Aborting training.'];
            if ( verbose )
               disp( wrn{ end } )
            end
         elseif ( train )
            msg = [ ' BFO error: training required but training-problems and',             ...
                     ' training-problems-data are incompatible. Terminating.'];
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

      if ( training_size == 0 ) 
         if ( solve )
            train = 0;
            msg   = [ ' BFO error: training required but training-problems unspecified.',  ...
                      ' Aborting training.'];
            if ( verbose )
               disp( msg )
            end
         else
            msg = [ ' BFO error: training required but training-problems unspecified.',    ...
                    ' Terminating.' ];
            if ( verbose )
               disp( msg )
            end
            return
         end
      end

      %  Check that the complete data for training is available for a user-supplied library.
      %  and transform the training problems to condensed form.

      if ( ~condensed_train_probs &&  ~training_set_cutest )

         %  Verify the presence of complete data.
	 
         if ( ~training_set_data_ok )
            if ( solve )
               train = 0;
               msg   = [' BFO error: training required but training-problems-data',        ...
                        ' unspecified or incompatible.  Aborting training.' ];
               if ( verbose )
                  disp( msg )
               end
            else
               msg = [ ' BFO error: training required but training-problems-data',         ...
                       ' unspecified or incompatible.  Terminating.' ];
               if ( verbose )
                  disp( msg )
               end
               return
            end
         end

         %  Transform the user-supplied training problems in condensed form.

         condensed_training_problems = {};

         for i = 1:length( training_problems )
	    

            %  The problems are available in the old extensive format.
	    %  Note that the use of lattice-based and/or categorical variables
	    %  is not allowed in this formulation.

            tpi = struct( 'objf', training_problems{ i } );
            if ( length( training_problems_data ) == 1 )
               fun_data = training_problems_data{ 1 };
            else
               fun_data = training_problems_data{ i };
            end

            %  Read the data associated with the current problem, depending on how
            %  much information is provided by the user-supplied data function.

            nout = nargout( fun_data ); % the number of outputs of associated data function

            switch ( nout )
            case 1
               tpx0       = feval( fun_data );
               tpi.x0     = tpx0;
            case 3
               [ tpx0, tpxlower, tpxupper ] = feval( fun_data );
               tpi.x0     = tpx0;
               tpi.xlower = tpxlower;
               tpi.xupper = tpxupper;
            case 4
               [ tpx0, tpxlower, tpxupper, tpxtype ] = feval( fun_data );
               tpi.x0     = tpx0;
               tpi.xlower = tpxlower;
               tpi.xupper = tpxupper;
               tpi.xtype  = tpxtype;
            case 5
               [ tpx0, tpxlower, tpxupper, tpxtype, tpxscale ] = feval( fun_data );
               tpi.x0     = tpx0;
               tpi.xlower = tpxlower;
               tpi.xupper = tpxupper;
               tpi.xtype  = tpxtype;
               tpi.xscale = tpxscale;
            case 6
               [ tpx0, tpxlower, tpxupper, tpxtype, tpxscale, tpmax_or_min ] =             ...
	       feval( fun_data );
               tpi.x0     = tpx0;
               tpi.xlower = tpxlower;
               tpi.xupper = tpxupper;
               tpi.xtype  = tpxtype;
               tpi.xscale = tpxscale;
               tpi.max_or_min = tpmax_or_min;
            case 7
               [ tpx0, tpxlower, tpxupper, tpxtype, tpxscale, tpmax_or_min, tpxlevel ] =   ...
	               feval( fun_data );
               tpi.x0     = tpx0;
               tpi.xlower = tpxlower;
               tpi.xupper = tpxupper;
               tpi.xtype  = tpxtype;
               tpi.xscale = tpxscale;
               tpi.max_or_min = tpmax_or_min;
               tpi.xlevel = tpxlevel;
            case 8
               [ tpx0, tpxlower, tpxupper, tpxtype, tpxscale, tpmax_or_min, tpxlevel,      ...
	               tpvb_name ] = feval( fun_data );
               tpi.x0     = tpx0;
               tpi.xlower = tpxlower;
               tpi.xupper = tpxupper;
               tpi.xtype  = tpxtype;
               tpi.xscale = tpxscale;
               tpi.max_or_min = tpmax_or_min;
               tpi.xlevel = tpxlevel;
               tpi.variable_bounds = tpvb_name;
            otherwise
                disp( [ ' BFO warning: error in output sequence of the ', int2str( i ),    ...
                        '-th training_problem_data function. Results unreliable.'] );
            end

            %  Verify the newly condensed training problem.

            if ( verbose >= 4 )
               disp( [ ' Verifying training problem ', int2str( tproblem),                 ...
	               ' in condensed form.'] )
            end
	    [ tpi, ~, ~, ~, ~, ~, ~, msg, wrnf ] =                                         ...
                            bfo_verify_prob( tpi, i, verbose, 1.0e25, '', {} );
            if ( ~isempty( wrnf ) )
               wrn{ end+1 } = wrnf;
            end
            if ( ~isempty( msg ) || ~isempty( wrnf ) )
               if ( verbose )
	          disp( ' Training problem ', int2str( tproblem), ' ignored.' )
               end
            else
	       condensed_training_problems{ i } = tpi;
            end
         end
         training_problems      = condensed_training_problems;
	 training_problems_data = {};
         condensed_train_probs  = 1;
      end
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Set default parameters before optional arguments are allowed to modify them.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( ~restart  || ~readall )
   
   %  Default training parameters (skipped because too late if during restart).

   use_trained = 0;                               % no use of trained parameters by default 

   if ( train  )
     training_strategy   = 'average';             % training strategy based on the average
                                                  % ... number of function evaluations
     training_epsilon    = [ 1e-2 1e-2 ];         % accuracy for the training phase
     training_maxeval    = [ 200 100 ];           % maximum number of training evals
     training_verbosity  = { 'silent', 'silent' };% no output from the parameters optimization
     nbr_train_params    = 6;                     % a priori choice of training parameters
     training_problem_epsilon   = 1e-4;           % accuracy for each training problem
     training_problem_maxeval   = 5000;           % max evals for each training problem
     training_problem_verbosity = 'silent';       % training each problem is silent
     training_parameters    = { 'alpha', 'beta', 'gamma', 'delta', 'eta', 'inertia' };
     trained_bfo_parameters = 'trained.bfo.parameters';  % default file name for ...
                                                  % ... trained parameters
     xtype = 'c';                                 % training for continuous problems by def.
     training_perfprofile_window = [1,20];        % default window for perfprofile training
     training_dataprofile_window = [0,2000];      % default window for dataprofile training
     user_training_profile_window     = 0;        % no user specified profile training window
     training_profile_cutoff_fraction = 1e-4;     % default cutoff for profile training
   end

   %  The general (non trainable) BFO default values

   if ( solve  )
      xlower          = -myinf * ones(n,1);   % no lower bound by default
      xupper          =  myinf * ones(n,1);   % no upper bound by default
      xtype(1:n)      = 'c';                  % the default variable type is continuous
      epsilon         = 1e-4;                 % the default accuracy requ. in the solve phase
      bfgs_finish     = 0;                    % no BFGS finish requested
      ftarget         = -0.99999999 * myinf;  % the target objective function value
      user_ftarg      = 0;                    % objective target not specified by the user
      term_basis      = 5;                    % the number of random basis used for ...
                                              % ... assessing termination
      user_tbasis     = 0;                    % true if the user specifies term_basis 
      ps_verbosity    = 'silent';             % the verbosity for ps subset optimization...
      ps_verbose      = 0;                    % ... also as an integer
  end
end
fxok              = 0;                        % no supplied f(x0) by default
fx0               = [];
use_tracker       = 0;                        % no user tracking function by default
max_npoll         = 50;                       % the maximum number of continuous polling
                                              % directions (for now somewhat arbitrary)
npoll             = max_npoll;                % the default
vb_name           = '';                       % empty variable bound function by default
cn_name           = '';                       % no default categorical neighboorhood function
fcallt            = 'simple';                 % the default function call type
withbound         = 0;                        % by default, the call for f does not use fbound
fbound            = myinf;                    % default for minimization
user_fbound       = 0;                        % the unsuccessful f bound is not user specified
l_hist            = 1;                        % the length of the memorized evaluation history
cat_states        = {};                       % the static neigbouhoods for categorical vars
num_cat_states    = {};                       % idem (numeric version)
user_fhist        = 0;                        % no user-supplied global history
user_elhist       = 0;                        % no user-supplied element history
use_cps           = 1;                        % by default, exploit CPS structure if present
topmost           = 1;                        % unless otherwise specified by the user, 
                                              % this is the topmost call to BFO
allow_core        = 0;                        % allows the use of bfo_core for solving the
                                              % cps subspace optimization
allow_min1d       = 0;                        % allows the use of bfo_min1d for solving the
                                              % cps subspace optimization

%  Default random setting

%%%%%  For normal runs using randomness, uncomment the next line.
%reset_random_seed = 'no-reset';
%%%%   For testing purposes (e.g. for running test_bfo), uncomment the next line.
reset_random_seed = 'reset';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%  Unpack the condensed form of the problem  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  If the problem is in condensed form, unpack it to allow individual modification to
%  its fields. This possibly overwrites the default values of variables directly
%  corresponding to the prob struct fields, but NOT the values derived from those fields
%  (n, f, objname, shfname, user_scl, maximize, multilevel, cat_dictionnary): these
%  have been unpacked when verifying the prob struct above). 

if ( condensed_main_prob )
   if ( iscell( prob.objf ) )
      if ( isfield( prob, 'eldom' ) )
         eldom   = prob.eldom;
      else
         eldom   = {};
      end
      f          = prob.objf;
      sum_form   = 1;
      n_elements = length( f );
   end
   x0 = prob.x0;
   if ( isfield( prob, 'xlower' ) )
      xlower = prob.xlower;
   end
   if ( isfield( prob, 'xupper' ) )
      xupper = prob.xupper;
   end
   if ( isfield( prob, 'xtype' ) )
      xtype = prob.xtype;
   end
   if ( isfield( prob, 'xscale' ) )
      xscale = prob.xscale;
   end
   if ( isfield( prob, 'max_or_min' ) )
      max_or_min = prob.max_or_min;
   end
   if ( isfield( prob, 'xlevel' ) )
      xlevel = prob.xlevel;
   end

   if ( isfield( prob, 'variable_bounds' ) )
      vb_name = prob.variable_bounds;
   end
   if ( isfield( prob, 'lattice_basis' ) )
      latbasis = prob.lattice_basis;
   end
   if ( isfield( prob, 'cat_states' ) )
      cat_states = prob.cat_states;
   end
   if ( isfield( prob, 'cat_neighbours' ) )
      cn_name = prob.cat_neighbours;
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Process the (remaining) variable argument list  %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Note that, when the condensed form is used, the problem's characteristics (xlower,
%  xupper, xscale, xtype, max-or-min, xlevel, variable-bounds, lattice_basis, cat-states
%  and cat-neighbours) specified individually in the calling sequence overwrite those
%  passed in the prob struct.

for i = first_optional:2:noptargs-1

   %  Check that varargin{ i } is a keyword (already done above if training data is read).

   if ( ~ischar( varargin{ i } ) )
      msg = [' BFO error: argument ', int2str( i ),' should be a keyword. Terminating.'];
      wrn{ end+1 } = msg;
      if ( verbose )
         disp( msg )
      end
      return
   end

   switch ( varargin{ i } )

%  The indicator of topmost entry

   case 'topmost'

      if ( ischar( varargin{ i+1 } ) )
         switch( varargin{ i+1 } )
         case 'yes'
            topmost = 1;
         case 'no'
            topmost = 0;
         otherwise
            msg = ' BFO internal error: unrecognized value of topmost.  Terminating.';
            disp( ' BFO internal error: unrecognized value of topmost: ' )
            varargin{ i+1 }
            disp( ' Terminating.' );
            return
         end
      else
         msg = ' BFO internal error: unrecognized value of topmost.  Terminating.';
         disp( ' BFO internal error: unrecognized value of topmost: ' )
         varargin{ i+1 }
         disp( ' Terminating.' );
         return
      end

%  The lower bounds

   case 'xlower'

      if ( solve  ) 
         [ xlower, wrnf ] = bfo_verify_xlower( varargin{ i+1 }, n, verbose, myinf );
         if ( ~isempty( wrnf ) )
            wrn{ end+1 } = wrnf;
         end
      end

%  The upper bounds

   case 'xupper'

      if ( solve )
         [ xupper, wrnf ] = bfo_verify_xupper( varargin{ i+1 }, n, verbose, myinf );
         if ( ~isempty( wrnf ) )
            wrn{ end+1 } = wrnf;
         end
      end

   %  The element domains for a sum-form objective function

   case 'eldom'

      if ( iscell(  varargin{ i+1 } ) )
         eldom = varargin{ i+1 }{ 1 };
         if ( ~iscell( eldom ) )  %  not a value cell array
            msg = ' BFO error: wrong type of input for parameter eldom. Terminating.' ;
            if ( verbose )
               disp( msg )
            end
            return
         end
         if ( ~isempty( eldom ) )                   % skip default setting eldom = {{}}
            [ eldom, msg, not_in_eldom ] = bfo_verify_eldom( eldom, n, length( f ),verbose ); 
            if ( ~isempty( msg ) )
               return
            end
         end
      else 
         msg = ' BFO error: wrong type of input for parameter eldom. Terminating.' ;
         if ( verbose )
            disp( msg )
         end
         return
      end
       
   %  The exploitation of coordinate-partially-separable structure

   case 'use-cps'

      if ( isnumeric( varargin{ i+1 } ) )
         use_cps = varargin{ i+1 };
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter use-cps.',    ...
	                  ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The solver option for coordinate-partially-separable subset optimization

   case 'cps-subspace-optimizer'

      if ( ischar( varargin{ i+1 } ) )
         switch ( varargin{ i+1 } )
         case 'full'
            allow_core  = 0;
            allow_min1d = 0;
            user_cps    = 1;
         case 'core'
            allow_core  = 1;
            allow_min1d = 0;
            user_cps    = 1;
         case 'min1d'
            allow_core  = 1;
            allow_min1d = 1;
            user_cps    = 1;
         otherwise
            wrn{ end+1 } = ' BFO warning: unknown cps-subspace-optimizer. Default used.';
            if ( verbose )
               disp( wrn{ end } )
            end

         end
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter',               ...
                          ' cps-subspace-optimizer. Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The verbosity for coordinate-partially-separable subset optimization

   case 'ps-subset-verbosity'

      if ( ischar( varargin{ i+1 } ) )
         ps_verbose = bfo_get_verbosity( varargin{ i+1 } );
         if ( ps_verbose >= 0 )
            ps_verbosity = varargin{ i+1 };
         else
            wrn{ end+1 } = ' BFO warning: unknown ps-subset-verbosity level. Default used.';
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter',               ...
                          ' ps-subset-verbosity. Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end
      
   %  The list of relevant elements

   case 'elset'

      if ( isnumeric( varargin{ i+1 } ) )
         elset = varargin{ i+1 };
      else
         msg   = ' BFO error: wrong type of input for parameter elset. Teminating.';
         if ( verbose )
            disp( msg )
         end
         return
      end

   %  The expansion shrinking factor

   case 'alpha'

      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            alpha      = max( min( 1.e8, abs( varargin{ i+1 } ) ), 1 );
            user_alpha = 1;
         else
            wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter alpha.',     ...
	                     ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The grid shrinking factor

   case 'beta'

      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            beta      = min( max( 1.e-8, abs( varargin{ i+1 } ) ), 0.999999 );
            user_beta = 1;
         else
            wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter beta.',      ...
	                     ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The coordinate-partially separable shrinking factor exponent

   case 'iota'

      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            iota      = min( max( 1, abs( varargin{ i+1 } ) ), 3 );
            user_iota = 1;
         else
            wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter iota.',      ...
	                     ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The bracket expansion factor in bfo_min1d (no quadratic interpolation)

   case 'kappa' 

      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            kappa      = min( max( 1, abs( varargin{ i+1 } ) ), 5 );
            user_kappa = 1;
         else
            wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter kappa.',     ...
	                     ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The minimum bracket expansion factor in bfo_min1d (quadratic interpolation)

   case 'lambda'

      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            lambda      = min( max( 0.1, abs( varargin{ i+1 } ) ), 0.9 );
            user_lambda = 1;
         else
            wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter lambda.',    ...
	                     ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The maximum bracket expansion factor in bfo_min1d (quadratic interpolation)

   case 'mu'

      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            mu      = min( max( 1, abs( varargin{ i+1 } ) ), 100 );
            user_mu = 1;
         else
            wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter mu.',        ...
	                     ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The maximum grid expansion factor

   case 'gamma'

      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            gamma      = max( 1, abs( varargin{ i+1 } ) );
            user_gamma = 1;
         else
            msg = ' BFO error: wrong type of input for parameter gamma. Default used.';
            disp( msg )
         end
      end

   %  The initial increment factor

   case 'delta'

      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            delta      = abs( varargin{ i+1 } );
            user_delta = 1;
            [ s1, s2 ] = size( delta );
            if ( s1 == 1 && s2 == 1 )
            elseif ( min( s1, s2 ) ~= 1 )
               wrn{ end+1 } = [ ' BFO error: wrong size of input for parameter delta.',    ...
	                        ' Default used.' ];
               if ( verbose )
                  disp( wrn{ end } )
               end
               user_delta = 0;
            elseif ( ( max( s1, s2 ) ~= n ) && solve )
               wrn{ end+1 } = [ ' BFO error: wrong size of input for parameter delta.',    ...
	                        ' Default used.' ];
               if ( verbose )
                  disp( wrn{ end } )
               end
               user_delta = 0;
            end
         else
            msg = ' BFO error: wrong type of input for parameter delta. Default used.';
            disp( msg )
         end
      end

   %  The sufficient decrease fraction

   case 'eta'

      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            eta      = max( 1.e-4, abs( varargin{ i+1 } ) );
            user_eta = 1;
         else
            wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter eta.',       ...
	                     ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The multilevel grid expansion factor for re-exploration of a previously
   %  visited level.

   case 'zeta'

      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            zeta      = max( 1, abs( varargin{ i+1 } ) );
            user_zeta = 1;
         else
            wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter zeta.',      ...
	                     ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The inertia for continuous step accumulation

   case 'inertia'

      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            inertia      = abs( round( varargin{ i+1 } ) );
            user_inertia = 1;
         else
            wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter inertia.',   ...
	                     ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The discrete tree search strategy

   case 'search-type'

      if ( not_restarting_training )
         if ( ischar( varargin{ i+1 } ) )
            searchtype = varargin{ i+1 };
            switch( searchtype )
            case 'breadth-first'
               stype           = 0;
               user_searchtype = 1;
            case 'depth-first'
               stype           = 1;
               user_searchtype = 1;
            case 'none'
               stype           = 2;
               user_searchtype = 1;
            otherwise
               wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter',         ...
	                        ' search-type. Default used.' ];
               if ( verbose )
                  disp( wrn{ end } )
               end
            end
         else
            wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter',            ...
	                     ' search-type. Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The random number generator's seed

   case 'random-seed'

      if ( not_restarting_training )
         if ( isnumeric( varargin{ i+1 } ) )
            rseed      = round( abs( varargin{ i+1 } ) );
            user_rseed = 1;
         else
            wrn{ end+1 }=[ ' BFO warning: wrong type of input for parameter random-seed.', ...
                           ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The variable scaling

   case 'xscale'

      [ xscale, user_scl, wrnf ] = bfo_verify_xscale( varargin{ i+1 }, n, solve, verbose );
      if ( ~isempty( wrnf ) )
         wrn{ end+1 } = wrnf;
      end

   %  The value of the objective function at the starting point

   case 'fx0'

      fx0  = varargin{ i+1 };
      if ( ~isnumeric( fx0 ) )
         wrn{ end+1 } = ' BFO warning: wrong initial objective value(s). Ignoring.'
	 if ( verbose )
	    disp( wrn{ end+1 } )
	 end
      elseif ( abs( sum( fx0 ) ) >= Inf )
         wrn{ end+1 } = ' BFO warning: infinite initial objective value(s). Ignoring.'
	 if ( verbose )
	    disp( wrn{ end+1 } )
	 end
      else
         [ nlfx0, ncfx0 ] = size( fx0 );
	 if ( min( nlfx0, ncfx0 ) ~= 1 )
            wrn{ end+1 } = ' BFO warning: wrong initial objective value(s). Ignoring.'
	    if ( verbose )
	       disp( wrn{ end+1 } )
	    end
	 elseif ( nlfx0 > 1  || ncfx0 > 1 )
	    if ( nlfx0 > ncfx0 )
	       fx0 = fx0';
	    end
            fxok = 1;
	 else
            fxok = 1;
	 end
      end
      
   %  The multilevel grid spacing (should only occur when entering levels > 1 )

   case 'xincr'

      if ( isnumeric( varargin{ i+1 } ) )
         xincr      = varargin{ i+1 };
	 user_xincr = 1;
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter xincr.',        ...
	                  ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The variable types

   case 'xtype'

      [ xtype, msgf, wrnf ] = bfo_verify_xtype( varargin{ i+1 }, n, solve, verbose );
      if ( ~isempty( msgf ) )
         msg = msgf;
         return
      end
      if ( ~isempty( wrnf ) )
          wrn{ end+1 } = wrnf;
       end

   %  The categorical states
  
   case 'cat-states'

      cst = varargin{ i+1 };
      if ( ~isempty( cst ) )
         [ cat_states, cat_dictionnary, msg ] =                                            ...
             bfo_verify_cat_states( cst{ 1 }, n, xtype, verbose, cat_dictionnary );
         if ( ~isempty( msg ) )
            return
         end
      end
      
   %  The user function defining the categorical neighbourhood
  
   case 'cat-neighbours'

      [ cn_name, msg ] = bfo_verify_cat_neighbours( varargin{ i+1 }, verbose );
      if ( ~isempty( msg ) )
         return
      end

   %  The lattice basis for integer variables

   case 'lattice-basis'

      latbasis = varargin{ i+1 };

%  The maximum number of objective function's evaluations

   case 'maxeval'

      if ( isnumeric( varargin{ i+1 } ) )
         maxeval     = abs( round( varargin{ i+1 } ) );
         user_maxeval = 1;
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter maxeval.',      ...
	                  ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The (maximum) number of polling directions

   case 'npoll'

      arg = varargin{ i+1 };
      if ( isnumeric( arg ) && length( arg ) == 1 )
         npoll = round( abs( arg ) );
      else
         wrn{ end+1 } = [ ' BFO warning: wrong size of input for parameter npoll.',        ...
                          ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
         npoll = max_npoll;
      end

   %  The requested accuracy of the mesh size

   case 'epsilon'

      if ( isnumeric( varargin{ i+1 } ) )
         epsilon    = abs( varargin{ i+1 } );
         [ s1, s2 ] = size( epsilon );
         if ( s1 ~= 1  || s2 ~= 1 )
            wrn{ end+1 } = [ ' BFO warning: wrong size of input for parameter epsilon.',   ...
	                     ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter epsilon.',      ...
	                  ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The quasi-Newton BFGS finish meshsize

   case 'bfgs-finish'

      if ( isnumeric( varargin{ i+1 } ) )
         bfgs_finish =  varargin{ i+1 };
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter bfgs-finish.',  ...
	                  ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The length of the memorized evaluation history

   case 'l-hist'

      if ( isnumeric( varargin{ i+1 } ) )
         l_hist = ceil( varargin{ i+1 } );
	 if ( l_hist <= 0 )
	    l_hist = Inf;
	 end
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter l-hist.',       ...
	                  ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
	 l_hist = 1;
      end

   %  The possible reset of the random number generator

   case 'reset-random-seed'

      if ( ischar( varargin{ i+1 } ) )
         reset_random_seed = varargin{ i+1 };
         if ( ~ismember( varargin{ i+1 }, {'reset', 'no-reset' } ) )
            wrn{ end+1 } = [ ' BFO warning: meaningless input for parameter',              ...
	                     ' reset-random-seed. Default used.' ];
           if ( verbose )
               disp( wrn{ end } )
            end
         end
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter',               ...
	                  ' reset-random-seed. Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The objective function target

   case 'f-target'

      if ( isnumeric( varargin{ i+1 } ) )
         ftarget    = varargin{ i+1 };
         user_ftarg = 1;
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter f-target.',     ...
	                  ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The saved information about discrete subspaces

   case 'sspace-hist'

      if ( isstruct( varargin{ i+1 } ) )
         s_hist = varargin{ i+1 };
      else
         wrn{ end+1 } = [ ' BFO internal warning: wrong type of input for parameter',      ...
	                  'sspace-hist. Default used.'];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

%  The elementwise history

   case 'element-hist'

      if ( iscell( varargin{ i+1 } ) )
         el_hist     = varargin{ i+1 };
	 user_elhist = ~isempty( el_hist );
      else
         wrn{ end+1 } = [ ' BFO internal warning: wrong type of input for parameter',      ...
	                  'element-hist. Default used.'];
         if ( verbose )
            disp( wrn{ end } )
         end
      end
      
   %  The recurrence path and depth

   case 'rpath'

      if ( isnumeric( varargin{ i+1 } ) )
         rpath = varargin{ i+1 };
      else
         wrn{ end+1 } =[ ' BFO internal warning: wrong type of input for parameter rpath.',...
                         '  Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end
      depth = length( rpath );

   %  The number of function calls higher in the recursion

   case 'nevr'

      if ( isnumeric( varargin{ i+1 } ) )
         neval = varargin{ i+1 };
      else
         wrn{ end+1 } = [ ' BFO internal warning: wrong type of input for parameter nevr.',...
                          ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end
      
   %  The vector of function values higher in the recursion

   case 'f-hist'

      if ( isnumeric( varargin{ i+1 } ) )
         f_hist     = varargin{ i+1 };
         user_fhist = 1;
      else
         wrn{ end+1 } = [ ' BFO internal warning: wrong type of input for parameter',      ...
	                  ' f-hist.  Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The number of random basis used for testing final termination

   case 'termination-basis'

      if ( isnumeric( varargin{ i+1 } ) )
         term_basis  = max( 1, round( varargin{ i+1 } ) );
         user_tbasis = 1;
      else
         wrn{ end+1 } = [' BFO warning:  wrong input for the number of termination basis.',...
                         ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  Maximize or minimize 

   case 'max-or-min'

      if ( restart )
         [ max_or_min, maximize, wrnf ] = bfo_verify_maxmin( varargin{ i+1 }, verbose,     ...
                                                            max_or_min );
      else
         [ max_or_min, maximize, wrnf ] = bfo_verify_maxmin( varargin{ i+1 }, verbose, '' );
      end
      if ( ~isempty( wrnf ) )
         wrn{ end+1} = wrnf;
      end

%  The call type for f(x)

   case 'f-call-type'

       if ( ischar( varargin{ i+1 } ) )
         fcallt = varargin{ i+1 };
         switch ( fcallt )
         case 'simple'
         case 'with-bound'
            withbound = 1;
         otherwise
            wrn{ end+1 } = [' BFO warning: wrong type of input for parameter f-call-type.',...
                            ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter f-call-type.',  ...
                          ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
       end
  
   %  The "unsuccessful" bound for f(x)

   case 'f-bound'

       if ( isnumeric( varargin{ i+1 } ) )
         fbound      = varargin{ i+1 };
         user_fbound = 1;
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter f-bound.',      ...
                          ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
       end

   %  The user-supplied tracker function

   case 'tracker'

      trfarg = varargin{ i+1 };
      if ( ischar( trfarg ) )
         trfname = trfarg;
      elseif ( isa( trfarg, 'function_handle' ) )
         trfname = func2str( trfarg );
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for tracker function.',      ...
                          ' No tracker used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
         trfname = '';
      end
      if ( ~isempty( trfname ) )
         if ( length( trfname ) > 3 && strcmp( trfname( 1:4 ), 'bfo_' ) )
            msg =[' BFO error: the name of the tracker function starts with ''bfo_''.',   ...
	          ' Terminating.' ];
            if ( verbose )
	       disp( msg )
	    end
	    return
         else
            if ( bfo_exist_function( trfname ) )
               if ( ischar( trfarg ) )
                  bfo_trf = str2func( trfname );
               else
                  bfo_trf = trfarg;
               end
               use_tracker = 1;
            else
               wrn{ end+1 } = [ ' BFO warning: m-file for tracker function ', trfname,    ...
                                ' not found. No tracker used. '];
               if ( verbose )
                  disp( wrn{ end } )
               end
            end
         end
      end

   %  The save strategy

   case 'save-freq'

      if ( isnumeric( varargin{ i+1 } ) )
         savef = varargin{ i+1 };
      else
         wrn{ end+1 } = [ ' BFO warning:  wrong input for the save-freq parameter.',       ...
	                  ' Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The name of the user-supplied search-step function

   case 'search-step'

      ssarg = varargin{ i+1 };
      if ( ischar( ssarg ) )
         ssfname = ssarg;
      elseif ( isa( ssarg, 'function_handle' ) )
         ssfname = func2str(  ssarg );
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for search-step.',            ...
                          ' No search-step used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
	 ssfname = '';
      end
      if ( ~isempty( ssfname ) )
         if ( length( ssfname ) > 3 && strcmp( ssfname( 1:4 ), 'bfo_' ) )
            msg =[' BFO error: the name of the search-step function starts with ''bfo_''.',...
	          ' Terminating.' ];
            if ( verbose )
	       disp( msg )
	    end
	    return
         else
            if ( bfo_exist_function( ssfname ) )
               if ( ischar( ssarg ) )
                  bfo_srch = str2func( ssfname );
               else
                  bfo_srch = ssarg;
               end
            else
               wrn{ end+1 } = [ ' BFO warning: m-file for search-step function ', ssfname, ...
                                ' not found. No search-step used. '];
               if ( verbose )
                  disp( wrn{ end } )
               end
	       ssfname = '';
            end
         end
      end
      
   %  The multilevel structure and assignment of variables to levels
       
   case 'xlevel'

      [ xlevel, msg, wrnt ] = bfo_verify_xlevel( varargin{ i+1 }, verbose );
       if ( ~isempty( msg ) )
         return
      end
      if ( ~isempty( wrnt ) )
         wrn{ end+1 } = wrnt;
      else
         multilevel = ( length( xlevel ) > 0 );
      end

%  The current level in a multilevel framework
      
    case 'level'

      if ( isnumeric( varargin{ i+1 } ) )
         level  = abs( round( varargin{ i+1 } ) );
      else
         wrn{ end+1 } = ' BFO warning:  wrong input for level parameter. Ignored.';
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The potential use of level dependent bounds
       
   case 'variable-bounds'

      [ vb_name, msg, wrnf ] = bfo_verify_variable_bounds( varargin{ i+1 }, verbose );
      if ( ~isempty( wrnf ) )
         wrn{ end+1 } = wrnf;
      end
      if ( ~isempty( msg ) )
         return
      end

%  The training strategy

   case 'training-strategy'

      if ( ~strcmp( restart_training, 'use' ) )
         if ( ischar( varargin{ i+1 } ) )
            training_strategy = varargin{ i+1 };
            if ( ismember( training_strategy,                                              ...
                           { 'average', 'robust', 'perfprofile', 'dataprofile' } ) )
            else
               training_strategy = 'average';
               wrn{ end+1 } = ' BFO warning: wrong type of training strategy. Default used.';
               if ( verbose )
                  disp( wrn{ end } )
               end
            end
         else
            training_strategy = 'average';
            wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter',            ...
	                     ' training-strategy.  Default used.'];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The training parameters

   case 'training-parameters'

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
                      switch( tpi )
                      case { 'alpha', 'beta', 'gamma', 'delta', 'eta', 'zeta', 'inertia',  ...
                             'search_type', 'random_seed', 'iota', 'kappa', 'lambda', 'mu' }
                         nbr_train_params = nbr_train_params + 1;
                         training_parameters{ nbr_train_params } = tpi;
                      otherwise
                         wrn{ end+1 } = [ ' BFO warning: unknown ', int2str( itp ),        ...
                                          '-th training parameter. Ignored.' ];
                         if ( verbose )
                            disp( wrn{ end } )
                         end
                      end
                   else
                      wrn{ end+1 } = [ ' BFO warning: unknown ', int2str( itp ),           ...
                                       '-th training parameter. Ignored.' ];
                      if ( verbose )
                         disp( wrn{ end } )
                      end
                   end
               end
            else
               nbr_train_params = 0;
            end
         else
            wrn{ end+1 } = [ ' BFO warning: wrong type of input for training-parameters.', ...
                             ' Default used.'];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      end

   %  The name of the file containing previously trained parameters (in 'solve'
   %  training mode) or where new trained parameters will be saved (in 'train'
   %  and 'train-and-solve' training modes)

   case 'trained-bfo-parameters'

      if ( ischar( varargin{ i+1 } ) )
         trained_bfo_parameters = varargin{ i+1 }; 
         use_trained = 1;       
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter',               ...
	                  ' trained-bfo-parameters.  Default name used.'];
         if ( verbose )
            disp( wrn{ end } )
         end
      end
   
   %  The accuracy threshold for the training process

   case 'training-epsilon'

      if ( isnumeric( varargin{ i+1 } ) )
         training_epsilon = abs( varargin{ i+1 } );
         if ( length( training_epsilon ) == 1 )
            training_epsilon = [ training_epsilon training_epsilon ];
         end
      else
         wrn{ end+1 } = [ ' BFO warning:  wrong input for the training-epsilon',           ...
	                  ' parameter. Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The maximum number of parameter configuration allowed in the training phase

   case 'training-maxeval'

      if ( isnumeric( varargin{ i+1 } ) )
         training_maxeval = varargin{ i+1 };
         if ( min( training_maxeval ) < 0 )
            wrn{ end+1 } = ' BFO warning: training-maxeval is negative. Default used.';
            if ( verbose )
               disp( wrn{ end } )
            end
         end
         if ( length( training_maxeval ) == 1 )
            training_maxeval = [ training_maxeval training_maxeval ];
         end
      else
         wrn{ end+1 }=[' BFO warning: wrong type of input for parameter training-maxeval.',...
                       ' Default used.'];
         if ( verbose )
            disp( wrn{ end } )
         end
      end
      
   %  The verbosity of the training process

   case 'training-verbosity'

      tmp = varargin{ i+1 };
      if ( iscell( tmp ) )
         tverbose = bfo_get_verbosity( tmp{ 1 } );
         if ( tverbose >= 0 )
            training_verbosity{ 1 } = tmp{ 1 };
            user_training_verbosity = 1;
         else
            wrn{ end+1 } = ' BFO warning: unknown training verbosity level. Default used.';
            if ( verbose )
               disp( wrn{ end } )
            end
         end
         if ( length( tmp ) > 1 )
            tverbose = bfo_get_verbosity( tmp{ 2 } );
            if ( tverbose >= 0 )
               training_verbosity{ 2 } = tmp{ 2 };
               user_training_verbosity = 1;
            else
               wrn{ end+1 } = ' BFO warning: unknown training verbosity level. Default used.';
               if ( verbose )
                  disp( wrn{ end } )
               end
               user_training_verbosity = 0;
            end
         end
      elseif ( ischar( tmp ) )
         tverbose = bfo_get_verbosity( tmp );
         if ( tverbose >= 0 )
            training_verbosity = { tmp, tmp };
            user_training_verbosity = 1;
         else
            wrn{ end+1 } = ' BFO warning: unknown training verbosity level. Default used.';
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      else
         wrn{ end+1 } = [ ' BFO warning:  wrong input for the training-verbosity',         ...
	                  ' parameter. Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The accuracy threshold for optimizing each test problem during training

   case 'training-problem-epsilon'

      if ( isnumeric( varargin{ i+1 } ) )
         training_problem_epsilon = abs( varargin{ i+1 } );
      else
         wrn{ end+1 } = [ ' BFO warning:  wrong input for the training-problem-epsilon',   ...
	                  ' parameter. Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The maximum number of objective function's evaluations for each problem
   %  during training

   case 'training-problem-maxeval'

      if ( isnumeric( varargin{ i+1 } ) )
         training_problem_maxeval = abs( round( varargin{ i+1 } ) );
      else
         wrn{ end+1 } = [ ' BFO warning: wrong type of input for parameter',               ...
                          ' training-problem-maxeval. Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The verbosity of the each problem training

   case 'training-problem-verbosity'

      tmp = varargin{ i+1 };
      if ( ischar( tmp ) )
         if ( bfo_get_verbosity( tmp ) >= 0 )
            training_problem_verbosity = tmp;
         else
            wrn{ end+1 } = [ ' BFO warning: unknown training problem verbosity level.',    ...
	                     ' Default used.' ];
            if ( verbose )
               disp( wrn{ end } )
            end
         end
      else
         wrn{ end+1 } = [ ' BFO warning:  wrong input for the training-problem-verbosity', ...
	                  ' parameter. Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The performance profile window (for profile training)

   case 'training-profile-window'

      tmp  = varargin{ i+1 };
      if ( isnumeric( tmp ) )
         if ( length( tmp ) == 1 )
            training_profile_window      = max( 1.1, tmp );
            user_training_profile_window = 1;
         else
            if ( tmp(2) > tmp(1) )
               training_profile_window      = tmp( 1:2 );
               user_training_profile_window = 2;
            else
               wrn{ end+1 } = [ ' BFO warning:  empty window for profile training.',       ...
	                        ' Default used.' ];
               if ( verbose )
                  disp( wrn{ end } )
               end
            end
         end
      else
         wrn{ end+1 } = [ ' BFO warning:  wrong input for the training-profile-window',    ...
	                  ' parameter. Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The performance profile cutoff fraction (for profile training)

   case 'training-profile-cutoff-fraction'

      tmp = varargin{ i+1 };
      if ( isnumeric( tmp ) )
         training_profile_cutoff_fraction = min( max( 1e-10, tmp ), 1-1e-10 );
      else
         wrn{ end+1 } = [ ' BFO warning:  wrong input for the training-profile-window',    ...
	                  ' parameter. Default used.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  The indeces of discrete variables across levels

   case 'idall'

      if ( isnumeric ( varargin{ i + 1 } ) )
         idall = varargin{ i + 1 };
      else
         wrn{ end+1 } = [ ' BFO warning:  wrong input for the idall parameter. Ignored.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
      end

   %  Keywords already handled before the current parsing are just ignored 
   %  without generating any error or warning.
   
   case { 'options-file', 'verbosity', 'restart', 'restart-file', 'training-mode',         ...
          'training-problems', 'training-problems-data', 'training-problems-library' }

   %  Unidentified keyword: ignore it and issue a warning.

   otherwise

      wrn{ end+1 } = [ ' BFO warning: undefined ', int2str( i ),'-th keyword ',            ...
                       varargin{ i }, '. Ignoring.' ];
      if ( verbose )
         disp( wrn{ end } )
      end

   end
end

%  Verify the user-supplied lattice basis.  This could not be done directly at reading
%  because it depends on the level, the specification of which may occur after that of
%  the lattice basis.

   if ( ~isempty( latbasis ) )
      [ latbasis, msg ] = bfo_verify_lattice_basis( latbasis, verbose, level );
      if ( ~isempty( msg ) )
         return
      end
   end

%  Possibly redefine the length of the memorized history if a search-step is required.

if ( ~isempty( ssfname ) )
   if ( l_hist == 1 )
      l_hist = 250;
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Choose so far unspecified algorithmic parameters according to the type of problem.     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   alpha        is the grid expansion factor                   
%   beta         is the grid shrinking factor                   
%   gamma        is the max grid expansion factor              
%   delta        is the initial scale for continuous variables
%   NOTE: the parameter delta is only used at *first* entry in BFO (that when topmost
%         == 1), the exchange of variable increment information between subsequent
%         depths or levels being handled via the increment table xincr, whose first
%         column contains the current step-sizes and the second the initial ones.  The latter
%         are used when starting a depth-first search for discrete variables.
%   eta          is the sufficient descent/ascent fraction             
%   zeta         is the increase in grid size when reoptimizing at a level previously visited
%   inertia      is the inertia for continuous step averaging  
%   stype        is the tree search  strategy for discrete variables (in integer form)
%   searchtype   is the tree search  strategy for discrete variables (in string form)
%   rseed        is the random number generator's seed
%   iota         is the CPS stepsize shrinking exponent
%   kappa        is the bracket expansion factor in bfo_min1d without quadrattic interpolation
%   lambda       is the min bracket expansion factor in bfo_min1d with quadrattic interp.
%   mu           is the max bracket expansion factor in bfo_min1d with quadrattic interp.

%   "Reasonable" values:

%   alpha           = 2;                   % the grid expansion factor                   
%   beta            = 0.5;                 % the grid shrinking factor                   
%   gamma           = 4;                   % the max grid expansion factor              
%   delta           = 1;                   % the initial scale for continuous variables
%   eta             = 1e-4;                % the sufficient descent fraction             
%   zeta            = 2;                   % the increase in grid size when ...
                                           % ... reoptimizing at a level previously visited
%   inertia         = 10;                  % the inertia for continuous step averaging  
%   stype           = 0;                   % breadth-first search for discrete variables 
%   searchtype      = 'breadth-first';     % ... also in string form
%   rseed           = 0;                   % random number generator's seed
%   iota            = 1.5;                 % the grid acceleration factor for CPS problems
%   kappa           = 2;                   % bracket expansion without interpolation
%   lambda          = 0.1;                 % min bracket expansion with interpolation
%   mu              = 50;                  % max bracket expansion with interpolation

%   Flag which is positive in the integer/categorical case

if ( solve )
   micase = any( ismember( xtype, { 'i', 'j', 's' } ) );
else
   micase = ismember( 'inertia'    , training_parameters ) ||                              ...
            ismember( 'search-type', training_parameters ) ||                              ...
            ismember( 'random-seed', training_parameters );
end

%   Define the default algorithmic parameters.

[ def_alpha, def_beta, def_gamma, def_delta, def_eta, def_zeta, def_inert, def_srcht,      ...
             def_stype, def_rseed, def_iota, def_kappa, def_lambda, def_mu ]               ...
      =  bfo_default_algorithmic_parameters( micase );

%  Use them for the present run unless the user has specified them explicitly.

if ( ~user_alpha )
   alpha = def_alpha;
end
if ( ~user_beta )
   beta  = def_beta;
end
if ( ~user_gamma )
   gamma = def_gamma;
end
if ( ~user_delta )
   delta = def_delta;
   user_delta = 1;
end
if ( ~user_eta )
   eta = def_eta;
end
if ( ~user_zeta )
   zeta = def_zeta;
end
if ( ~user_inertia )
   inertia = def_inert;
end
if ( ~user_searchtype )
   searchtype = def_srcht;
   stype      = def_stype;
end
if ( ~user_rseed )
   rseed = def_rseed;
end
if ( ~user_iota )
   iota = def_iota;
end
if ( ~user_kappa )
   kappa = def_kappa;
end
if ( ~user_lambda )
   lambda = def_lambda;
end
if ( ~user_mu )
   mu = def_mu;
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
      bfo_print_banner( this_version );
   end

   %  If the training strategy is profile, make sure each problem is solved accurately
   %  in order to allow recomputation of data profiles on the fly.

   if ( strcmp( training_strategy, 'perfprofile' ) )                                       ...
      training_problem_epsilon = min( training_problem_epsilon, 1e-12 );
      if ( user_training_profile_window )
         training_profile_window = [ training_perfprofile_window(1),                       ...
                                     training_profile_window ];
      else
         training_profile_window = training_perfprofile_window;
      end
   elseif( strcmp( training_strategy, 'dataprofile' )   )
      training_problem_epsilon = min( training_problem_epsilon, 1e-12 );
      if ( user_training_profile_window )
         training_profile_window = [ training_dataprofile_window(1),training_profile_window ];
      else
         training_profile_window = training_dataprofile_window;
      end
   end

   %  The current default parameters, with their lower/upper bounds and their scale.

   if ( nbr_train_params > 0 )
%     p0     = [  alpha  beta   gamma  delta  eta    zeta  inertia  stype rseed  iota  kappa lambda  mu  ];
      p0     = [  alpha  beta     2    delta  1e-3   zeta  inertia  stype rseed  iota  kappa lambda  mu  ];
%     plower = [    1    0.01     1    0.25   1e-4     1        5      0     0    1    1.1    0.1    10  ];
      plower = [    1    0.01    1.1   0.25   1e-4     1        5      0     0    1    1.1    0.1    10  ];
      pupper = [    2    0.95    10     10     0.5    10       30      2   100    3     5     0.9    50  ];
      pscale = [    1      1      1      1       1     1        1      1     1    1     1      1      1  ];
      pdelta = [        (pupper(1:6)-plower(1:6))/10            1      1     1    0.1  0.1    0.1     1  ];

      %   If the user has chosen which parameters to train, identify them in the list
      %   by releasing their type from fixed to continuous or integer, according to 
      %   their nature.

      ptype  = char( double('f') * ones( size( p0 ) ) );           % ptype  = 'ffffffffffffff'
      for  i = 1:nbr_train_params 
          switch ( training_parameters{ i } )
          case 'alpha'
             ptype( 1 )  = 'c';
          case 'beta'
             ptype( 2 )  = 'c';
          case 'gamma'
             ptype( 3 )  = 'c';
          case 'delta'
             ptype( 4 )  = 'c';
          case 'eta'
             ptype( 5 )  = 'c';
          case 'zeta'
             ptype( 6 )  = 'c';
          case 'inertia'
             ptype( 7 )  = 'i';
          case 'search-type'
             ptype( 8 )  = 'i';
          case 'random-seed'
             ptype( 9 )  = 'i';
          case 'iota'
             ptype( 10 )  = 'c';
          case 'kappa'
             ptype( 11 )  = 'c';
          case 'lambda'
             ptype( 12 )  = 'c';
          case 'mu'
             ptype( 13 )  = 'c';
          end
      end

   %  The user has supplied and empty list of parameters to train.

   else
      if ( solve)
         wrn{ end+1 } = ' BFO warning: empty set of training parameters. Aborting training.';
         if ( verbose )
            disp( wrn{ end } )
         end
         train = 0;
      else
         msg = ' BFO error: empty set of training parameters. Terminating.';
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
                                    training_problems,                                     ...
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
         disp( [ ' BFO training is running...  (', training_strategy, ' training strategy)' ])
         disp( ' ' )
         fprintf( '%s ', '     parameters:' )
         fprintf( '%s ', training_parameters{ 1:nbr_train_params } )
         fprintf( '\n\n' )
         if ( verbose > 2 )
            if ( strcmp( training_strategy, 'average' )      ||                            ...
                 strcmp( training_strategy, 'perfprofile' )  ||                            ...
                 strcmp( training_strategy, 'dataprofile' )   )
               disp( [ '    training epsilon           = ',                                ...
                                               num2str( training_epsilon( 1 ) ) ] )
               disp( [ '    training maxeval           = ',                                ...
                                               num2str( training_maxeval( 1 ) ) ] )
               disp( [ '    training verbosity         = ',                                ...
                                               training_verbosity{ 1 } ] )
            else
               disp( [ '    training epsilon           = [ ',                              ...
                                               num2str( training_epsilon( 1 ) ), ', ',     ...
                                               num2str( training_epsilon( 2 ) ), ' ]' ] )
               disp( [ '    training maxeval           = [ ',                              ...
                                               num2str( training_maxeval( 1 ) ), ', ',     ...
                                               num2str( training_maxeval( 2 ) ) , ' ]' ] )
               disp( [ '    training verbosity         = { ',                              ...
                                               training_verbosity{ 1 } , ', ',             ...
                                               training_verbosity{ 2 }, ' }'  ] )
            end
            if ( strcmp( training_strategy,   'perfprofile' ) ||                           ...
                 strcmp( training_strategy,   'dataprofile' )     )                        ...
               disp( [ '    profile window             = [',                               ...
                                               num2str( training_profile_window(1)),','    ...
                                               num2str( training_profile_window(2) ), ' ]' ] )
               disp( [ '    cutoff_fraction            = [ 1,',                            ...
                                               num2str( training_profile_cutoff_fraction), ...
                                                 ' ]' ] )
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
                  training_size = length( training_problems );
                  t_prob_names  = cell( training_size );
		  for ip = 1:training_size
		     t_prob_names{ ip } = func2str( training_problems{ ip }.objf );
		  end
                  bfo_print_cell( '   ', 'user specified training problems',               ...
                                  t_prob_names );
               end
            end
            fprintf( '\n %30s \n\n', 'Value of the starting algorithmic parameters:' );
            fprintf( ' %.12e   %-12s\n', alpha,     'alpha'   );
            fprintf( ' %.12e   %-12s\n', beta,      'beta'    );
            fprintf( ' %.12e   %-12s\n', gamma,     'gamma'   );
            fprintf( ' %.12e   %-12s\n', delta,     'delta'   );
            fprintf( ' %.12e   %-12s\n', eta,       'eta'     );
            fprintf( ' %.12e   %-12s\n', zeta,      'zeta'    );
            fprintf( ' %18d   %-12s\n',  inertia,   'inertia' );
            fprintf( ' %18d   %-12s\n',  stype,     'stype'   );
            fprintf( ' %18d   %-12s\n',  rseed,     'rseed'   );
            fprintf( ' %.12e   %-12s\n', iota,      'iota'    );
            fprintf( ' %.12e   %-12s\n', kappa,     'kappa'   );
            fprintf( ' %.12e   %-12s\n', lambda,    'lambda'  );
            fprintf( ' %.12e   %-12s\n', mu,        'mu'      );
            fprintf( '\n' );
         end
      end

      %  Train the parameters using the "average" training strategy.

      if ( strcmp( training_strategy, 'average' ) )
	 [ trained_parameters, ~, msgt, wrnt, ~, ~, ~, ~, training_history ] =             ...
               bfo( @(p,bestperf)bfo_average_perf( p, bestperf,                            ...
                                          'average',                                       ...
                                          training_problems,                               ...
                                          training_set_cutest,                             ...
                                          training_verbosity{ 1 },                         ...
                                          training_problem_epsilon,                        ...
                                          training_problem_maxeval,                        ...
                                          training_problem_verbosity,                      ...
                                          []                          ),                   ...
                    p0,  'xlower', plower, 'xupper', pupper, 'xtype', ptype, 'xscale',     ...
                    pscale, 'epsilon', training_epsilon( 1 ), 'maxeval',                   ...
                    training_maxeval( 1 ), 'verbosity', training_verbosity{ 1 },           ...
                    'termination-basis', 1, 'search-type', 'none', 'save-freq', savef,     ...
                    'restart', restart_training, 'restart-file', sfname, 'f-call-type',    ...
                    'with-bound', 'delta', pdelta );

      %  Train the parameters using the "robust" training strategy.

      elseif ( strcmp( training_strategy, 'robust' ) )
	 [ trained_parameters, ~, msgt, wrnt, ~, ~, ~, ~, training_history ] =             ...
               bfo( @(p,bestperf)bfo_robust_perf( p,  bestperf,                            ...
                                          training_parameters, training_problems,          ...
					  training_set_cutest,                             ...
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

      %  Train the parameters using the "perfprofile" training strategy.

      elseif ( strcmp( training_strategy, 'perfprofile' )  ||                              ...
               strcmp( training_strategy, 'dataprofile' )   )

	 [ trained_parameters, ~, msgt, wrnt, ~, ~, ~, ~, training_history] =              ...
             bfo( @(p, fstar, tpval)bfo_dpprofile_perf( p, fstar, tpval,                   ...
                                                        training_strategy,                 ...
                                                        training_profile_window,           ...
                                                        training_profile_cutoff_fraction,  ...
                                                        training_problems,                 ...
                                                        training_set_cutest,               ...
                                                        training_verbosity{ 1 },           ...
                                                        training_problem_epsilon,          ...
                                                        training_problem_maxeval,          ...
                                                        training_problem_verbosity ),      ...
                    p0,  'xlower', plower, 'xupper', pupper, 'xtype', ptype, 'xscale',     ...
                    pscale, 'epsilon', training_epsilon( 1 ), 'maxeval',                   ...
                    training_maxeval( 1 ), 'verbosity', training_verbosity{ 1 },           ...
                    'termination-basis', 1, 'search-type', 'none', 'save-freq', savef,     ...
                    'restart', restart_training, 'restart-file', sfname, 'f-call-type',    ...
                    'with-bound', 'delta', pdelta, 'max-or-min', 'max',                    ...
                    'training-strategy', training_strategy );

      end

      if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ) )
         msg   = msgt;
         return
      end
      if ( length( wrnt ) >= 11 && strcmp( wrnt(1:11), ' BFO warning' ) )
	 wrn{ end+1 } = wrnt;
      end

      %  Update the current algorithmic parameters for further use in train-and-solve 
      %  training mode.

      for i = 1:nbr_train_params
         switch  ( training_parameters{ i } )
         case 'alpha'
            alpha   = trained_parameters( 1 );
         case 'beta'
            beta    = trained_parameters( 2 );
         case 'gamma'
            gamma   = trained_parameters( 3 );
         case 'delta'
            delta   = trained_parameters( 4 );
         case 'eta'
            eta     = trained_parameters( 5 );
         case 'zeta'
            zeta    = trained_parameters( 6 );
         case 'inertia'
            inertia = trained_parameters( 7 );
         case 'search-type'
            stype   = trained_parameters( 8 );
         case 'random-seed'
            rseed   = trained_parameters( 9 );
         case 'iota'
            iota    = trained_parameters( 10 );
         case 'kappa'
            kappa   = trained_parameters( 11 );
         case 'lambda'
            lambda  = trained_parameters( 12 );
         case 'mu'
            mu      = trained_parameters( 13 );
         end      
      end  

      %  Save the trained parameters for future runs.

      fid = fopen( trained_bfo_parameters, 'w' );
      if ( fid == -1 )
         msg = [ ' BFO warning: cannot open file ', trained_bfo_parameters,                ...
                 '. Not saving trained parameters.' ];
         disp( msg )
      else
         fprintf( fid, ' *** BFO trained parameters file %s\n', date );
         fprintf( fid, ' *** (c) Ph. Toint and M. Porcelli\n' );
         fprintf( fid,' %.12e   %-12s\n', alpha,   'alpha'  );
         fprintf( fid,' %.12e   %-12s\n', beta,    'beta'   );
         fprintf( fid,' %.12e   %-12s\n', gamma,   'gamma'  );
         fprintf( fid,' %.12e   %-12s\n', delta,   'delta'  );
         fprintf( fid,' %.12e   %-12s\n', eta,     'eta'    );
         fprintf( fid,' %.12e   %-12s\n', zeta,    'zeta'   );
         fprintf( fid,' %18d   %-12s\n',  inertia, 'inertia');
         fprintf( fid,' %18d   %-12s\n',  stype,   'stype'  );
         fprintf( fid,' %18d   %-12s\n',  rseed,   'rseed'  );
         fprintf( fid,' %.12e   %-12s\n',  iota,   'iota'   );
         fprintf( fid,' %.12e   %-12s\n',  kappa,  'kappa'  );
         fprintf( fid,' %.12e   %-12s\n',  lambda, 'lambda' );
         fprintf( fid,' %.12e   %-12s\n',  mu,     'mu'     );
      end
      fclose(fid);

      msg = [ ' Training successful: trained parameters saved in file ',                   ...
              trained_bfo_parameters, '.' ];

      if ( verbose )
         fprintf( '\n%60s\n\n', msg )
         lh      = size( training_history, 1 );
         perf0   = training_history(  1, 4 );
         perfend = training_history( lh, 4 );
         if ( strcmp( training_strategy, 'perfprofile' ) )
            disp( [ ' Performance improved to ', num2str( perfend ),                       ...
                    '% in ', num2str( training_history( lh, 3 ) ),                         ...
                    ' problem function evaluations.' ] )
         else
            if ( strcmp( training_strategy, 'dataprofile' )   )
               improvement = round( 100 * ( perfend / perf0 - 1 ) );
             else
               improvement = round( 100 * ( 1 - perfend / perf0 ) );
            end
            disp( [ ' Performance improved by ', num2str( improvement ),                   ...
                    '% in ', num2str( training_history( lh, 3 ) ),                         ...
                    ' problem function evaluations.' ] )
         end
      end

      %  Possibly print the trained parameters and return these to the user,

      if ( verbose > 2 )
         fprintf( '\n %30s \n\n', 'Value of the trained algorithmic parameters:' );
         fprintf( ' %.12e   %-12s\n', alpha,   'alpha'   );
         fprintf( ' %.12e   %-12s\n', beta,    'beta'    );
         fprintf( ' %.12e   %-12s\n', gamma,   'gamma'   );
         fprintf( ' %.12e   %-12s\n', delta,   'delta'   );
         fprintf( ' %.12e   %-12s\n', eta,     'eta'     );
         fprintf( ' %.12e   %-12s\n', zeta,    'zeta'    );
         fprintf( ' %18d   %-12s\n',  inertia, 'inertia' );
         fprintf( ' %18d   %-12s\n',  stype,   'stype'   );
         fprintf( ' %18d   %-12s\n',  rseed,   'rseed'   );
         fprintf( ' %.12e   %-12s\n', iota,    'iota'    );
         fprintf( ' %.12e   %-12s\n', kappa,   'kappa'   );
         fprintf( ' %.12e   %-12s\n', lambda,  'lambda'  );
         fprintf( ' %.12e   %-12s\n', mu,      'mu'      );
      end

      %   Define the string for the discrete search type from the optimized 
      %   value of stype.

      switch ( stype )
      case 1
         searchtype = 'depth-first';
      case 0
         searchtype = 'breadth-first';
      case 2
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
                                         %  after each higher level evaluation
   if ( iscell( verbosity ) )
       lverbosity = length( verbosity );
   else
       lverbosity = 1;
   end
   if ( lverbosity > 1 ) 
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

   if ( level < 1  || level > nlevel )
      msg = [ ' BFO error: the level input parameter is not between 1 and max(xlevel).',   ...
              ' Terminating.' ];
      if ( verbose )
         disp( msg )
      end
      return

   elseif ( level == 1 )

      %  Check the size of the multilevel arguments.
    
      xlevel = abs( round( xlevel) );
      if ( length( xlevel ) ~= n )
         msg = [ ' BFO error: the xlevel input parameter has a size different from n.',    ...
                 ' Terminating.' ];
         if ( verbose )
            disp( msg )
         end
         return
      end
   
      %  Check the max-min specification size and content.
   
      if ( lmaxmin > 1 )
         if ( lmaxmin ~= nlevel )
            msg = [ ' BFO error: the multilevel max-min specification has the wrong size.',...
                    ' Terminating.'];
            if ( verbose )
               disp( msg )
            end
            return
         end
         for lev = 1:nlevel
            if ( strcmp( max_or_min( lev ,: ), 'max' ) == 0 &&                             ...
                 strcmp( max_or_min( lev, : ), 'min' ) == 0  )
               msg = [ ' BFO error: incorrect multilevel max-min specification.',          ...
                       ' Terminating.' ];
               if ( verbose )
                  disp( msg )
               end
               return
            end
         end
      end

      %  Determine if the number of levels is acceptable.
   
      if ( nlevel > max_nlevel )
         msg = ' BFO error: too many levels. Terminating.';
         if ( verbose )
            disp( msg )
         end
      end
   
      %  Compute the number of active variables per level.

      levsize  = zeros( 1, nlevel );
      inactive = [];
      for i = 1:n
         if ( ismember( xtype( i ), { 'c', 'i', 'j', 's', 'r', 'd', 'k' } ) )
            ilevel            = xlevel( i );
            levsize( ilevel ) = levsize( ilevel ) + 1;
         else
            inactive( end+1 ) = i;
         end
      end

      %  If not every level has an active variable, redefine the levels by
      %  assigning all inactive variables to the first level containing active
      %  variables.

      renumber = 0;
      if ( min( levsize ) == 0 )
         wrn{ end+1 } = [ ' BFO warning: not every level is assigned an active variable.', ...
                          ' Redefining levels.' ];
         if ( verbose )
            disp( wrn{ end } )
         end
         actlevels = find( levsize );
         xlevel( inactive ) = actlevels( 1 );
         renumber = 1;
      end

      %  Renumber the levels if their definition has changed.

      if ( renumber )
         lvl     = find( levsize == 0 );
         while ( ~isempty( lvl ) )
            ilvl = lvl( 1 );
            for j = 1:n
               if ( xlevel( j ) >  ilvl )
                  xlevel( j ) = xlevel( j ) - 1;
               end
            end
            if ( size( max_or_min, 1 ) > 1 )
               max_or_min( lvl, : ) = [];
            end
            levsize( lvl ) = -1;   % level lvl is now empty
            lvl = find( levsize( lvl+1:end ) == 0 );
         end
      end
      clear levsize;
      nlevel  = max( xlevel );              %  the new number of levels
      multilevel = ( nlevel > 1 );
   end
end

if ( multilevel )

   %  Construct the distribution of variables within levels.

   for lev = 1:nlevel
      vlevel{ lev } = [];
   end
   for i = 1:n
      vlevel{ xlevel( i ) }( end+1 ) =  i;
   end

   %  Decide which of maximization or minimization applies at the current level.

   if ( size( max_or_min, 1 ) > 1 )
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

%  If categorical variables are present, set a the flag categorical and transform
%  the starting point into useable numerical values (while performing a few checks). 

categorical = ~isnumeric( x0 );
if ( categorical )
   x0ref = x0;                % Remember the default values for categorical variables
   [ x0, cat_dictionnary, errx0 ] = bfo_numerify( x0, xtype, cat_dictionnary );
   numx0 = x0;
   if ( errx0 > 0 )
      msg = [ ' BFO error: component ', int2str( errx0 ), ' of x0 of the wrong type.',     ...
              ' Terminating.' ];
      if ( verbose )
         disp( msg )
      end
      return
   elseif ( errx0 < 0 )
      return
   end
else
   x0ref = 0;                 %  No categorical variables (could use x0, but more expensive)
   numx0 = x0;
end
   
%   Find the indices of each type of variable and project the initial
%   point onto the feasible set.  More specifically,
%      icont   contains the indices of the active continuous variables,
%      idisc   contains the indices of the active discrete variables,
%      icate   contains the indices of the active categorical variables,
%      ifixd   contains the indices of the variables fixed by the user,
%      ixed    contains the indices of the continuous variables whose value is fixed 
%              in the course of multilevel optimization,
%      iyed    contains the indices of the discrete variables whose value is fixed 
%              in the course of multilevel optimization,
%      ized    contains the indices of the categorical variables whose value is fixed 
%              in the course of multilevel optimization,
%      iwait   contains the indices of the variables whose value is fixed
%              by a recursive call in the presence of discrete/categorical variables.
%      ired    contains the indices of the continuous variables deactivated
%              for the current values of the categorical ones
%      ided    contains the indices of the integer variables deactivated
%              for the current values of the categorical ones
%      iked    contains the indices of the categorical variables deactivated
%              for the current values of the categorical ones
%   For each type of variable assign the correct scale to the multilevel increments.

nactive    = 0;   % to contain the number of active variables (at current level)
nactivd    = 0;   % to contain the number of discrete active variables (at current level)
nactivs    = 0;   % to contain the number of categorical active variables (at current level)
icont      = [];  % to contain the indices of continuous variables (at current level)
idisc      = [];  % to contain the indices of discrete variables (at current level)
icate      = [];  % to contain the indices of categorical variables (at current level)
ifixd      = [];  % to contain the indices of fixed variables (at current level)
ixed       = [];  % to contain the indices of continuous frozen variables (at current level)
iyed       = [];  % to contain the indices of discrete frozen variables (at current level)
ized       = [];  % to contain the indices of categorical frozen variables (at current level)
iwait      = [];  % to contain the indices of waiting variables (at current level)
ired       = [];  % to contain the indices of continuous variables inactive for the...
                  %         ... current states of the categorical variables
ided       = [];  % to contain the indices of discrete variables inactive for the...
                  %         ... current states of the categorical variables
iked       = [];  % to contain the indices of categorical variables inactive for the...
                  %         ... current states of the categorical variables
ncbound    = 0;   % to contain the number of continuous variables...
                  %         ... with a finite lower/upper bound (at current level)
ndbound    = 0;   % idem for discrete variables

%  Save the original status of the variables.  This is passed to the multilevel
%  recursive call to bfo and allows the original status to be activated when the
%  level variables are activated.

   xtmulti  = xtype;

%  Build the necessary information for the lattice basis in the multilevel case.

if ( topmost )
   idall = [];
   build_idall = ~isempty( latbasis ) && multilevel;
else
   build_idall = 0;
end

%  Loop on the variables.

%  Find the type of the i-the variable, assign its associated increment to the 
%  multilevel increment structure xincr, freeze it if it does not belong to the 
%  current level and determine the subsets of variables of each type
%  at the current level. Finally project the non-fixed components of the 
%  starting point onto their feasible interval. 
%  Note also that the second column of xincr is set for later
%  reference to the values of the initial stepsizes defined from delta (and xscale). 

incb        = [];                                     % no inconsistent bounds so far
equb        = [];

if ( length( delta ) == 1  )
  common_delta = delta;
   delta        = ones( n, 1 );
else
    common_delta = 0;
end

for i = 1:n

   switch ( xtype( i ) )

   %  Continuous variables

   case { 'c', 'r' }

      % Assign the multilevel increments at first entry in BFO.

      if ( topmost )
         if ( common_delta  )
            xincr( i, cur ) = common_delta;
         else
            xincr( i, cur ) = delta(i);
         end
      end

      % Freeze variable i if it is not assigned to the current level.

      if ( multilevel &&  xlevel( i ) ~= level )
          xtype( i )     = 'x';
          ixed( end + 1 ) = i;

      %   Deactivate it if indicated so by the user.

      elseif( xtype( i ) == 'r' )
          ired( end+1 ) = i;

      %   Variable i is possibly active at the current level. Check if it is truly 
      %   active or fixed because its lower and upper bounds are the same (using
      %   the unscaled bounds, if relevant).

      else
         if ( xlower( i ) > xupper( i ) )
            xlower( i )     = -myinf;
            xupper( i )     =  myinf;
            incb( end+1 )   = i ;
            icont ( end+1 ) = i;
         elseif ( ( xupper( i ) >= 0 && xlower( i ) < xupper( i ) * ( 1 - eps ) ) ||       ...
                  ( xupper( i ) <  0 && xlower( i ) < xupper( i ) * ( 1 + eps ) )    )
            icont ( end+1 ) = i;
            nactive         = nactive + 1;
            x0( i )         = max( xlower( i ), min( x0( i ), xupper( i ) ) );
            if ( xlower( i ) > - myinf  || xupper( i ) < myinf )
                ncbound    = ncbound + 1;
            end
         else 
            ifixd( end+1 ) = i;
            x0( i )        = 0.5 *( xlower( i ) + xupper( i ) );
            xtype( i )     = 'f';
            equb( end+1 )  = i;
         end
      end

   %  Discrete variables

   case { 'i', 'j', 'd' }

      if ( build_idall )
         idall( end + 1 ) = i;
      end

      % Assign the multilevel increments at first entry in BFO.

      if ( topmost )
         if ( user_scl == 1 && isempty( latbasis ) ) % avoids using single number spec.
            xincr( i, cur ) = xscale( i );
         else
            xincr( i, cur ) = 1;
         end
      end

      % Freeze variable i if it is not assigned to the current level.

      if ( multilevel &&  xlevel( i ) ~= level )
         xtype( i )     = 'y';
         iyed( end + 1 ) = i;

      %   Deactivate it if indicated so by the user.

      elseif( xtype( i ) == 'd' )
          ided( end+1 ) = i;

      %   Variable i is possibly active at the current level. Check if it is truly 
      %   active or fixed because its lower and upper bounds are the same.

      else
         if ( xlower( i ) > xupper( i ) )
            xlower( i )    = -myinf;
            xupper( i )    =  myinf;
            incb( end+1 )  = i;
         elseif ( xlower( i ) < xupper( i ) * ( 1 - eps ) )
            idisc( end+1 ) = i;
            nactive        = nactive + 1;
            nactivd        = nactivd + 1;
            if ( xlower( i ) > - myinf  || xupper( i ) < myinf )
                ndbound = ndbound + 1;
            end
            
            %  The variable is truly active: make sure it lies between its lower and
            %  upper bounds.  This can be done one variable at a time if the discrete
            %  variables vary on the (default) canonical lattice.

            if ( isempty( latbasis ) )

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

         %  The lower and upper bound are the same: fix the i-th variable.

         else 
            ifixd( end+1 ) = i;
            x0( i )        = 0.5 * ( xlower( i ) + xupper( i ) );
            xtype( i )     = 'f';
	    equb( end+1 )  = i;
         end
      end

   %  Categorical variables

   case { 's', 'k' }

      % Assign the multilevel increments at first entry in BFO.
      % Note that these increments will never be effectively used because the
      % neighbourhoods of categorical variables is defined by the user.

      if ( topmost )
         xincr( i, cur ) = 1;
      end

      % Freeze variable i if it is not assigned to the current level.

      if ( multilevel &&  xlevel( i ) ~= level )
         xtype( i )     = 'z';
         ized( end + 1 ) = i;

      %   Deactivate it if indicated so by the user.

      elseif( xtype( i ) == 'k' )
          iked( end+1 ) = i;

      %   Variable i is possibly active at the current level.

      else
         icate( end+1 ) = i;
         nactive        = nactive + 1;
         nactivs        = nactivs + 1;

         %  Make sure the categorical variables are unconstrained.

         xlower( i ) = -myinf;
	 xupper( i ) =  myinf;
	 
      end

   %  Fixed variables

   case 'f'
      if ( topmost )
         ifixd( end+1 ) = i;
         if ( xlower( i ) > x0( i )  || xupper( i ) < x0( i ) )
            msg =  [ ' BFO error: variable ', int2str( i ),                                ...
                     ' is fixed but outside of its bounds. Terminating.' ];
            if ( verbose )
               disp( msg )
            end
            return
         end
         xincr( i, cur ) = 0;
      end

   %  Waiting variables (i.e. temporarily fixed variables within subspace recursion)

   case 'w'
      iwait( end+1 ) = i;
      if ( topmost )
         xincr(  i, cur ) = 1;
      end

   %  Frozen continuous variables (i.e. continuous variables inactive at the current level)

   case 'x'
      ixed( end+1 )    = i;
      if ( topmost )
         xincr(  i, cur ) = 1;
      end

   %  Frozen discrete variables (i.e. discrete variables inactive at the current level)

   case 'y'
      iyed( end+1 )    = i;
      if ( topmost )
         xincr(  i, cur ) = 1;
      end

   %  Frozen categorical variables (i.e. categorical variables inactive at the current level)

   case 'z'
      ized( end+1 )    = i;
      if ( topmost )
         xincr(  i, cur ) = 1;
      end

  %  Something went wrong.
   
   otherwise
      if ( topmost )
         msg = [ ' BFO internal error: impossible variable type: ', xtype( i ),            ...
                 ' Terminating.' ];
         disp( msg )
         return
      end
   end
end

%  Terminate if inconsistent bounds were found.

lincb = length( incb );
if ( lincb > 0 )
   msg = [ ' BFO error: inconsistent bounds for variable(s) ', mat2str( incb ),            ...
                       '. Terminating.' ];
   if ( verbose )
      disp( ' BFO error: inconsistent bounds for variables' );
      is = 1;
      for iii = 1:ceil( lincb/10 )
         it = min( is + 9, lincb );
         fprintf( '  %5d   %5d   %5d   %5d   %5d   %5d   %5d   %5d   %5d   %5d',           ...
                  incb( is:it ) );
         fprintf( '\n' );
         is = is + 10;
      end
      disp( ' Terminating.' )
   end
   return
end

%  The role of delta is completed.  Deallocate it.

if ( topmost )
   clear delta
end

%  Output warning if equal bounds were found.

lequb = length( equb );
if ( lequb > 0 )
   if ( lequb == 1 )
      wrn{ end+1 } = [ ' BFO warning: equal bounds for variable ', num2str( equb ),        ...
                       '. Fixing this variable.' ];
   else
      wrn{ end+1 } = [ ' BFO warning: equal bounds for ', int2str( lequb ),                ...
                       ' variables. Fixing these variables.' ];
   end
   if ( verbose )
      if ( lequb == 1 )
         disp( wrn{ end } )
      else
         disp( ' BFO warning: equal bounds for variables' );
         is = 1;
         for iii = 1:ceil( equb / 10 )
            it = min( is + 9, lequb );
            fprintf( '  %5d   %5d   %5d   %5d   %5d   %5d   %5d   %5d   %5d   %5d',        ...
                     equb( is:it ) );
            fprintf( '\n' );
            is = is + 10;
         end
         disp( ' Fixing these variables.' )
      end
   end
end

%  Get the number of variables of each type.

ncont = length( icont );   % the number of continuous variables
ndisc = length( idisc );   % the number of discrete variables
ncate = length( icate );   % the number of categorical variables
nfixd = length( ifixd );   % the number of fixed variables
nxed  = length( ixed  );   % the number of frozen continuous variables
nyed  = length( iyed  );   % the number of frozen discrete variables
nzed  = length( ized  );   % the number of frozen categorical variables
nwait = length( iwait );   % the number of waiting variables
nred  = length( ired  );   % the number of cat-deactivated continuous variables
nded  = length( ided  );   % the number of cat-deactivated discrete variables
nked  = length( iked  );   % the number of cat-deactivated discrete variables
ndall = length( idall );   % the number of discrete varaibles across levels

%  Return if all variables are inactive.

if ( ncont + ndisc + ncate == 0 )
   msg = ' BFO error: all variables are fixed or inactive: nothing to optimize. Terminating.';
   if ( verbose )
      disp( msg )
   end
end

%  Make a copy of the initial stepsizes (contained in column 1 of xincr) to a 
%  second column, if requested for the depth-first search recursion strategy 
%  for discrete variables, or for restarting the optimization at a higher level
%  in the multilevel case.

if ( ( ndisc && stype == 1 ) || multilevel )
   ini = 2;
   xincr( :, ini ) = xincr( :, cur );
end

%  Construct the combined index set for discrete and categorical variables.

idc = [ idisc icate ];
icd = [ icont idisc ];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%  Determine the status of the continuous variables wrt to scaling  %%%%%%%%%%%%%%%
%%%%%%%%%%%%       depending on the entry value of cscaled (if any)            %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  1) On the first top-most entry in BFO in solve mode, either xscale is specified by
%     the user or not.
%     - If xscale is specified, it must contain the scaling to be applied to the continuous
%       variables. If any of the scaling factors xscale(j) for continuous j is different 
%       from one, this means that BFO must first transform the problem by scaling the 
%       continuous variables to define new internal scaled variables x(j)/xscale(j) 
%       (the lower and upper bounds also need the same scaling). Note that, in this case, 
%       the new continuous variables  must be unscaled  (ie multiplied by xscale(j)) 
%       before calls to user-supplied functions (because the user (and his/her functions)
%       only know about original unscaled variables). Because xscale is set to 1 for all
%       non continuous variables, it has no effect on these.
%     - If xscale is not specified or all scaling factors xscale(j) are equal to 1, 
%       then an initial scaling transformation is not required, and consequently there
%       is no need for unscaling before calling user-supplied functions. 
%       xscale is then redefined to be the empty vector [].
%  2) On entry not at the topmost level, any required scaling has been applied already
%     and, if non-trivial, unscaling is required before calling the user-supplied
%     functions.
%  Note that variable scaling occurs only once at the top level of any BFO recursion,
%  when xscale is specified on entry. 

if ( topmost )
   if( ncont && user_scl && norm( xscale( icont ) - ones( ncont, 1 ) ) > eps )
      x0( icont )      = x0( icont ) ./ xscale( icont );
      finite           = intersect( icont, find( xlower > myinf ) );
      xlower( finite ) = xlower( finite ) ./ xscale( finite );
      finite           = intersect( icont, find( xupper < myinf ) );
      xupper( finite ) = xupper( finite ) ./ xscale( finite );
      clear finite
      xscale( setdiff( [ 1:n ], icont ) ) = 1;
   else
      xscale = [];
   end
else
   if ( ~user_scl )
      xscale = [];
   end
end

%  Verify the consistency of xtype and x0 for numeric x0.

if ( topmost && ncate  && ~categorical )
   msg = [ ' BFO error: inconsistent definition of categorical variables in x0 and xtype.',...
           ' Terminating.' ];
   if ( verbose )
      disp( msg )
   end
   return
end

%  Verify that there are 'r', 'd' or 'k' variables only if there are categorical ones

if ( nred + nded + nked && ~categorical )
   nred_nded_nked_ncate = [ nred nded nked ncate ]
   msg = [ ' BFO error: some variables are deactivated by the user given the current',     ...
           ' state of the categorical variables, but there are no categorical variables.'  ...
           ' Terminating.' ];
   disp( msg )
   return
end

%  Initialize the history of previous iterates.

if ( categorical )
   x_hist = {};
else
   x_hist = [];
end
if ( sum_form && ~user_elhist )
   explicit_domains = length( eldom );
   el_hist          = cell( 1, n_elements );
   for iel = 1:n_elements
      if ( categorical )
         if ( explicit_domains )
            el_hist{ iel } = struct( 'xel', {{}}, 'fel', [],  'eldom', eldom{ iel },       ...
                                     'fbest', [] );
         else
            el_hist{ iel } = struct( 'xel', {{}}, 'fel', [], 'eldom', [], 'fbest', [] );
         end
      else
         if ( explicit_domains )
            el_hist{ iel } = struct( 'xel', [], 'fel', [],'eldom', eldom{ iel },           ...
                                     'fbest', [] );
         else
            el_hist{ iel } = struct( 'xel', [], 'fel', [], 'eldom', [], 'fbest', [] );
         end
      end
   end
end

%  If categorical variables are present, verify that the user has specified neighbourhoods
%  in one or the other way.

if ( categorical && topmost && isempty( cat_states ) && isempty( cn_name ) )
   msg = [ ' BFO error: categorical variables are present but no neighbourhood has been',  ...
           ' specified by the user. Terminating.' ];
   if ( verbose )
      disp( msg )
   end
   return
end
dynamical = categorical && ~isempty( cn_name );

%  If categorical variables are present using the static neighbourhood structure
%  cat_states, transform it to num_cat_states, a useable numerical structure.

if ( categorical && isempty( cn_name ) )
   for i = 1:n
      num_cat_states{ i } = [];
      if ( xtype( i ) == 's'  || xtype( i ) == 'k' )
         icats = cat_states( i );
         for j = 1: length( icats ) 
            [ ~, index ] = ismember( icats{ j }, cat_dictionnary );
            num_cat_states{ i } = [ num_cat_states{ i } index ];
         end
      end
   end
   if ( verbose >= 10 )
      num_cat_states = num_cat_states
   end
end

%  If the starting point is infeasible for its discrete bounds and a lattice is used,
%  apply BFO for finding the (locally) minimal infeasibility on these variables,
%  and exit if no discrete feasible point can be found on the lattice.

if ( nactivd > 0 && topmost && norm( x0( idisc )' - max( [ xlower( idisc )';               ...
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

   %  Fix the continuous and categorical variables and remember the evaluation count.

   xts          = xtype;
   xts( icont ) = 'f';
   zz           = zeros( size( x0 ) );

   if ( categorical )
      xts( icate ) = 'f';
      xu = x0ref;
   else
      xu = x0;
   end

   %  Minimize the ell_1 norm of the discrete infeasibilities with respect to
   %  the user-supplied lattice.

   [ xtry, infeasible, msg0, wrn0, nevalinf, f_hist_ell_1 ] =                              ...
        bfo( @(x)norm(max([xlower'-x';zz'])+max([x'-xupper';zz']),1), xu,                  ...
             'xscale', xscale, 'xtype', xts, 'maxeval', maxeval, 'verbosity', fverb,       ...
             'alpha', alpha, 'beta', beta, 'gamma', gamma, 'eta', eta, 'zeta', zeta,       ...
	     'inertia', inertia, 'search-type', searchtype, 'random-seed', rseed,          ...
	     'iota', iota, 'kappa', kappa, 'lambda', lambda, 'mu', mu, 'lattice-basis',    ...
             latbasis, 'xincr', xincr, 'reset-random-seed', reset_random_seed );

   if ( ( length( msg0 ) >= 10 && strcmp( msg0(1:10), 'BFO error' ) )  ||                  ...
        isnan( infeasible ) )
      msg = [ msg0( 1:12 ), ' search for lattice feasible x0 returned the message: ',      ...
              msg0( 13:length( msg0 ) ) ];
      return
   end
   if ( length( wrn0 ) >= 11 && strcmp( wrn0(1:11), 'BFO warning' ) )
      wrna = [ wrn0( 1:13 ), ' search for lattice feasible x0 returned the warning ',      ...
              wrn0( 14:length( wrn0 ) ) ];
      wrn{ end+1 } = wrna;
   end

   if ( verbose >= 2 )
      disp( [ ' discrete infeasibility = ', num2str( infeasible ), ' computed in ',        ...
              int2str( nevalinf ), ' BFO evaluations.' ] )
      if ( verbose > 3 )
         disp( 'Successive ell_1 norms of the discrete infeasibilities :' )
	 f_hist_ell_1
      end
   end

   %  Terminate if locally infeasible.

   if ( infeasible )
      msg = [ ' BFO error: no feasible discrete point found on the lattice near x0.',      ...
              ' Terminating.' ];
      if ( verbose )
         disp( msg )
      end
      xbest = x0ref;
      fbest = infeasible;
      return
   else
   end

   %  Use the feasible starting point.

   if ( categorical )
      x0 = bfo_numerify( xtry, xtype, cat_dictionnary );
   else
      x0 = xtry;
   end
   bfo_print_x( '', 'feasible x0', x0, [], verbose, xtype, cat_dictionnary, x0ref )
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  Verify the use of optimized algorithmic parameters %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Check if the file trained_bfo_parameters exists if specified by the user
%  for the solve training mode. If the file exists, read the trained parameters and
%  update the current BFO algorithmic parameters.

if ( use_trained && ~train )
   fid = fopen( trained_bfo_parameters, 'r' );
   if ( fid == -1 )
      wrn{ end+1 } = [ ' BFO warning: cannot open file ', trained_bfo_parameters,          ...
                       '. Using default values for BFO parameters.' ];
      if ( verbose )
         disp(' ')
         disp( wrn{ end } )
      end
   else
      filetitle = fscanf( fid, '%s', 13 );
      alpha     = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      beta      = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      gamma     = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      delta     = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      eta       = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      zeta      = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      inertia   = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
      stype     = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
      rseed     = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
      iota      = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      kappa     = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      lambda    = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      mu        = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
      fclose( fid );
      if ( verbose > 1 )
         disp( ' ' )
         disp( [ ' Trained parameters read from file: ', trained_bfo_parameters, '.' ] )
      end
      xincr( icont, cur:ini ) = delta;
      if ( user_scl )
         xincr( idisc, cur:ini ) = xscale( i );
      else
         xincr( idisc, cur:ini ) = 1;
      end
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Miscellaneous initializations  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Maximization is requested.

if ( maximize )

   %  Adapt the unsuccessful bound.

   if ( withbound && ~user_fbound )
      fbound = -Inf;
   end
   
   %   Adapt the default target objective function value.

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
   if ( use_cps < 0 )
       indent = [ '     ' indent ];
   end
end

%   Initialize the termination checks counter.

term_loops = 0;

%   Determine if search-setp cleanup is necessary.

search_step_needs_cleaning_up = topmost && ~isempty( ssfname );

%   Possibly reset the random number generator.

if ( strcmp( reset_random_seed, 'reset' ) && ~restart )
   rng( rseed, 'twister' )
end

%  Verify the structure of the lattice basis. If multilevel optimization is
%  requested, the lattice basis must be separable between levels.
%  Only perform this verification once (at topmost entry).

if ( topmost && ~isempty( latbasis ) )
   [ slb1, slb2 ] = size( latbasis );
   if ( (  ndall &&          ( slb1 ~= ndall || slb2 ~= ndall ) ) ||                       ...
        ( ~ndall && ndisc && ( slb1 ~= ndisc || slb2 ~= ndisc ) )   )
        
      msg = ' BFO error: wrong dimension for the lattice-basis. Terminating.';
      if ( verbose )
         disp( msg )
      end
      return
   end

   if ( multilevel )
      for i = 1:ndall
         id = idall( i );
         for j = 1:ndall
            jd = idall( j );
            if ( xlevel( id ) ~= xlevel( jd ) && latbasis( i, j ) ~= 0 )
               msg = [ ' BFO error: lattice-basis is not separable between levels''',      ...
	               ' definition. Terminating.' ];
               if ( verbose )
                  disp( msg )
               end
               return
            end
         end
      end
   end 
end

%   Define the initial mesh size for the current level.  Note that one needs to define 
%   cmesh even if there is no continuous variable, because cmesh defines the sufficient 
%   objective function decrease  which, if unachievable, causes convergence. In this case,
%   cmesh is chosen to make a decrease of 10*eps  sufficient for BFO to pursue 
%   optimization.

if ( ncont > 0 )
   cmesh = max( xincr( icont, cur ) );
else
   cmesh = sqrt( 10 * eps ) / eta ;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%   Possibly print the initial data.  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( verbose >= 10 )
   if ( topmost )
      fprintf( '\n%18s   %-12s\n\n', ' Value',       'Parameter'          );
      fprintf( '%18s   %-12s\n',       shfname,      'shfname'            );
      if ( fxok )  
         fprintf( '%+.11e   %-12s\n',  sum(fx0),     'fx0'                ); 
      end   
      fprintf( '%18d   %-12s\n',       n,            'n'                  );
      fprintf( '%18d   %-12s\n',       ncont,        'ncont'              );
      fprintf( '%18d   %-12s\n',       ndisc,        'ndisc'              );
      fprintf( '%18d   %-12s\n',       ncate,        'ncate'              );
      fprintf( '%18d   %-12s\n',       nfixd,        'nfixd'              );
      fprintf( '%18d   %-12s\n',       nxed,         'nxed'               );
      fprintf( '%18d   %-12s\n',       nyed,         'nyed'               );
      fprintf( '%18d   %-12s\n',       nzed,         'nzed'               );
      if ( categorical )
         fprintf( '%18d   %-12s\n',    nred,         'nred'               );
         fprintf( '%18d   %-12s\n',    nded,         'nded'               );
         fprintf( '%18d   %-12s\n',    nked,         'nked'               );
      end
      if ( multilevel )
         fprintf( '%18d   %-12s\n',    nwait,        'nwait'              );
      end
      fprintf( '%18d   %-12s\n',       ncbound,      'ncbound'            );
      fprintf( '%18d   %-12s\n',       ndisc,        'ndisc'              );
      fprintf( '%18s   %-12s\n',       xtype,        'variables'' type'   );
      fprintf( '%18d   %-12s\n',       nactive,      'nactive'            ); 
      fprintf( '%18s   %-12s\n',       fcallt,       'f-call-type'        );
      if ( user_fbound )
         fprintf( '%+.11e   %-12s\n',  fbound,       'f-bound'            );
      end     
      fprintf( '%18d   %-12s\n',       maxeval,      'maxeval'            );
      if ( ~isempty( ssfname ) )
         fprintf( '%18s   %-12s\n',    ssfname,      'ssfname'            );
      end
      fprintf( '%18s   %-12s\n',  reset_random_seed, 'reset-random-seed'  );
      fprintf( '%18d   %-12s\n',       verbose,      'verbose'            );
      fprintf( '%18d   %-12s\n',       nlevel,       'nlevel'             );
      if ( ~isempty( vb_name' ) )
         fprintf( '%18s   %-12s\n',    vb_name,      'vb_name'            );
      end
      if ( dynamical )
         fprintf( '%18s   %-12s\n',    cn_name,      'cn_name'            );
      end
      fprintf( '%+.11e   %-12s\n',     ftarget,      'ftarget'            ); 
      fprintf( '%.12e   %-12s\n',      epsilon,      'epsilon'            ); 
      fprintf( '%18d   %-12s\n',       l_hist,       'l_hist'             );
      fprintf( '%18d   %-12s\n',       bfgs_finish,  'bfgs_finish'        ); 
      fprintf( '%.12e   %-12s\n',      alpha,        'alpha'              ); 
      fprintf( '%.12e   %-12s\n',      beta,         'beta'               ); 
      fprintf( '%.12e   %-12s\n',      gamma,        'gamma'              ); 
      fprintf( '%.12e   %-12s\n',      eta ,         'eta'                );
      fprintf( '%.12e   %-12s\n',      zeta ,        'zeta'               );
      fprintf( '%18d   %-12s\n',       inertia,      'inertia'            );
      fprintf( '%18d   %-12s\n',       rseed,        'rseed'              );
      fprintf( '%.12e   %-12s\n',      iota,         'iota'               ); 
      fprintf( '%.12e   %-12s\n',      kappa,        'kappa'              ); 
      fprintf( '%.12e   %-12s\n',      lambda,       'lambda'             ); 
      fprintf( '%.12e   %-12s\n',      mu,           'mu'                 ); 
   end
   if ( ndisc > 0 )
      fprintf( '%18s   %-12s\n',       searchtype,   'search-type'        );
      if ( ~isempty( latbasis ) )
         bfo_print_matrix( indent, 'latbasis', latbasis                   );
      end
   end   
   fprintf( '%18d   %-12s\n',          term_basis,   'term_basis'         );
   fprintf( '%18d   %-12s\n',          restart,      'restart'            );
   fprintf( '%18d   %-12s\n\n',        savef,        'save-freq'          );
   
   for i = 1:length( eldom )
      fprintf( '%s%s = ', indent, [ ' eldom{', int2str( i ), '}' ] );
      for j = 1:length( eldom{ i } )
         fprintf( ' %3d', eldom{ i }( j ) );
      end
      fprintf( '\n' );
   end

   if ( isempty( xscale ) )
      if ( ncbound )
         bfo_print_vector( indent, 'lower bounds', xlower );
      end
      bfo_print_vector( indent, 'starting point',  x0, xtype, cat_dictionnary, x0ref );
      if ( ncbound )
         bfo_print_vector( indent, 'upper bounds', xupper );
      end
   else
     if ( ncbound )
         bfo_print_vector( indent, 'lower bounds', xscale.*xlower );
      end
      bfo_print_vector( indent, 'starting point', xscale.*x0, xtype, cat_dictionnary,     ...
                        x0ref );
      if ( ncbound )
         bfo_print_vector( indent, 'upper bounds', xscale.*xupper );
         bfo_print_vector( indent, 'scaled lower bounds', xlower );
         bfo_print_vector( indent, 'scaled upper bounds', xupper );
      end
   end
   bfo_print_vector( indent, '(scaled) increments', xincr(:,cur) );
   if ( multilevel )
      bfo_print_vector( indent, 'xlevel', xlevel );
   end
   fprintf( '\n \n')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%  Search for a better starting point, if possible.  %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fbloop   = 1;
fxok     = 0;

if ( ( ( ndisc  || categorical ) && depth )  || restart )
   nh = length( s_hist );               %  the length of the subspace history
   if ( depth )

      %   Loop over all previously explored subspaces and attempt to match the waiting
      %   variables with those of the current subspace. 

      for j = 1:nh
         been_there = 1;
         if ( categorical )
            for i = rpath
               if ( ismember( xtype( i ), { 's','k','w' } ) &&                             ...
                    ( ( isnumeric( s_hist( j ).x{ i } ) &&                                 ...
                              abs( s_hist( j ).x{ i } - x0( i ) ) > eps ) ||               ...
                    ( ischar( s_hist( j ).x{ i } ) &&                                      ...
	                      ~strcmp( s_hist( j ).x{ i }, x0ref{ i } ) ) ) )
                   been_there = 0;
	           break;
               end
            end
         else
            for i = rpath
               if ( ismember( xtype( i ), { 'i', 'w' } ) &&                                ...
                              abs( s_hist( j ).x( i ) - x0( i ) ) > eps )
                   been_there = 0;
	           break;
               end
            end
         end

         %  This subspace has been explored already.  Move to best found so far.

	 if ( been_there )
            if ( categorical )
               x0ref = s_hist( j ).x;
               [ x0, cat_dictionnary, errx0 ] = bfo_numerify( x0ref, xtype, cat_dictionnary );
               if ( errx0 > 0 )
                  msg = [ ' BFO internal error: component ', int2str( errx0 ),             ...
		           ' of x0 of the wrong type on subspace restart. Terminating.' ];
                  if ( verbose )
                     disp( msg )
                  end
                  return
               elseif ( errx0 < 0 )
                  msg = [ ' BFO internal error in numerifying x0 after subspace move.',    ...
		           ' Terminating.' ];
                  if ( verbose )
                     disp( msg )
                  end
                  return
               end
	    else
               x0     = s_hist( j ).x;
	    end
            fxok      = 1;
            fbest     = s_hist( j ).fx;
            s_xincr   = s_hist( j ).xincr;
	    
            %  Restore the optimization context, if relevant.
	    
	    if ( dynamical )
	       xtype       = s_hist( j ).context.xtype;
	       xlower      = s_hist( j ).context.xlower;
	       xupper      = s_hist( j ).context.xupper;
	       opt_context = s_hist( j ).context;
               [ icont, idisc, icate, ifixd, ixed, iyed, ized, iwait, ired, ided, iked,    ...
                 ncont, ndisc, ncate, nfixd, nxed, nyed, nzed, nwait, nred, nded, nked,    ...
	         sacc, Q ] = bfo_switch_context( xtype, npoll, verbose );
	    end
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
         if ( ncont == 0  || max( xincr( icont, cur ) - s_xincr( icont ) ) >= 0 )
            msg   = 'This subspace has been explored already.';
            if ( verbose > 3 )
               disp( msg)
            end
            xbest = bfo_pack_x( x0, xscale, xtype, cat_dictionnary, x0ref);
            return
         else
            xincr( icont, cur ) = s_xincr( icont, cur );
         end
      end

%  On restart, start the iteration with the best value saved so far.

   elseif ( nh > 0 )
      fbest = Inf;
      for j = nh:-1:1
         if ( s_hist( j ).fx <= fbest )
            if ( categorical )
               x0ref = s_hist( j ).x;
	       xu    = x0ref;
               [ x0, cat_dictionnary, errx0 ] = bfo_numerify( x0ref, xtype, cat_dictionnary );
               if ( errx0 > 0 )
                  msg = [ ' BFO error: component ', int2str( errx0 ),                      ...
		           ' of x0 of the wrong type after restart. Terminating.' ];
                  if ( verbose )
                     disp( msg )
                  end
                  return
               elseif ( errx0 < 0 )
                  msg = [ ' BFO internal error in numerifying x0 after restart.',          ...
		           ' Terminating.' ];
                  if ( verbose )
                     disp( msg )
                  end
                  return
               end
	    else
               x0           = s_hist( j ).x;
               xu           = x0;
	    end
            fxok            = 1;
            fbest           = s_hist( j ).fx;
            xincr( :, cur ) = s_hist( j ).xincr;
	    
            %  Restore the optimization context, if relevant.
	    
	    if ( dynamical )
	       xtype       = s_hist( j ).context.xtype;
	       xlower      = s_hist( j ).context.xlower;
	       xupper      = s_hist( j ).context.xupper;
	       opt_context = s_hist( j ).context;
               [ icont, idisc, icate, ifixd, ixed, iyed, ized, iwait, ired, ided, iked,    ...
                 ncont, ndisc, ncate, nfixd, nxed, nyed, nzed, nwait, nred, nded, nked,    ...
	         sacc, Q ] = bfo_switch_context( xtype, npoll, verbose);
	    end
         end
      end

      %  Compute the associated grid size.

      if ( ncont > 0 )
         cmesh = max( xincr( icont, cur ) );
      else
         cmesh = sqrt( 10 * eps ) / eta;
      end

      %  Disable the forward/backward loop if all continuous variables have 
      %  converged already.

      if ( ncont > 0 && max( xincr( icont, cur ) - epsilon * ones( ncont, 1 ) ) <= 0   )
         fbloop = 0;
      end

   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Analyze the coordinate partially-separable structure, if any %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  See if the user has specified a coordinate-partially-separable structure.

if ( ~isempty( eldom ) )
   max_domain_dim = 0;
   for iel = 1:length( eldom )
      max_domain_dim   = max( max_domain_dim, length( eldom{ iel } ) );
   end
   partially_separable = ( max_domain_dim < nactive ) && use_cps > 0;
   explicit_domains    = 1;

   %  Fix all variables not occurring explictly in eldom.

   if ( partially_separable && ~isempty( not_in_eldom ) )
      wrn{ end+1 } = [ ' BFO warning: variables ', mat2str( not_in_eldom ),                ...
                       ' not found in eldom. Fixing them.' ];
      if ( verbose )
         disp( wrn{ end } )
      end
      for ivj = 1:length( not_in_eldom )
         iv = not_in_eldom( ivj );
         switch( xtype( iv ) )
         case 'c'
            ncont = ncont - 1;
            icont( find( iv == icont ) ) = [];
         case 'i'
            ndisc = ndisc - 1;
            idisc( find( iv == idisc ) ) = [];
         case 's'
            ncate = ncate - 1;
            icate( find( iv == icate ) ) = [];
         case 'r'
            nred  = nred  - 1;
            ired( find( iv == ired ) ) = [];
         case 'd'
            nded  = nded  - 1;
            ided( find( iv == ided ) ) = [];
         case 'k'
            nked  = nked  - 1;
            iked( find( iv == iked ) ) = [];
         case 'f'
         otherwise
            disp( [ ' BFO internal error: unknown type ', xtype( iv ) ] );
            return
         end
      end
      xtype( not_in_eldom )  = 'f';
   end

else
   partially_separable = 0;
   explicit_domains    = 0;
end

%  For now, ignore coordinate-partially-separable structure if there are categorical variables
%  with dynamic neighbourhoods.

if ( partially_separable && dynamical )
   wrn{ end+1 } = [ ' BFO warning: partially separable structure is present but ignored',  ...
                    ' because of dynamic categorical neighbourhoods.' ];
   if ( verbose )
     disp( wrn{ end } )
   end
   partially_separable = 0;
end

%  A coordinate partially-separable structure has been detected: print it and analyze it.

if ( partially_separable )


   if ( verbose > 3 )
      disp(   ' *** Coordinate partially-separable structure detected: ' )
      disp( [ ' Found ', int2str( n_elements ),                                            ...
              ' element functions of domains of size at most ', int2str(max_domain_dim),'.']);
      if ( verbose > 4 )
	 for j = 1:n_elements
	    fprintf( '  Element %3d involves variable(s) ', j );
	    for k = 1:length( eldom{ j } )
	       fprintf( ' %3d', eldom{ j }( k ) );
	    end
	    fprintf( '\n' )
	 end
      end 
   end

   %  Analyze the coordinate partially separable structure specified by eldom for the active 
   %  variables, to produce n_groups groups (xgroups) of sets (xsets) of active 
   %  variables, each involving certain elements functions (esets). Two sets of active 
   %  variables in the same group involve disjoint ensembles of elements (and can 
   %  therefore be used in parallel).

   active = [ icont, idisc, icate, ired, ided, iked ];
   [ n_groups, xgroups, n_sets, xsets, esets, xinel ] =                                    ...
                          bfo_analyze_cps_structure( n, n_elements, eldom, active, verbose );

   partially_separable = ( n_sets > 1 );

   if ( verbose > 3 )
      max_sub_dim = 0;
      for numset = 1:n_sets
         max_sub_dim = max( max_sub_dim, length( xsets{ numset } ) );
      end
      disp( [ ' The CPS structure has ', int2str( n_sets ),                                ...
              ' subspaces of dimension at most ', int2str( max_sub_dim ), ' in ',          ...
              int2str( n_groups ), ' groups.' ] )
   end

   %  If the problem is multilevel, the variables at level higher that 1 have been
   %  ignored in the analysis.  One still needs to to determine the list of elements
   %  involving variables of level > 1.

   if ( multilevel && partially_separable )
      high_level_elts = [];
      for iel = 1:n_elements
          for ivar = eldom{ iel }
             if ( xlevel( ivar ) > level )
                high_level_elts = union( high_level_elts, iel, 'stable' );
             end
          end
      end
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                            %
%                Phase 2: Apply the BFO algorithm to the verified problem.                   %
%                                                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Set up the initial basis of continuous directions (and a default for other types).
%   Note that Q is not used as a full matrix for CPS problems, and therefore not created 
%   because of its potentially very large size.

if ( ncont > 0 )
   sacc = [];
   if ( partially_separable )
      Q = eye( ncont, min( ncont, npoll ) );
   else
      Q     = eye( ncont, ncont );
      npoll = ncont;
   end
else
   Q    = 1;
end

%  Set flags if the current minimization uses one of the training performance functions.

within_average_training       = strcmp( shfname, 'bfo_average_perf' );
within_robust_training        = strcmp( shfname, 'bfo_robust_perf'  );
within_profile_training       = strcmp( shfname, 'bfo_dpprofile_perf' );
if ( within_profile_training )
  within_perfprofile_training = strcmp( training_strategy, 'perfprofile' );
  within_dataprofile_training = strcmp( training_strategy, 'dataprofile' );
end
within_training               = within_average_training  || within_robust_training  ||     ...
                                within_profile_training;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Print header, optimization direction, objective function's name and starting point value.  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( verbose > 1 && ~within_training )
   if ( depth == 0 ) 
      if ( level == 1 && ~train && use_cps >= 0 )
         bfo_print_banner( this_version );
      else
         fprintf( '\n')
         fprintf( '%s ********************************************************\n', indent)
         fprintf( '\n')
      end
      if ( maximize )
         fprintf( '%s Maximizing %s', indent, shfname )
      else
         fprintf( '%s Minimizing %s', indent, shfname )
      end
      if ( multilevel  || use_cps < 0 )
         if ( multilevel )
            if ( length( vlevel{ level } ) == 1 )
               disp( [ ' on variable ', int2str( vlevel{ level } ),                        ...
                       ' (level ',int2str( level ),')' ] )
            else
               disp( [ ' on variables ', int2str( vlevel{ level } ),                       ...
                       ' (level ',int2str( level ),')' ] )
            end
         end
         if ( use_cps < 0 )
            disp( [ ' on subset ', int2str( -use_cps ) ] )
         end
      end
      if ( partially_separable )
         fprintf( '\n%s  (using %4d sets of independent variables in %4d groups)\n',       ...
                  indent, n_sets, n_groups );
      end
      fprintf( '\n')
   end

   if ( fxok && verbose >= 3 )          % the starting point is not 
                                        % that specified on input
      if ( depth > 0 )
         disp( [ indent, '  Switching to a better starting point.' ] )
      else
         disp( [ indent, '  Restarting from a previous run.' ] )
      end
      fprintf( '\n')
   end

end

%  Build and store the optimization context, which is a struct containing 
%  the vectors which are user-updatable via dynamical categorical
%  neighbourhoods (variable's types and bounds).

if ( categorical )
   opt_context = struct( 'xtype', xtype, 'xlower', xlower, 'xupper', xupper );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Evaluate the objective function at the initial point (which is the best
%   point so far).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xbest = x0;
if ( ~fxok  || sum_form )

   %  Multilevel case: evaluate the objective function by (recursively) performing 
   %  optimization on the next level.

   if ( multilevel && level < nlevel )
      checking = level > 1 && term_basis > 1;
      
      if ( categorical )
         xbest = { bfo_cellify( xbest, xtype, cat_dictionnary, x0ref ) };
      end

      [ xbest, fbest, msglow, ~, neval, f_hist, xincr, el_hist ] =                         ...
          bfo_next_level_objf( level, nlevel, xlevel, neval, f, xbest, checking,           ...
                               f_hist, el_hist, xtmulti, xincr, xscale, xlower, xupper,    ...
                               eldom, max_or_min, vb_name, epsilon, bfgs_finish, maxeval,  ...
                               verbosity, fcallt, alpha, beta, gamma, eta, zeta, inertia,  ...
                               searchtype, rseed, iota, kappa, lambda, mu, term_basis,     ...
                               latbasis, idall, reset_random_seed, ssfname, cn_name,       ...
                               cat_states );
      if ( verbose >= 2 && blank_line )
         disp( ' ' )
      end
                     
      %  Return if an error occurred down in the recursion.

      if ( length( msglow ) >= 10 && strcmp( msglow(1:10), ' BFO error' ) )
         msg = msglow;
         if ( ~isempty( xscale ) )
            xbest = xscale.*xbest;
         end

         % Possibly call for cleanup in the search-step function.

         if ( search_step_needs_cleaning_up ) 
            bfo_search_step_cleanup( bfo_srch, categorical, sum_form);
         end

         return;
      end

      %  If the objective is in sum form, recover the values of the element 
      %  functions at the evaluated point.

      if ( sum_form )
         for iel = 1:n_elements
            fibest( iel ) = el_hist{ iel }.fbest;
         end
      end

      % Re-numerify xbest, if the problem has categorical variables

      if ( categorical )
         xbest = bfo_numerify( xbest{ 1 }, xtype, cat_dictionnary );
      end

   %  Evaluate the (single level) objective function.
       
   else

      %  Unscale xbest before evaluation, if it has been scaled by BFO.
      
      if ( isempty( xscale ) )
         xu = xbest;
      else
         xu = xscale.*xbest;
      end

      %  Transform xu to a vector state, if the problem contains categorical variables

      xp = xu;
      if ( categorical )
         xu = { bfo_cellify( xp, xtype, cat_dictionnary, x0ref ) };
      end

      %  The objective function value has been supplied by the user.

      if ( ~isempty( fx0 ) && norm( xp - numx0 ) < 1e-14 ) 
         if ( sum_form  )
            fibest = fx0;
            fbest  = sum( fibest );
         else
            fbest  = fx0;
         end

      %  The objective function is the (internal) function used for "average" training.

      elseif ( within_average_training )
         [ fbest, msgt, wrnt] = f( xu, fbound );
         if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ))
            if ( isempty( xscale ) )
              xbest = xu;
            else
              xbest = xscale .* xu;
            end
                       
            % Possibly call for cleanup in the search-step function.

            if ( search_step_needs_cleaning_up ) 
               bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
            end

            return
         end
         nevalt = fbest;
         [ f_hist, x_hist, neval ] = bfo_ehistupd( f_hist, x_hist, l_hist, neval, fbest,   ...
                                                   xu, categorical );
         training_history = [ training_history; [ 1, 0, nevalt, fbest, xbest' ] ];
			      

      %  The objective function is the (internal) function used for "robust" training.

      elseif ( within_robust_training )
         [ fbest, msgt, wrnt, t_neval ] = f( xu, fbound );
         if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ))
            if ( isempty( xscale ) )
              xbest = xu;
            else
              xbest = xscale .* xu;
            end

            % Possibly call for cleanup in the search-step function.

            if ( search_step_needs_cleaning_up ) 
               bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
            end

            return
         end
         nevalt = t_neval;
         [ f_hist, x_hist, neval ] = bfo_ehistupd( f_hist, x_hist, l_hist, neval, fbest,   ...
                                                   xu, categorical );
         training_history = [ training_history; [ 3, 0, nevalt, fbest, xbest' ] ];

      %  The objective function is the (internal) function used for "profile" training.

      elseif ( within_profile_training )
         [ fbest, msgt, wrnt, t_neval, fstar, tpval ] = f( xu, fstar, tpval );
         if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ))
            if ( isempty( xscale ) )
              xbest = xu;
            else
              xbest = xscale .* xu;
            end

            % Possibly call for cleanup in the search-step function.

            if ( search_step_needs_cleaning_up ) 
               bfo_search_step_cleanup(bfo_srch,categorical,sum_form );
            end

            return

         end
         nevalt = t_neval;
         [ f_hist, x_hist, neval ] = bfo_ehistupd( f_hist, x_hist, l_hist, neval, fbest,   ...
                                                   xu, categorical );
         training_history = [ training_history; [ 2, 0, nevalt, fbest, xbest' ] ];

      %  The objective function is that specified by the user in sum form.

      elseif ( sum_form )
         if ( withbound )
            [ fbest, fibest, nevali ] = bfo_sum_objf( xu, f, elset, eldom,                 ...
	                                              fbound, maximize );
         else
            [ fbest, fibest, nevali ] = bfo_sum_objf( xu, f, elset, eldom );
         end
         [ f_hist, x_hist, neval, el_hist, ev_hist ] = bfo_ehistupd( f_hist, x_hist,       ...
                                 l_hist, neval, fbest, xu, categorical, el_hist,           ...
	     		         fibest, eldom, nevali, ev_hist );

      %  The objective function is that specified by the user in simple form

      else
         if ( withbound )
            fbest = f( xu, fbound );
         else
            fbest = f( xu );
         end
         [ f_hist, x_hist, neval ] = bfo_ehistupd( f_hist, x_hist, l_hist, neval, fbest,   ...
                                                   xu, categorical );
      end

      if ( verbose >= 10 )
         f_hist  = f_hist
	 x_histp = x_hist'
	 neval   = neval
	 if ( sum_form )
	    nfcalls = length( f_hist )
	    for iel = 1:n_elements
	       iel
	       el_hist{ iel }
	    end
	 end
      end
      
   end

   %  Check for undefined function value.

   if ( isnan( fbest ) )
      fbest = Inf;
   end

   %  Check if BFO should terminate because the objective function's target
   %  has been reached.

   if ( (  maximize && fbest >= ftarget )  || ( ~maximize && fbest <= ftarget ) )
      msg = ' The objective function target value has been reached. Terminating.';
      if ( verbose )
         disp( msg )
      end
      xbest = bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref );

      % Possibly call for cleanup in the search-step function before termination.

      if ( search_step_needs_cleaning_up ) 
         bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
      end

      return
   end
end

%  Check there are active variables at the current level.  Return if not.

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
   xbest = bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref );

   % Possibly call for cleanup in the search-step function.

   if ( search_step_needs_cleaning_up ) 
      bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
   end

   return
end

%  Print zero-th iteration summary.

if ( verbose > 1 )
   if ( within_training )
      if ( verbose > 2 )
         fprintf( '\n' );
      end
      fprintf( '%s train    prob.obj.     training       training\n', indent );
      fprintf( '%s neval     neval       performance      cmesh      status\n', indent );
      fprintf( '%s%5d  %11d   %+.6e  %4e\n', indent, neval, nevalt, fbest, cmesh );
      if ( verbose > 2 )
         fprintf( '\n' );
      end
   else
      if ( verbose > 3 )
         fprintf( '\n' );
      end
      if ( partially_separable )
         fprintf( '%s neval        fx       nimp neltry    cmesh       ig:iel\n', indent );
      else
         fprintf( '%s neval        fx        est.crit       cmesh       status\n', indent );
      end
      fprintf( '%s%5d  %+.6e                %4e\n',                                        ...
                                                indent, ceil( neval ), fbest, cmesh );
      if ( verbose > 3 )
         fprintf( '\n' );
      end
   end
   bfo_print_x( indent, 'x0', xbest, xscale, verbose, xtype, cat_dictionnary, x0ref )
end

%  Save information for possible restart, if requested.

if ( savef > 0 && mod( ceil( neval ), savef ) == 0 && ~restart )
   s_hist  = bfo_shistupd( s_hist, idc, xbest, fbest, xincr( :, cur ), xtype,              ...
                           cat_dictionnary, x0ref, cn_name, opt_context );
   savok = bfo_save( sfname, shfname, maximize, epsilon, ftarget, maxeval, neval, f_hist,  ...
                     xtype, xincr, xscale, xlower, xupper, verbose, alpha, beta, gamma,    ...
                     eta, zeta, inertia, stype, rseed, iota, kappa, lambda, mu, term_basis,...
                     use_trained, s_hist, latbasis, bfgs_finish, training_history,  fstar, ...
                     tpval, ssfname, cn_name, cat_states, cat_dictionnary );
   if ( ~savok )
      msg = [ ' BFO error: checkpointing file ', sfname',                                  ...
              ' could not be opened. Skipping checkpointing at ',                          ...
              int2str( ceil( neval ) ), ' evaluations.' ];
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
   xbest = bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref );

   % Possibly call for cleanup in the search-step function.

   if ( search_step_needs_cleaning_up ) 
      bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
   end

   return;
end

%  Record the initial function value in the history, unless supplied by the user.

if ( ~user_fhist )
   f_hist = fbest;
end

%  Record the initial point as best point in the element-wise history.

if ( sum_form && ( ~multilevel  || level == nlevel ) && ( ~fxok  || restart ) )
   for iel = 1:n_elements
      if ( explicit_domains )
         if ( categorical )
            xtry = {{ xu{ 1 }{ eldom{ iel } } }};
         else
            xtry = xbest( eldom{ iel } );
         end
      else
         xtry = xbest;
      end
      el_hist{ iel }.fbest = fibest( iel );
   end
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

%   Initialize the current and previous iterates, the number of successive
%   refinement steps and the subspace successive increments.

x       = xbest;                       % the iterate
fx      = flow;
xp      = x;                           % the previous iterate
fxp     = flow;                        % the function value at the previous iterate
if ( partially_separable )
   nrefine  = zeros( 1, n_sets ); 
   optstage = Inf*ones( 1, n_sets );   % This vector is meant to contain the increment
                                       % already used to optimize in the different subsets
                                       % from the same point.
else
   nrefine  = 0;
end                                    % the number of successive refinements
iqn     = 0;                           % no quasi-Newton matrix yet

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%         THE OPTIMIZATION       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

checking = 0;                          % no checking for optimality yet

%%%%%%%%%%%%%%%%%%%%%%%%  This the main iteration's loop  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for itg = 1:maxeval

   %  Call the user tracking function, if specified, and terminate cleanly if termination is
   %  requested.


   if ( use_tracker && topmost && use_cps >= 0 &&                                          ...
        bfo_trf( bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref ),               ...
                 fbest, xlower, xupper, neval )                                  )
      msg = [' Terminating at user request in ', trfname, '.' ];
      if ( verbose && level == 1 )
         disp( msg )
      end
      if ( savef >= 0 )
         s_hist  = bfo_shistupd( s_hist, idc, xbest, fbest, xincr( :, cur ), xtype,        ...
                          cat_dictionnary, x0ref, cn_name, opt_context );
         savok   = bfo_save( sfname, shfname, maximize,  epsilon, ftarget,                 ...
                             maxeval, neval, f_hist, xtype,  xincr,                        ...
                             xscale, xlower, xupper,  verbose, alpha, beta,                ...
                             gamma, eta, zeta, inertia, stype, rseed, iota, kappa, lambda, ...
                             mu, term_basis, use_trained, s_hist, latbasis, bfgs_finish,   ...
                             training_history, fstar, tpval, ssfname, cn_name, cat_states, ...
                             cat_dictionnary );
         if ( ~savok )
            msg = [ ' BFO error: checkpointing file ', sfname',                            ...
                    ' could not be opened. Skipping checkpointing at ',                    ...
                     int2str( ceil( neval ) ), ' evaluations.' ];
            if ( verbose )
               disp( msg )
            end
         end
      end
      xbest = bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref );

      % Possibly call for cleanup in the search-step function.

      if ( search_step_needs_cleaning_up ) 
         bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
      end

      return;
   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  User-defined SEARCH STEP  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   successful_search_step = 0;
   at_optimizer           = 0;         %  Seemingly not yet at a optimizer!

   if ( ~isempty( ssfname ) && ~isempty( x_hist ) )

      %  Build the summary vector of variable types for use within the search-step.

      xts = xtype;
      if ( nlevel > 1 || ndisc )
         xts( find( ismember( xts, { 'w', 'x', 'y', 'z' } ) ) ) = 'f';
      end

      %  Possibly print data on entry of the user searc-step routine.

      if ( verbose >= 4 )
         disp( ' ' )
         disp( [ ' Calling the user-supplied search function ', ssfname ] )
         if ( verbose >= 10 )
            x_hist = x_hist
            if ( categorical )
               lfh = length( f_hist ) - length( x_hist ) + 1;
            else
               lfh = length( f_hist ) - size( x_hist, 2 ) + 1;
            end
            f_hist( lfh:end )
            xts
            xlower
            xupper
            latbasis
            if ( sum_form )
               for iel = 1:n_elements
                   el_hist{ iel }.fel
                   el_hist{ iel }.xel
               end
            end
         end
      end

      %  Call the user search function with categorical variables after
      %  truncating f_hist to make it correspond to x_hist.

      l_hist_prev = length( f_hist );
      if ( categorical )
         lfh = length( f_hist ) - length( x_hist ) + 1;
         if ( lfh > 1 )
            f_hist_old = f_hist( 1:lfh-1 );   % Save the part of f_hist not sent to the user.
         end
         if ( sum_form )
            [ xtry, fisearch, nevalss, exc, x_hist, f_hist, xtss2, xlss, xuss, el_hist ] = ...
                    bfo_srch( level, {f},                                                  ...
                              bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref),   ...
                              max_or_min, xincr( 1:n, cur ), x_hist, f_hist( lfh:end ),    ...
                              xts, xlower, xupper, latbasis, {cat_states}, cn_name, el_hist );
            fsearch = sum( fisearch );
         else
            [ xtry, fsearch, nevalss, exc, x_hist, f_hist, xtss2, xlss, xuss ] =           ...
                    bfo_srch( level, f,                                                    ...
                              bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref),   ...
                              max_or_min, xincr( 1:n, cur ), x_hist, f_hist( lfh:end ),    ...
                              xts, xlower, xupper, latbasis, {cat_states}, cn_name );
         end
         if ( size( xlss, 2 ) > 1 )
            xlss = xlss';
         end
         if ( size( xuss, 2 ) > 1 )
            xuss = xuss';
         end

         %  If the history completed by the search-step is too long, truncate it.

         lxh = length( x_hist );
         if ( lxh > l_hist )
            x_hist = x_hist{ lxh-l_hist+1:lxh }; 
            if ( sum_form )
               for iel = 1:n_elements
                   el_hist{ iel }.fel = el_hist{ iel }.fel( lxh-l_hist+1:lxh );
                   el_hist{ iel }.xel = el_hist{ iel }.xel{ lxh-l_hist+1:lxh };
               end
            end
	 end

      %  Call the user search function without categorical variables after
      %  truncating f_hist to make it correspond to x_hist.
      
      else

         lfh = length( f_hist ) - size( x_hist, 2 ) + 1;
	 if ( lfh > 1 )
            f_hist_old = f_hist( 1:lfh-1 );   % Save the part of f_hist not sent to the user.
         end
         if ( sum_form )
            [ xtry, fisearch, nevalss, exc, x_hist, f_hist, el_hist ] =                    ...
                    bfo_srch( level, {f},                                                  ...
                              bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref),   ...
                              max_or_min, xincr( 1:n, cur ), x_hist, f_hist( lfh:end ),    ...
                              xts, xlower, xupper, latbasis, el_hist );
            fsearch = sum( fisearch );
         else
            [ xtry, fsearch, nevalss, exc, x_hist, f_hist ] =                              ...
                    bfo_srch( level, f,                                                    ...
                              bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref),   ...
                              max_or_min, xincr( 1:n, cur ), x_hist, f_hist( lfh:end ),    ...
                              xts, xlower, xupper, latbasis );
         end
         if ( size( xtry, 2 ) > 1 )
            xtry = xtry';
         end

         %  If the history complete by the search-step is too long, truncate it.

	 lxh   = size( x_hist, 2 );
	 if ( lxh > l_hist )               
	    x_hist = x_hist( 1:n,lxh-l_hist+1:lxh ); 
         end
         if ( sum_form )
            for iel = 1:n_elements
               lelh = length( el_hist{ iel }.fel );
               if ( lelh > l_hist )
                   el_hist{ iel }.fel = el_hist{ iel }.fel( lelh-l_hist+1:lelh );
                   el_hist{ iel }.xel = el_hist{ iel }.xel( :, lelh-l_hist+1:lelh );
               end
            end
	 end
      end

      %  Reconstruct the complete f_hist by concatenating the part that was not sent to
      %  the user with that resulting from the search step.

      if ( lfh > 1 )
         f_hist = [ f_hist_old f_hist ];
      end
      if ( ~isempty( ev_hist ) )
         ev_hist = [ ev_hist,                                                              ...
                    (ev_hist( end )+max(0,nevalss))*ones( 1, length( f_hist)-l_hist_prev ) ];      
      end

      %  Update the total number of objective function evaluations.

      neval  = neval + nevalss;

      %  Interpret exc, the exit code from the search-step function.

      switch ( exc )

      case -3      %  Termination requested by user

         msg = ' Optimization terminated by the user.';
         if ( verbose )
            disp( msg )
         end
         xbest = bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref);

         % Call for cleanup in the search-step function.

         bfo_search_step_cleanup( bfo_srch, categorical, sum_form );

         return

      case { -1, -2, 1 }  % Unsuccessful returns (various causes).

         successful_search_step = 0;
         if ( verbose >= 4 )
            disp( [ ' Unsuccessful return from ', ssfname ] )
         end

      case 0              %  Search point found, but decision to use it up to BFO

         if     (  maximize && fsearch > fbest + eta*cmesh^2 ) 
            successful_search_step = 1;
         elseif ( ~maximize && fsearch < fbest - eta*cmesh^2 )
            successful_search_step = 1;
         end

      case 2              %  Successful return with significant improvement

         %  Move to ( xsearch, fsearch ).
         %  Note that this involves setting fbest (the best value irrespective of min/max)
         %  and flow (the smallest value after sign change for the max).
         %  The index ibest is also set to zero, in order to identify successful
         %  search steps in the one-line iteration summary (showing as +0 in the status
         %  column).

         successful_search_step = 1;
         ibest                  = 0;

      case 3             %  Seemingly at optimizer, terminate cleanly.

         successful_search_step = 0;
         at_optimizer           = 1;

      end

      %  Manage the successful returns (exc = 0 or exc = 2)
       
      if ( successful_search_step )

         if (  maximize ) 
            flow = -fsearch;
         else
            flow =  fsearch;
         end

         fbest = fsearch;
         if ( sum_form )
            fibest = fisearch;
         end

         if ( dynamical )

            %  Update optimization context if necessary.
	 
            xlss = max( [ xlss'; -myinf * ones( 1, n ) ] )';
            xuss = min( [ xuss';  myinf * ones( 1, n ) ] )';
            if ( ~strcmp( xts, xtss2 ) || norm( xlower - xlss ) || norm( xupper-xuss ) )
               xtype              = xtss2;
	       opt_context.xtype  = xtype;
               if ( isempty( xscale ) )
  	          xlower = xlss;
	          xupper = xuss;
               else
                  finite = intersect( icd, find( xlss > myinf ) );
                  xlower( finite ) = xlss( finite ) ./ xscale( finite );
                  finite = intersect( icd, find( xuss < myinf ) );
                  xupper( finite ) = xuss( finite ) ./ xscale( finite );
               end
	       opt_context.xlower = xlower;
	       opt.context.xupper = xupper;
               [ icont, idisc, icate, ifixd, ixed, iyed, ized, iwait, ired, ided, iked,    ...
                 ncont, ndisc, ncate, nfixd, nxed, nyed, nzed, nwait, nred, nded, nked,    ...
	         sacc, Q ] = bfo_switch_context( xtype, npoll, verbose );
  	    end

            %   Move to the best point (categorical)

	    [ xbest, cat_dictionnary, err ] = bfo_numerify( xtry{ 1 }, xtype,              ...
                                                              cat_dictionnary );
            if ( err > 0 )
               msg = [ ' BFO error: component ', int2str( err ),                           ...
	               ' of xsearch of the wrong type. Terminating.' ];
               if ( verbose )
                  disp( msg )
               end
            end

            if ( err )

               % Possibly call for cleanup in the search-step function.

               if ( search_step_needs_cleaning_up ) 
                  bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
               end
               return

            end

         %  Move to the best point (numerical)

         else 
            xbest = xtry;
         end

         %  Rescale xbest, if necessary.

         if ( ~isempty( xscale ) )
            xbest( icd ) = xbest( icd ) ./ xscale( icd );
         end

         %  Record the best point elementwise, if necessary.

         if ( ~isempty( el_hist ) )
            for iel = 1:n_elements
               el_hist{ iel }.fbest = fibest( iel );
	    end
         end
	    
	 ibest = 0;
         if ( verbose >= 4 )
            disp( [ ' Successful return from ', ssfname ] )
         end

      end

%     Terminate if the maximum number of function evaluations has been reached.

      if ( neval >= maxeval )
         msg = [' Maximum number of ', int2str( maxeval ), ' evaluations of ', shfname,    ...
                ' reached.' ];
         if ( verbose && level == 1 )
            disp( msg )
         end
         if ( savef >= 0 )
            s_hist  = bfo_shistupd( s_hist, idc, xbest, fbest, xincr( :, cur ), xtype,     ...
	                            cat_dictionnary, x0ref, cn_name, opt_context );
            savok   = bfo_save( sfname, shfname, maximize,  epsilon, ftarget, maxeval,     ...
                                neval, f_hist, xtype,  xincr, xscale, xlower, xupper,      ...
                                verbose, alpha, beta, gamma, eta, zeta, inertia, stype,    ...
                                rseed, iota, kappa, lambda, mu, term_basis, use_trained,   ...
                                s_hist, latbasis, bfgs_finish, training_history, fstar,    ...
                                tpval, ssfname, cn_name, cat_states, cat_dictionnary );
            if ( ~savok )
               msg = [ ' BFO error: checkpointing file ', sfname',                         ...
                       ' could not be opened. Skipping checkpointing at ',                 ...
                        int2str( ceil( neval ) ), ' evaluations.' ];
               if ( verbose )
                  disp( msg )
               end
            end
         end
         xbest = bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref );

         % Possibly call for cleanup in the search-step function.

         if ( search_step_needs_cleaning_up ) 
            bfo_search_step_cleanup( bfo_srch, categorical, sum_form )
         end

         return;
      end

   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%% Coordinate-partially-separable POLL LOOP %%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   if ( partially_separable && checking >= 0 && ~successful_search_step && ~at_optimizer )

      %  Redefine nrefine after a checking loop.

      if ( nrefine == -1 )
         nrefine = -ones( 1, length( esets ) );
      end

      %  Loop on the independent groups of variables

      loopbreak  = 0;                  % indicator to force breaking the groups' loop
      if ( verbose )
         gbest   = 0;                  % index of successful group
         sbest   = 0;                  % index of successful set within group
         impbest = 0;                  % the best set improvement
      end
      nimp       = 0;                  % the number of successful independent sets
      ntry       = 0;                  % the number of subsets attempted

      zgroup     = [ 1:n_groups ] + itg;
      for igroup = ones( 1, n_groups ) + mod( zgroup, n_groups * ones( 1, n_groups ) )

         %  Loop on the variables' sets of the igroup-th group.  Remember that two such
         %  variables' sets involve disjoint ensemble of elements, and thus that the
	 %  decreases obtained in the corresponding partial objectives can be summed.

         %  ftry and fitry contain the *complete* function value, that is the sum of
         %  all element functions (in all sets/groups).

         ftry     = fbest;
         fitry    = fibest;
	 xtry     = xbest;
         impgroup = 0;

         for iset = xgroups{ igroup }

            %  **************************************************************************
            %  ******* Ideally, this loop on iset should be executed in parallel. *******
            %  **************************************************************************

            %  Obtain a decrease for the partial objective by taking a step in
	    %  the variables of xsets{ iset } with elements in esets{ iset }.

            xset  = xsets{ iset };
            if ( multilevel && level < nlevel && ~isempty( high_level_elts ) ) 
               elsetset = union( esets{ iset }, high_level_elts, 'sorted' );
            else
               elsetset = esets{ iset };
            end
	    nset    = length( xset );
            l_elset = length( elsetset );
	    
            %  Build the partial objective function for the current set of variables
	    %  (given by xsets{ iset }) and, possibly, the variables of a higher level.
            %  This partial objective is the set of elements indexed by elset.
            %  Also construct (in fxset) the complete set of variables involved in 
            %  the elements of elsetset, including those that are not included in 
            %  xset{ iset } (they are needed  because one cannot call the element 
            %  function within subset optimization without giving those variables a value).
            %  Finally, construct the set of current element values for the element 
            %  functions belonging to the partial objective function.

	    fset       = cell( 1, l_elset );
	    fisetval   = zeros( 1, l_elset );
            fxset      = [];
    	    for elidx  = 1:l_elset
	       iel                  = elsetset( elidx );
	       fset{ elidx }        = f{ iel };
               fxset                = [ fxset, eldom{ iel } ]; 
               fisetval( elidx )    = fibest( iel );
	    end
            fxset = unique( fxset );
            l_fxset     = length( fxset );
            cost_ratio  = l_elset / n_elements;

            %  Compute the set of continuous variables at the current level for the
            %  subset and its size.
           
            xsetcont = [];
            for i = 1:nset
              xseti = xset( i );
              if( xtype( xseti ) == 'c' )
                 xsetcont = [ xsetcont, xseti ];
              end
            end
            if ( multilevel )
               xsetcont = intersect( vlevel{ level }, xsetcont ); 
            end
            lxsetcont   = length( xsetcont );

            %  Determine the minimal stepsize for active continuous variables in the subset.

            minincset  = min( xincr( xsetcont, cur ) );

            %  If there are active continuous variable for the subset optimization, 
            %  make sure their increments are not all larger than that already used
            %  previously for optimizing from the same subset point.

            if ( lxsetcont && optstage( iset ) <= minincset )
               onesetcont             = ones( 1, lxsetcont );        
               xincr( xsetcont, cur ) = max( [ epsilon * onesetcont                    ;
                                               beta * min( [ xincr( xsetcont, cur )'   ;
                                                             minincset * onesetcont ] ) ] )';
               nrefine( iset )   = nrefine( iset ) + 1;
               minincset         = min( xincr( xsetcont, cur ) );
            end

            %  Construct the variable's types for the subset optimization by freezing
            %  all variables in fxset which are not in xset (i.e. are not optimization
            %  variables).  Also compute the number of active optimization variables 
            %  for this subset, possibly including variables at higher levels.

	    xtypeset   = '';
            nsetactive = 0;
            for j = 1:l_fxset
               ivar = fxset( j );
	       if ( ismember( ivar, xset ) || ( multilevel && xlevel( ivar ) > level ) )
	          xtypeset( j ) = xtype( ivar );
                  switch( xtypeset( j ) )
                  case { 'c', 'i', 's' } 
                     nsetactive = nsetactive + 1;
                  end
	       else
                  if ( xtype( ivar ) == 's' )
	             xtypeset( j ) = 'k';
                  elseif ( xtype( ivar ) == 'f' )
	             xtypeset( j ) = 'f';
                  else
	             xtypeset( j ) = 'z';
                  end
	       end
      	    end

            %  Avoid performing subset optimization if the smallest increment is not smaller
            %  than that used previously. Also avoid reoptimizing for a subset in a checking
            %  loop if there is only one continuous variable and a checking loop has already
            %  been performed within subsets, which is detected by the fact that 
            %  checking >= 2 in this case.

            if  ( ( nsetactive > lxsetcont || optstage( iset ) > minincset ) &&           ...
                  ( checking < 2 || lxsetcont > 1 )                            )

               %  Build the element domains corresponding to the selected elements.

               eldom_set = cell( 1, l_elset );
    	       for elidx = 1:l_elset
                  [ ~, eldom_set{ elidx } ] = ismember( eldom{ elsetset( elidx ) }, fxset );
               end
               maxevalset = floor( ( maxeval - neval ) / cost_ratio );
               
               %  Printout the main objective of the current subset optimization.

               if ( verbose >= 10  )
	          fprintf( [ ' Partial objective for set %3d in group %3d\n',              ...
	                     '  composed of element(s)       ' ], iset, igroup );
                  for iel = 1:l_elset
	             fprintf( ' %3d', elsetset( iel ) );
	          end
	          fprintf( '\n  involving variable(s)        ' );
                  for j = 1:length( fxset )
	             fprintf( ' %3d', fxset( j ) );
	          end
	          fprintf( '\n  and to optimize on variable(s)' );
                  for j = 1:nset
	             fprintf( ' %3d', xset( j ) );
	          end
	          fprintf( '\n' );
	       end

               %  Define accuracy and termination for the subset optimization.

	       if ( checking )                     % require subset checking loops
	          termloops = term_basis;
                  epsset    = epsilon;
	       else
	          termloops = 0;
                  epsset    = max( cmesh, epsilon );
	       end


               %.............................................................................%
               %                                                                             %
               %  Optimization is the CPS subspace can be conducted in three different ways. %
               %  1) In the (frequent) case where:                                           %
               %           * there are no active discrete or categorical variables           %
               %           * the (continuous) variables are unscaled                         %
               %           * the element function calls are simple                           %
               %           * the optimization is not multilevel                              %
               %     the 'core version' of BFO (bfo_core) can be used, that is a version of  %
               %     BFO without many of the bells and whistles that are necessary to        %
               %     handle complicated cases.                                               %
               %  2) If, in addition to the above four conditions,                           %
               %           * the dimension of the CPS subspace is equal to 1,                %
               %     an even simpler code (bfo_min1d) can be used.                           %
               %  3) In all other cases, a recursive call to the full BFO is used.           %
               %                                                                             %
               %  As both specialized routines only handle minimization, maximization is     %
               %  taken care of by changing the sign of the objective function.              %
               %.............................................................................%

               use_core = ( lxsetcont == nset && isempty( xscale ) && ~categorical &&      ...
                            ~multilevel &&  strcmp( fcallt, 'simple' ) );

               if ( use_core && allow_core )

                  %  Obtain (in ovars) the positions in fxset of the variables in xset.

                  [ ~, ovars ] = ismember( xset, fxset );

                  if ( maximize )

                     %  Change the sign of the element functions and initial values.

                     for elidx = 1:l_elset
	                iel               = elsetset( elidx );
	                fset{ elidx }     = -f{ iel };
                        fisetval( elidx ) = -fibest( iel );
                     end

                  end

                  %  Call the (hopefully) faster minimizer.
                  %  Note that the optimal function value is not returned, as it would
                  %  only represent the elements in elset.  But the optimal values for
                  %  each of these elements is stored in el_hist{.}.fbest, from which the
                  %  complete function value can be reconstructed.

                  if ( nset == 1 && allow_min1d && ( n > 500 || user_cps ) )

                     %  Call the simple 1D minimizer.

                     warning off
                     [ xp, fxp1, msgs, nevalset, el_hist_set ] =                           ...
                              bfo_min1d( {fset}, xtry( fxset ), fisetval, ovars,           ...
                                          xlower( fxset ), xupper( fxset ),                ...
                                          xincr( xset, cur ), epsset, maxevalset,          ...
                                          elsetset, eldom_set, kappa, lambda, mu );
                     warning on
                  else

                     %  Call the core version of BFO.

                     [ xp, fxp2, msgs, nevalset, ~, ~, el_hist_set ] =                     ...
                               bfo_core( {fset}, xtry( fxset ), fisetval, ovars, epsset,   ...
                                         maxevalset, 'xlower', xlower( fxset ), 'xupper',  ...
                                         xupper( fxset ), 'eldom', eldom_set, 'elset',     ...
                                         elsetset, 'termination-basis', termloops, 'alpha',...
                                         alpha, 'beta', beta, 'gamma', gamma, 'delta',     ...
                                         minincset, 'eta', eta, 'inertia', inertia,        ...
                                         'indentation', indent );
                  end

                  %  Recover the correct element history for the elements which have 
                  %  been used (and possibly modified) by the call to bfo_core, and
                  %  update the element values for those elements. Also update the
                  %  element-wise values of the best objective function so far.

                  for elidx = 1:l_elset
                     iel = elsetset( elidx );                  % the original element index
                     cl  = length( el_hist{ iel }.fel );       % the length of the old history
                     nl  = length( el_hist_set{ elidx }.fel ); % the length of the new history

                     %  The sum of the lengths is within limits: just concatenate the
                     %  values and point histories.

                     if ( cl + nl <= l_hist )
                        if ( maximize )
	                   el_hist{iel}.fel = [ el_hist{iel}.fel -el_hist_set{elidx}.fel ];
                        else
	                   el_hist{iel}.fel = [ el_hist{iel}.fel  el_hist_set{elidx}.fel ];
                        end
	                el_hist{iel}.xel = [ el_hist{iel}.xel  el_hist_set{elidx}.xel ];

                     %  The new history is already too long, truncate it to replace the
                     %  old one.

                     elseif( nl >= l_hist )
                        if ( maximize )
                           el_hist{iel}.fel = -el_hist_set{elidx}.fel(nl-l_hist+1:nl);
                        else
                           el_hist{iel}.fel =  el_hist_set{elidx}.fel(nl-l_hist+1:nl);
                        end
	                el_hist{iel}.xel =  el_hist_set{elidx}.xel(:,nl-l_hist+1:nl);

                     %  The sum of the lengths is too large: update the histories to
                     %  contain the most recent ones but yet removing enough of the
                     %  older ones to avoid repeating this shuffle too often.

                     else
                        if ( nl >= cl )
                           svef = [];
                           svex = [];
                        else
                           brk = max( nl, floor( cl/2 ) );
                           svef = el_hist{iel}.fel(brk:cl);
                           svex = el_hist{iel}.xel(:,brk:cl);
                        end
                        if ( maximize )
                           el_hist{iel}.fel = [svef  -el_hist_set{elidx}.fel];                                          ...
                        else
                           el_hist{iel}.fel = [svef  el_hist_set{elidx}.fel];                                  ...
                        end
                        el_hist{iel}.xel = [svex  el_hist_set{elidx}.xel];
                     end

                     %  Update the element-wise memory of the best objective function
                     %  value so far.

                     if ( ~isempty( el_hist_set{ elidx }.fbest ) )
                        if ( maximize )
                           fitry( iel ) = -el_hist_set{ elidx }.fbest;
                        else
                           fitry( iel ) = el_hist_set{ elidx }.fbest;
                        end
                     end
                  end

               %.............................................................................%
               %                                                                             %
               %      The above conditions do not hold: the core minimization cannot be      %
               %      applied. Use a recursive call to the full BFO instead.                 %
               %.............................................................................%

               else

                  %  Construct the set element histories

                  el_hist_set = cell( 1, l_elset );
                  for elidx = 1:l_elset
	             el_hist_set{ elidx } = el_hist{ elsetset( elidx ) };
                  end

                  %  Define the levels for the subset optimization, if relevant, and
                  %  add to the information passed to the subset optimization everything
                  %  related to levels higher than the current.

                  if ( multilevel && level < nlevel )
                     xlevelset = xlevel( fxset );
                  else
                     xlevelset = [];
                  end

                  %  Define the subset scaling, if any.
	      
                  if ( isempty( xscale ) )
                     xscset = [];
                  else
                     xscset = xscale( fxset );
                  end

                  %  If categorical variables are present, construct their associated
                  %  cat_states and vector state xu. Otherwise, set cat_states to the
                  %  empty set and pass on the vector of set variables.

                  if ( categorical )

                     %  Construct the associated cat_states if the problem has categorical 
                     %  variables (or the empty one otherwise).

                     cat_states_set = cell( 1, l_fxset );
                     for j = 1:l_fxset
                        ivar = fxset( j );
                        cat_states_set{ j } = cat_states{ ivar };
                     end
                     xu = bfo_cellify( xtry, xtype, cat_dictionnary, x0ref );
                     xu = {{ xu{ fxset } }};
                  else
                     xu = xtry( fxset );
                     cat_states_set = {};
                  end

                  %  Then call BFO to optimize fset in variables indexed by xset on the
	          %  current mesh.
                  %  Note that the optimal function value is not returned, as it would
                  %  only represent the elements in elset.  But the optimal values for
                  %  each of these elements is stored in el_hist{.}.fbest, from which the
                  %  complete function value can be updated.

                  [ xp, ~, msgs, wrns, nevalset, ~, ~, ~, ~, ~, ~, ~, el_hist_set, ~ ]=    ...
                     bfo( {fset}, xu, 'fx0', fisetval, 'topmost', 'no', 'xscale',          ...
                      xscset, 'xtype', xtypeset, 'xlower', xlower(fxset),                  ...
                      'xupper', xupper(fxset),  'epsilon', epsset, 'maxeval', maxevalset,  ...
                      'verbosity', ps_verbosity, 'termination-basis', termloops,           ...
                      'f-call-type', fcallt, 'max-or-min', max_or_min, 'alpha', alpha,     ...
                      'beta', beta, 'gamma', gamma, 'eta', eta,'zeta', zeta, 'inertia',    ...
                      inertia, 'search-type', searchtype, 'random-seed', rseed, 'iota',    ...
                      iota, 'kappa', kappa, 'lambda', lambda, 'mu', mu,                    ...
                      'reset-random-seed', reset_random_seed, 'lattice-basis',             ...
                      latbasis,  'xincr', xincr(fxset,:), 'bfgs-finish', bfgs_finish,      ...
                      'cat-states', {cat_states_set}, 'element-hist', el_hist_set, 'elset',...
                      elsetset, 'eldom', {eldom_set}, 'use-cps', -iset, 'xlevel',xlevelset,...
                      'level', level );

                  %  Verify that this optimization succeeded. 

                  if ( length( msgs ) >= 10 && strcmp( msgs(1:10), ' BFO error' ) )
                     msg = [ ' BFO error: optimizing on subset ', int2str( iset ),         ...
                             ' returned the message:', msgs(12:end) ];
                     if ( tverbose )
                        disp( msg )
                     end

                     % Possibly call for cleanup in the search-step function.

                     if ( search_step_needs_cleaning_up ) 
                     bfo_search_step_cleanup( bfo_srch,categorical, sum_form );
                     end

                     return
                  end

                  if ( ~isempty( wrns ) && length( wrns{ 1 } ) >= 11 &&                    ...
	               strcmp( wrns{ 1 }(1:11), ' BFO warning' ) )
                     wrn{ end +1 } = [ ' BFO warning: training on problem ',               ...
                                       func2str( fun ), ' issued the warning:',            ...
                                       wrns{ 1 }(13,:) ];   
                     if ( verbose )
                        disp( wrn{ end } )
                     end
                  end


                  %  Update the best value so far in this set and recover the correct 
                  %  element history for the elements which have been used (and possibly
                  %  modified) by the call to BFO. Also update the element values for 
                  %  those elements.

                  for elidx = 1:l_elset
                     iel            = elsetset( elidx );
	             el_hist{ iel } = el_hist_set{ elidx };
                     fitry( iel )   = el_hist{ iel }.fbest;
	          end

               end
 
               %.............................................................................%

               ftry_prev = ftry;
               ftry      = sum( fitry );    %  Recompute ftry for improved accuracy

               %  Compute the improvement obtained by optimizing on xset.

               ntry      = ntry + 1;
               impset    = ftry_prev - ftry;
               impgroup  = impgroup + impset;

               if ( impset )
                  nimp  = nimp + 1;
               end

               if ( verbose >= 10 )
	          disp( [ ' Achieved improvement ', num2str( impset ),                     ...
	                  ' in ', num2str( nevalset*cost_ratio ), ' evaluations ',         ...
                          '( ftry: ', num2str( ftry_prev ), ' => ', num2str( ftry),' )' ] )
	       end

               %  Compute the associated full vector.

               if ( categorical )
                  xtry( fxset ) = bfo_numerify( xp{ 1 }, xtypeset, cat_dictionnary );
               else
                  xtry( fxset ) = xp;
               end

               %  Maintain the evaluation history.

               [ f_hist, x_hist, neval ] = bfo_ehistupd( f_hist, x_hist, l_hist, neval,    ...
                                                         ftry, xtry, categorical );

               costeval = nevalset*cost_ratio;
               neval    = neval + ( costeval - 1 );  % -1 corrects the wrong increment of 1 
                                                     %  in bfo_ehistupd

               if ( ~isempty( ev_hist ) )
                  ev_hist = [ ev_hist, ev_hist( end ) + costeval ];
               else
                  ev_hist = costeval;
               end

               %  Terminate with the best point found if the maximum number of evaluations
	       %  has been reached.

               time_to_exit = 0;
               if ( neval > maxeval - 1/n_elements )
                  msg = [ ' Maximum number of ', int2str( maxeval ), ' evaluations of ',   ...
                         shfname, ' reached.' ];
	          if ( verbose )
	             disp( msg )
	          end
                  time_to_exit = 1;
               end

               %  Save information for possible restart, if requested.

               if ( savef > 0 && ( mod( ceil( neval ), savef ) == 0  || time_to_exit ) )

                  s_hist  = bfo_shistupd( s_hist, idc, xbest, fbest, xincr( :, cur ),      ...
			               xtype, cat_dictionnary, x0ref, cn_name, opt_context );
                  savok   = bfo_save( sfname, shfname, maximize, epsilon, ftarget,         ...
                                   maxeval, neval, f_hist, xtype, xincr, xscale, xlower,   ...
                                   xupper, verbose, alpha, beta, gamma, eta, zeta,         ...
                                   inertia, stype, rseed, iota, kappa, lambda, mu,         ...
                                   term_basis, use_trained, s_hist, latbasis, bfgs_finish, ...
                                   training_history, fstar,  tpval, ssfname, cn_name,      ...
                                   cat_states, cat_dictionnary );
                  if ( ~savok )
                     msg = [ ' BFO error: checkpointing file ', sfname',                   ...
                             ' could not be opened. Skipping checkpointing at ',           ...
                              int2str( ceil( neval ) ), ' evaluations.' ];
                     if ( verbose )
                        disp( msg )
                     end
                  end
               end
 
               %  Terminate if the optimization on xset generated an error.

               if ( length( msgs ) > 10 && strcmp( msgs(1:10), ' BFO error' ) )
                  msg = [ ' BFO error: optimization of set ', int2str( iset ),' in group ',...
                          int2str( igroup ), ' returned an error message. Terminating.' ];
	          if ( verbose )
	             disp( msg )
	          end
	          time_to_exit = 1;
	       end

               %  Effectively terminate the minimization.

               if ( time_to_exit  )
                  if ( search_step_needs_cleaning_up ) 
                     bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
                  end
                  return;
               end

            %  The optimization of the variables of the iset-th independent set of 
            %  variables in the current group has been skipped.  Set the obtained 
            %  improvement to zero.

            else
                impset = 0;
            end

            %  Handle the mesh if there are continuous variables in the current set.

            if ( lxsetcont )

               %  Significant improvement is obtained: expand the relevant mesh.

               meshset    = max( xincr( xsetcont, cur ) );
               onesetcont = ones( 1, lxsetcont );        
               if ( impset > eta * meshset^2 )

                  xincr( xsetcont, cur )= min([ xupper( xsetcont )' - xlower( xsetcont )';
                                                alpha * xincr( xsetcont, cur )'          ;
                                                gamma * onesetcont                        ])';
                  nrefine( iset ) = -1;

                  optstage( elsetset ) = Inf * ones( 1, l_elset );

               %  No or too small improvement: shrink the relevant mesh.
               %  Refine the grid for continuous variables, accelerating when more  
               %  than 2 successive refinements have taken place. Note that the actual
               %  (reduced) increments for discrete variables are never used.

               else 
                  if ( nrefine( iset ) > 2 )
                     shrink = beta * beta;
                  else
                     shrink = beta;
                  end
                  xincr( xsetcont, cur ) = max( [ epsilon * onesetcont              ;
                                                  shrink * xincr( xsetcont, cur )'  ;
                                                  (shrink^iota) * cmesh * onesetcont  ] )';
                  nrefine( iset )  = nrefine( iset ) + 1;
                  optstage( iset ) = meshset;

               end

               %  Update the best subset improvement and the indeces of the best 
               %  group and set.

	       if ( verbose && impset > impbest )
	          impbest = impset;
   	          gbest   = igroup;
	          sbest   = iset;
	       end

            end

         end  % of the loop on the independent sets of variables in group igroup.
	 
         %  Sufficient improvement has been obtained: 

         if ( impgroup > eta*cmesh^2 )

            %  1) move to the best point,

	    xbest  = xtry;
	    fbest  = ftry;
            fibest = fitry;

            if ( verbose >= 10 )
	       fibest = fibest
	    end

            %  Check if BFO should terminate because the objective function's target
            %  has been reached.

            if ( (  maximize && fbest >= ftarget )  || ( ~maximize && fbest <= ftarget ) )
               msg = ' The objective function target value has been reached. Terminating.';
               if ( verbose )
                  disp( msg )
               end
               xbest = bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref );

               %  Possibly call for cleanup in the search-step function before termination.

               if ( search_step_needs_cleaning_up ) 
                  bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
               end

               return
            end

            %  2) mark this point as best in the element-wise history,

            if ( categorical )
               xp = { bfo_cellify( xbest, xtype, cat_dictionnary, x0ref ) };
            end
            for iel = 1:n_elements
               el_hist{ iel }.fbest = fibest( iel );
            end

            % 3 ) break the loop on groups.

            loopbreak = 1;
	    break;
          end

      end % of the loop on groups

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  TERMINATION STEP %%%%%%%%%%%%%%%%%%%%%%%%%

      %  Determine the global continuous mesh resulting from the CPS poll loop.

      cmesh = max( xincr( icont, cur ) );

      %  Sufficient decrease has been obtained: print out and continue.

      if ( loopbreak )

	 checking = 0;             % reset termination ckecks
	    
         %  Print the one-line iteration summary.

         if ( verbose > 1  )
            if ( verbose > 3 )
               fprintf( '\n' );
               fprintf( '%s neval        fx       nimp neltry    cmesh       ig:iel\n',    ...
                        indent );
            end
            fprintf( '%s%5d  %+.6e  %5d %5d   %4e %3d:%3d\n',                 ...
                      indent, ceil( neval ), fbest, nimp, ntry, cmesh, gbest, sbest );
            if ( verbose > 3 )
               fprintf( '\n' );
            end
            bfo_print_x( indent, 'x', xbest, xscale, verbose, xtype, cat_dictionnary,      ...
	                 x0ref )
         end

      %  No sufficient decrease: test for termination on the current grid and overall.

      else

         %  Recompute the current mesh.

         cmesh = min( xincr( icont, cur ) );

         %  Reallow subset optimization irrespective of increment size.

         optstage = Inf*ones( 1, n_sets );

         %  Further grid refinement is possible, possibly leading to further minimization.

         if ( ncont && cmesh > epsilon ) 

            %  Print the one-line iteration summary.

            if ( verbose > 1 )
               if ( verbose > 3 )
                  fprintf( '\n' );
                  fprintf( '%s neval        fx       nimp neltry    cmesh       status\n', ...
                           indent );
               end
               fprintf( '%s%5d  %+.6e  %5d %5d   %4e  refine\n',                           ...
                        indent, ceil( neval ), fbest,  nimp, ntry, cmesh );
               if ( verbose > 3 )
                  fprintf( '\n' );
               end
               bfo_print_x( indent, 'x', xbest, xscale, verbose, xtype, cat_dictionnary,   ...
	                    x0ref )
            end

         %  Convergence is achieved on the finest grid.

         else

            %  Printout

	    if ( verbose > 1 )
               if ( verbose > 3 )
                  fprintf( '\n' );
                  fprintf( [ '%s neval        fx       nimp neltry',                       ...
                             '    cmesh        status\n'], indent );
               end
               fprintf( '%s%5d  %+.6e  %5d %5d   %4e  checking (s)\n',                     ...
                        indent, ceil( neval ), fbest, nimp, ntry, cmesh  );
               if ( verbose > 3 )
                  fprintf( '\n' );
               end
            end
            bfo_print_x( indent, 'x', xbest, xscale, verbose, xtype, cat_dictionnary, x0ref );

            term_loops = term_loops + 1;

            %  Termination has been verified in that no decrease can be obtained
            %  (after term_basis termination loops in each subspace) by steps 
            %  preserving the CPS structure. It now possible to
            %  strengthen the assessment of the current point as a minimizer by
            %  attempting to obtain decrease along unstructured polling directions,
            %  that is directions not preserving the CPS structure.
            %  We choose to build them as a small number (term_basis) of random 
            %  directions in the whole space. Polling along randomly generated basis
            %  is not attempted since any move in an unstructured direction requires
            %  at least one full function evaluation (and therefore a standard 
            %  checking step would typically involve 2*n function evaluations),
            %  which is much too expensive in the context of CPS problems.
            %  The mechanism used for this additional checking phase is to rely on
            %  a single non-CPS polling loop of the non-CPS algorithm, but using a 
            %  incomplete basis (containing only term_basis directions).  The next 
            %  paragraph thus generates this basis  and prepares to reenter the non-CPS
            %  part of the algorithm. In particular, setting the checking flag to -1, 
            %  indentifies this particular use of the standard loop and forces 
            %  termination after a single unsuccessful loop.

            if ( term_loops >= min( 2, term_basis ) || ncont == 0 )
               Q     = bfo_new_continuous_basis( ncont, [], [], term_basis );
               npoll = size( Q, 2 );
               xp    = xbest;
               x     = xbest;
               sacc  = [];
               fxp   = fbest; % not ok for maximization
               if ( maximize )
                  flow  = - fbest;
               else
                  flow  =   fbest;
               end
               fx       = flow;
               checking = -1;

            %  Further termination loops are requested
	    
            else
	    
               %  Reset the number of successive refinements to 3 (= 2 -(-1)) before 
               %  acceleration occurs again.

               nrefine = -1;

               %  Require termination checking inside the subspace for the next loops.
               %  Note that incrementing checking allows to distinguish the first checking
               %  loop (within subsets) from the subsequent ones, and that this role cannot
               %  be played by term_loops, which counts termination loops within subset and
               %  in the whole space together.

	       checking = checking + 1;
            end
         end
      end
   end   %  end of the CPS optimization part of the optimization loop

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%  Non-coordinate-partially separable POLL LOOP  %%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   %  Note: checking < 0 allows general random polling directions after having converged 
   %  with the coordinate partially separable poll loop. The loop is also entered when 
   %  the search step has been successful in order to allow proper printout and 
   %  termination tests.

   if ( ~partially_separable  || checking < 0 || successful_search_step || at_optimizer )

      %% The poll step

      %  Initialize the direction of "hopeful descent" to nothing, as it may be 
      %  required even if the forward/backward loop is bypassed after a restart.

      ddir = [];

      %  Initialize the index of the best move in the cycle

      ibest = 0;

      %  Verify that the forward/backward iteration loop is necessary, as this may 
      %  not be the case after a restart or after a successful search step.

      if ( fbloop && ~successful_search_step && ~at_optimizer )

         %  Initialize the estimate of the gradient and the full loop indicator
         %  (a full loop is necessary for the estimate of the gradient to be
         %  meaningful, which is detected by setting the full_loop flag.)

         gcdiff    = zeros( ncont, 1 );
         full_loop = 0;
 
         %  Set a flag if the algorithm is in the convergence checking mode.

         if ( checking >= 0 )
            checking = ( level == 1 && term_basis > 1 );
         end

         %  Compute the unknown values of the objective function at the
         %  neighbouring points on the current grid.

         loopbreak = 0;                % no break for sufficient decrease yet
         ic        = 0;                % initialize the continuous variable counter
         if ( ncont == 0 )             % set a default continuous basis if there are no    ...
            Q = 1;                     % ... continuous variables
         end
         dneigh    = 0;                % dynamic neighbourhood not yet considered
         ineigh    = 0;                % Counter for the number of neighbourhood effectively
                                       % computed

         for i = 1:n                   % loop on the variables

            %  Avoid categorical variables beyond the first if the categorical
	    %  neighbourhoods are defined dynamically. 
            
            xtypei = xtype( i );
            if ( xtypei == 'c' || xtypei == 'i' || ( xtypei ~= 's' || ~dneigh  ) )

               ineigh = ineigh + 1;

               [ neighbours, alphaf, alphab,xtypes,xlowers,xuppers,cat_dictionnary,msg ] = ...
	              bfo_build_neighbours( x, i, xtype, xlower, xupper, Q, icont, idisc,  ...
                                            idall, x0ref, cat_dictionnary, cn_name,        ...
                                            num_cat_states, 0, xincr(:,cur), ncbound,      ...
                                            ndbound, latbasis, verbose, indent, myinf );
               if ( ~isempty( msg ) )

                  % Possibly call for cleanup in the search-step function.

                  if ( search_step_needs_cleaning_up ) 
                     bfo_search_step_cleanup( bfo_srch,categorical, sum_form );
                  end

	          return
	       end

               %  Remember that the dynamical neighbourhood of categorical variable has been
	       %  examined.

               if ( xtypei == 's' && dynamical )
                  dneigh = 1;
	          if ( verbose >= 4 )
	             disp( ' Dynamical categorical neighbourhood analyzed.')
	          end
	       end

               %  Obtain the number of neighbours relative to variable i.
	       
	       nnghbri = size( neighbours, 2);

               %  Print the neighbours, if requested.

               if ( verbose >= 4 )
                  for inghbr = 1:nnghbri	 
                      bfo_print_x( indent, [ 'neighbour ', int2str(inghbr) ],              ...
	                           neighbours( 1:n, inghbr ), xscale, verbose, xtype,      ...
			           cat_dictionnary, x0ref )
                  end
               end
	 
               %  Loop on the neighbouring values of the current iterate
               %  relative to variable i.

               for inghbr = 1:nnghbri

                  xtry = neighbours( 1:n, inghbr );

                  %  Compute the minimum distance between the trial point and the l_hist
	          %  previous iterates.


                  dxxp = norm( xp - xtry, 'inf' );
	          if ( dxxp <= eps )
                     if ( categorical )
                        nprev = length( x_hist );
                     else
                        nprev = size( x_hist, 2 );
                     end
	             for ip = 1:nprev - 1
	                if ( categorical )
	                   dxxp = min ( dxxp, norm( xtry - bfo_numerify(                   ...
			                  x_hist{ max( 1, end-ip ) }{ 1 }, xtype,          ...
                                          cat_dictionnary ),'inf' ) );
		        else
		           dxxp = min ( dxxp, norm( xtry -                                 ...
		             x_hist( 1:n, max( 1, end-ip ) ), 'inf' ) );
		        end
                     end
	          end
	       
                  %  Verify that the trial point is different from the l_hist
		  %  previous iterates.

                  if ( dxxp > eps ) 

                     %  Verify that the forward point is different from the current iterate. 

                     time_to_exit = 0;
                     dxx = norm( xtry - x );
                     if (  dxx > eps  )

                        %  Evaluate the objective function at the forward point.

                        %  1) Evaluate the objective function by (recursively) performing
		        %     optimization on the next level.

                        if ( multilevel && level < nlevel )

                           %  Transform xtry to a vector state if categorical variables
			   %  are present.
			
                           if ( categorical )
                              if ( xtypei == 's' )
		                 xu = { bfo_cellify( xtry, xtypes{ inghbr },               ...
			                             cat_dictionnary, x0ref ) };
			      else
			         xu = { bfo_cellify( xtry, xtype, cat_dictionnary, x0ref ) };
			      end
			   else
			      xu = xtry;
		           end

                           %  Call the objective function for the next level.

                           [ xu, fnghbr, msglow, ~, neval, f_hist, xincr, el_hist_nghbr] = ...
                               bfo_next_level_objf( level, nlevel, xlevel, neval, f, xu,   ...
                                 checking, f_hist, el_hist, xtmulti, xincr, xscale, xlower,...
			         xupper, eldom, max_or_min, vb_name, epsilon, bfgs_finish, ...
			         maxeval, verbosity, fcallt, alpha, beta, gamma, eta, zeta,...
			         inertia, searchtype, rseed, iota, kappa, lambda, mu,      ...
                                 term_basis, latbasis, idall, reset_random_seed, ssfname,  ...
                                 cn_name, cat_states );

                           if ( verbose >= 2 && blank_line )
                              disp( ' ' )
                           end

                           %  Return if an error occured down in the recursion.

                           if ( length(msglow) >= 10 && strcmp( msglow(1:10),' BFO error' ) )
                              msg = msglow;
                              xbest = bfo_pack_x( xbest, xscale, xtype, cat_dictionnary,   ...
			                          x0ref );

                              % Possibly call for cleanup in the search-step function.

                              if ( search_step_needs_cleaning_up ) 
                                 bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
                              end

                              return
                           end
       
                           %  Retrieve the final element function values from el_hist and 
                           %  update the evaluation history of the original elements.
                           %  Note that the best functon value and corresponding point
                           %  are not saved in the original element's history, because
                           %  there is no guarantee that the best point from the higher
                           %  level optimization will result in a best point at the
                           %  current level.

                           if ( sum_form )
                              for iel = 1:n_elements
                                  finghbr( iel )     = el_hist_nghbr{ iel }.fbest;
                                  el_hist{ iel }.fel = el_hist_nghbr{ iel }.fel;
                                  el_hist{ iel }.xel = el_hist_nghbr{ iel }.xel;
                              end
	                   end

                        %  2) Evaluate the (single level) objective function.
       
                        else

                           %  Unscale xtry if relevant and transform it to a vector state
			   %  if categorical variables are present.
			
                           if ( xtypei == 's' )
			      xu = bfo_pack_x( xtry, xscale, xtypes{ inghbr },             ...
			                       cat_dictionnary, x0ref );
                           else
			      xu = bfo_pack_x( xtry, xscale, xtype,                        ...
			                       cat_dictionnary, x0ref );
                           end

                           %  The objective function is the (internal) function used for 
                           %  "average" training.

                           if ( within_average_training )
                              [ fnghbr, msgt, wrnt ] = f( xu, min( fbound, fbest ) );
                              if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ))
                                 if ( isempty( xscale ) )
                                    xbest = xu;
                                 else
                                    xbest = xscale .* xu;
                                 end

                                 % Possibly call for cleanup in the search-step function.

                                 if ( search_step_needs_cleaning_up ) 
                                    bfo_search_step_cleanup( bfo_srch,categorical, sum_form );
                                 end

                                 return
                              end
                              nevalt = nevalt + fnghbr;
                              [ f_hist, x_hist, neval ] = bfo_ehistupd( f_hist,            ...
			              x_hist, l_hist, neval, fnghbr, xu, categorical );
			      
                           %  The objective function is the (internal) function used for 
                           %  "robust" training.

                           elseif ( within_robust_training )
                              [ fnghbr, msgt, wrnt, t_neval ] = f( xu, min( fbound, fbest ) );
                              if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ))
                                 if ( isempty( xscale ) )
                                    xbest = xu;
                                 else
                                    xbest = xscale .* xu;
                                 end

                                 % Possibly call for cleanup in the search-step function.

                                 if ( search_step_needs_cleaning_up ) 
                                    bfo_search_step_cleanup( bfo_srch,categorical, sum_form );
                                 end

                                 return
                              end
                              nevalt = nevalt + t_neval;
                              [ f_hist, x_hist, neval ] = bfo_ehistupd( f_hist,            ...
			              x_hist, l_hist, neval, fnghbr, xu, categorical );

                           %  The objective function is the (internal) function used for 
                           %  "robust" training.

                           elseif ( within_profile_training )
                              [ fnghbr, msgt, wrnt, t_neval, fstar, tpval ] =              ...
                                                              f( xu, fstar, tpval );
                              if ( length( msgt ) >= 10 && strcmp( msgt(1:10), ' BFO error' ))
                                 if ( isempty( xscale ) )
                                    xbest = xu;
                                 else
                                    xbest = xscale .* xu;
                                 end

                                 % Possibly call for cleanup in the search-step function.

                                 if ( search_step_needs_cleaning_up ) 
                                    bfo_search_step_cleanup( bfo_srch,categorical, sum_form ); 
                                 end

                                 return
                              end
                              nevalt = nevalt + t_neval;
                              [ f_hist, x_hist, neval ] = bfo_ehistupd( f_hist,            ...
			              x_hist, l_hist, neval, fnghbr, xu, categorical );

                           %  The objective function is specified by the user in sum form.

                           elseif( sum_form )
			      if ( withbound )
                                 if ( maximize )
				    curfb = max( fbound, fbest );
				 else
				    curfb = min( fbound, fbest );
				 end
                                 [ fnghbr, finghbr, nevali ] =                             ...
				            bfo_sum_objf( xu, f, elset, eldom, curfb,      ...
                                                          maximize );
                              else
                                 [ fnghbr, finghbr, nevali ] =                             ...
                                            bfo_sum_objf( xu, f, elset, eldom );
                              end

                              %  Maintain the evaluation history globally and elementwise.

                              [ f_hist, x_hist, neval, el_hist, ev_hist ] = bfo_ehistupd(  ...
                                            f_hist, x_hist, l_hist, neval, fnghbr, xu,     ...
					    categorical, el_hist, finghbr, eldom,          ...
                                            nevali, ev_hist );

                           %  The objective function is specified by the user in simple form.

                           else
			      if ( withbound )
                                 if ( maximize )
                                    fnghbr = f( xu, max( fbound, fbest ) );
                                 else
                                    fnghbr = f( xu, min( fbound, fbest ) );
                                 end
                              else
                                 fnghbr = f( xu );
                              end
                              [ f_hist, x_hist, neval ] = bfo_ehistupd( f_hist,            ...
			              x_hist, l_hist, neval, fnghbr, xu, categorical );
			   end

                        end

                        %  Print the result, if in debug mode.

                        if ( verbose >= 4 )
                           fprintf( '%s value of f at neighbour %3d is %.12e\n',           ...
                                    indent, inghbr, fnghbr );
                           if ( sum_form )
                              bfo_print_vector( indent, 'fi', finghbr );
                           end
                        end

                        %  Take maximization into account, if relevant.

                        if ( maximize )
                          fnghbr = - fnghbr;
                        end

                        %  Check for undefined function values.

                        if ( isnan( fnghbr ) )
                           if ( maximize )
                              fnghbr = -Inf;
                           else
                              fnghbr = Inf;
                           end
                        end

                        %  Update the best value so far.

                        if ( fnghbr < flow - eta * cmesh^2 )

                           %  Retransform the best point found back to numerical form, if
			   %  categorical variables are present. Note that this is only
			   %  necessary for levels below the last in the multilevel case
			   %  since then this point results from optimization at a higher
			   %  level and may therefore differ from the point specified for
			   %  evaluation.
			
		           if ( multilevel  &&  level < nlevel )
			      if ( categorical )
   			         xbest = bfo_numerify( xu{ 1 }, xtype, cat_dictionnary );
			      else
			         xbest = xu;
			      end
			   else
                              xbest = xtry;
			   end
                           flow   = fnghbr;
                           if ( maximize )
                              fbest = -fnghbr;
                           else
                              fbest =  fnghbr;
                           end
                           if ( sum_form )
                              fibest = finghbr;
                           end
                           ibest = ineigh;

                           %  Record the best point elementwise, if necessary.

                           if ( sum_form )
	                      for iel = 1:n_elements
	                         el_hist{ iel }.fbest = fibest( iel );
	                      end
                           end
	    
                           %  Reconstruct the new optimization context if a move has been
		           %  made in a categorical variable that alters the context.

		           if ( dynamical && xtypei == 's' )
			      xlss=max( [ xlowers( 1:n, inghbr )'; -myinf * ones( 1, n ) ] )';
			      xuss=min( [ xuppers( 1:n, inghbr )';  myinf * ones( 1, n ) ] )';
			      if ( ~strcmp( xtype, xtypes{ inghbr } )   ||                 ...
			           norm( xlower - xlss )  || norm( xupper - xuss ) )
		                 xtype              = xtypes{ inghbr };
			         opt_context.xtype  = xtype;
		                 xtmulti            = xtype;
                                 if ( isempty( xscale ) )
  	                            xlower = xlss;
	                            xupper = xuss;
                                 else
                                    finite = intersect( icd, find( xlss > myinf ) );
                                    xlower( finite ) = xlss( finite ) ./ xscale( finite );
                                    finite = intersect( icd, find( xuss < myinf ) );
                                    xupper( finite ) = xuss( finite ) ./ xscale( finite );
                                 end
	                         opt_context.xlower = xlower;
	                         opt.context.xupper = xupper;
                                 [ icont, idisc, icate, ifixd, ixed, iyed, ized, iwait,    ...
                                   ired, ided, iked, ncont, ndisc, ncate, nfixd, nxed,     ...
                                   nyed, nzed, nwait, nred, nded, nked,	sacc, Q ]          ...
                                    = bfo_switch_context( xtype, npoll, verbose );
                              end
		           end
                        end

                        %  Terminate if the user-defined target has been reached.

                        if ( (  maximize && fbest >= ftarget )  ||                         ...
		             ( ~maximize && fbest <= ftarget )    )
                           msg = [' The objective function target value has been reached.',...
                                  ' Terminating.' ];
                           if ( verbose )
                             disp( msg )
                           end
			   time_to_exit = 1;
                        end

                     else
                        fnghbr = fx;
                     end

                     %  Terminate if the maximum number of function evaluations
                     %  has been reached.
		     
                     if ( neval >= maxeval )
                        msg = [ ' Maximum number of ', int2str( maxeval ),                 ...
                                ' evaluations of ', shfname, ' reached.' ];
                        if ( verbose && level == 1 )
                           disp( msg )
                        end
		        time_to_exit = 1;
		     end

                     %  Save information for possible restart, if requested.

                     if ( ( savef >  0 && mod( ceil( neval ), savef ) == 0 )  ||           ...
                          ( savef >= 0 && time_to_exit )                       )

                        s_hist  = bfo_shistupd( s_hist, idc, xbest, fbest, xincr( :, cur ),...
			              xtype, cat_dictionnary, x0ref, cn_name, opt_context );
                        savok   = bfo_save( sfname, shfname, maximize, epsilon, ftarget,   ...
                                      maxeval, neval, f_hist, xtype, xincr, xscale, xlower,...
                                      xupper, verbose, alpha, beta, gamma, eta, zeta,      ...
                                      inertia, stype, rseed, iota, kappa, lambda, mu,      ...
                                      term_basis, use_trained, s_hist, latbasis,           ...
                                      bfgs_finish, training_history, fstar, tpval, ssfname,...
                                      cn_name, cat_states, cat_dictionnary );
                        if ( ~savok )
                           msg = [ ' BFO error: checkpointing file ', sfname',             ...
                                   ' could not be opened. Skipping checkpointing at ',     ...
                                   int2str( ceil( neval ) ), ' evaluations.' ];
                           if ( verbose )
                              disp( msg )
                           end
                        end
                     end
 
                     %  Effectively terminate the minimization.
		  
                     if ( time_to_exit )
                        xbest = bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref );

                        % Possibly call for cleanup in the search-step function.

                        if ( search_step_needs_cleaning_up ) 
                           bfo_search_step_cleanup( bfo_srch,categorical, sum_form );
                        end

                        return;
                     end

                     %  Terminate the poll loop on the variables if sufficient 
                     %  decrease has already been found.  This is achieved by setting the
	             %  loopbreak flag, breaking the neighbours' loop and testing the
	             %  flag on exit of that loop, for possibly breaking the poll loop.

                     if ( fx - flow >= eta * cmesh^2 )
	                loopbreak = 1;
                        break; 
                     end
		  
                     %  Avoid computing the backward value if the forward value does not give
		     %  a sufficient increase (note that the forward value does not give a
		     %  sufficient decrease either).

                     if ( xtypei == 'c' && inghbr == 1 &&  dxx > eps  &&               ...
		          fnghbr - fx <= eta * cmesh^2 )
                         ffwd  = fnghbr;
		         xifwd = xtry( i );
		         fbwd  = Inf;
                         break;
                     end
		  
                  else   
                     fnghbr = fxp;
                  end

                  %  Save the forward and backward values for continuous variables.

                  if ( xtypei == 'c' )
	             if ( inghbr == 1 )
	                ffwd   = fnghbr;
		        xifwd  = xtry( i );
	             else
	                fbwd   = fnghbr;
		        xibwd  = xtry( i );
	             end
	          end

               end  %  End of the loop on the neighbours of variable i

               %  Break from the loop on the variables if sufficient decrease has been found.
	
               if ( loopbreak )
	          if ( verbose >= 4 )
	             disp( [ ' sufficient decrease obtained:',                             ...
	                     ' breaking the poll loop after variable ', int2str( i ) ] )
                  end
	          break;
	       end
	
               %  Update the estimate of the gradient, avoiding the use of
               %  meaningless function values.

               if ( xtypei== 'c' )
	          ic = ic + 1;
                  if ( ffwd < Inf && fbwd < Inf )
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

               %  Set the full-loop indicator.

               full_loop = ( i == n );

            end  

         end  %  End of the loop on the dimensions

         %  The forward-backward loop on continuous variables has been performed 
         %  completely. As a consequence, an central difference approximation of the
         %  gradient (if it exists) can be computed and possibly used within a
         %  quasi-Newton update.

         if ( ncont > 0 && full_loop && npoll >= ncont )

            %  Compute the approximate gradient and estimate for the criticality measure.

            grad = Q' * gcdiff;
            estcrit = norm( x( icont ) - max( [ xlower( icont )';                          ...
                                                min( [ ( x( icont ) - grad )';             ...
                                                        xupper( icont )' ] ) ] )', 'inf' );

            ddir = - grad;                          % set the hopeful descent direction to the
                                                    % negative gradient

            %  Compute a very approximate quasi-Newton (BFGS) step.

            if ( max( xincr( icont, cur ) - bfgs_finish * ones( ncont, 1 ) ) <= 0 )
                                                    % the current cont; mesh is small enough
               if ( iqn  == 0 )                     % set the first 'full grad' iterate
                  iqn   = 1;
                  gradp = grad;                     % remember the gradient
                  xqnp  = x;                        % remember the iterate
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
                  iqn  = itg;
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

         if ( n > 1 && ndisc + ncate > 0 && nactive > 0 )
            s_hist = bfo_shistupd( s_hist, idc, xbest, fbest, xincr( :, cur ), xtype,      ...
	                          cat_dictionnary, x0ref, cn_name, opt_context );
            if ( verbose >= 10 )
               s_history = s_hist
            end

            %  If integer variables are present and no progress was made,
            %  consider the neighbouring continuous or discrete subproblems.
            %  This is no longer necessary for termination checking loops 
            %  or (obviously) if recursion is not wanted. The test 
            %  checking >= 0 is to avoid recursion when checking using 
            %  unstructured poll directions for the  coordinate-partially-separable case.
            %  NOTE: more work is probably approriate to decide when branching
            %        should occur.

%           if ( ( nactivd  || nactivs )  && stype ~= 2 &&  checking >=0 && ...
%              ( ( stype == 0 && term_loops < 2 )  || ( stype == 1 && term_loops == 1  ) ) )
%           if ( ( nactivd  || nactivs )  && stype ~= 2 &&  ~term_loops &&  checking >=0  )

            if ( ( nactivd  || nactivs )  && stype ~= 2 &&  checking >=0 && ...
                 ( ( ncont && cmesh >= epsilon ) || ( ~ncont && term_loops == 1  ) ) )

               %   Print the result of the current iteration.

               if ( verbose > 1 && ndisc + ncate > 0 )
                  if ( ~full_loop  || ndisc == n || estcrit < 0 )
                      fprintf( '%s%5d  %+.6e  ------------  %4e    %+3d\n',                ...
                               indent, ceil( neval ), fbest, cmesh, ibest );
                  else
                      fprintf( '%s%5d  %+.6e  %4e  %4e    %+3d\n',                         ...
                              indent, ceil( neval ), fbest, estcrit, cmesh, ibest  );
                  end
                  bfo_print_x( indent, 'x', xbest, xscale, verbose, xtype, cat_dictionnary,...
                               x0ref )
               end

               %%%%%%%%%%%%%%%%%%%%%%%%%  RECURSIVE STEP %%%%%%%%%%%%%%%%%%%%%%%%%%
               %% The recursive step

               %  Define the parameters for the recursive calls.

               xis = xincr;
               if ( stype == 1 )
                  xis( icont, cur ) = xincr( icont, ini );
               end

               %  Only perform the checking loops on neighbouring discrete
               %  subspaces during the first top level checking loop.

               if ( term_loops == 1 )
                  tbs = term_basis;
               else
                  tbs = 1;
               end

               %  Recursively call the minimizer for all neigbouring vectors of
	       %  the current iterate relative to (discrete or categorical)
	       %  changes in variable j (j = 1:n).
               %  Avoid reconsidering more than a single categorical variable if
               %  there are more than one and dynamic categorical neighbourhoods
               %  are used.
	    
               for j = 1:n
                  xtj = xtype( j );
                  if ( ( xtj == 'i'  || ( xtj == 's' ) )            &&                     ...
                       ( depth == 0  || j >  rpath( depth ) )       &&                     ...
                       xupper( j ) - xlower( j ) > eps               )

                     %  Build the list of neighbouring values of the current iterates
	             %  by considering all possible neighbours of variable j.  These
	             %  vectors are stored in the columns of the array neighbours, whose
	             %  number of columns must be at least 2 (for integer or lattice
	             %  variables) and at most the maximum number of neighbouring values
	             %  for a categorical variable.
	       
                     [ neighbours, ~, ~, xtypes, xlowers, xuppers, cat_dictionnary, msg ] =...
	                bfo_build_neighbours( x, j, xtype, xlower, xupper, Q, icont, idisc,...
                                              idall, x0ref, cat_dictionnary, cn_name,      ...
                                              num_cat_states, 1, xincr(:,cur), ncbound,    ...
                                              ndbound, latbasis, verbose, indent, myinf );

                     if ( ~isempty( msg ) )

                        % Possibly call for cleanup in the search-step function.

                        if ( search_step_needs_cleaning_up ) 
                           bfo_search_step_cleanup( bfo_srch,categorical, sum_form );
                        end
	                return
	             end
	       
                     %  Remember that the dynamical neighbourhood of categorical
		     %  variable has been examined.
	    
                     if ( xtype( i ) == 's' && dynamical )
	                if ( verbose >= 4 )
	                   disp( ' Dynamical categorical neighbourhood analyzed.')
	                end
	             end

                     %  Obtain the number of neighbours relative to variable j.
	       
	             nnghbrj = size( neighbours, 2); 

                     %  Reset the variable types.

                     xts = xtype;

                     %  Loop on the neighbouring values of the current iterate
	             %  relative to variable j.

                     for inghbr = 1:nnghbrj

                        xs = neighbours( 1:n, inghbr );

                        %  Declare the j-th variable to be fixed for minimization in
                        %  the associated subspaces (make it a "waiting variable") and
                        %  augment the recursion path accordingly.  If categorical variables
		        %  with dynamical neighbourhoods are used, the set of all categorical
		        %  variables results in a single neighbourhood and hence all these
		        %  variables must be declared waiting for the recursion.

                        xts( j ) = 'w';
                        xtl = xlower;
                        xtu = xupper;
                        rpj = [ rpath j ];

                        %  Recursive minimization for the inghbr-th neighbour of variable j.

                        if ( verbose > 1 )
                           disp( [indent,'      ====  recursive call for neighbour ',      ...
                               int2str( inghbr ), ' of variable ', int2str( j ), '  =====' ] )
                        end

                        %  Call BFO recursively if the trial point differs from 
                        %  the current iterate.  Note that specifying xlevel is unnecessary
		        %  as integer/categorical recursion occurs in each level
			%  independently.

                        if ( norm ( xs - x ) > 0 )
		     
                           %  Transform xs to a vector state, if the problem contains
			   %  categorical variables.

                           if ( categorical )
                             if ( xtype( i ) == 's' )
   		                 xs = { bfo_cellify( xs, xtypes{ inghbr }, cat_dictionnary,...
			                x0ref ) };
		              else
   		                 xs = { bfo_cellify( xs, xtype, cat_dictionnary, x0ref ) };
			      end
                           end

                           [ xtry, fxtry, msgs, ~, neval, f_hist, ~, ~, th, s_hist, ~,     ...
                             nopt_context] =                                               ...
                                bfo( f, xs, 'topmost', 'no','eldom', {eldom},              ...
			        'xscale',  xscale, 'xtype', xts, 'epsilon', 2*cmesh,       ...
				'maxeval', maxeval, 'verbosity', verbosity, 'xlower', xtl, ...
				'xupper', xtu, 'termination-basis', tbs, 'sspace-hist',    ...
				s_hist, 'element-hist', el_hist, 'rpath', rpj,  'nevr',    ...
				neval, 'save-freq', savef, 'f-call-type', fcallt,          ...
				'max-or-min',max_or_min, 'f-hist', f_hist, 'alpha',        ...
                                alpha, 'beta', beta, 'gamma', gamma, 'eta', eta, 'zeta',   ...
                                zeta, 'inertia', inertia, 'search-type', searchtype,       ...
                                'random-seed', rseed, 'iota', iota, 'kappa', kappa,        ...
                                'lambda', lambda, 'mu', mu, 'lattice-basis',               ...
                                latbasis, 'xincr', xis, 'bfgs-finish',   bfgs_finish,      ...
                                'reset-random-seed', reset_random_seed, 'search-step',     ...
                                ssfname, 'cat-states', {cat_states}, 'cat-neighbours',     ...
                                cn_name );

                           %  Return if an error occured down in the recursion.

                           if ( length( msgs ) >= 10 &&                                    ...
			        ( strcmp( msgs(1:10), ' BFO error' )  ||                   ...
			          strcmp( msgs(1:10), ' Maximum n' )   ) )
                              msg = msgs;
			      if ( ~isempty( xscale ) )
			         xbest = xscale .* xbest;
			      end

                              % Possibly call for cleanup in the search-step function.

                              if ( search_step_needs_cleaning_up ) 
                                 bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
                              end
                              return
                           end

                           %  Update the training history for what happened in the recursion.

                           if ( within_training )
                              nevalt = nevalt + th( end, 3 );
                           end

                           %  See if a better point has been found. If yes,
                           %  update the lowest value so far and determine if a
			   %  a change of minimization context is possible.
			   
                           new_context  = 0;
			   better_point = 0;
                           if ( maximize && -fxtry < flow )
                              flow  = -fxtry;
			      new_context = ( xtype( i ) == 's' );
			      better_point = 1;
                           elseif ( ~maximize &&  fxtry < flow ) 
                              flow  = fxtry;
			      new_context = ( xtype( i ) == 's' );
			      better_point = 1;
                           end

                           %  A change of optimization context is possible.
                           %  Check that this is the case, and, if positive,
                           %  perform the context change.
			
		           if ( new_context                                 &&             ...
			        ( ~strcmp( xtype, nopt_context.xtype  )  ||                ...
			          norm( xlower - nopt_context.xlower )   ||                ...
			          norm( xupper - nopt_context.xupper )     )   )
			      xtype      = nopt_context.xtype;
			      xtype( j ) = xtj;  % restore original status of the hinge var.
			      xlower     = nopt_context.xlower;
			      xupper     = nopt_context.xupper;
		              xtmulti    = xtype;
			      opt_context.xtype  = xtype;
			      opt_context.xlower = xlower;
			      opt.context.xupper = xupper;
                              [ icont, idisc, icate, ifixd, ixed, iyed, ized, iwait, ired, ...
                                ided, iked, ncont, ndisc, ncate, nfixd, nxed, nyed, nzed,  ...
                                nwait, nred, nded, nked, sacc, Q ]                         ...
                                = bfo_switch_context( xtype, npoll, verbose );
                           end

                           %  Update the best approximate solution (in the new
                           %  context, if relevant).

                          if ( better_point )
                              fbest = fxtry;
                              ibest = j;

                              %  If the problem contains categorical variables,
			      %  re-transform the best point found in the recursion
			      %  to numerical form.
			   
                              if ( categorical )
			         xbest = bfo_numerify( xtry{ 1 }, xtype, cat_dictionnary );
			      else
                                 xbest = xtry;
			      end
                           end
			   

                        else
                           if ( verbose >= 10 )
                               disp( ['  recursive call skipped because |x - xs| = ',      ...
                                      num2str( norm( x - xs ) ) ]);
                           end                     
                        end

                        %  Terminate if the maximum number of function evaluations
                        %  has been reached.

                        if ( neval >= maxeval )
                           msg   = [ ' Maximum number of ', int2str( maxeval ),            ...
                                     ' evaluations of ', shfname, ' reached.' ];
                           if ( verbose && level == 1 )
                              disp( msg )
                           end
                           if ( savef >= 0 )

                              s_hist = bfo_shistupd( s_hist, idc, xbest, fbest,            ...
			                             xincr(:,cur), xtype, cat_dictionnary, ...
						     x0ref, cn_name, opt_context );
                              savok  = bfo_save( sfname, shfname, maximize,epsilon,ftarget,...
                                           maxeval, neval, f_hist, xtype, xincr, xscale,   ...
                                           xlower, xupper, verbose, alpha, beta, gamma,    ...
                                           eta, zeta, inertia, stype, rseed, iota, kappa,  ...
                                           lambda, mu, term_basis, use_trained,            ...
                                           s_hist, latbasis, bfgs_finish, training_history,...
                                           fstar, tpval, ssfname, cn_name, cat_states,     ...
                                           cat_dictionnary );
                              if ( ~savok )
                                 msg = [ ' BFO error: checkpointing file ', sfname',       ...
                                        ' could not be opened. Skipping checkpointing at ',...
                                        int2str( ceil( neval ) ), ' evaluations.' ];
                                 if ( verbose )
                                    disp( msg )
                                 end
                              end
                           end
                           xbest = bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref );
                           % Possibly call for cleanup in the search-step function.

                           if ( search_step_needs_cleaning_up ) 
                              bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
                           end

                           return;
	                end
                     end
                  end
               end
            end
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
      elseif ( within_profile_training )
         if ( within_perfprofile_training )
            ittrain = training_history( end, 2 ) + 1;
            training_history = [ training_history; [ 3, ittrain, nevalt, fbest, xbest' ] ];
         elseif ( within_dataprofile_training )
            ittrain = training_history( end, 2 ) + 1;
            training_history = [ training_history; [ 4, ittrain, nevalt, fbest, xbest' ] ];
         end
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

            if ( ismember( xtype( i ), { 'c', 'i' } )              &&                      ...
	         ( xbest( i )  - xlower( i ) <= xincr( i, cur )  ||                        ...
                   xupper( i ) - xbest( i )  <= xincr( i, cur )   )  )
               if ( verbose >= 10 )
                  disp( [' variable ', int2str(i), ' has a (nearly) saturated bound'] )
               end
               isat( end+1 )      = ic;
               nsat               = nsat + 1;
               if ( nsat <= npoll )
                  N( 1:ncont, nsat ) = zeros( ncont, 1 );
                  N( ic, nsat )      = 1;
               end
            end
         end

         if ( verbose >= 10 )
            disp( [' a total of ', int2str( nsat ), ' continuous variables (out of ',      ...
                     int2str( ncont ),') are nearly saturated'] )
            if ( nsat > 0  )
               corresponding_N = N( 1:ncont,1:nsat )
            end
         end

      end

      %  Progress has been made.

      if ( flow < fx - eta * cmesh^2 )

         %  Reset the termination loop counter.

         term_loops = 0;

         %  Print the one-line iteration summary.

         if ( verbose > 1  )
            if ( within_training )
               if ( verbose > 2 )
                  fprintf( '\n' );
                  fprintf( '%s train     prob.obj.   training\n', indent );
                  fprintf( '%s neval       neval    performance       cmesh     status\n', ...
                           indent );
               end
               fprintf( '%s%5d  %11d   %+.6e  %4e    %+3d\n',                              ...
                        indent, ceil( neval ), nevalt, fbest, cmesh, ibest );
               if ( verbose > 2 )
                  fprintf( '\n' );
               end
            else
               if ( verbose > 3 )
                  fprintf( '\n' );
                  fprintf( '%s neval        fx        est.crit       cmesh       status\n',...
                           indent );
               end
               if ( ~full_loop  || ncont == 0 )
                  fprintf( '%s%5d  %+.6e   -----------  %4e    %+3d\n',                    ...
                           indent, ceil( neval ), fbest, cmesh, ibest );
               else
                  fprintf( '%s%5d  %+.6e  %4e  %4e    %+3d\n',                             ...
                            indent, ceil( neval ), fbest, estcrit, cmesh, ibest  );
               end
               if ( verbose > 3 )
                  fprintf( '\n' );
               end
            end
            bfo_print_x( indent, 'x', xbest, xscale, verbose, xtype, cat_dictionnary,    ...
	                 x0ref )
         end

         %  Accumulate the average direction of descent over the last 
         %  inertia iterations.

         if ( ncont > 0 && ~partially_separable ) 
            ns = size( sacc, 2 );
            s  = xbest - x;
            if ( inertia > 0 )
               if ( ns == 0 )
                  sacc = s( icont );
               elseif ( ns < inertia )
                  sacc( :, end+1 ) = s( icont );
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

            Q =  bfo_new_continuous_basis( ncont, N, ddir, npoll );
            if ( verbose >= 10 )
               newQ1 = Q
            end
         end

         %  Expand the grid for continuous variables after a successful iteration, except
         %  if success is due to the user search-step, in which case grid size is irrelevant.
         %  Note that the actual (expanded) increments for discrete variables 
         %  are never used.

         if ( ncont > 0 && ~successful_search_step )
            xincr( icont, 1 ) = min( [ xupper( icont )' - xlower( icont )';                ...
                                       alpha * xincr( icont, cur )'       ;                ...
                                       gamma * ones( ncont, 1 )'            ] )';
            cmesh   = max( xincr( icont, cur ) );
            nrefine = -1;              % reset the nbr of successive refinements to 3 
                                       % (= 2 -(-1)) before acceleration occurs again.
         end

         %  Move to the best point.

         xp       = x;
         fxp      = fx;
         x        = xbest;
         fx       = flow;
         checking = 0;

      %  Test for termination on the current grid and overall.

      else

         %  Further grid refinement is possible, possibly leading 
         %  to further minimization.

         if ( ( ncont > 0 && cmesh > epsilon ) & ~at_optimizer )

            %  Print the one-line iteration summary.

            if ( verbose > 1 )
               if ( within_training )
                  if ( verbose > 2 )
                     fprintf( '\n' );
                     fprintf( '%s train     prob.obj.   training\n', indent );
                    fprintf('%s neval       neval    performance       cmesh     status\n',...
                            indent );
                  end
                  fprintf( '%s%5d  %11d   %+.6e  %4e  refine\n',                           ...
                           indent, neval, nevalt, fbest, cmesh );
                  if ( verbose > 2 )
                     fprintf( '\n' );
                  end
               else
                  if ( verbose > 3 )
                     fprintf( '\n' );
                   fprintf('%s neval        fx        est.crit       cmesh       status\n',...
                           indent );
                  end
                  fprintf( '%s%5d  %+.6e  %4e  %4e  refine\n',                             ...
                           indent, ceil( neval ), fbest, estcrit, cmesh );
                  if ( verbose > 3 )
                     fprintf( '\n' );
                  end
               end
               bfo_print_x( indent, 'x', x, xscale, verbose, xtype, cat_dictionnary, x0ref );
            end
         
            %  Refine the grid for continuous variables, accelerating when more  
            %  than 2 successive refinements have taken place. Note that the actual
            %  (reduced) increments for discrete variables are never used.

            nrefine = nrefine + 1;
            if ( nrefine > 2 )
               xincr( icont, cur ) = max( [ 0.5 * epsilon * ones( ncont, 1 )';
                                            beta * beta * xincr( icont, cur )' ] );
            else
               xincr( icont, cur ) = max( [ 0.5 * epsilon * ones( ncont, 1 )';
                                            beta * xincr( icont, cur )' ] );
            end
            cmesh  = max( xincr( icont, cur ) );

            %   Choose a new random basis for the continuous variables.

            Q = bfo_new_continuous_basis( ncont, N, ddir, npoll );
            if ( verbose >= 10 )
                newQ2 = Q
            end

         %  Convergence is achieved on the finest grid.

         else


            %  Increment the number of termination loops performed so far.

            term_loops = term_loops + 1;

            %  Termination has been verified for the required term_basis random 
            %  basis for the continuous variables.
            %  Note that another termination loop is useless if there are no continuous 
            %  variables.  If there is only one continuous variable, a further termination 
            %  loop is only potentially useful for exploring neighbouring discrete 
            %  subspaces (if any) for depth-first search, or for exploring the neighbouring
	    %  categorical variables (if any).
            %  The detection of checking < 0 is a flag allowing to terminate after the
            %  the prescribed number of unstructured checking poll direction is the
            %  coordinate-partially-separable case.

            if  ( term_loops >= term_basis  || ~ncont || checking < 0  || at_optimizer ||  ...
                  ( ncont == 1 &&                                                          ...
	            ( term_loops > 1  || ( ( ndisc == 0  || stype ~= 1 ) && ncate == 0 ) ) ) )

               if ( verbose > 1 )
                  if ( within_training )
                     if ( verbose > 2 )
                        fprintf( '\n' );
                        fprintf( '%s train     prob.obj.   training\n', indent );
                        fprintf( [ '%s neval       neval    performance',                  ...
                                   '       cmesh     status\n'], indent );
                     end
                     fprintf( '%s%5d  %11d   %+.6e  %4e  converged\n',                     ...
                              indent, neval, nevalt, fbest, cmesh, ibest );
                     if ( verbose > 2 )
                        fprintf( '\n' );
                     end
                  else
                     if ( verbose > 3 )
                        fprintf( '\n' );
                        fprintf( [ '%s neval        fx        est.crit',                   ...
                                   '       cmesh       status\n'],  indent );
                     end
		     if ( estcrit >= 0 )
                        fprintf( '%s%5d  %+.6e  %4e  %4e  converged\n',                    ...
                                 indent, ceil( neval ), fbest, estcrit, cmesh  );
	             else
                        fprintf( '%s%5d  %+.6e                %4e  converged\n',           ...
                                 indent, ceil( neval ), fbest, cmesh  );
                     end
                     if ( verbose > 3 )
                        fprintf( '\n' );
                     end
                  end
                  bfo_print_x( indent, 'xbest', xbest, xscale, verbose, xtype,             ...
	                       cat_dictionnary, x0ref )
               end

               if ( depth == 0 )
                  if ( multilevel )
                     msg = [ indent, ' Convergence at level ', int2str( level ), ' in ',   ...
                             int2str( ceil( neval ) ), ' evaluations of ', shfname, '.'];
                  else
                     msg = [ indent, ' Convergence in ', int2str( ceil( neval ) ),         ...
                          ' evaluations of ', shfname, '.'];
                  end
                  if ( verbose > 1 )
                     disp( msg )
                  end
               end

               if ( verbose > 1 && use_cps < 0 )
                  fprintf( '\n')
                  fprintf( '%s ********************************************************\n',...
                           indent)
                  fprintf( '\n')
               end

               if ( savef >= 0 )
                  s_hist = bfo_shistupd( s_hist, idc, xbest, fbest, xincr( :, cur ),       ...
	                                xtype, cat_dictionnary, x0ref, cn_name, opt_context );
                  savok  = bfo_save( sfname, shfname, maximize, epsilon, ftarget,          ...
                                     maxeval, neval, f_hist, xtype, xincr, xscale,         ...
                                     xlower, xupper, verbose, alpha, beta, gamma,  eta,    ...
                                     zeta, inertia, stype, rseed, iota, kappa, lambda, mu, ...
                                     term_basis, use_trained, s_hist, latbasis,bfgs_finish,...
                                     training_history, fstar, tpval, ssfname, cn_name,     ...
                                     cat_states, cat_dictionnary );
                  if ( ~savok )
                     msg = [ ' BFO error: checkpointing file ', sfname',                   ...
                             ' could not be opened. Skipping final checkpointing.' ];
                     if ( verbose )
                        disp( msg )
                     end
                  end
               end
               xbest = bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref );

               %  Before the final return to the user, make sure to return the absolutely
               %  best point, as points marginally better than xbest may have been found 
               %  in the checking phase.

               if ( topmost && use_cps >= 0 )
                  if ( categorical ) 
                     l =  size( x_hist, 1 );
                     if ( l > 0 )
                        [ ffinal, ivbest ] = min( f_hist( end-l+1:end ) );
                        if ( ffinal < fbest )
                           fbest = ffinal;
                           xbest = x_hist{ ivbest }{ 1 };
                        end
                     end
                  else
                     l = size( x_hist, 2 );
                     if ( l > 0 )
                        [ ffinal, ivbest ] = min( f_hist( end-l+1:end ) );
                        if ( ffinal < fbest )
                           fbest = ffinal;
                           xbest = x_hist( 1:n, ivbest );
                        end
                     end
                  end
               end

               %  Possibly call for cleanup in the search-step function.

               if ( search_step_needs_cleaning_up ) 
                  bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
               end

               %  Terminate with convergence

               return

            %  More loops on random basis are required to assert termination.

            else

               if ( verbose > 1 )
                  if ( within_training )
                     if ( verbose > 2 )
                        fprintf( '\n' );
                        fprintf( '%s train     prob.obj.   training\n', indent );
                        fprintf( [ '%s neval       neval    performance',                  ...
                                   '       cmesh     status\n' ], indent );
                     end
                     fprintf( '%s%5d  %11d   %+.6e  %4e  checking\n',                      ...
                              indent, neval, nevalt, fbest, cmesh, ibest );
                     if ( verbose > 2 )
                        fprintf( '\n' );
                     end
                  else
                     if ( estcrit >= 0 )
                        if ( verbose > 3 )
                           fprintf( '\n' );
                           fprintf( [ '%s neval        fx        est.crit',                ...
                                      '       cmesh       status\n'],  indent );
                        end
                        fprintf( '%s%5d  %+.6e  %4e  %4e  checking\n',                     ...
                                 indent, ceil( neval ), fbest, estcrit, cmesh  );

                     %  estcrit may (exceptionally) be -1 when BFO has been
                     %  restarted after convergence has occurred in the continuous
                     %  variables : the forward-backward loop on these variables
                     %  is then skipped at the first iteration of the restarted
                     %  algorithm and estcrit is not assigned a meaningful
                     %  value. Moreover printing a new function value is only
                     %  informative if there are discrete/categorical variables.

                     elseif ( ndisc + ncate > 0 ) 
                        if ( verbose > 3 )
                           fprintf( '\n' );
                           fprintf( [ '%s neval        fx                ',                ...
                                      '       cmesh       status\n'],  indent );
                        end
                        fprintf( '%s%5d  %+.6e                %4e  checking\n',            ...
                                 indent, ceil( neval ), fbest, cmesh  );
                     end
                     if ( verbose > 3 )
                        fprintf( '\n' );
                     end
                  end
                  bfo_print_x( indent, 'xbest', xbest, xscale, verbose, xtype,             ...
	                       cat_dictionnary, x0ref );
               end

               % Reset the number of successive refinements to 3 (= 2 -(-1)) before 
               % acceleration occurs again.

               nrefine = -1;

               %   Again, if there is only one continuous variable, the new termination 
               %   loop is only potentially useful for exploring neighbouring discrete 
               %   subspaces (if any) for depth-first search.  If this the case, unset the
               %   fbloop flag to skip the continuous variables in the next termination 
               %   loop. 

               fbloop = ( ncont > 1 || ndisc == 0 || stype ~= 1 );

               %   Choose a new random basis for the continuous variables.

               if ( fbloop  && ncont > 1 )
                  Q = bfo_new_continuous_basis( ncont, N, ddir, npoll );
                  if ( verbose >= 10 ) 
                     newQ3 = Q
                  end
               end

            end
         end
      end

   end %  end of the main (non-CPS) part of the optimization loop

end % of the main optimization loop



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

   s_hist = bfo_shistupd( s_hist, idc, xbest, fbest, xincr( :, cur ), xtype,               ...
                          cat_dictionnary, x0ref, cn_name, opt_context );
   savok  = bfo_save( sfname, shfname, maximize, epsilon, ftarget, maxeval,  neval,        ...
                      f_hist, xtype, xincr, xscale, xlower, xupper, verbose,               ...
                      alpha, beta, gamma, eta, zeta, inertia, stype, rseed, iota,          ...
                      kappa, lambda, mu, term_basis, use_trained, s_hist, latbasis,        ...
                      bfgs_finish, training_history, fstar, tpval, ssfname, cn_name,       ...
                      cat_states, cat_dictionnary );
   if ( ~savok )
      msg = [ ' BFO error: checkpointing file ', sfname',                                  ...
              ' could not be opened. Skipping final checkpointing.' ];
      if ( verbose )
         disp( msg )
      end
   end
end

xbest = bfo_pack_x( xbest, xscale, xtype, cat_dictionnary, x0ref );

%  Possibly call for cleanup in the search-step function.

if ( search_step_needs_cleaning_up ) 
   bfo_search_step_cleanup( bfo_srch, categorical, sum_form );
end

%  Terminate with convergence

return

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 


function [ xbest, fbest, msg, neval, x_hist, f_hist, hist ] =                              ...
                             bfo_core( f, x0, fx0, ovars, epsilon, maxeval, varargin )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  A 'core' BFO, i.e. a minimizing routine where the function f is minimized as a function
%  of the continuous variables of index given by ovars, possibly subject to bounds.
%  The function f can be defined in sum_form (allowing for coordinate partial separability).
%  ************* No coherence checks are performed on input arguments. *********************
%
%  INPUT:
%
%  f       : the objective function's handle
%  x0      : the initial point
%  f0      : the value(s) of the (element) objective function(s) at x0
%  ovars   : the indeces of the continuous optimization variables
%  xlower  : the lower bounds on the variables (same size as x0)
%  xupper  : the upper bounds on the variables (same size as x0)
%  epsilon : the length of the mesh under which termination occurs
%  maxeval : the maximum number of function evaluations
%  elset   : the indeces of the element functions occurring in f
%  eldom   : the element domains, if relevant, not used or referenced otherwise.
%  varargin: an optional list of (keyword,value) pairs, where the only keywords
%            allowed are 
%              'xlower', 'xupper', 'eldom', 'elset', 'alpha', 'beta', 'gamma', 
%              'delta', 'eta', 'intertia', 'termination-basis' and 'indentation'.
%           The definitions and corresponding allowed values are identical to those
%           of the main BFO call, except for the last one, where the value associated
%           with the 'indentation' keyword is a string (typically a few blanks).
%
%  OUTPUT:
%
%  fbest : the (approximate) optimal value,
%  xbest : the (approximate) minimizer,
%  neval : the number of objective function evaluations during the minimization,
%  msg   : a termination message,
%  x_hist: the history of the evaluations points,
%  f_hist: the corresponding history of objective function values,
%  hist  : the element-wise history of the evaluations, if the objective function
%          was defined in sum-form.  For element iel,
%          hist{iel}.fel  : is the history of the values of the iel-th element function
%          hist{iel}.xel  : is the history of the associated points
%          hist{iel}.fbest: is the value of the iel-th element occurring in the best value
%                           of the objective function found (fbest).
%
%  PROGRAMMING: Ph. Toint, January 2018 (this version 4 III 2018).
%
%  DEPENDENCIES: bfo_print_x, bfo_feasible_cstep, bfo_print_vector, bfo_new_continuous_basis
%
%  CONDITIONS OF USE: Use at your own risk! No guarantee of any kind given or implied.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


verbose    = 0;                    % verbosity for this function
myinf      = 1.0e25;               % a numerical value for plus infinity

%  Set default (empty) return values.

msg       = 'Unexpected exit.';
neval     = 0;

%  Ckeck for sum-form objective function.

nvars     = length( ovars );
n         = length( x0 );
sum_form  = iscell( f );
if ( sum_form )
   eldom  = {};
   elset  = [];
   f      = f{1};
   nel    = length( f );
   hist   = cell( 1, nel );
   fbest  = 0;
   for iel = 1:nel
      hist{ iel } = struct( 'fel', [], 'xel', [], 'fbest', [] ) ;
      fbest       = fbest + fx0( iel );
   end
else
   fbest  = fx0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Set default parameters before optional arguments are allowed to modify them.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xlower     = -Inf * x0;            % no lower bound by default
xupper     =  Inf * x0;            % no upper bound by default
alpha      = 1.4248;               % the optimized (trained) algorithmic parameters
beta       = 0.1997;
gamma      = 2.3599;
cmesh      = 1.0368;
eta        = 0.0001;
inertia    = 10;
indent     = '';                   % the desired indentation for output
term_basis = 5;                    % the number of random basis used for termination

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Process the (remaining) variable argument list  %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Note that the only acceptable keywords are:
%       'xlower', 'xupper', 'eldom', 'elset', 'alpha', 'beta', 'gamma', 'delta',
%       'eta', 'inertia', 'indentation' and 'termination-basis'
%  No checks!!!

for i = 1:2:length( varargin )
   switch( varargin{ i } )
   case 'xlower'
      xlower = varargin{ i+1 };
   case 'xupper'
      xupper = varargin{ i+1 };
   case 'eldom'
      eldom  = varargin{ i+1 };  
   case 'elset'
      elset  = varargin{ i+1 };  
   case 'alpha'
      alpha  = varargin{ i+1 };  
   case 'beta'
      beta   = varargin{ i+1 };  
   case 'gamma'
      gamma  = varargin{ i+1 };  
   case 'delta'
      cmesh  = varargin{ i+1 };  
   case 'eta'
      eta    = varargin{ i+1 };  
   case 'inertia'
      inertia = varargin{ i+1 };  
   case 'indentation'
      indent  = varargin{ i+1 };
   case 'termination-basis'
      term_basis = varargin{ i+1 };  
   end
end

%  Use implicit domains and all elements if not otherwise specified.

if ( sum_form )
   if ( isempty( eldom ) )
      for iel = 1:nel
          eldom{ iel } = [ 1:n ];
      end
   end
   if ( isempty( elset ) )
      elset = [ 1:nel ];
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      Optimization     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%   Analyze the various types of variables and the starting point.  %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Define the optimization variables.

xlower = xlower( ovars );
xupper = xupper( ovars );
is_bounded  = ( max( xlower ) > - myinf ) || ( min( xupper ) <   myinf );
if ( is_bounded )
   maxcmesh = min( xupper - xlower );
else
   maxcmesh = Inf;
end

%  Initialize the history of previous iterates.

x_hist = [ ];
f_hist = [ ];

%   Initialize the termination checks counter.

term_loops = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                            %
%                Phase 2: Apply the BFO algorithm to the verified problem.                   %
%                                                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Set up the initial basis of continuous directions (and a default for other types).
%   Note that Q is not used as a full matrix for CPS problems, and therefore not created 
%   because of its potentially very large size.

Q     = eye( nvars, nvars );
sacc  = [];
npoll = nvars;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Print header, optimization direction, objective function's name and starting point value.  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Print zero-th iteration summary.

if ( verbose )
   fprintf( '\n%s --------------- BFO_core ----------------\n', indent )
   fprintf(   '%s neval        fx         cmesh      status\n', indent );
   fprintf( '%s%5d  %+.6e  %4e\n', indent, ceil( neval ), fbest, cmesh );
   if ( verbose > 3 )
      fprintf( '\n' );
      bfo_print_x( indent, 'x0', x0, [], verbose )
   end
end

%   Initialize the current and previous iterates, and the number of successive
%   refinement steps. 

x       = x0;                         % the iterate
xbest   = x0;
fx      = fbest;                      % the function value
xp      = x0;                         % the previous iterate
fxp     = fbest;                      % the function value at the previous iterate
nrefine = 0;                          % if > 0, the index of the current refinement iteration

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%         THE OPTIMIZATION       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%  This the main iteration's poll loop  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

checking = 0;                          % no checking for optimality yet

for itg = 1:maxeval

   %  Initialize the index of the best move in the poll cycle.

   ibest = 0;

   %  Set a flag if the algorithm is in the convergence checking mode.

   if ( checking >= 0 )
       checking =  term_basis > 1;
   end

   %  Compute the unknown values of the objective function at the
   %  neighbouring points on the current grid.

   loopbreak = 0;                % no break for sufficient decrease yet
   ic        = 0;                % initialize the continuous variable counter
   gcdiff    = [];
   
   for i = ovars                 % loop on the variables

      ic = ic + 1;

      if ( verbose >= 4 )
         disp( ' ' )
         disp( ' Considering neighbours of ' )
         bfo_print_x( indent, 'x', x, [], verbose )
         disp( [ ' relative to component ', int2str( i ), ' (', int2str( ic ),             ...
	         '-th active continous variable)' ] )
      end

      %  Loop on the neighbouring values of the current iterate
      %  relative to variable i.

      xtry = x;

      afwd_abwd = [];
      ffwd_fbwd = [ Inf, Inf ];
      for inghbr = 1:2

         %  Construct the forward/backward neighbour.

         if ( inghbr == 1 )
            step = cmesh * Q( 1:nvars, ic );
	 else
	    step = -step;
	 end

         if ( is_bounded )
            [ xtry( ovars ), stepsize ] = bfo_feasible_cstep( x( ovars ), step, xlower, xupper );
            afwd_abwd = [ afwd_abwd, stepsize ];
         else
            xtry( ovars ) = x( ovars ) + step;
            afwd_abwd     = [ afwd_abwd, norm( xtry( ovars ) - x( ovars ) ) ];
         end
    
         %  Print the neighbour, if requested.

         if ( verbose >= 4 )
             bfo_print_x( indent, [ 'neighbour ', int2str(inghbr) ], xtry, [], verbose )
         end
	 
         %  Compute the minimum distance between the trial point and the 
         %  previous iterates.

         dxxp = norm( xp( ovars ) - xtry( ovars ), 'inf' );
	       
         %  Verify that the trial point is different from previous ones.

         if ( dxxp > eps ) 

            %  Verify that the trial point is different from the current iterate. 

            dxx = norm( xtry( ovars ) - x( ovars ) );
            if (  dxx > eps  )

               %  Evaluate the objective function at the neighbouring point.
			
               if ( sum_form )
	          fnghbr  = 0;
		  fiel    = zeros( 1, nel );
                  for iel = 1:nel
                     xiel        = xtry( eldom{ iel } );
                     fiel( iel ) = f{iel}( elset( iel ), xiel );
                     fnghbr      = fnghbr + fiel( iel );
                     inext       = length( hist{ iel }.fel ) + 1;
                     hist{ iel }.fel( inext )    = fiel( iel );
                     hist{ iel }.xel( :, inext ) = xiel;
                  end
               else
                  fnghbr = f( xtry );
    	       end 
	       
               %  Print the result, if in debug mode.

               if ( verbose >= 4 )
                  fprintf( '%s value of f at neighbour %3d is %.12e\n',                    ...
		           indent, inghbr, fnghbr );
                  if ( sum_form )
                     bfo_print_vector( indent, 'fi', fiel );
                  end
               end

               %  Check for undefined function values.

               if ( isnan( fnghbr ) )
                  fnghbr = Inf;
               end
               ffwd_fbwd( inghbr ) = fnghbr;

               %  Update histories.

               neval                = neval + 1;
	       f_hist( neval )      = fnghbr;
               x_hist( 1:n, neval ) = xtry;

               %  Update the best value so far.

               if ( fnghbr < fbest - eta * cmesh^2 )
                  xbest = xtry;
                  fbest = fnghbr;
                  if ( sum_form )
                     for iel = 1:nel
	                hist{ iel }.fbest = fiel( iel );
                     end
                  end
                  ibest = i;
               end
  
            else   %   No move from x (cmesh tiny or active bounds)
	    
               fnghbr = fx;
	       
            end

            %  Terminate if the maximum number of function evaluations has been reached.
		     
            if ( neval >= maxeval )
               msg = [ ' Maximum number of ', int2str( maxeval ), ' evaluations reached.' ];
               if ( verbose )
                  disp( msg )
               end
               return;
            end

            %  Terminate the poll loop on the variables if sufficient 
            %  decrease has already been found.  This is achieved by setting the
	    %  loopbreak flag, breaking the neighbours' loop and testing the
	    %  flag on exit of that loop, for possibly breaking the poll loop.

            if ( fx - fbest >= eta * cmesh^2 )
	       loopbreak = 1;
               break; 
            end
		  
            %  Avoid computing the backward value if the forward value does not give
	    %  a sufficient increase (note that the forward value does not give a
	    %  sufficient decrease either).

            if ( inghbr == 1 &&  dxx > eps  && fnghbr - fx <= eta * cmesh^2 )
               break;
            end
		  
         else   %  Return to xp
	 
            fnghbr = fxp;
	    
         end

      end %  End of the loop on the neighbours of variable i

      %  Break from the loop on the variables if sufficient decrease has been found.
	
      if ( loopbreak )
         if ( verbose >= 4 )
            disp( [ ' sufficient decrease obtained:',                                      ...
	            ' breaking the poll loop after variable ', int2str( i ) ] )
         end
	 break;
	 
      end  

      %  Update the estimate of the gradient (in base Q),
      %  avoiding the use of meaningless function values.

      if ( ffwd_fbwd( 1 ) < Inf && ffwd_fbwd( 2 ) < Inf )
         gcdiff( ic ) = ( ffwd_fbwd(1) - ffwd_fbwd(2) ) / ( afwd_abwd(1) + afwd_abwd(2) );
      elseif ( ffwd_fbwd( 1 ) < Inf )      % avoid using meaningless function values
         gcdiff( ic ) = ( ffwd_fbwd(1) - fx ) / afwd_abwd( 1 );
      elseif ( ffwd_fbwd( 2 ) < Inf )      % avoid using meaningless function values
         gcdiff( ic ) = ( fx - ffwd_fbwd(2) ) / afwd_abwd( 2 );
      end

   end  %  End of the loop on the dimensions

   %   Set the hopeful descent direction to the negative gradient.

   full_grad = ( length( gcdiff ) == nvars );
   if ( full_grad )
      ddir = - ( gcdiff * Q )';
   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  TERMINATION STEP %%%%%%%%%%%%%%%%%%%%%%%%%
   %% The termination step

   %  Determine which bounds on continuous variables are nearly saturated at 
   %  xbest, if any, as well as the matrix of normal to these bounds. (A bound 
   %  is nearly saturated when its distance from the current iterate is not 
   %  larger than the current grid spacing).

   isat = [];              % to be the set of nearly saturated continuous variables
   nsat = 0;               % the number of nearly saturated bounds
   N    = [];              % the normals to nearly saturated bounds
   if ( is_bounded )
      ic = 0;
      for i = ovars
         ic = ic + 1;      % the index of i in the set of continuous variables

         %  Check if nearly saturated.

         if ( xbest( ic ) - xlower( ic ) <= cmesh || xupper( ic ) - xbest( ic ) <= cmesh )
            if ( verbose >= 10 )
               disp( [' variable ', int2str(ic), ' has a (nearly) saturated bound'] )
            end
            isat( end+1 ) = ic;
            nsat          = nsat + 1;
            if ( nsat <= npoll )
               N( 1:nvars, nsat ) = zeros( nvars, 1 );
               N( ic, nsat )      = 1;
            end
         end
      end

      if ( verbose >= 10 )
         disp( [' a total of ', int2str( nsat ), ' continuous variables (out of ',         ...
                  int2str( nvars ),') are nearly saturated'] )
         if ( nsat > 0  )
            corresponding_N = N( 1:nvars,1:nsat )
         end
      end
      

   end

   %  Progress has been made.

   if ( fbest < fx - eta * cmesh^2 )

      %  Reset the termination loop counter.

      term_loops = 0;

      %  Print the one-line iteration summary.

      if ( verbose  )
         if ( verbose > 3 )
            fprintf( '\n' );
            fprintf( '%s neval        fx         cmesh       status\n', indent );
         end
         fprintf( '%s%5d  %+.6e  %4e   %+3d\n', indent, ceil( neval ), fbest, cmesh, ibest );
         if ( verbose > 3 )
            fprintf( '\n' );
         end
         bfo_print_x( indent, 'x', xbest, [], verbose );
      end

      %  Accumulate the average direction of descent over the last 
      %  inertia iterations.

      if ( inertia > 0 )
         ns = size( sacc, 2 );
         s  = xbest( ovars ) - x( ovars );
         if ( ns == 0 )
            sacc = s;
         elseif ( ns < inertia )
            sacc( :, ns+1 ) = s;
         else
            sacc = [ sacc( 1:nvars,2:inertia ), s ];
         end
         avs = sum( sacc, 2 );

         %  Use the average direction if an approximate gradient is not available.

         if ( ~full_grad )
             ddir = avs;
         end
      end

      %  Expand the grid for continuous variables after a successful iteration.

      cmesh   = min( [ gamma,  alpha*cmesh,  maxcmesh ] );
      Q       =  bfo_new_continuous_basis( nvars, N, ddir, npoll );
      nrefine = -1;                  % reset the nbr of successive refinements to 3 
                                     % (= 2 -(-1)) before acceleration occurs again.

      %  Move to the best point.

      xp       = x;
      fxp      = fx;
      x        = xbest;
      fx       = fbest;
      checking = 0;

   %  Test for termination on the current grid and overall.

   else

      %  Further grid refinement is possible, possibly leading to further minimization.

      if ( cmesh > epsilon )

         %  Print the one-line iteration summary.

         if ( verbose )
            if ( verbose > 3 )
                fprintf( '\n' );
                fprintf('%s neval        fx         cmesh       status\n', indent );
            end
            fprintf( '%s%5d  %+.6e  %4e  refine\n', indent, ceil( neval ), fbest, cmesh );
            if ( verbose > 3 )
               fprintf( '\n' );
               bfo_print_x( indent, 'x', x, [], verbose );
            end
         end
         
         %  Refine the grid for continuous variables, accelerating when more  
         %  than two successive refinements have taken place.

         nrefine = nrefine + 1;
         if ( nrefine > 2 )
            cmesh = max( [ 0.5*epsilon, beta*beta*cmesh ] );
         else
            cmesh = max( [ 0.5*epsilon, beta*cmesh ] );
         end

         %  Compute a new basis taking advantage of a finite-difference
	 %  approximation of the gradient.
	 
         Q = bfo_new_continuous_basis( nvars, N, ddir, npoll );

      %  Convergence is achieved on the finest grid.

      else

         %  Increment the number of termination loops performed so far.

         term_loops = term_loops + 1;

         %  Termination has been verified for the required term_basis random 
         %  basis for the continuous variables.
         %  Note that another termination loop is useless if there are no continuous 
         %  variables.  

         if  ( term_loops >= term_basis || nvars == 1 )

            if ( verbose )
               if ( verbose > 3 )
                  fprintf( '\n' );
                  fprintf( [ '%s neval        fx         cmesh       status\n'],  indent );
               end
               fprintf( '%s%5d  %+.6e  %4e  converged\n', indent, ceil(neval), fbest, cmesh );
               if ( verbose > 3 )
                  fprintf( '\n' );
                  bfo_print_x( indent, 'xbest', xbest, [], verbose );
               end
            end
            msg = [ indent, ' Convergence in ', int2str( ceil( neval ) ), ' evaluations.'];
            if ( verbose )
               disp( msg )
            end
            if ( verbose > 1  )
               fprintf( '\n')
               fprintf( '%s --------------------------------------------------------\n',   ...
                        indent)
               fprintf( '\n')
            end

            return

         %  More loops on random basis are required to assert termination.

         else

            if ( verbose > 1 )
               bfo_print_x( indent, 'xbest', xbest, [], verbose );
            end

            % Reset the number of successive refinements to 3 (= 2 -(-1)) before 
            % acceleration occurs again.

            nrefine = -1;

            %   Choose a new random basis for the continuous variables.

            Q = bfo_new_continuous_basis( nvars, N, ddir, npoll );

         end
      end
   end  %  of the termination analysis

end % of the main optimization loop

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Algorithm's termination  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Exit because the maximum number of iterations was reached

msg = [' Maximum number of ', int2str( maxeval ), ' evaluations reached.' ];
if ( verbose )
   disp( msg )
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 


function [ xbest, fbest, msg, nf, hist ] =                                                 ...
         bfo_min1d( f, x0, f0, j, xlower, xupper, alpha0, epsilon, nfmax, elset, eldom,    ...
                    kappa, lambda, mu )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  A 'no frills' unidimensional minimizing routine, where the function f( x0 + alpha*e_j )
%  is minimized as a function of alpha,  subject to bounds on x0 + alpha*e_j. 
%  The function f can be defined in sum_form (allowing for coordinate partial separability).
%
%  INPUT:
%
%  f       : the objective function's handle
%  x0      : the initial point
%  f0      : the value of the objective function at the initial point
%  j       : the coordinate along which the one-dimensional minimization is required
%  xlower  : the lower bounds on the variables
%  xupper  : the upper bounds on the variables
%  alpha0  : the initial stepsize along e_j from x0
%  epsilon : the length of the bracket under which termination occurs
%  nfmax   : the maximum number of function evaluations
%  elset   : the indeces of the element functions occurring in f
%  eldom   : the element domains, if relevant, not used or referenced otherwise.
%  kappa   : the interval expansion factor when quadratic interpolation is not used
%  lambda  : the min interval expansion factor when quadratic interpolation is used
%  mu      : the max interval expansion factor when quadratic interpolation is used
%
%  OUTPUT:
%
%  fbest: the (approximate) optimal value
%  xbest: the (approximate) minimizer
%  msg  : a termination message
%  nf   : the number of objective function evaluations during the minimization
%  hist : the element-wise history of the evaluations, if the objective function
%         was defined in sum-form.  For element iel,
%         hist{iel}.fel  : is the hsotory of the values of the iel-th element function
%         hist{iel}.xel  : is the hostoty of the associated points
%         hist{Iel}.fbest: is the vale of the iel-th element occurring in the best value
%                          of the objective function found (fbest).
%
%  PROGRAMMING: Ph. Toint, November 2017 (this version 20 II 2018).
%
%  DEPENDENCIES: ~
%
%  CONDITIONS OF USE: Use at your own risk! No guarantee of any kind given or implied.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

msg     = ''; %  Exit message set to 'ok'.
nf      = 0;  %  No function value computed so far.
verbose = 0;  %  Verbosity level

if ( verbose )
   disp( ' ***********  new call to min1d  *********' )
end

%  Check for sum-form objective function.

n        = length(x0);
sum_form =  iscell( f );
if ( sum_form )
   f = f{1};
   nel = length( f );
   if ( isempty( eldom ) )
      for iel = 1:nel
        eldom{ iel } = [ 1:n ];
      end
   end
   nel      = length( f );
   hist     = cell( 1, nel );
   for iel  = 1:nel
      hist{ iel }     = struct( 'fel', [], 'xel', [], 'fbest', [] ) ;
      hist{ iel }.fel = [ f0( iel ) ];
      hist{ iel }.xel = x0( eldom{ iel } );
   end
end

%  Find lower and upper bounds on alpha.

if ( isempty( xlower ) && isempty( xupper ) )
   alower = -Inf;
   aupper =  Inf;
else
   alower =  - ( x0( j ) - xlower( j ) );
   aupper =      xupper( j ) - x0( j )  ;
end

if ( verbose )
   disp( [ ' I   alower = ', num2str( alower ), ' aupper = ', num2str( aupper ),           ...
           ' epsset = ', num2str( epsilon ), ' alpha0 = ', num2str( alpha0 ) ] )
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Phase 1: find an orientation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( sum_form )
   fleft = sum( f0 );
else
   fleft = f0;
end
aleft = 0;
xleft = x0;
ileft = 1;

%  Evaluate the first point on the right.

bracket     = 0;
aright      = min ( alpha0, aupper );
xright      = x0;
xright( j ) = xright( j ) + aright;
if ( sum_form )
   fright  = 0;
   for iel   = 1:nel
      xiel   = xright( eldom{ iel } );
      fiel   = f{iel}( elset( iel ), xiel );
      fright = fright + fiel;
      iright = length( hist{ iel }.fel ) + 1;
      hist{ iel }.fel( iright )    = fiel;
      hist{ iel }.xel( :, iright ) = xiel;
   end
else
   fright  = f( xright );
end
if ( isnan( fright ) )
   fright = Inf;
end
nf = nf + 1;

%  Check for nfmax and return the best point.

if ( nf >= nfmax )
   if ( fleft <= fright )
      xbest = xleft;
      fbest = fleft;
      if ( sum_form )
         for iel = 1:nel
	    hist{ iel }.fbest = hist{ iel }.fel( ileft );
	 end
      end
   else
      xbest = xright;
      fbest = fright;
      if ( sum_form )
         for iel = 1:nel
	    hist{ iel }.fbest = hist{ iel }.fel( iright );
	 end
      end
   end
   msg = [' Maximum number of ', int2str( nfmax ), ' evaluations reached.' ];
   return
end

%  Determine orientation if the two points have different function values.

aprev = [];
fprev = [];
if ( fright > fleft )                          %  Move on the left.       
   dir = -1;
   if ( verbose )
      disp( [ ' DL  fleft = ', num2str( fleft ), ' fright = ', num2str( fright ) ] )
      disp( [ ' DL  aleft = ', num2str( aleft ), ' aright = ', num2str( aright ) ] )
   end 
elseif ( fright < fleft )                      %  Move on the right.
   dir =  1;
   if ( verbose )
      disp( [ ' DR  fleft = ', num2str( fleft ), ' fright = ', num2str( fright ) ] )
      disp( [ ' DR  aleft = ', num2str( aleft ), ' aright = ', num2str( aright ) ] )
   end
else                                           %  Evaluate the first point on the left.
   amid       = aleft;
   fmid       = fleft;
   xmid       = xleft;
   imid       = ileft;
   aleft      = max( -aright, alower );
   xleft      = x0;
   xleft( j ) = xleft( j ) + aleft;
   if ( sum_form )
      fleft = 0;
      for iel  = 1:nel
         xiel  = xleft( eldom{ iel } );
         fiel  = f{iel}( elset( iel ), xiel );
         fleft = fleft + fiel;
         ileft = length( hist{ iel }.fel ) + 1;
         hist{ iel }.fel( ileft )    = fiel;
         hist{ iel }.xel( :, ileft ) = xiel;
      end
   else
      fleft   = f( xleft );
   end
   if ( isnan( fleft ) )
      fleft = Inf;
   end
   
   nf = 2;
   if ( verbose )
      disp( [ ' D?  fleft = ', num2str( fleft ), ' fmid = ', num2str( fmid ),              ...
              ' fright = ', num2str( fright ) ] )
      disp( [ ' D?  aleft = ', num2str( aleft ), ' amid = ', num2str( amid ),              ...
              ' aright = ', num2str( aright ) ] )
   end
   
   %  Check for nfmax and return the best point.

   if ( nf >= nfmax )
      if ( fleft < fmid )
         xbest = xleft;
	 fbest = fleft;
         if ( sum_form )
            for iel = 1:nel
	       hist{ iel }.fbest = hist{ iel }.fel( ileft );
	    end
         end
      else
         xbest = xmid
	 fbest = fmid;
         if ( sum_form )
            for iel = 1:nel
	       hist{ iel }.fbest = hist{ iel }.fel( imid );
	    end
         end
      end
      msg = [' Maximum number of ', int2str( nfmax ), ' evaluations reached.' ];
      return
   end

   %  Make another attempt at determining the orientation.
   
   if ( fleft < fmid )                         %  Move on the left.
      aprev  = aright;
      fprev  = fright;
      aright = amid;
      xright = xmid;
      fright = fmid;
      iright = imid;
      dir    = -1;
   else                                        %  All 3 points are identical: bracket.
      bracket = 1;
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Phase 2: find a bracket containing a minimizer, given a direction and an interval
%           [ aleft, aright] with associated function values.
%           The flag bracket is 0 as long as the desired minimizer is outside the interval
%           [ aleft, aright ].  It is 1 if, for [ aleft, amid, aright ] and corresponding
%           function values [ fleft, fmid, fright ], one has that fmid <= min( fleft, fright),
%           in which case one knows that the minimum of the interpolating quadratic lies in
%           [ aleft, aright ].  When fmid > min( fleft, fright ) and either aleft = alower or
%           aright = aupper, the sought minimizer is also in [ aleft, aright] and bracket is 
%           set to 2.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

flat = 0;%D

while ( ~bracket ) 

   %  Moving to the right
   
   if ( dir > 0 )
      if ( verbose )
         disp( [ ' MR  fleft = ', num2str( fleft ), ' fright = ', num2str( fright ) ] )
         disp( [ ' MR  aleft = ', num2str( aleft ), ' aright = ', num2str( aright ) ] )
      end

      % The right point is not at the upper bound.

      if ( ( aupper - aright ) > eps )             

         if ( isempty( aprev ) ) 
            aplus = aright + ( aright - aleft );
         else                                     % Approx the min by quadratic interpolation.
            q = [ 0.5*(aprev-aright)^2 aprev-aright ;
	          0.5*(aleft-aright)^2 aleft-aright ] \ [ fprev-fright; fleft-fright ];
	    if ( q(1) > eps )
	       aqmin = aright - q(2)/q(1);
	       aplus = max( aright + lambda * ( aright - aleft ),                          ...
         	                min( aqmin, aright + mu * ( aright - aleft ) ) );
            else
               aplus = aright + kappa * ( aright - aleft );
	    end
         end

         if ( aplus >= aupper )                   %  At upper bound: bracket.
            amid    = aright;
	    xmid    = xright;
	    fmid    = fright;
	    imid    = iright;
            aright  = aupper;
            xright  = x0;
	    xright( j ) = xright( j ) + aupper;
            if ( sum_form )
               fright = 0;
               for iel   = 1:nel
	          xiel   = xright( eldom{ iel } );
                  fiel   = f{iel}( elset( iel ), xiel );
                  fright = fright + fiel;
	          iright = length( hist{ iel }.fel ) + 1;
                  hist{ iel }.fel( iright )   = fiel;
                  hist{ iel }.xel( :, iright ) = xiel;
               end
            else
               fright  = myf( xright );
            end
            if ( isnan( fright ) )
               fright = Inf;
            end
            nf      = nf + 1;
	    bracket = 2;
         else                                     %  Not at upper bound
            xplus = x0;
	    xplus( j ) = xplus( j ) + aplus;
            if ( sum_form )
               fplus = 0;
               for iel  = 1:nel
                  xiel  = xplus( eldom{ iel } );
                  fiel  = f{iel}( elset( iel ), xiel );
                  fplus = fplus + fiel;
	          iplus = length( hist{ iel }.fel ) + 1;
                  hist{ iel }.fel( iplus )    = fiel;
                  hist{ iel }.xel( :, iplus ) = xiel;
               end
            else
               fplus = f( xplus );
            end
            if ( isnan( fplus ) )
               fplus = Inf;
            end
            nf = nf + 1;
            if ( fplus <= min( fleft, fright ) + eps )
               flat   = 1;
               fmid   = fright;
               fright = fplus;
               break;
            end
            if ( fplus > min( fleft, fright ) )   %  Bracket found: [ aleft amid aright ]
               bracket = 1; 
	       amid    = aright;
	       xmid    = xright;
	       fmid    = fright;
	       imid    = iright;
            else                                  % No bracket yet: move further on the right.
               aprev   = aleft;
	       fprev   = fleft;
	       aleft   = aright;
	       xleft   = xright;
	       fleft   = fright;
	       ileft   = iright;
            end
            aright = aplus;
	    xright = xplus;
            fright = fplus;
	    iright = iplus;
         end

      %  The right point is at the upper bound.

      else

         %  Test for temination.

         if ( aright - aleft <= 0.5 * epsilon )
            fmid = Inf; % a fake value to make sure it is not chosen in 
                        % the final termination test
            break

         %  Bisect the interval

         else
            aplus = ( aright + aleft ) / 2;
            xplus = x0;
	    xplus( j ) = xplus( j ) + aplus;
            if ( sum_form )
               fplus = 0;
               for iel  = 1:nel
                  xiel  = xplus( eldom{ iel } );
                  fiel  = f{iel}( elset( iel ), xiel );
                  fplus = fplus + fiel;
	          iplus = length( hist{ iel }.fel ) + 1;
                  hist{ iel }.fel( iplus )    = fiel;
                  hist{ iel }.xel( :, iplus ) = xiel;
               end
            else
               fplus = f( xplus );
            end
            if ( isnan( fplus ) )
               fplus = Inf;
            end
            nf = nf + 1;
            if ( fplus <= min( fleft, fright ) + eps )
               flat  = 1;
               fmid  = fleft;
               fleft = fplus;
               break;
            end
            if ( fplus <= min( fleft, fright ) )  %  Bracket found
               bracket = 1;                       % [ aleft amid aright ]
               amid    = aplus;
               xmid    = xplus;
               fmid    = fplus;
               imid    = iplus;
            else                                  %  No bracket, halve the interval.
               aleft   = aplus;
               xleft   = xplus;
               fleft   = fplus;
               ileft   = iplus;
            end
         end
      end

   %  Moving to the left
   
   else % if ( dir < 0 )

      if ( verbose )
         disp( [ ' ML  fleft = ', num2str( fleft ), ' fright = ', num2str( fright ) ] )
         disp( [ ' ML  aleft = ', num2str( aleft ), ' aright = ', num2str( aright ) ] )
      end

      %  The left point is not at the lower bound.

      if ( aleft - alower > eps )

         if ( isempty( aprev ) ) 
            aplus = - aright;
         else                                     % Approx the min by quadratic interpolation.
            q = [ 0.5*( aprev-aleft)^2  aprev-aleft ;
	          0.5*(aright-aleft)^2 aright-aleft ] \ [ fprev-fleft; fright-fleft ];
	    if ( q( 1 ) > eps )
	       aqmin = aleft - q(2)/q(1);
	       aplus = max( aleft - mu * ( aright-aleft ),                                ...
         	            min( aqmin, aleft - lambda * ( aright - aleft ) ) );
            else
               aplus = aleft - kappa * ( aright - aleft );
	    end
         end

         if ( aplus <= alower )                   %  At lower bound: bracket.
            amid    = aleft;
	    xmid    = xleft;
	    fmid    = fleft;
	    imid    = ileft;
            aleft   = alower;
            xleft   = x0;
	    xleft( j ) = xleft( j )+ alower;
            if ( sum_form )
               fleft    = 0;
               for iel  = 1:nel
	          xiel  = xleft( eldom{ iel } );
                  fiel  = f{iel}( elset( iel ), xiel );
                  fleft = fleft + fiel;
	          ileft = length( hist{ iel }.fel ) + 1;
                  hist{ iel }.fel( ileft )    = fiel;
                  hist{ iel }.xel( :, ileft ) = xiel;
               end
            else
               fleft = f( xleft );
	    end
            if ( isnan( fleft ) )
               fleft = Inf;
            end
            nf       = nf + 1;
	    bracket  = 2;
         else                                     %  Not at lower bound
            xplus      = x0;
	    xplus( j ) = xplus( j ) + aplus;
            if ( sum_form )
               fplus = 0;
               for iel  = 1:nel
	          xiel  = xplus( eldom{ iel } );
                  fiel  = f{iel}( elset( iel ), xiel );
                  fplus = fplus + fiel;
	          iplus = length( hist{ iel }.fel ) + 1;
                  hist{ iel }.fel( iplus )    = fiel;
                  hist{ iel }.xel( :, iplus ) = xiel;
               end
            else
	       fplus = f( xplus );
            end
	    nf    = nf + 1;
            if ( isnan( fplus ) )
               fplus = Inf;
            end
            if ( fplus > min( fleft, fright ) )   %  Bracket found: [ aleft amid aright ]
               bracket = 1;               
               amid    = aleft;
	       xmid    = xleft;
	       fmid    = fleft;
	       imid    = ileft;
            else                                  %  No bracket yet: move further on the left.
               aprev   = aright;
	       fprev   = fright;
	       aright  = aleft;
	       xright  = xleft;
	       fright  = fleft;
	       iright  = ileft;
            end
            aleft  = aplus;
	    xleft  = xplus;
            fleft  = fplus;
	    ileft  = iplus;
         end

     %  The left point is at the lower bound.

      else

         %  Test for temination.

         if ( ( aright - aleft ) <= 0.5 * epsilon )
            fmid = Inf; % a fake value to make sure it is not chosen in 
                        % the final termination test
            break

         %  Bisect the interval.

         else
            aplus = ( aright + aleft ) / 2;
            xplus = x0;
	    xplus( j ) = xplus( j ) + aplus;
            if ( sum_form )
               fplus = 0;
               for iel  = 1:nel
                  xiel  = xplus( eldom{ iel } );
                  fiel  = f{iel}( elset( iel ), xiel );
                  fplus = fplus + fiel;
	          iplus = length( hist{ iel }.fel ) + 1;
                  hist{ iel }.fel( iplus )    = fiel;
                  hist{ iel }.xel( :, iplus ) = xiel;
               end
            else
               fplus = f( xplus );
            end
            if ( isnan( fplus ) )
               fplus = Inf;
            end
            nf = nf + 1;
            if ( fplus <= min( fleft, fright ) )  %  Bracket found: [ aleft amid aright ]
               bracket = 1; 
               amid    = aplus;
               xmid    = xplus;
               fmid    = fplus;
               imid    = iplus;
            else                                  %  No bracket, halve the interval.
               aright  = aplus;
               xright  = xplus;
               fright  = fplus;
               iright  = iplus;
            end
         end
      end
   end

   %  Check for nfmax and return the currently best point.
   
   if ( nf >= nfmax )
      if ( bracket )
         fvals = [ fleft fright fmid ];
      else
         fvals = [ fleft fright ];
      end
      [ fbest, ibest ] = min( fvals );
      switch ( ibest )
      case 1
         xbest = xleft;
         if ( sum_form )
            for iel = 1:nel
	       hist{ iel }.fbest = hist{ iel }.fel( ileft );
	    end
         end
      case 2
         xbest = xright;
         if ( sum_form )
            for iel = 1:nel
	       hist{ iel }.fbest = hist{ iel }.fel( iright );
	    end
         end
      case 3
         xbest = xmid;
         if ( sum_form )
            for iel = 1:nel
	       hist{ iel }.fbest = hist{ iel }.fel( imid );
	    end
         end
      end
      msg = [' Maximum number of ', int2str( nfmax ), ' evaluations reached.' ];
      return
   end
   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Phase 2: reduce the bracket [ aleft, amid, aright] to a bracket of smaller length
%           still containing a minimizer. Note that fmid is not always smaller than
%           the minimum of fleft and fright when the bracket contains a bound on a.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while ( ( aright - aleft ) > epsilon && ~flat )

   if ( verbose )
      disp( [ ' B   fleft = ', num2str( fleft ), ' fmid = ', num2str( fmid ),              ...
              ' fright = ', num2str( fright ) ] )
      disp( [ ' B   aleft = ', num2str( aleft ), ' amid = ', num2str( amid ),              ...
              ' aright = ', num2str( aright ) ] )
      disp( [ ' B   bracket length = ', num2str( (aright-aleft) ) ]  )
   end
   
   %  Find the interpolating quadratic and its minimizer.

if ( abs( aleft - aright ) < eps || ...
     abs( aleft - amid   ) < eps || ...
     abs( aright - amid  ) < eps    )%D
disp( 'PROBLEMATIC INTERVAL' )%D
disp( [ 'aleft = ', num2str(aleft), ' amid = ', num2str(amid), ' aright = ', num2str(aright) ] )%D
keyboard
end%D

   %  The objective's value at a mid is at most equall to the minimum of the values
   %  at the bracket extremities, and hence the minimum of the interpolating quadratic
   %  is in [ aleft, aright ]. It therefore pays to compute it.

   if ( bracket == 1 )
      q = [ 0.5*(aleft -amid)^2  aleft-amid ;
            0.5*(aright-amid)^2 aright-amid ] \ [ fleft-fmid; fright-fmid ];
      aqmin = amid - q(2) / q(1);      % The minimum of the interpolating quadratic

      %  Perturb it if too close to amid.
   
      acenter = 0.5 * ( aleft + aright );
      alength = 0.5 * ( aright - aleft );
      if ( abs( aqmin-amid ) < epsilon )
         if ( amid - aleft < aright - amid )
            aqmin = amid + 0.4999 * epsilon;
         else
            aqmin = amid - 0.4999 * epsilon;
         end
      elseif ( abs( aqmin-amid ) < 0.15 * alength )
         if ( amid - aleft < aright - amid )
            aqmin = amid + 0.5 * ( aright - amid );
         else
            aqmin = amid - 0.5 * ( amid  - aleft );
         end
      end

      %  Compute aplus to ensure decrease in bracket length.
      %  NOTE: the coefficient 0.9 could be trained, but has to be bounded away from 1.
   
if ( aqmin < aleft || aqmin > aright )%D
   disp( 'STRANGE BRACKET' )%D
   keyboard%D
end%D

      aplus       = max( acenter-0.9*alength, min( aqmin, acenter+0.9*alength) );
%      aplus       = max( acenter-0.75*alength, min( aqmin, acenter+0.75*alength) );
%      aplus       = max( acenter-0.5*alength, min( aqmin, acenter+0.5*alength) );

   %  The minimum of the interpolating quadratic is not in [ aleft, aright ].  This
   %  may happen if alower = aleft or aupper = aright.  Bisect the bracket.

   else
      aplus = 0.5 * ( aleft + aright);
   end

   xplus       = x0;
   xplus( j )  = xplus( j ) + aplus;
   if ( sum_form )
      fplus    = 0;
      for iel  = 1:nel
         xiel  = xplus( eldom{ iel } );
         fiel  = f{iel}( elset( iel ), xiel );
         fplus = fplus + fiel;
         iplus = length( hist{ iel }.fel ) + 1;
         hist{ iel }.fel( iplus )    = fiel;
         hist{ iel }.xel( :, iplus ) = xiel;
      end
   else
      fplus = f( xplus );
   end
   nf = nf + 1;
   if ( isnan( fplus ) )

      fplus = Inf;
   end

   if ( verbose )
      disp( [ ' B   aplus = ', num2str( aplus ), ' fplus = ', num2str( fplus ) ]  )
      disp( ' ' )
   end
   
   %  Select the next bracket.
   
   if ( fplus < fmid )
      if ( aplus < amid )
         aright = amid;
	 xright = xmid;
	 fright = fmid;
	 iright = imid;
      elseif ( aplus > amid )
         aleft  = amid;
	 xleft  = xmid;
	 fleft  = fmid;
	 ileft  = imid;
      end
      amid = aplus;
      xmid = xplus;
      fmid = fplus;
      imid = iplus;
   else
      if ( aplus < amid )
         aleft  = aplus;
	 xleft  = xplus;
	 fleft  = fplus;
	 ileft  = iplus;
      else
         aright = aplus;
	 xright = xplus;
	 fright = fplus;
	 iright = iplus;
      end
   end

   if ( verbose )
      disp( [ ' B   new bracket length = ', num2str( aright-aleft ) ]  )
   end
   
   %  Check for nfmax and return the currently best point.
   
   if ( nf >= nfmax )
      fvals = [ fleft fmid fright ];
      [ fbest, ibest ] = min( fvals );
      switch ( ibest )
      case 1
         xbest = xleft;
         if ( sum_form )
            for iel = 1:nel
	       hist{ iel }.fbest = hist{ iel }.fel( ileft );
	    end
         end
      case 2
         xbest = xmid;
         if ( sum_form )
            for iel = 1:nel
	       hist{ iel }.fbest = hist{ iel }.fel( imid );
	    end
         end
      case 3
         xbest = xright;
         if ( sum_form )
            for iel = 1:nel
	       hist{ iel }.fbest = hist{ iel }.fel( iright );
	    end
         end
      end
      msg = [' Maximum number of ', int2str( nfmax ), ' evaluations reached.' ];
      return
   end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Phase 3: successful termination.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Select the best point point in the last bracket.

fvals = [ fleft fright fmid ];
[ fbest, ibest ] = min( fvals );
switch ( ibest )
case 1
   xbest = xleft;
   if ( sum_form )
      for iel = 1:nel
	  hist{ iel }.fbest = hist{ iel }.fel( ileft );
      end
   end
case 2
   xbest = xright;
   if ( sum_form )
      for iel = 1:nel
	  hist{ iel }.fbest = hist{ iel }.fel( iright );
      end
   end
case 3
   xbest = xmid;
   if ( sum_form )
      for iel = 1:nel
	  hist{ iel }.fbest = hist{ iel }.fel( imid );
      end
   end
end

if ( verbose )
   disp( [ ' T   fbest = ', num2str( fbest ) ] )
end

%  Remove the initial point from the element history.

for iel = 1:nel
   hist{ iel }.fel = hist{ iel }.fel( 2:end );
   hist{ iel }.xel = hist{ iel }.xel( :,2:end );
end

%  Check the coherence of the returned element function values.

if( 0 )
   fbest2 = 0;
   fiel   = [];
   for iel = 1:nel
      fiel = [ fiel hist{iel}.fbest];
      fbest2 = fbest2 + hist{iel}.fbest;
   end
   if ( abs(fbest2-fbest ) > 1e-10 )
      fvals
      ibest
      for it = 1:length( hist{1}.fel )
         fit = 0;
         fitel = [];
         for iel = 1:nel
            fitel = [ fitel hist{iel}.fel(it)];
            fit = fit + hist{iel}.fel( it );
         end
         it
         fitel_fit = [ fitel, fit ]
         fiel
      end
      fbest_fbest2 = [ fbest fbest2 ]
      fiel
      pause
   else
      fprintf( '\n ============== OK ==============\n\n' );
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function  [ n_groups, xgroups, n_sets, xsets, esets, xinel ] =                             ...
                         bfo_analyze_cps_structure( n, n_elements, eldom, active, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Analyze the coordinate partially separable structure specified by eldom to produce 
%  n_groups groups (xgroups) of sets (xsets) of variables involving certain elements 
%  functions (esets). Two sets of variables in the same group involve disjoint ensembles 
%  of elements (and can therefore be used in parallel).
%  See: C. Price and Ph.L. Toint,
%     "Exploiting problem structure in pattern-search methods for unconstrained optimization",
%      Optimization Methods and Software, vol. 21(3), pp. 479-491, 2006.

%  INPUT:

%  n         : the total number of variables in the problem
%  n_elements: the number of element function in the objective function
%  eldom     : a cell array of length n_elements, whose i-th entry is a
%              vector containg the indices of the variables defining the domain
%              of the i-th element
%  active    : the indices of the currently active variables (including categorically
%              deactivated ones)
%  verbose   : the current verbosity level

%  OUTPUT:

%  n_groups  : the number of groups of independent elements
%  x_groups  : a cell array of length n_groups whose i-th entry is a vector containing
%              the indeces of the variable sets belonging to the i-th group
%  n_sets    : the number of variable sets
%  xsets     : a cell array of length n_sets, whose i-th entry is a vector containing
%              the indeces of the variables in the i-th set
%  esets     : a cell array of length n_sets, whose i-th entry is a vector containing
%              the indeces of the elements involving variables of the i-th set.

%  PROGRAMMING: Ph. L. Toint, October 2016 (This version 16 XII 2016).

%  DEPENDENCIES: ~

%  REFERENCE:  C. Price and Ph.L. Toint,
%              "Exploiting problem structure in pattern-search methods for 
%              unconstrained optimization",
%              Optimization Methods and Software, vol. 21(3), pp. 479-491, 2006.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Determine which elements do involve each variable.

slice   = min( 50, n_elements );  % the value 50 here is a more or less arbitrary
xinel   = zeros( n, slice );
maxrow  = 0;
for iel = 1:n_elements
   eldomiel = eldom{ iel };
   n     = max( n, max( eldomiel ) );
   for i = 1:length( eldomiel )
      iv = eldom{ iel }( i );
      if ( ismember( iv, active )  )
         pos      = find( xinel( iv,: ) == 0, 1 );
         if ( isempty( pos ) )
            xinel = [ xinel, zeros( n, slice ) ];
            pos   = find( xinel( iv,: ) == 0, 1 );
         end
         xinel( iv, pos ) = iel;
         maxrow = max( maxrow, pos );
      end
   end
end
xinel = xinel( active, 1:maxrow );

% Sort the lists of variables per element to ensure they are lexicographically
% ordered. Hence rows corresponding to variables occuring in the same set
% of elements follow each other.

[ xinel, perm ] = sortrows( xinel );
ivar            = active( perm );

%  Find the independent sets of variables corresponding to identical
%  list of elements (in xsets) and their associated list of elements (in esets).
   
xsets{ 1 } = [ ivar( 1 ) ];
esets{ 1 } = xinel( 1, : ); 
n_sets     = 1;
for i = 2:size( xinel, 1 )
   if ( norm( xinel( i, : ) - xinel( i - 1, : ), 1 ) )
      esets{ n_sets}  = esets{ n_sets }( find( esets{ n_sets } ) );
      n_sets          = n_sets + 1 ;
      xsets{ n_sets } = ivar( i );
      esets{ n_sets } = xinel( i, : );
   else
      xsets{ n_sets }( end+1 ) = ivar( i );
   end
end
esets{ n_sets } = esets{ n_sets }( find( esets{ n_sets } ) );

if ( verbose > 3 )
   disp( [ ' The ', int2str( n_elements ), ' elements have been combined in ',             ...
           int2str( n_sets ), ' sets of variables']);
   if ( verbose > 4 )
      for j = 1:n_sets
         fprintf( '  Set %3d involves \n       variable(s)', j );
         for k = 1:length( xsets{ j } )
            fprintf( ' %3d', xsets{ j }( k ) );
         end
         fprintf( '\n        element(s)' )
         for k = 1:length( esets{ j } )
            fprintf( ' %3d', esets{ j }( k ) );
         end
         fprintf( '\n' );
      end
   end 
end

%  Find groups of independent xsets, that is collections of xsets such that they involve
%  differents elements.  Use a greedy strategy to build those sets.

xgroups    = {};
n_groups   = 0;
unassigned = [ 1:n_sets ];
while( ~isempty( unassigned ) )
   n_groups = n_groups + 1;

   %  Include the first unassigned set in the next group.

   xgroups{ n_groups } = [ unassigned( 1 ) ];
   egroups{ n_groups } = esets{ unassigned( 1 ) };
   unassigned( 1 )     = 0;

   %  Check for the next unassigned set ...

   for idx = 2:length( unassigned )
      iset = unassigned( idx );
        
      %  ... if it is independent from the already included ones, in which case
      %  the group is extended.

      if ( isempty( intersect( esets{ iset }, egroups{ n_groups } ) ) )
         xgroups{ n_groups }( end+1 ) =  iset;
         egroups{ n_groups }          = [ egroups{ n_groups }, esets{ iset } ];
         unassigned( idx )            = 0;
      end
   end
   unassigned = unassigned( find( unassigned ) );
end

if ( verbose > 3 )
   disp( [ ' The ', int2str( n_sets), ' variable''s sets have been combined in ',          ...
           int2str( n_groups ), ' independent groups']);
   if ( verbose > 4 )
      for j = 1:n_groups
         fprintf( '  Group %3d involves \n            set(s)', j );
         for k = 1:length( xgroups{ j } )
            fprintf( ' %3d', xgroups{ j }( k ) );
         end
	 fprintf( '\n       variable(s)' );
	 for i = 1:length( xgroups{ j } )
            for  k = 1:length( xsets{ xgroups{ j }( i ) } )
              fprintf( ' %3d', xsets{ xgroups{ j }( i ) }( k ) );
	    end
         end
	 fprintf( '\n        element(s)' );
         for k = 1:length( egroups{ j } )
            fprintf( ' %3d', egroups{ j }( k ) );
         end
	 fprintf( '\n' );
      end
   end
end
clear egroups unassigned ivar perm;

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [ neighbours, alphaf, alphab, xtypes, xlowers, xuppers, cat_dictionnary, msg ] =  ...
                bfo_build_neighbours( x, j, xtype, xlower, xupper, Q, icont, idisc, idall, ...
                                      x0ref, cat_dictionnary, cn_name, num_cat_states,     ...
			              recursive, xincrcur, ncbound, ndbound, latbasis,     ...
				      verbose, indent, myinf )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Builds the vectors of neighbouring values of the current iterate x by considering
%  all possible feasible neighbours of variable j.  These vectors are stored in the
%  columns of the array neighbours, whose number of columns must be at least 2
%  (for continuous, integer or lattice variables) and at most the maximum number of
%  neighbouring values for a categorical variable.  For continuous variables, the index
%  (j) of the variable merely allows to loop over the directions of the random basis
%  Q, and does not generate a neighbour which is more in component j than any other.
%  This makes using an incomplete basis (a matrix Q with a number of columns less than ncont)
%  possible. 

%  INPUT:

%  x                 : the current iterate
%  j                 : the component relative to which neighbours must be considered
%  xtype             : the variables types
%  xlower            : the lower bounds on the variables
%  xupper            : the upper bounds on the variables
%  Q                 : the current (possibly incomplete) basis for continuous variables
%  icont             : the indeces of the continuous variables
%  idisc             : the indeces of the active discrete variables
%  idall             : the indeces of the discrete variables (active or not)
%  x0ref             : the reference vector for vector state values
%  cat_dictionnary   : the current dictionnary of categorical values
%  cn_name           : the name of the dynamical categorical neighbourhood routine
%  num_cat_states    : the numerical version of cat_states
%  recursive         : true if in the recursive part of the poll step
%  xincrcur          : the current table of increments
%  ncbound           : the number of bounded continuous variables
%  nbbound           : the number of bounded discrete variables
%  latbasis          : the basis for th lattice
%  verbose           : the verbosity level
%  indent            : the current printout indentation
%  myinf             : infinity for bounds

%  OUTPUT:

%  neighbours        : a cell array containing the vector states defining the neighbours
%                      of x wrt component j
%  alphaf            : the forward stepsize for continuous variables
%  alphab            : the backward stepsize for continuous variables
%  xtypes            : a cell array whose i-th element contains the types of the
%                      components of neighbours(:,i)
%  xlowers           : xlowers(:,i) contains the lower bounds on the components of
%                      neighbours(:,i)
%  xuppers           : xuppers(:,i) contains the upper bounds on the components of
%                      neighbours(:,i)
%  msg               : an error message (if any)

%  DEPENDENCIES (internal): bfo_print_matrix, bfo_print_vector, bfo_numerify, bfo_cellify,
%                           bfo_feasible_cstep
%  DEPENDENCIES (external): cat_neighbours (optional)

%  PROGRAMMING: Ph. Toint, August 2016. (This version 7 III 2016)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Define defaults for all outputs.

neighbours = [];
xtypes     = {};
xlowers    = [];
xuppers    = [];
alphaf     = 1; 
alphab     = 1;
msg        = '';

%  Get the problem's size.

n          = length( x );               %  the number of variables
nnghbrj    = 0;                         %  the number of neighbours relative to variable j

if ( verbose >= 4 )
   disp( ' Considering neighbours of ' )
   bfo_print_vector( indent, 'x', x, xtype, cat_dictionnary, x0ref );
   disp( [ ' relative to component ', int2str( j ) ] )
   disp( ' ' )
end

%  Consider the continuous variables (only in the non-recursive part of the poll step).

if ( xtype( j ) == 'c' &&  ~recursive )

   ncont = length( icont);              %  the number of continuous variables
   ic    = find( icont == j, 1 );       %  the index of variable i in the set 
                                        %  of continuous variables
   spoll = size( Q, 2 );                %  the number of columns in Q

   if ( ic <= spoll )

      neighbours = zeros( n, 2 );

      %  Construct the forward neighbour.
   
      xs   = x;
      step = xincrcur( icont ) .* Q( 1:ncont, ic );

      if ( ncbound )
         [ xs( icont ), alphaf ]  = bfo_feasible_cstep( x( icont ), step,                  ...
                                                     xlower( icont ), xupper( icont ) );
      else
         xs( icont ) = x( icont ) + step;
         alphaf      = norm( xs( icont ) - x( icont ) );
      end
    
      %  Store the new (forward) neighbour.

      nnghbrj                    = nnghbrj + 1;
      neighbours( 1:n, nnghbrj ) = xs;

      %  Construct the backward neighbour.
   
      xs   = x;
      step = - xincrcur( icont ) .* Q( 1:ncont, ic );
      if ( ncbound )
         [ xs( icont ), alphab ] = bfo_feasible_cstep( x( icont ), step,                   ...
                                                    xlower( icont ), xupper( icont ) );
      else
         xs( icont ) = x( icont ) + step;
         alphab      = norm( xs( icont ) - x( icont ) );
      end
    
      %  Store the new (backward) neighbour.

      nnghbrj                    = nnghbrj + 1;
      neighbours( 1:n, nnghbrj ) = xs;

   end

%  Consider the discrete variables.

elseif ( xtype( j ) == 'i' )

   neighbours = zeros( n, 2 );

   %  Construct the forward neighbour.
      
   xs  = x;
   if ( ~isempty( latbasis ) )
      if ( isempty( idall ) )
         [ ~, id ]     = ismember( j, idisc );        %  the index of variable j in the set 
                                                      %  of active discrete variables
%         id            = find( idisc == j, 1 );
         xs( idisc )   = x( idisc ) + xincrcur( j ) * latbasis( idisc, id ); 
      else
         [ ~, id ]     = ismember( j, idall );        %  the index of variable j in the set 
                                                      %  of discrete variables across levels
         [ ~, irange ] = ismember( idisc , idall );   %  the indeces of the current discrete
                                                      %  variables in the same list
         xs( idisc )   = x( idisc ) + xincrcur( j ) * latbasis( irange, id );
      end 
   else
      xs( j )          = x( j ) + xincrcur( j );
   end
   
   % Enforce feasibility.

    xs_feasible = 1;
    if ( ndbound )
       for ii = 1:n 
          if ( xs( ii ) < xlower( ii )  || xs( ii ) > xupper( ii ) )
             xs = x;
             xs_feasible = 0;
             break
          end
       end
    end

   %  Store the new (forward) neighbour if feasible.

   if ( xs_feasible )
      nnghbrj                    = nnghbrj + 1;
      neighbours( 1:n, nnghbrj ) = xs;
   end
                  
   %  Construct the backward neighbour.
      
   xs = x;
   if ( ~isempty( latbasis ) )
      if ( isempty( idall ) )
         xs( idisc ) = x( idisc ) - xincrcur( j ) * latbasis( idisc, id );
      else
         xs( idisc ) = x( idisc ) - xincrcur( j ) * latbasis( irange, id );
      end
   else
      xs( j )        = x( j ) - xincrcur( j );
   end

   % Enforce feasibility.

   xs_feasible = 1;
   if ( ndbound )
      for ii = 1:n 
         if ( xs( ii ) < xlower( ii )  || xs( ii ) > xupper( ii ) )
            xs          = x;
	    xs_feasible = 0;
            break
         end
      end
   end

  %  Store the new (backward) neighbour if feasible.

   if ( xs_feasible )
      nnghbrj                    = nnghbrj + 1;
      neighbours( 1:n, nnghbrj ) = xs;
   end
   neighbours = neighbours( 1:n, 1:nnghbrj );

%  Consider the categorical variables.
                  
elseif ( xtype( j ) == 's' )

   %  Case 1: neighbourhoods of categorical variables are dynamically defined.
   %          They are subject to the restrictions (i)--(vi) in the header's comments.
   %          In particular, the restriction that forbids mixing continuous and integer or
   %          categorical variables avoids the redefinition of the associated stepsizes.

   if ( ~isempty( cn_name ) )

      %  Transform the current vector of numerical variables into a proper vector state
      %  before passing it to the user.
  
      vsx = bfo_cellify( x, xtype, cat_dictionnary, x0ref );

      %  Call the user-supplied function to define neighbourhoods of categorical variables.
  
      cat_neighbours = str2func( cn_name );
      [ cneighbours, xtypes, xlowers, xuppers ] = cat_neighbours( vsx, xtype, xlower, xupper);
      [ ~, nnghbrj ] =  size( cneighbours );

      %  Check the neighbouring rules for each user-defined neighbour in turn.

      for inghbrj = 1:nnghbrj
      
         %  Check that the problem's total dimension is unchanged.

         if ( length( cneighbours{ inghbrj }{ 1 } ) ~= n )
            msg = [ ' BFO error: total problem dimension altered in new neighbourhood.',   ...
	            ' (neighbour ' int2str( inghbrj),' of variable ', int2str( j), ').'    ...
		    ' Terminating.' ];
            if ( verbose )
               disp( msg )
            end
	    return
         end

         %  Check there is at least one categorical variable left.

	 if ( ~any( xtypes{ inghbrj } == 's' ) )
            msg = [ ' BFO error: no categorical variable left in neighbour ',              ...
	             int2str( inghbrj ),' of variable ', int2str( j), '. Terminating.' ];
            if ( verbose )
               disp( msg )
            end
	    return
	 end

         %  Check the lengths of the type and of the lower and upper bounds.

         if ( length( xtypes{ inghbrj }  ) ~= n )
            msg = [ ' BFO error: the length of xtypes differs from the total problem'      ...
	            ' dimension for neighbour ', int2str( inghbrj ), ' of variable ',      ...
		    int2str( j ),'. Terminating.' ];
            if ( verbose )
               disp( msg )
            end
	    return
	 end
         if ( length( xlowers( :, inghbrj ) ) ~= n )
            msg = [ ' BFO error: the length of xlowers differs from the total problem'     ...
	            ' dimension for neighbour ', int2str( inghbrj ), ' of variable ',      ...
		    int2str( j),'. Terminating.' ];
            if ( verbose )
               disp( msg )
            end
	    return
	 end
         if ( length( xuppers( :, inghbrj ) ) ~= n )
            msg = [ ' BFO error: the length of xuppers differs from the total problem'     ...
	            ' dimension for neighbour ', int2str( inghbrj ), ' of variable ',      ...
		    int2str( j), '.  Terminating.' ];
            if ( verbose )
               disp( msg )
            end
	    return
	 end
            
         %  Check the activations/deactivations by variable type. Also avoid bounds
	 %  equal to +/- Inf.
	 
         for i = 1:n

            inconsistent = 0;
	    
	    %  1) continuous: { 'c', 'r' } variables must remain { 'c', 'r' }
	    
            if ( (  ismember( xtype( i ),             { 'c', 'r' } ) &&                    ...
                   ~ismember( xtypes{ inghbrj }( i ), { 'c', 'r' } )    )  ||              ...
                 (  ismember( xtypes{ inghbrj }( i ), { 'c', 'r' } ) &&                    ...
                   ~ismember( xtype( i ),             { 'c', 'r' } )    )    )
               inconsistent = 1;
	    end
	    
	    %  2) integer: { 'i', 'd' } variables must remain { 'i', 'd' }

            if ( (  ismember( xtype( i ),             { 'i', 'd' } ) &&                    ...
	           ~ismember( xtypes{ inghbrj }( i ), { 'i', 'd' } )    )  ||              ...
                 (  ismember( xtypes{ inghbrj }( i ), { 'i', 'd' } ) &&                    ...
	           ~ismember( xtype( i ),             { 'i', 'd' } )    )   )
               inconsistent = 2;
            end
	    
	    %  3) categorical: { 's', 'k' } variables must remain { 'd', 'k' }
	    
            if ( (  ismember( xtype( i ),             { 's', 'k' } ) &&                    ...
	           ~ismember( xtypes{ inghbrj }( i ), { 's', 'k' } )   )  ||               ...
                 (  ismember( xtypes{ inghbrj }( i ), { 's', 'k' } ) &&                    ...
	           ~ismember( xtype( i ),             { 's', 'k' } )   )   )
               inconsistent = 3;
            end
	    
	    %  4) waiting: 'w' variables should remain 'w' ...

            if ( ( xtype( i ) == 'w' && xtypes{ inghbrj }( i ) ~= 'w'  )   ||              ...
                 ( xtypes{ inghbrj }( i ) == 'w' && xtype( i ) ~= 'w'  )    )
               inconsistent = 4;
            end

            % ... and keep the same value.
	    
	    if ( xtype( i ) == 'w' )
	       ni = cneighbours{ inghbrj }{ 1 };
	       if ( isnumeric( ni ) && ni( i ) ~= x( i )     )
                  inconsistent = 5;
	       elseif ( iscell( ni )  )
	          if ( ( isnumeric( ni{i} ) && ni{i} ~= vsx{ i } )  ||                     ...
		       ( iscell( ni{i} )    && ~strcmp( ni{ i }, vsx{ i } ) ) )     
                     inconsistent = 6;
		  end
	       end
            end

	    %  4) frozen: 'x', 'y', 'z' variables should remain 'x', 'y', 'z' ...
	    
            if ( ( xtype( i ) == 'x' && xtypes{ inghbrj }( i ) ~= 'x'  )   ||              ...
                 ( xtypes{ inghbrj }( i ) == 'x' && xtype( i ) ~= 'x'  )   ||              ...
                 ( xtype( i ) == 'y' && xtypes{ inghbrj }( i ) ~= 'y'  )   ||              ...
                 ( xtypes{ inghbrj }( i ) == 'y' && xtype( i ) ~= 'y'  )   ||              ...
                 ( xtype( i ) == 'z' && xtypes{ inghbrj }( i ) ~= 'z'  )   ||              ...
                 ( xtypes{ inghbrj }( i ) == 'z' && xtype( i ) ~= 'z'  )      )
               inconsistent = 4;
            end

            % ... and keep the same value.
	    
	    if ( ismember( xtype( i ), { 'x', 'y', 'z' } ) )
	       ni = cneighbours{ inghbrj }{ 1 };
	       if ( isnumeric( ni ) && ni( i ) ~= x( i )     )
                  inconsistent = 7;
	       elseif ( iscell( ni )  )
	          if ( ( isnumeric( ni{i} ) && ni{i} ~= vsx{ i } )  ||                     ...
		       ( iscell( ni{i} )    && ~strcmp( ni{ i }, vsx{ i } ) ) )     
                     inconsistent = 8;
		  end
	       end
            end
	    
	    if ( inconsistent )
	       msg = [ ' BFO error: inconsistent variable type/bound returned by ',        ...
		       cn_name, ' (component ', int2str( i ), ' of neighbour ',            ...
		       int2str( inghbrj ), ' of variable ', int2str( j ), ').',            ...
		       ' Terminating.' ];
               if ( verbose )
                  disp( msg )
               end
  	       return
	    end

	 end

         %  Make the bounds finite.

         imininf = find( xlowers( :, inghbrj ) < -myinf );
	 limininf = length( imininf );
	 if ( limininf )
	    xlowers( imininf, inghbrj ) = -myinf * ones( limininf, 1 );
	 end
         imaxinf = find( xuppers( :, inghbrj ) >  myinf );
	 limaxinf = length( imaxinf );
	 if ( limaxinf )
	    xuppers( imaxinf, inghbrj ) =  myinf * ones( limaxinf, 1 );
	 end

         %  Build its corresponding numerical variants.

         [ neighbours( 1:n, inghbrj ), cat_dictionnary, errnum ] =                         ...
              bfo_numerify( cneighbours{ inghbrj }{ 1 }, xtypes{ inghbrj }, cat_dictionnary );

         if ( errnum > 0 )
            msg = [ ' BFO error: component ', int2str( errnum ), ' of neighbour ',         ...
	              int2str( inghbrj ),  ' of variable ', int2str( j ),                  ...
		    ' is a string for a non-categorical variable. Terminating.' ];
            if ( verbose )
               disp( msg )
            end
  	    return
	 elseif ( errnum < 0 )
            msg = [ ' BFO error: error in format of neighbour ', int2str( inghbrj ),       ...
	            ' of variable ', int2str( j ),'. Terminating.' ];
            if ( verbose )
               disp( msg )
            end
  	    return
	 end

      end
      
   %  Case 2: neighbourhoods of categorical variables are statically defined by the
   %  cat_states structure.
  
   else
      for jj = 1:length( num_cat_states{ j } )
	 nsj = num_cat_states{ j }( jj );
	 if ( nsj ~= x( j ) )
            nnghbrj                    = nnghbrj + 1;
            neighbours( 1:n, nnghbrj ) = x;
            neighbours( j  , nnghbrj ) = nsj;
	    xlowers(    1:n, nnghbrj ) = xlower;
	    xuppers(    1:n, nnghbrj ) = xupper;
  	    xtypes{ nnghbrj }          = xtype;
	 end
      end
   end
end


return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function x = bfo_cellify( numx, xtype, cat_dictionnary, x0ref )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Construct an output version of numx which is either a numerical vector (= numx) when
%  none of the components of x is the index of a categorical state, or a cell array
%  containing numerical components and categaorical states, depending on xtype.

%  INPUT:

%  numx             : numx(i) contains a numerical value if xtype(i) is 'c' or 'i',
%                     or the index of a categorical state (in cat_dictionnary) if
%                     xtype(i) is 's' or 'k'.
%  xtype            : the type of the components of numx
%  cat_dictionnary  : the ordered list of current categorical states
%  x0ref            : a vector state providing default values for variables whose
%                     type is not 'c', 'i', 's' or 'k'.
%  OUTPUT:

%  x                : a vector state, corresponding to numx.

%  DEPENDENCIES : -

%  PROGRAMMING: Ph. Toint, August 2016. (This version 23 VIII 2016)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n  = length( numx );  % the dimension of numx
if ( isnumeric( x0ref ) )
   x = numx;
else
   for i = 1:n
      switch ( xtype( i ) )
      case { 's', 'k' }
         x{ i } = cat_dictionnary{ numx( i ) }; 
      case { 'c', 'i' }
         x{ i } = numx( i );
      otherwise
         x{ i } = x0ref{ i };
      end
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [ numx, cat_dictionnary, err ] = bfo_numerify( x, xtype, cat_dictionnary )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Construct an output version of x which is a numerical vector such that
%  numx(i) = x(i) if xtype(i) == 'c' or == 'i'
%  numx(i) = the index of the state x{i} in cat_dictionnary if xtype(i)=='s'
%  numx(i) = 0 otherwise.
%  Note that cat_dictionnary is potentially updated to contain new categorical
%  states occuring for the first time in x.  Also note that variables with
%  status 'w', 'x', 'y' or 'z' may occur if the function is called within a recursion,
%  in which case the type is derived from x itself.

%  INPUT:

%  x                : either a numerical vector, or a vector state
%  xtype            : the expected types of the components of x.
%  cat_dictionnary  : the ordered list of current categorical states.

%  OUTPUT:

%  numx             : a numerical vector or a cell array, corresponding to x.
%  cat_dictionnary  : the (possibly updated) list of categorical states.
%  err              : the smallest index in x such that x{i} is a string but
%                        xtype(i)~= 's' or 'k',
%                     -1 if no entry should be a string but the input is not a numeric vector,
%                     -2 the input should be a cell array but is not
%                      O if no problem was detected.

%  DEPENDENCIES : -

%  PROGRAMMING: Ph. Toint, August 2016. (This version 18 IX 2016)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

err  = 0;                          % no wrong component detected yet
n    = length( x ); 
numx = zeros( n, 1 );              % prepare the corresponding numerical array
for i = 1:n
   xi = x{ i };
   if ( xtype( i ) == 'w'  ||  xtype( i ) == 'x'  || xtype( i ) == 'y'  ||                 ...
        xtype( i ) == 'z'  || xtype( i ) == 'f' )
      iscat = ischar( xi );
   else
      iscat = ( xtype( i ) == 's'  || xtype( i ) == 'k' );
   end
   if ( iscat )
      if ( ischar( xi ) )
         [ existing, index ] = ismember( xi, cat_dictionnary ); % see if the state is known
         if ( existing )              % if yes, record its index
            numx(i)   = index;
         else                         % if not, create a new state and record its index
            cat_dictionnary = union( cat_dictionnary, xi, 'stable' );
            numx( i )       = length( cat_dictionnary );
	 end
      else
         err = i;
	 return
      end
   else
      if ( ischar( xi ) )          % ... but it is a string...
         err = i;                  % ... and termination occurs with an error
         return
      else
         numx( i ) = xi;
      end
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function xout = bfo_pack_x( numx, xscale, xtype, cat_dictionnary, x0ref )
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Scales and cellify the numerical version of a vector state.

%  INPUT:

%  numx           : the numerical version of the vector state to reconstruct
%  xscale         : the scaling vector
%  xtype:         : the type of the variables
%  cat_dictionnary: the current categorical value dictionnary
%  x0ref          : the reference vector for vector state values

%  OUTPUT:

%  xout           : the unscaled and cellified numx

%  DEPENDENCIES : -

%  PROGRAMMING: Ph. Toint, September 2016. (This version 11 I 2018)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( ~isempty( xscale ) )
  numx = xscale .* numx;
end
if( isnumeric( x0ref ) )
  xout = numx;
else
  xout = { bfo_cellify( numx, xtype, cat_dictionnary, x0ref ) };
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function bfo_search_step_cleanup( bfo_srch, categorical, partially_separable )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  This function is called if a user search-step routine has been used and just before the
%  final return to the user (as identified by level = 1, depth = 0 and use_cps >= 0). It calls 
%  the search-step function bfo_srch a last time with a special value of level = -1, in order 
%  to allow the search-step function to perform internal clean up. The call's form depends on 
%  whether categorical variables are present or on the possible sum-form of the objective
%  function, but all arguments (except the first) are dummies. The second argument is a
%  (dummy) cell when the objective function is in sum form in order to all the search-step
%  function to distinguish this case. No output is expected from this call.
%
%  Note that

%  INPUT:

%  depth              : the recursion depth in mix-integer search
%  bfo_srch           : the handle for the search-step function
%  categorical        : true iff the problem involves categorical variables
%  partially_separable: true if coordinate-partially separable, in which case the element 
%                       domains must be passed to the search-step cleanup.

%  OUTPUT:

%  DEPENDENCIES : bfo_srch (user supplied)

%  PROGRAMMING: Ph. Toint, October 2011. (This version 29 I 2018)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( categorical )
   if ( partially_separable )
      dum_el_hist = { struct( 'eldom', [ 1 ] ) };
      bfo_srch( -1, {@bfo_dummy}, [], [], [], [], [], [], [], [], [], [], [],dum_el_hist );
   else
      bfo_srch( -1, @bfo_dummy, [], [], [], [], [], [], [], [], [], [], [] );
   end
else
   if ( partially_separable )
      dum_el_hist = { struct( 'eldom', [ 1 ] ) };
      bfo_srch( -1, {@bfo_dummy}, [], [], [], [], [], [], [], [], [], dum_el_hist  );
   else
      bfo_srch( -1, @bfo_dummy, [], [], [], [], [], [], [], [], [] );
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function Q = bfo_new_continuous_basis( ncont, N, ddir, npoll )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Compute a new random (potentially partial) basis Q for the continuous variables, 
%  taking care to include normals to the (nearly) saturated bound constraints (N) 
%  and possibly including a hopeful descent direction (ddir).

%  INPUT:

%  ncont   : the number of continuous variables
%  N       : an array whose columns contain the normalized normals to the (nearly) saturated 
%            constraints, [] is no constraint is nearly saturated
%  ddir    : a hopeful direction of descent in the continuous variables, [] if none available
%  npoll   : the maximum number of orthonormal polling directions
%  varargin: an (optional) basis for a termination subspace (in the checking phase)

%  OUTPUT:

%  Q     : the (possibly partial) new basis, with ncont rows and min( npoll, ncont ) columns

%  DEPENDENCIES : -

%  PROGRAMMING: Ph. Toint and M. Porcelli, May 2010. (This version 9 I 2017)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

use_ddir = ~isempty( ddir );
spoll    = min( ncont, npoll );
nsat     = size( N, 2 );
if ( nsat > 0 )
   if ( nsat < ncont )
      if ( use_ddir && nsat < spoll-1 )
         [ Q, ~ ] = qr( [ N( 1:ncont, 1:nsat ) ddir rand( ncont, spoll-nsat-1 ) ], 0 );
      else
         [ Q, ~ ] = qr( [ N( 1:ncont, 1:nsat ) rand( ncont, spoll-nsat ) ], 0 );
      end
   else
      Q = N( 1:ncont, 1:spoll );
   end
else
   if ( use_ddir && ncont > 1 )
      [ Q, ~ ] = qr( [ ddir rand( ncont, spoll-1 ) ], 0 );
   elseif ( ncont > 1 )
      [ Q, ~ ] = qr( rand( ncont, spoll ), 0 );
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



function [ totalnf, msg, wrn, tpval, tpdim, tpevs ] =                                      ...
                                 bfo_average_perf( p0, bestperf,                           ...
                                                   training_strategy,                      ...
                                                   training_problems,                      ...
                                                   training_set_cutest,                    ...
                                                   training_verbosity,                     ...
                                                   training_problem_epsilon,               ...
                                                   training_problem_maxeval,               ...
                                                   training_problem_verbosity,             ...
                                                   fstar )
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Optimize BFO parameters using the Audet-Orban formulation, that is
%  minimizing the total number of function evaluations (or, equivalently, its
%  average per problem).

%  INPUT:

%  p0                     : the initial value of the parameters to be trained
%  betsperf               : the value of the best performance seen so far
%  training_problems      : the list of objective functions of problems to be used 
%                           for training
%  training_set_cutest    : the associated problems library 
%  training_maxeval       : the maximum numbers of training function evaluations
%  training_verbosity     : the verbosity of the training process
%  training_problem_verbosity : the verbosities for the solution of the training problems
%  training_problem_epsilon : the accuracy at which each of the training
%                           problems must be solved
%  training_problem_maxeval  : the maximum number of objective evaluations in the
%                           solution of each training problem
%  fstar                  : the current target value for the training problems


%  OUTPUT:

%  totalnf : the total number of function evaluation to solve all problems in
%            the training set
%  msg     : a message returned from the training process
%  wrn     : a warning returned from the training process
%  tpval   : a cell arry whose i-th entry is a vector containing the successive values
%            of the objective function of test problem i during its solution
%  tpdim   : a vector whose i-th entry is the dimension of the i-th test problem.
%  tpval   : a cell array whose i-th entry is a vector containing the successive evaluation
%            counts for test problem i during its solution
%  tpevs   : a cell whose i-th element is a vector containing the evaluation history for 
%            test problems i in terms of full function evaluations, if problem i is 
%            coordinate-partially separable and a profile training is computed; {} if not.

%  DEPENDENCIES : bfo, bfo_cutest_data, bfo_get_verbosity, bfo_exist_function,
%                 cutest_terminate
%
%  PROGRAMMING: Ph. Toint and M. Porcelli, May 2010. (This version 17 XII 2016)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Set default output.

msg     = '';
wrn     = {};
totalnf = 0;

%  Decode the meaning of each parameter

alpha   = p0(1);
beta    = p0(2);
gamma   = p0(3);
delta   = p0(4);
eta     = p0(5);
zeta    = p0(6);
inertia = p0(7);
rseed   = p0(9);
iota    = p0(10);
kappa   = p0(11);
lambda  = p0(12);
mu      = p0(13);

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

nproblems  = length( training_problems );

if ( strcmp( training_strategy, 'perfprofile' )  ||                                        ...
     strcmp( training_strategy, 'dataprofile' )    )
   tpval = cell( 1, nproblems );
else
   tpval = {};
end
tpevs = {};

for i = 1:nproblems

      %  Decode the CUTEst training problems
      
      if ( training_set_cutest ) 
         shpbname   = training_problems{ i };
         tpi        = bfo_cutest_data( shpbname );

      %  Check the presence of the file(s) describing the objective function.

      else
         tpi =  training_problems{ i };
         if ( isfield( tpi, 'name' ) )
            shpbname = tpi.name;
         elseif ( iscell( tpi.objf ) )
            [ ~, shpbname ] = bfo_exist_function( func2str( tpi.objf{ 1 } ) );
         else 
            [ ~, shpbname ] = bfo_exist_function( func2str( tpi.objf ) );
         end
      end
      tpdim( i ) = length( tpi.x0 );
      
      %  Show the name of the training problem.
   
   if ( tverbose > 2 )
      fprintf( '%-40s', [' BFO training: running ', shpbname, ' ...' ] );
   end

   %  Solve the current test problem with the current set of algorithmic parameters.

   if ( ismember( training_strategy, { 'perfprofile', 'dataprofile' } ) )
      if ( ~isempty( fstar ) )
         [ ~, ~, msgp, wrnp, neval, tpval{ i }, ~, ~, ~, ~, ~, ~, ~, ev_hist ] = bfo( tpi, ...
             'epsilon', training_problem_epsilon, 'maxeval', training_problem_maxeval,     ...
	     'verbosity', training_problem_verbosity, 'alpha', alpha, 'beta', beta,        ...
	     'gamma', gamma, 'delta', delta, 'eta', eta, 'zeta', zeta, 'inertia', inertia, ...
	     'search-type', stype, 'random-seed', rseed, 'iota', iota, 'kappa', kappa,     ...
             'lambda', lambda, 'mu', mu, 'f-target', fstar( i ) );
      else
         [ ~, ~, msgp, wrnp, neval, tpval{ i }, ~, ~, ~, ~, ~, ~, ~, ev_hist ] = bfo( tpi, ...
             'epsilon', training_problem_epsilon, 'maxeval', training_problem_maxeval,     ...
	     'verbosity', training_problem_verbosity, 'alpha', alpha, 'beta', beta,        ...
	     'gamma', gamma, 'delta', delta, 'eta', eta, 'zeta', zeta, 'inertia', inertia, ...
	     'search-type', stype, 'random-seed', rseed, 'iota', iota, 'kappa', kappa,     ...
             'lambda', lambda, 'mu', mu );
      end
      if ( ~isempty( ev_hist ) )
         if ( isempty( tpevs ) )
            tpevs = cell( 1, nproblems );
            for ip = 1:i-1
               tpevs{ ip } = [];
            end
         end
         tpevs{ i } = ev_hist;
      end
   else
      [ ~, ~, msgp, wrnp, neval ] = bfo( tpi,                                              ...
             'epsilon', training_problem_epsilon, 'maxeval', training_problem_maxeval,     ...
	     'verbosity', training_problem_verbosity, 'alpha', alpha, 'beta', beta,        ...
	     'gamma', gamma, 'delta', delta, 'eta', eta, 'zeta', zeta, 'inertia', inertia, ...
	     'search-type', stype, 'random-seed', rseed, 'iota', iota, 'kappa', kappa,     ...
             'lambda', lambda, 'mu', mu );
   end

   %  Verify the result of this optimization.

   if ( length( msgp ) >= 10 && strcmp( msgp(1:10), ' BFO error' ) )
       msg = [ ' BFO error: training on problem ', func2str( tpi.objf ),                   ...
               ' returned the message:', msgp(12:end) ];
      if ( tverbose )
         disp( msg )
      end
      return
   end

   if ( ~isempty( wrnp ) && length( wrnp{ 1 } ) >= 11 &&                                   ...
        strcmp( wrnp{ 1 }(1:11), ' BFO warning' ) )
      wrn{ end +1 } = [ ' BFO warning: training on problem ', func2str( fun ),             ...
                        ' issued the warning:', wrnp{ 1 }(13,:) ];   
      if ( verbose )
         disp( wrn{ end } )
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
              training_parameters, training_problems, training_set_cutest,                 ...
	      training_epsilon, training_maxeval, training_verbosity,                      ...
              training_problem_epsilon, training_problem_maxeval, training_problem_verbosity )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Optimize BFO parameters using the robust optimization formulation, that is
%  minimizing the worst behaviour of the algorithm on the training problems for a set
%  of algorithmic parameters differing from at most 5% from its nominal value 
%  (i.e. min max (total number of fevals))).

%  INPUT:

%  p0                     : the initial value of the parameters to be trained
%  bestperf               : the best "worst performance" so far
%  training_parameters    : a cell containing the names of the parameters to be trained
%  training_problems      : the list of objective functions of problems to be used 
%                           for training
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

%  PROGRAMMING: Ph. Toint,and M. Porcelli, May 2010. (This version 10 X 2017)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fup = 1.05;                 %  the relative upper limit of the box (1 + 5%)
fdo = 0.95;                 %  the relative lower limit of the box (1 - 5%)

tverbose = bfo_get_verbosity( training_verbosity );  % the local verbosity

%  Define the box (for the continuous algorithmic parameters to be trained) in which 
%  the worst performance is sought.

sp0   = size( p0 );                     
ptyp  = char( double('f') * ones( sp0 ) ); % ptyp  = 'ffffffffff': all are fixed by default
xlow  = p0;
xupp  = p0;
xsca  = ones( sp0 );
delt  = ones( sp0 );
for  i = 1:length( training_parameters )      % loop on the selected training parameters
   if (     strcmp( training_parameters{ i }, 'alpha'       ) )
      ptyp( 1 ) = 'c';
      xlow( 1 ) = fdo * p0( 1 );
      xupp( 1 ) = fup * p0( 1 );
      delt( 1 ) = 0.025;
   elseif ( strcmp( training_parameters{ i }, 'beta'        ) )
      ptyp( 2 ) = 'c';
      xlow( 2 ) = fdo * p0( 2 );
      xupp( 2 ) = fup * p0( 2 );
      delt( 2 ) = 0.025;
   elseif ( strcmp( training_parameters{ i }, 'gamma'       ) )
      ptyp( 3 ) = 'c';
      xlow( 3 ) = fdo * p0( 3 );
      xupp( 3 ) = fup * p0( 3 );
      delt( 3 ) = 0.025;
   elseif ( strcmp( training_parameters{ i }, 'delta'       ) )
      ptyp( 4 ) = 'c';
      xlow( 4 ) = fdo * p0( 4 );
      xupp( 4 ) = fup * p0( 4 );
      delt( 4 ) = 0.025;
   elseif ( strcmp( training_parameters{ i }, 'eta'         ) )
      ptyp( 5 ) = 'c';
      xlow( 5 ) = fdo * p0( 5 );
      xupp( 5 ) = fup * p0( 5 );
      delt( 5 ) = 0.025;
   elseif ( strcmp( training_parameters{ i }, 'zeta'        ) )
      ptyp( 6 ) = 'c';
      xlow( 6 ) = fdo * p0( 6 );
      xupp( 6 ) = fup * p0( 6 );
      delt( 6 ) = 0.025;
   elseif ( strcmp( training_parameters{ i }, 'inertia'     ) )
   elseif ( strcmp( training_parameters{ i }, 'search-type' ) )
   elseif ( strcmp( training_parameters{ i }, 'random-seed' ) )
   elseif ( strcmp( training_parameters{ i }, 'iota'        ) )
      ptyp( 10 ) = 'c';
      xlow( 10 ) = fdo * p0( 10 );
      xupp( 10 ) = fup * p0( 10 );
      delt( 10 ) = 0.025;
   elseif ( strcmp( training_parameters{ i }, 'kappa'       ) )
      ptyp( 11 ) = 'c';
      xlow( 11 ) = fdo * p0( 11 );
      xupp( 11 ) = fup * p0( 11 );
      delt( 11 ) = 0.025;
   elseif ( strcmp( training_parameters{ i }, 'lambda'      ) )
      ptyp( 12 ) = 'c';
      xlow( 12 ) = fdo * p0( 12 );
      xupp( 12 ) = fup * p0( 12 );
      delt( 12 ) = 0.025;
   elseif ( strcmp( training_parameters{ i }, 'mu'          ) )
      ptyp( 13 ) = 'c';
      xlow( 13 ) = fdo * p0( 13 );
      xupp( 13 ) = fup * p0( 13 );
      delt( 13 ) = 0.025;
   end
end

%  Find the worst performance in this box by maximizing the total number of
%  function evaluations.

if ( tverbose > 1 )
   disp ( ' ---------- Evaluating worst case in the box ----------------' )
end

[ ~, fworst, msg, wrn, ~, ~, ~, ~, th ] =                                                  ...
     bfo( @(x,bestperf)bfo_average_perf( x, bestperf, 'robust', training_problems,         ...
                                training_set_cutest,                                       ...
                                training_verbosity, training_problem_epsilon,              ...
                                training_problem_maxeval, training_problem_verbosity,      ...
                                [] ),                                                      ...
          p0, 'xscale', xsca, 'xtype', ptyp, 'xupper', xupp, 'xlower', xlow, 'epsilon',    ...
          training_epsilon, 'termination-basis', 1, 'verbosity', training_verbosity,       ...
          'search-type', 'none', 'max-or-min', 'max', 'maxeval', training_maxeval,         ...
          'f-target', bestperf, 'f-call-type', 'with-bound', 'f-bound', bestperf,          ...
          'delta', delt );

if ( tverbose > 1 )
   disp ( ' ------------------------------------------------------------' )
end

nevalt = th( size( th, 1 ), 3 );

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [ surface, msg, wrn, nevalt, fstar, tpval ] =                                     ...
                          bfo_dpprofile_perf( p0, fstar, tpval,                            ...
                                              training_strategy,                           ...
                                              training_profile_window,                     ...
                                              training_profile_cutoff_fraction,            ...
                                              training_problems,                           ...
                                              training_set_cutest,                         ...
                                              training_verbosity,                          ...
                                              training_problem_epsilon,                    ...
                                              training_problem_maxeval,                    ...
                                              training_problem_verbosity )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Optimize BFO parameters using the data profile or performance profile area formulation.
%  that is maximizing the surface below the graph of the data profile on the training
%  problems for a set of algorithmic parameters, or maximizing the surface between the 
%  performance profile for the current variant and that for the initial variant.


%  INPUT:

%  p0                     : the initial value of the parameters to be trained
%  fstar                  : at first training evaluation, a vector containing the current 
%                           best function value for each training problem;
%                           at later iterations, a vector containing the target values for
%                           each training problem
%  tpval                  : at first training iteration, a cell whose i-th entry is the
%                           vector of objective values history for the i-th training problem,
%                           at later iterations, a cell whose i-th entry is the 
%                           number of evaluations required by the initial variant to solve
%                           problem i
%  training_profile_window: the range of abscissas for which the data- or performence profile
%                           is computed ( given as [range lower bound, range upper bound])
%  training_profile_cutoff_fraction: the fraction of |f(x0)-f_*| determining the (accuracy
%                           dependent) solution value for each training problem
%  training_strategy      : the requested training strategy
%  training_parameters    : a cell containing the names of the parameters to be trained
%  training_problems      : the list of objective functions of problems to be used 
%                           for training
%  training_set_cutest    : the associated problems library 
%  training_verbosity     : the verbosity of the training problem itself 
%  training_problem_epsilon : the accuracy at which each of the training
%                           problems must be solved
%  training_problem_maxeval : the maximum number of objective evaluations in the
%                           solution of each training problem
%  training_problem_verbosity : the verbosity of the solution of each test problem


%  OUTPUT:

%  surface            :
%  msg                    : a message returned from the training process
%  wrn                    : a warning returned from the training process
%  nevalt                 : the number of training problem's function
%                           evaluation during the call to bfo_robust_perf
%  fstar                  : a vector containing the target values for each training problem
%  tpval                  : a cell whose i-th entry is the number of evaluations required
%                           by the initial variant to solve problem i
%  

%  DEPENDENCIES : bfo_average_perf, bfo_profile_area

%  PROGRAMMING: Ph. Toint,and M. Porcelli, December 2016. (This version 1 II 2018)
%               Considerably inspired by Jorge More and Stefan Wild's 
%               "Benchmarking Derivative-Free Optimization Algorithms".

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Interpret some arguments.

tverbose = bfo_get_verbosity( training_verbosity );  % the local verbosity
data     = strcmp( training_strategy, 'dataprofile' );

%  Find the surface below the data or performance profile.

if ( tverbose > 2 )
   disp ( ' ---------- Evaluating surface performance ----------------' )
end

nprob    = length( training_problems );  
bestperf = Inf;

%  First iteration

if ( isempty( fstar ) )

   %  Run all test problems to establish their reference objective function values in fstar.
 
   [ nevalt, msg, wrn, tpval, tpdim, tpevs ] = bfo_average_perf( p0, Inf,                  ...
                             training_strategy, training_problems, training_set_cutest,    ...
                             training_verbosity, training_problem_epsilon,                 ...
                             training_problem_maxeval, training_problem_verbosity, [] );

   % For each problem and solver, determine the number of evaluations
   % required to reach the cutoff value.

   T     = zeros( nprob, 1 );
   for i = 1:nprob
      fstar( i ) = tpval{ i }( end );
      f0( i )    = tpval{ i }( find( abs( tpval{ i } ) < Inf, 1 ) ); 
      if ( f0( i ) > fstar( i ) )
         cutoff  = fstar( i ) + training_profile_cutoff_fraction * ( f0( i ) - fstar( i ) );
         ncut    = find( tpval{ i } < cutoff, 1 );
      else
         cutoff  = fstar( i ) - training_profile_cutoff_fraction * ( fstar( i ) - f0( i ) );
         ncut    = find( tpval{ i } > cutoff, 1 );
      end
      if ( isempty( ncut ) )
         T( i ) = NaN;
      else
         if ( ~isempty( tpevs ) && ~isempty( tpevs{ i } ) )
            nfevs = tpevs{ i }( ncut );
         else
            nfevs = ncut;
         end
         if ( data )
            T( i ) = nfevs  / ( tpdim( i ) + 1 );
         end
      end
      tpval{ i } = nfevs;
      fstar( i ) = cutoff;
   end

   %  If data profile are requested, establish the first one and compute its 
   %  associated surface. 

   if ( data )
      T( isnan( T ) ) = 2 * max( T );
      [ xs, ys ] = stairs( sort( T ), ( 1:nprob )/nprob );
      surface    = bfo_profile_area( xs, ys, training_profile_window );

   %  If performance profile are requested, directly compute the area below the
   %  constant performance profile.

   else
      surface = 0;
   end

   %  Show plot, if requested.

   if ( 0 )

      % Plot stair graph.

      clf
      if ( data )
         plot( [ 0; xs(1); xs ], [ 0; 0; ys ], 'b-', 'LineWidth' ,2 );
         hold on;   
         title( 'Data Profiles' )
         axis([ training_profile_window , 0, 1 ] );
         legend( 'BFO - current' )
         pause(0.01)
      end

   end

%  Iterations beyond the first.

else

   %  Run the test problems with the current set of algorithmic parameters.

   [ nevalt, msg, wrn, ntpval, tpdim, tpevs ] = bfo_average_perf( p0, Inf,                 ...
                             training_strategy, training_problems, training_set_cutest,    ...
                             training_verbosity, training_problem_epsilon,                 ...
                             training_problem_maxeval, training_problem_verbosity,         ...
                             fstar );

   % For each problem and solver, determine the number of evaluations
   % required to reach the cutoff value.

   T = zeros( nprob, 2 ); 
   for i = 1 : nprob
      f0( i ) = ntpval{ i }( 1 );  
      n       = tpdim( i );
      if ( data )
         T( i, 1 ) = tpval{ i } / ( n + 1 );
      else
         T( i, 1 ) = tpval{ i };
      end
      cutoff = fstar( i );
      if ( f0( i ) > fstar( i ) )
        ncut = find( ntpval{ i } < cutoff, 1 );
      else
        ncut = find( ntpval{ i } > cutoff, 1 );
      end
      if ( isempty( ncut ) )
         T( i, 2 ) = NaN;
      else
         if ( ~isempty( tpevs ) && ~isempty( tpevs{ i } ) )
            nfevs = tpevs{ i }( ncut );
         else
            nfevs = ncut;
         end
         if ( data )
            T( i, 2 ) = nfevs / ( n + 1 );
         else
            T( i, 2 ) = nfevs;
         end
      end
   end

   %  If performance profiles are requested, scale the performance by that of the best
   %  variant.

   if ( ~data )
      T = diag( 1./min( T' ) ) * T;
   end

   % Replace all NaN's with twice the max_ratio and sort.

   max_data = max( max( T ) );
   T( isnan( T ) ) = 2 * max_data;
   T = sort( T, 1 );

   %  Compute the data profile for the given profile window

   [ xn, yn ] = stairs( T(:,2), ( 1:nprob )/nprob );

   surfacen   = bfo_profile_area( xn, yn, training_profile_window );
   if ( data )
      surface = surfacen;
   else
      [ xp, yp ] = stairs( T(:,1), ( 1:nprob )/nprob );
      surfacep   = bfo_profile_area( xp, yp, training_profile_window );
      surface    = surfacen - surfacep;   
   end

   %  Show plots, if requested.

   if ( 0 )

      % For each solver, plot stair graphs.

      clf
      if ( data )
         plot( [ 0; xp(1); xp ], [ 0; 0; yp ], 'r-', 'LineWidth' ,2 );
         hold on;   
         plot( [ 0; xn(1); xn ], [ 0; 0; yn ], 'b-', 'LineWidth', 2 );
         title( 'Data Profiles' )
         axis([ training_profile_window , 0, 1 ] );
      else
         plot( [ 1; xp(1); xp ], [ 0; 0; yp ], 'r-', 'LineWidth', 2 );
         hold on;   
         plot( [ 1; xn(1); xn ], [ 0; 0; yn ], 'b-', 'LineWidth', 2 );
         title( 'Performance Profiles' )
         axis([ training_profile_window, 0, 1 ] );
      end
      legend( 'BFO - current', 'BFO - new' )

%      pause(0.01)
     pause

   end
end

if ( tverbose > 2 )
   disp ( ' ------------------------------------------------------------' )
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function area = bfo_profile_area( x, y, window )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Computes the area below a staircase curve (given by a performance or data profile)
%  in the interval specified by window. In BFO, the curve is computed using the stairs 
%  Matlab function.

%  INPUT :

%  x     : the abscissas of the staircase curve (with repetitions at breakpoints)
%  y     : the values of the staircase curve
%  window: a vector whose first element is the lower bound on the considered window
%          and the second its upper bound

%  OUTPUT:

%  area : the area below the staircase curve within the woindow

%  DEPENDENCIES : -

%  PROGRAMMING: Ph. Toint,and M. Porcelli, December 2014. (This version 4 X 2017)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Remove the abscissas before the lower window bound and after the upper bound, if necessary.
%  Also complete the interval on the right.

before = find( x <= window( 1 ), 1, 'last' );
if ( length( before ) )
   x  = [ window( 1 ) ; x( before+1:end ) ];
   y  = y( before:end );
end
after = find( x >= window( 2 ), 1, 'first' );
if ( length( after ) )
   x  = [ x( 1:after-1 ) ; window( 2 ) ];
   y  = y( 1:after );
else
   x  = [ x ; window( 2 ) ];
   y  = [ y ; y(end) ];
end

%  Compute the area below the curve in the relevant window.

area = 0;
for i = 2 : length( x )
   if x( i ) > x( i - 1 )
      area = area + 0.5 * ( x( i ) - x( i - 1 ) ) * ( y( i ) + y( i - 1 ) );
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [ xbest, fbest, msg, wrn, neval, f_hist, xincr, el_hist ] =                       ...
         bfo_next_level_objf( level, nlevel, xlevel, neval, objf, x, checking, f_hist,     ...
                              el_hist, xtype, xincr, xscale, xlower, xupper, eldom,        ...
                              max_or_min, vb_name, epsilon, bfgs_finish,  maxeval,         ...
                              verbosity, fcallt, alpha, beta, gamma, eta, zeta, inertia,   ...
                              stype, rseed, iota, kappa, lambda, mu, term_basis,           ...
                              latbasis, idall, reset_random_seed, ssfname, cn_name,        ...
                              cat_states )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  This function provides a call to BFO for optimizing variables at a level beyond that
%  of the calling process.

%  INPUT:

%  level             : the level at which optimization must be performed
%  nlevel            : the total number of levels
%  xlevel            : the assignment of variables to levels
%  obj               : the objective function handle
%  checking          : true is the calling process is close to convergence
%  f_hist            : the history of objective function evaluations
%  el_hist           : the element-wise history
%  xtype             : the variables types
%  xincr             : the current set of (level dependent) stepsizes
%  xscale            : the variables' scalings
%  xlower            : the lower bounds on the variables
%  xupper            : the upper bounds on the variables
%  eldom             : the cell array containing the domain defining vectors
%  max-or-min        : the max/min program for the various levels
%  vb_name           : the name of the user-supplied variable bounds function
%  epsilon           : the increment accuracy
%  bfgs_finish       : the meshsize under which BFGS is attempted
%  maxeval           : the maximum number of function evaluations
%  verbosity         : the desired verbosity level at level level and beyond
%  fcallt            : the type of objective function call
%  alpha             : the grid expansion factor
%  beta              : the grid expansion/reduction factor
%  gamma             : the maximum grid expansion factor for continuous variables
%  delta             : the single mesh parameter
%  eta               : the sufficient decrease factor
%  zeta              : the multilevel re-expansion factor
%  inertia           : the number of iterations use for continuous step averaging
%  stype             : the discrete variables search type
%  rseed             : the random number generator's seed
%  iota              : the CPS stepsize shrinking exponent
%  kappa             : the bracket expansion factor in min1d without quadratic interpolation
%  lambda            : the min bracket expansion factor in min1d with quadratic interpolation
%  mu                : the max bracket expansion factor in min1d with quadratic interpolation
%  term_basis        : the number of random basis used for assessing termination
%  latbasis          : the lattice basis ([] if none)
%  idall             : the indeces of discrete varaibles across levels
%  reset_random_seed : the request for resetting the random seed
%  sssfname          : the name of the search step is requested
%  cn_name           : the name of the categorical neighbourhood fubction
%  cat_states        : the list of categorical values

%  OUTPUT:

%  xbest             : the best point resulting from optimization at level level and beyond
%  fbest             : the objective function value at xbest
%  msg               : a termination message
%  wrn               : a termination warning
%  neval             : the cumulated number of function evaluations
%  f_hist            : the cumulated history of objective function evaluations
%  xincr             : the (possibly updated) set of (level dependent) stepsizes
%  el_hist           : the element-wise history

%  DEPENDENCIES : bfo, vb_name

%  PROGRAMMING: Ph. Toint,and M. Porcelli, November 2014. (This version 7 III 2018)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  If needed, recompute the bounds as a function of the variables in the
%  levels up to the current one.

xlnxt = xlower;
xunxt = xupper;
xnxt  = x;
if ( ~isempty( vb_name ) )
   vbname          = str2func( vb_name );
   if ( isempty( xscale ) )
      [ xl, xu ]   = vbname( x, level, xlevel, xlower, xupper );
   else 
      [ xl, xu ]   = vbname( xscale.*x, level, xlevel, xscale.*xlower, xscale.*xupper );
      finite       = intersect( icd, find( xl > myinf ) );
      xl( finite ) = xl( finite ) ./ xscale( finite );
      finite       = intersect( icd, find( xu < myinf ) );
      xu( finite ) = xu( finite ) ./ xscale( finite );
   end
   inxt            = find( xlevel > level );                   % indices of next levels vars
   xlnxt( inxt )   = min( [ xl( inxt )'; xu( inxt )' ] )';     % only update those bounds...
   xunxt( inxt )   = max( [ xl( inxt )'; xu( inxt )' ] )';
   if ( isnumeric( x ) )
      xnxt( inxt ) = min( [ max( [ xlnxt( inxt )'; x( inxt )' ]); xunxt( inxt )' ] ); 
   else
      for i = inxt
         if ( xtype( i ) == 'c'  || xtype( i ) == 'd' )
            xnxt{ i } = min( max( xlnxt( i ), x{ i } ), xunxt( i ) );  % ... and variables
         end
      end
   end
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
%      xincr( i, 1 ) = min ( zeta * xincr( i, 1 ), xincr( i, 2 ) );
      xincr( i, 1 ) = zeta * xincr( i, 1 );
   end
end

%  If the problem is in sum form, make the objective function a value cell array.

if ( iscell( objf ) )
   objf = { objf };
end

%  Optimize the objective function corresponding to the variables of the next level.
%  Note that save-freq is not specified, preventing restarts at levels beyond 1.

[ xbest, fbest, msg, wrn, neval, f_hist,~,~,~, ssh, xincr, nopt_context, el_hist ] =       ...
         bfo( objf, xnxt, 'topmost', 'no', 'eldom', {eldom}, 'level', level+1,             ...
              'xlevel', xlevel, 'variable-bounds', vb_name, 'xtype', xtype,                ...
              'xincr', xincr, 'xscale', xscale, 'xupper', xunxt, 'xlower', xlnxt,          ...
              'max-or-min', max_or_min, 'epsilon', epsilon, 'maxeval', maxeval,            ...
              'verbosity', verbosity, 'f-call-type', fcallt, 'alpha', alpha, 'beta', beta, ...
              'gamma', gamma, 'eta', eta, 'zeta', zeta, 'inertia', inertia, 'search-type', ...
              stype, 'random-seed', rseed, 'iota', iota, 'kappa', kappa, 'lambda', lambda, ...
              'mu', mu, 'nevr', neval, 'f-hist', f_hist, 'element-hist', el_hist,          ...
              'lattice-basis', latbasis, 'idall', idall, 'termination-basis',              ...
              nexttermbasis, 'bfgs-finish', bfgs_finish, 'reset-random-seed',              ...
              reset_random_seed, 'search-step', ssfname, 'cat-neighbours', cn_name,        ...
              'cat-states', {cat_states} );

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function savok = bfo_save( filename, objname, maximize, epsilon, ftarget,  maxeval, neval, ...
                           f_hist, xtype, xincr, xscale, xlower, xupper, verbose,          ...
                           alpha, beta, gamma, eta, zeta, inertia, stype, rseed, iota,     ...
                           kappa, lambda, mu, term_basis, used_trained, hist,              ...
                           latbasis, bfgs_finish, training_history, fstar, tpval, ssfname, ...
                           cn_name, cat_states, cat_dictionnary )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Save the algorithmic parameters to the filename file, in order to allow for
%  future restart.

%  INPUT:

%  filename        : the name of the file for saving
%  objname         : the name of the objective function
%  maximize        : the maximization/minimization flag
%  epsilon         : the termination accuracy
%  ftarget         : the objective function target
%  maxeval         : the maximum number of evaluations
%  neval           : the current number of evaluations
%  f_hist          : the current vector of all computed function values 
%  xtype           : the variables' types
%  xincr           : the current increments
%  xscale          : the variables' scaling
%  xlower          : the current lower bounds on the variables
%  xupper          : the current upper bounds on the variables
%  verbose         : the printout quantity flag
%  alpha           : the grid expansion factor
%  beta            : the grid expansion/reduction factor
%  gamma           : the maximum grid expansion factor for continuous variables
%  eta             : the sufficient decrease factor
%  zeta            : the multilevel re-expansion factor
%  inertia         : the number of iterations use for continuous step averaging
%  stype           : the discrete variables search type
%  rseed           : the random number generator's seed
%  iota            : the CPS stepsize shrinking factor
%  kappa           : the bracket expansion factor in min1d without quadratic interpolation
%  lambda          : the min bracket expansion factor in min1d with quadratic interpolation
%  mu              : the max bracket expansion factor in min1d with quadratic interpolation
%  term_basis      : the number of random basis used for assessing termination
%  use_trained     : the flag indicating use of trained BFO parameters
%  hist            : the saved information of the explored subspaces ( a struct)
%  latbasis        : the lattice basis ([] if none)
%  bfgs_finish     : the meshsize under which BFGS is attempted
%  training_history: the history of training so far
%  fstar           : the list of objectiv function reference values for each training problem
%  tpval           : the last evaluation history for each training problem
%  ssfname         : the name of the user-defined search step
%  cn_name         : the name of the dynamical categorical neighbourhoods function
%  cat_states      : the static cateorical neighbourhoods
%  cat_dictionnary : the current dictionnary of categorical values

%  OUTPUT:

%  savok        : 1 if the parameters could be saved, 0 otherwise

%  DEPENDENCIES: -

%  PROGRAMMING: Ph. Toint, May 2009. (This version 4 II 2018)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Save data.

fid = fopen( filename, 'w' );
if ( fid == -1 )
   savok = 0;
else

   size_lattice = size( latbasis, 2 );
   categorical  = ~isnumeric( hist(1).x );
   n            = length( xlower);
   nh           = length( hist );
   nfcalls      = length( f_hist );
   empty_xsc    = isempty( xscale );

   fprintf( fid, ' *** BFO optimization checkpointing file %s\n', date );
   fprintf( fid, ' *** (c) Ph. Toint and M. Porcelli\n' );
   fprintf( fid,' %18s   %-12s\n',  objname,             'objname'         );
   fprintf( fid,' %18d   %-12s\n',  maximize,            'maximize'        );
   fprintf( fid,' %18d   %-12s\n',  verbose,             'verbose'         );
   fprintf( fid,' %.11e   %-12s\n', ftarget,             'ftarget'         );
   fprintf( fid,' %.12e   %-12s\n', epsilon,             'epsilon'         );
   fprintf( fid,' %18d   %-12s\n',  bfgs_finish,         'bfgs_finish'     );
   fprintf( fid,' %18d   %-12s\n',  term_basis,          'term_basis'      );
   fprintf( fid,' %18d   %-12s\n',  maxeval,             'maxeval'         );
   fprintf( fid,' %.12e   %-12s\n', neval,               'neval'           );
   fprintf( fid,' %.12e   %-12s\n', alpha,               'alpha'           );
   fprintf( fid,' %.12e   %-12s\n', beta,                'beta'            );
   fprintf( fid,' %.12e   %-12s\n', gamma,               'gamma'           );
   fprintf( fid,' %.12e   %-12s\n', eta,                 'eta'             );
   fprintf( fid,' %.12e   %-12s\n', zeta,                'zeta'            );
   fprintf( fid,' %18d   %-12s\n',  inertia,             'inertia'         );
   fprintf( fid,' %18d   %-12s\n',  stype,               'stype'           );
   fprintf( fid,' %18d   %-12s\n',  rseed,               'rseed'           );
   fprintf( fid,' %.12e   %-12s\n', iota,                'iota'            );
   fprintf( fid,' %.12e   %-12s\n', kappa,               'kappa'           );
   fprintf( fid,' %.12e   %-12s\n', lambda,              'lambda'          );
   fprintf( fid,' %.12e   %-12s\n', mu,                  'mu'              );
   fprintf( fid,' %18d   %-12s\n',  used_trained,        'used_trained'    );
   fprintf( fid,' %18d   %-12s\n',  n,                   'n'               );
   fprintf( fid,' %18s   %-12s\n',  xtype,               'xtype'           );
   fprintf( fid,' %18d   %-12s\n',  size_lattice,        'size_lattice'    );
   fprintf( fid,' %18d   %-12s\n',  categorical,         'categorical'     );
   fprintf( fid,' %18d   %-12s\n',  empty_xsc,           'empty_xsc'       );
   if ( ~isempty( ssfname ) )
      fprintf( fid,' %18s   %-12s\n',  ssfname,          'ssfname'         );
   else
      fprintf( fid,'           bfo_none   %-12s\n',      'ssfname'         );
   end
   fprintf( fid,' %18d   %-12s\n',  nfcalls,             'nfcalls'         );
   fprintf( fid,' %+.12e   ', f_hist( 1:nfcalls ) );
   fprintf( fid, '\n' );
   fprintf( fid,' %18d   %-12s\n',  size( xincr, 2 ),    'ncols_xincr'     );
   for icol = 1:size( xincr, 2 )
      fprintf( fid, '%+.12e   ', xincr( 1:n, icol ) );
      fprintf( fid, '\n' );
   end
   if ( ~empty_xsc )
      fprintf( fid, '%+.12e   ', xscale( 1:n ) );
   end
   fprintf( fid, '\n' );
   fprintf( fid, '%+.12e   ', xlower( 1:n ) );
   fprintf( fid, '\n' );
   fprintf( fid, '%+.12e   ', xupper( 1:n ) );
   fprintf( fid, '\n' );
   for j = 1:size_lattice
      fprintf( fid, '%+.12e   ', latbasis( 1:size_lattice, j ) );
      fprintf( fid, '\n' );
   end

   %  Save the categorical quantities.
   
   if ( categorical )
      if ( ~isempty( cn_name ) )
         fprintf( fid,' %18s   %-12s\n',  cn_name,             'cn_name'     );
      else
         fprintf( fid,'           bfo_none   %-12s\n',         'cn_name'     );
      end
      if ( isempty( cn_name ) )                             % the cat_states
         xref = hist( 1 ).x;
	 icat = [];
	 for j = 1:n                                        % get the vat variable's indices
	    if ( ischar( xref{ j } ) )
	       icat( end+1 ) = j;
	    end
	 end
         fprintf( fid, '%5d \n',length( icat) );
         for j = icat                                       % save the states for those
            ns = length( cat_states{ j } );
            fprintf( fid, '%5d %2d ', j, ns );
            for is = 1:ns
               fprintf( fid, '%-12s ', cat_states{ j }{ is } );
            end
            fprintf( fid, '\n' );
         end
      else
         lcatdico = length( cat_dictionnary );
         fprintf( fid, '%2d ', lcatdico );
	 for is = 1:lcatdico
            fprintf( fid, '%-12s ', cat_dictionnary{ is } );
         end
         fprintf( fid, '\n' );
      end
   end

   %  Save the subspace history.
   
   fprintf( fid,' %18d   %-12s\n',  nh,           'nh');
   for j = 1:nh
      if ( categorical )
         for i  = 1:n
	    cxi = hist( j ).x{ i };                                    % the vector states
            if ( isnumeric(  cxi ) )
               fprintf( fid, '%+.12e  ', cxi );
	    else
               fprintf( fid, '%12s  ', cxi );
            end
	  end
      else
         fprintf( fid, '%+.12e   ', hist(j).x( 1:n ) );
      end
      fprintf( fid, '\n' );
      fprintf( fid, '%+.12e   ', hist(j).fx, hist(j).xincr( 1:n) );  % objf and xincr
      fprintf( fid, '\n' );
      if ( categorical && length( cn_name ) )                        % the contexts
         fprintf( fid, '%s   '    , hist( j ).context.xtype  );
         fprintf( fid, '%+.12e   ', hist( j ).context.xlower( 1:n ) );
         fprintf( fid, '%+.12e   ', hist( j ).context.xupper( 1:n ) );
         fprintf( fid, '\n' );
      end
   end
   lh = size( training_history, 1 );
   fprintf( fid,' %18d   %-12s\n',  lh,                 'training history length');

   %  NOTE-TRAINING: 
   %  in the following paragraph, 17 is 4 + the total number of trainable parameters.
   %        The format fmt must accomodate the 17 columns of training_history.

   if ( lh > 0 )
      fmt = [ '%1d %3d %10d %10d %.12e %.12e %.12e %.12e %.12e %.12e %2d %1d %3d ',        ...
              '%.12e %.12e %.12e %.12e\n' ];
      for j = 1:lh
         fprintf( fid, fmt, training_history( j, 1:17 ) );
      end
   end
   lh = length( fstar );
   fprintf( fid,' %18d   %-12s\n',  lh,                  'fstar length');
   if ( lh > 0 )
      fprintf( fid, '%+.12e   ', fstar( 1:lh ) );
      fprintf( fid, '\n' );
   end
   lh = length( tpval );
   fprintf( fid,' %18d   %-12s\n',  lh,                  'tpval length');
   if ( lh > 0 )
      for ip = 1:lh
         lip = length( tpval{ ip } );
         fprintf( fid, ' %5d ', lip );
         fprintf( fid, '%+.12e   ', tpval{ ip }( 1:lip ) );
         fprintf( fid, '\n' );
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
                                    training_problems, training_set_cutest,                ...
				    trained_bfo_parameters, training_epsilon,              ...
				    training_maxeval, training_verbosity,                  ...
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
%  training parameters: a cell whose elements are strings which are the names of the
%                 parameters to train
%  training_problems : a cell whose entries specify (in condensed form)
%                 the test problems on which training is performed 
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

%  PROGRAMMING: Ph. Toint, December 2014. (This version 5 X 2017)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Save data.

fid = fopen( [ filename, '.training' ], 'w' );
if ( fid == -1 )
   savok = 0;
else
   fprintf( fid, ' *** BFO training checkpointing file %s\n', date );
   fprintf( fid, ' *** (c) Ph. Toint and M. Porcelli\n' );
   fprintf( fid,' %18s   %-12s\n',  training_strategy,   'training_strategy');
   fprintf( fid,' %18d   %-12s\n',  training_set_cutest, 'training_set_cutest');
   np = length( training_parameters );
   fprintf( fid,' %18d   %-12s\n',  np,                  'nbr_training_params');
   fprintf( fid, '%s  ', training_parameters{ 1:np } );
   fprintf( fid, '\n' );
   np = length( p );
   fprintf( fid, '%+.5e   ', p( 1:13 ) ); % NOTE-TRAINING: 13 is the total number 
                                          % of trainable params!
   fprintf( fid, '\n' );
   np = length( training_problems );
   fprintf( fid,' %18d   %-12s\n',  np,                  'nbr_training_problems');
   for j = 1:np
      if ( training_set_cutest )
         fprintf( fid, ' %18s ', training_problems{ j } );
      else
         bfo_print_prob( fid, training_problems{ j } );
      end
   end
   if ( training_set_cutest )
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


function  [ objname, maximize, epsilon, ftarget, maxeval, neval, f_hist, xtype,            ...
            xincr, xscale, xlower, xupper, verbose, alpha, beta, gamma, eta, zeta, inertia,...
            stype, rseed, iota, kappa, lambda, mu, term_basis, use_trained, hist, latbasis,...
            bfgs_finish, training_history, fstar, tpval, ssfname, cn_name, cat_states,     ...
            cat_dictionnary, restok ] =  bfo_restore( filename, readall )

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
%   f_hist       : the current vector of all computed function values 
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
%   iota         : the CPS stepsize shrinking exponent
%   kappa        : the bracket expansion factor in min1d without quadratic interpolation
%   lambda       : the min bracket expansion factor in min1d with quadratic interpolation
%   mu           : the max bracket expansion factor in min1d with quadratic interpolation
%   term_basis   : the number of random basis used for assessing termination
%   use_trained  : the flag indicating use of trained BFO parameters
%   hist         : the saved information of the explored subspaces
%   latbasis     : the lattice basis ([] if none )
%   bfgs_finish  : the meshsize under which BFGS is attempted
%   training_history : the training history so far
%   fstar        : the list of objectiv function reference values for each training problem
%   tpval        : the last evaluation history for each training problem
%   ssfname      : the name of the user-defined search step function
%   cn_name      : the name of the user-defined function for dynamical neighbourhoods
%   cat_states   : the static categorical neighbourhoods
%   restok       : 1 if restore successful, 0 if unsuccessful, 2 if successful in training


%   PROGRAMMING: Ph. Toint,and M. Porcelli, May 2010. (This version 4 II 2018)

%   DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Attempt to open the restart file.

fid = fopen( filename, 'r' );

%  The file can't be opened: return meangless parameters.

if ( fid == -1 )
   objname          = '';
   maximize         = NaN;
   epsilon          = NaN;
   ftarget          = NaN;
   maxeval          = NaN;
   neval            = NaN;
   f_hist           = [];
   xtype            = '';
   xincr            = [];
   xscale           = [];
   xlower           = [];
   xupper           = [];
   verbose          = NaN;
   alpha            = NaN;
   beta             = NaN;
   gamma            = NaN;
   eta              = NaN;
   zeta             = NaN;
   inertia          = NaN;
   stype            = NaN;
   rseed            = NaN;
   iota             = NaN;
   kappa            = NaN;
   lambda           = NaN;
   mu               = NaN;
   hist             = struct([]);
   use_trained      = NaN;
   term_basis       = NaN;
   latbasis         = [];
   restok           = 0;
   bfgs_finish      = NaN;
   training_history = [];
   fstar            = [];
   tpval            = [];
   ssfname          = '';
   cn_name          = '';
   cat_states       = {{}};
   cat_dictionnary  = {};
   return

%  The restart file opened fine: read the restart parameters.

else
   filetitle = fscanf( fid, '%s', 13 );
   objname   = fscanf( fid, '%s', 1 ); name = fscanf( fid, '%s\n', 1 );
   if ( ~readall &&  ismember( objname, { 'bfo_average_perf'  , 'bfo_robust_pertf',       ...
                                          'bfo_dpprofile_perf' } ) )
      maximize         = 0;
      verbose          = 0;
      epsilon          = NaN;
      ftarget          = NaN;
      maxeval          = NaN;
      neval            = 0;
      f_hist           = [];
      xtype            = '';
      xincr            = [];
      xscale           = [];
      xlower           = [];
      xupper           = [];
      alpha            = NaN;
      beta             = NaN;
      gamma            = NaN;
      eta              = NaN;
      zeta             = NaN;
      inertia          = NaN;
      stype            = NaN;
      rseed            = NaN;
      iota             = NaN;
      kappa            = NaN;
      lambda           = NaN;
      mu               = NaN;
      hist             = struct([]);
      use_trained      = NaN;
      term_basis       = NaN;
      latbasis         = [];
      bfgs_finish      = NaN;
      training_history = [];
      fstar            = [];
      tpval            = [];
      ssfname          = '';
      cn_name          = '';
      cat_states       = {{}};
      cat_dictionnary  = {};
      restok           = 2;
      return
   end
   maximize        = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   verbose         = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   ftarget         = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   epsilon         = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   bfgs_finish     = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   term_basis      = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   maxeval         = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   neval           = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   alpha           = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   beta            = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   gamma           = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   eta             = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   zeta            = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   inertia         = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   stype           = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   rseed           = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   iota            = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   kappa           = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   lambda          = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   mu              = fscanf( fid, '%e', 1 ); name = fscanf( fid, '%s\n', 1 );
   use_trained     = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   n               = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   xtype           = fscanf( fid, '%s', 1 ); name = fscanf( fid, '%s\n', 1 );
   size_lattice    = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   categorical     = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   empty_xsc       = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   ssfname         = fscanf( fid, '%s', 1 ); name = fscanf( fid, '%s\n', 1 );
   if ( strcmp( ssfname, 'bfo_none' ) )
      ssfname = '';
   end
   nfcalls      = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   f_hist(1:nfcalls) = fscanf( fid, '%e', nfcalls );
   ncols_xincr  = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   for icol = 1:ncols_xincr
      xincr(1:n,icol)  = fscanf( fid, '%e', n );
   end
   if ( empty_xsc )
      xscale = [];
   else
      xscale(1:n,1) = fscanf( fid, '%e', n );
   end
   xlower(1:n,1) = fscanf( fid, '%e', n );
   xupper(1:n,1) = fscanf( fid, '%e', n );
   if ( size_lattice )
      for j = 1:size_lattice
         latbasis(1:size_lattice,j) = fscanf( fid, '%e', n );
      end
   else
      latbasis = [];
   end
   if ( categorical )
      cn_name    = fscanf( fid, '%s', 1 ); name = fscanf( fid, '%s\n', 1 );
      if ( strcmp( cn_name, 'bfo_none' ) )
         cn_name = '';
         cat_dictionnary = {};
         cat_states      = cell( n, 1 ) ;
	 for j = 1:n
	    cat_states{ j } = '';
	 end
	 licat = fscanf( fid, '%d ', 1 );
	 for jl = 1:licat
            jstates = {};
  	    j       = fscanf( fid, '%2d ', 1 );
            ns      = fscanf( fid, '%2d ', 1 );
            for is  = 1:ns
	       str  = fscanf( fid, '%12s ', 1 );
	       str( find( str == ' ' ) ) = []; 
	       jstates{ is } = str;
	    end
	    fscanf( fid, '\n', 1 );
            cat_states{ j } = jstates;
	 end
      else
         cat_states = {{}};
	 lcatdico   = fscanf( fid, '%2d ', 1 );
	 for is = 1:lcatdico
	    str = fscanf( fid, '%12s ', 1 );
	    str( find( str == ' ' ) ) = []; 
	    cat_dictionnary{ is } = str;
	 end
	 fscanf( fid, '\n', 1 );
      end
   else
      cat_dictionnary = {};
      cat_states      = {{}};
      cn_name         = '';
   end
   nh   = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 1 );
   hist = struct( [] );
   if ( nh > 0 )
      for j = 1:nh
         if ( categorical )
            for i = 1:n
               str = fscanf( fid, '%s  ', 1 );
	       str( find( str == ' ' ) ) = [];
	       if ( str( 1 ) == '+'  || str( 1 ) == '-' )
                  hist(j).x{ i } = sscanf( str, '%e  ', 1 );
	       else
	          hist(j).x{ i } = sscanf( str, '%s  ', 1 );
               end	       
	    end
	 else
            hist( j ).x = fscanf( fid, '%e', n );
	 end
	 hist( j ).fx    = fscanf( fid, '%e', 1 );
	 hist( j ).xincr = fscanf( fid, '%e', n );
	 fscanf( fid, '\n', 1 );
	 if ( categorical && length( cn_name ) )
            hist( j ).context.xtype  = fscanf( fid, '%s ', 1 );
            hist( j ).context.xlower = fscanf( fid, '%e', n );
            hist( j ).context.xupper = fscanf( fid, '%e', n );
         end
      end
   end
   lh = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 3 );

   %  NOTE-TRAINING:
   %  in the following paragraph, 17 is 4 + the total number of trainable parameters.

   if ( lh > 0 )
      training_history = zeros( lh, 17 );
      for j = 1:lh
         training_history( j, 1:17 ) = fscanf( fid, '%e', 17 );
      end
   else
      training_history = [];
   end
   fscanf( fid, '\n', 1 );
   lh = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 2 );
   if ( lh > 0 )
      fstar = zeros( 1, lh );
      fstar( 1:lh )  = fscanf( fid, '%e', lh );
   else
      fstar = [];
   end
   fscanf( fid, '\n', 1 );
   lh = fscanf( fid, '%d', 1 ); name = fscanf( fid, '%s\n', 2 );
   if ( lh > 0 )
      tpval = cell( 1, lh );
      for ip = 1:lh
         lip = fscanf( fid, '%d', 1 );
         tpval{ ip } = zeros( 1, lip );
         tpval{ ip }( 1:lip )  = fscanf( fid, '%e', lip );
         fscanf( fid, '\n', 1 );
      end
   else
      tpval = {};
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
           training_set_cutest, trained_bfo_parameters, training_epsilon, training_maxeval,...
	   training_verbosity, training_problem_epsilon, training_problem_maxeval,         ...
	   training_problem_verbosity, restok ] = bfo_restore_training( filename )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  If training restart is desired, read the training algorithmic parameters and 
%  the environment of the whole training procedure.

%  INPUT :

%  filename   : the name of the file to be read

%  OUTPUT :

%  p            : the current value of the BFO algorithmic parameters
%  verbose      : the verbosity of the top level BFO routine
%  training_strategy : the strategy applied for training
%  training_problems : a cell whose entries specify (in condensed form)
%                 the test problems on which training is performed 
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

%  PROGRAMMING: Ph. Toint, December 2014. (This version 6 I 2018)

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

restok = 0;
fid    = fopen( filename, 'r');
if ( fid == -1 )
   p                          = [];
   verbose                    = 1;
   training_strategy          = '';
   training_parameters        = {};
   trained_bfo_parameters     = '';
   training_problems          = {};
   training_set_cutest        = 0;
   training_epsilon           = NaN;
   training_maxeval           = NaN;
   training_verbosity         = NaN;
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
   p                   = fscanf( fid, '%e', 13 ); % NOTE-TRAINING: 13 is the total number 
                                                  % of trainable parameters!!!
   nbr_train_probs     = fscanf( fid, '%d', 1 );   name = fscanf( fid, '%s\n', 1 );

   training_problems   = cell( nbr_train_probs );
   for j = 1:nbr_train_probs
      if ( training_set_cutest )
         training_problems{ j } = fscanf( fid, '%s', 1 );
      else
         training_problems{ j } = bfo_read_prob( fid );
      end
   end
   if ( training_set_cutest )
      fscanf( fid, '\n', 1 );
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




function bfo_print_vector( indent, name, x, varargin )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Prints a vector after its name and the string indent.

%  INPUT:

%  indent  : a string (typically containing indentation blanks) to be
%            printed at the beginning of the line
%  name    : the name of the vector to be printed
%  x       : the vector to be printed
%  varargin: an optional list of argument.  If present, it must consist of
%            xtype           : the type of the variables
%            cat_dictionnary : the current dictionnary for categorical values
%            x0ref           : the reference vector for default values

%  PROGRAMMING: Ph. Toint, April 2009. (This version 13 IX 2016)

%  DEPENDENCIES: bfo_cellify

%  TEST
%  bfo_print_vector( '   ', 'vec', [1:n] )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length(x);
if ( n )
   if ( ~isempty( varargin ) && ~isnumeric( varargin{ 3 } ) )
      bfo_print_cell( indent, name, bfo_cellify( x, varargin{ 1 }, varargin{ 2 },          ...
                      varargin{ 3 } ) );
   else
      disp( [ indent, ' ', name, ' = '] )
      is = 1;
      for i = 1:ceil( n/10 )
         it = min( is + 9, n );
         fprintf( '%s  %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e',       ...
                  indent, x( is:it ) );
         fprintf( '\n' );
         is = is + 10;
      end
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function bfo_print_index_vector( indent, name, x, varargin )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Prints an integer vector after its name and the string indent.

%  INPUT:

%  indent : a string (typically containing indentation blanks) to be
%           printed at the beginning of the line
%  name   : the name of the vector to be printed
%  x      : the vector to be printed

%  PROGRAMMING: Ph. Toint, September2016. (This version 12 IX 2016)

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length( x );
if ( n  )
   fprintf( '%s', [ indent, name, ' = '] );
   fprintf( '%3d', x( 1:n ) );
   fprintf( '\n');
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

%  PROGRAMMING: Ph. Toint, December 2014. (This version 25 IX 2016)

%  DEPENDENCIES: -

%  TEST
%  c = { 'alpha', 'beta', 'search-type', @a_rather_long_name, ...
%         'an_even_longer_name_than_before' };
%  bfo_print_cell( '   ', 'cell', { 'elt1', 'elt2'} )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length( c );
if ( n )
   disp( [ indent, ' ', name, ' = '] )
   fprintf( '%s    ', indent )
   for i = 1:n
      if ( ischar( c{ i } ) )
         fprintf( '%s  ', c{ i } )
      elseif ( isa( c{ i }, 'function_handle' ) )
         fprintf( '%s  ', func2str( c{ i } ) )
      elseif ( isnumeric( c{ i } ) )
         fprintf( '%+.6e  ', c{ i } )
      end
   end
   fprintf( '\n' );
end

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

%  PROGRAMMING: Ph. Toint, December 2014. (This version 22 VIII 2016)

%  DEPENDENCIES: -

%  TEST
%  for n=1:9
%     bfo_print_matrix( '   ', 'mat', [1:n;1:n] )
%  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[ m, n ] = size(x);

if ( n + m )
   disp( [ indent, ' ', name, ' = '] )
   for i = 1: m
      is = 1;
      for j = 1:ceil( n/10 )
         it = min( is + 9, n );
        fprintf('%s row %3d:  %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e %+.6e',...
                 indent, i, x( i, is:it ) );
         fprintf( '\n' );
         is = is + 10;
      end
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function bfo_print_summary_vector( indent, name, x, varargin  )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Prints (on a line) the first and last few (4) components of a vector after 
%  its name and the string indent.

%  INPUT:

%  indent  : a string (typically containing indentation blanks) to be
%            printed at the beginning of the line
%  name    : the name of the vector to be 'summary-printed'
%  x       : the vector to be 'summary-printed' 
%  varargin: an optional list of argument.  If present, it must consist of
%            xtype           : the type of the variables
%            cat_dictionnary : the current dictionnary for categorical values
%            x0ref           : the reference vector for default values

%  PROGRAMMING: Ph. Toint, April 2009. (This version 11 IX 2016)

%  DEPENDENCIES: bfo_cellify

%  TEST
%  for n=1:9
%     bfo_print_summary_vector( '   ', 'vec', [1:n] )
%  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n  = length( x );
if ( n )
   s1 = [ indent, ' ', name, ' = (' ];
   if ( ~isempty( varargin ) && ~isnumeric( varargin{ 3 } ) )
      cx = bfo_cellify( x, varargin{ 1 }, varargin{ 2 }, varargin{ 3 } );
      fprintf( '%s', s1 )
      if ( n <= 8 )
         fprintf( ' ' )
         for i = 1:min( 8, n )
            if ( ischar( cx{ i } ) )
               fprintf( '%s ', cx{ i } )
            elseif ( isa( cx{ i }, 'function_handle' ) )
               fprintf( '%s ', func2str( cx{ i } ) )
            elseif ( isnumeric( cx{ i } ) )
               fprintf( '%+.6e ', cx{ i } )
            end
         end
      else
         fprintf( ' ' )
         for i = 1:4
            if ( ischar( cx{ i } ) )
               fprintf( '%s  ', cx{ i } )
            elseif ( isa( cx{ i }, 'function_handle' ) )
               fprintf( '%s  ', func2str( cx{ i } ) )
            elseif ( isnumeric( cx{ i } ) )
               fprintf( '%+.6e  ', cx{ i } )
            end
         end	 
         fprintf( '...  ' );
         fprintf( ' ' )
         for i = n-3:n
            if ( ischar( cx{ i } ) )
               fprintf( '%s  ', cx{ i } )
            elseif ( isa( cx{ i }, 'function_handle' ) )
               fprintf( '%s  ', func2str( cx{ i } ) )
            elseif ( isnumeric( cx{ i } ) )
               fprintf( '%+.6e  ', cx{ i } )
            end
         end
      end
   else
      fprintf( '%s', s1 )
      if ( n <= 8 )
         fprintf( '%+.6e ', x( 1:min( 8, n ) ) );
      else
         fprintf( '%+.6e ', x( 1:4 ) );
         fprintf( '...  ' );
         fprintf( '%+.6e ', x( n-3:n ) );
      end
   end 
   fprintf( ')\n' );
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ objf_handle, objf_name, shfname, msg ] = bfo_verify_objf( objf, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies that the objective function is well-defined.

%  INPUT:

%  objf       : a string or function handle identifying the objective function
%  verbose    : the current verbosity level

%  OUTPUT:

%  objf_handle: the handle for the objective function
%  objf_name  : the objective function's full name
%  shfname    : the objective function's short name
%  msg        : an error message, if relevant ('' if no error)
%  wrn        : a warning, if relevant.

%  PROGRAMMING : Ph. Toint, September 2016 (this version 28 XII 2016).

%  DEPENDENCIES: bfo_is_reserved, bfo_exist_function, bfo_verify_objf

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

objf_handle = NaN;
objf_name   = '';
shfname     = '';
msg         = '';

if ( ischar( objf ) )
   [ objfok, shfname ] = bfo_exist_function( objf );
   if ( bfo_is_reserved( shfname ) )
      msg = [ ' BFO error: the name of the objective function starts with ''bfo_''.',      ...
              ' Terminating.' ];
      if ( verbose )
         disp( msg )
      end
      return
   end
   if ( objfok )
      objf_handle = str2func( objf );
      objf_name   = objf;
   else
      msg = [ ' BFO error: m-file for function ', objf, ' not found. Terminating. '];
      if ( verbose )
         disp( msg )
      end
      return
   end
elseif ( isa( objf, 'function_handle' ) )
   objf_name  = func2str( objf );
   if ( bfo_is_reserved( objf_name ) )
      msg = [ ' BFO error: the name of the objective function starts with ''bfo_''.',     ...
              ' Terminating.' ];
      if ( verbose )
         disp( msg )
      end
      return
   end
   [ objfok, shfname ] = bfo_exist_function( objf_name );
   if ( bfo_is_reserved( shfname ) )
      msg = [ ' BFO error: the name of the objective function starts with ''bfo_''.',     ...
              ' Terminating.' ];
      if ( verbose )
         disp( msg )
      end
      return
   end
   if ( objfok )
      objf_handle = objf;
   else
      msg = [ ' BFO error: m-file for function ', func2str( objf ), ' not found.',         ...
              ' Terminating.'];
      if ( verbose )
         disp( msg )
      end
      return
   end
elseif ( iscell( objf ) )
   p = length( objf );
   objf_handle = cell( 1, p );
   for i = 1:p
       [ objf_handle{ i }, objfn, shfn, msg ] = bfo_verify_objf( objf{ i }, verbose );
       if ( ~isempty( msg ) )
          return
       end
       if ( i == 1 )
          if ( p > 1 )
             objf_name = [ 'sum_{i=1:', int2str(p),'}(', objfn ];
	     shfname   = [ 'sum_{i=1:', int2str(p),'}(', shfn  ];
          else
             objf_name = objfn;
	     shfname   = shfn;
          end
       elseif ( i == p && p > 1 )
          objf_name = [ objf_name, '->', objfn, ')' ];
          shfname   = [ shfname,   '->', shfn , ')' ];
       end
   end
else
   msg = ' BFO error: objective function misspecified. Terminating.';
   if ( verbose )
      disp( msg )
   end
   return
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ xlower, wrn ] = bfo_verify_xlower( xlow, n, verbose, myinf )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies the user-supplied lower bounds.

%  INPUT:

%  xlow    : the user-supplied bounds
%  n       : the number of variables
%  verbose : the current verbosity level
%  myinf   : the "infinity value" for bounds

%  OUTPUT:

%  xlower  : the verified lower bound
%  wrn     : a possible warning

%  PROGRAMMING: Ph. Toint, December 2014. (This version 29 III 2018)

%  DEPENDENCIES: -

%  TEST
%  bfo_verify_xlower( [ 1 1 ], 3, 4, 1e+20 )
%  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wrn = '';
if ( isnumeric( xlow ) )
   [ s1, s2 ] = size( xlow );
   if ( s1 == 1 && s2 == 1 )
      xlower  = xlow * ones( n, 1 );
   elseif ( min( s1, s2 ) ~= 1  || max( s1, s2 ) ~= n )
       wrn = ' BFO warning: wrong size of input for parameter xlower. Default used.';
       if ( verbose )
          disp( wrn )
       end
       xlower = -myinf*ones( n, 1 );
   else
      if ( s2 > s1 )      % make sure the lower bound is a column vector 
         xlower = max( [ xlow ; -myinf*ones( 1, n ) ])';      
      else
         xlower = max( [ xlow'; -myinf*ones( 1, n ) ])';     
      end
   end
else
   wrn = ' BFO warning: wrong type of input for parameter xlower. Default used.';
   if ( verbose )
      disp( wrn )
   end
   xlower = -myinf*ones( n, 1 );
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ xupper, wrn ] = bfo_verify_xupper( xupp, n, verbose, myinf )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies the user-supplied upper bounds.

%  INPUT:

%  xupp    : the user-supplied bounds
%  n       : the number of variables
%  verbose : the current verbosity level
%  myinf   : the "infinity value" for bounds

%  OUTPUT:

%  xupper  : the verified upper bounds
%  wrn     : a possible warning

%  PROGRAMMING: Ph. Toint, December 2014. (This version 11 III 2018)

%  DEPENDENCIES: -

%  TEST
%  bfo_verify_xupper( [ 1 1 ], 3, 4, 1e+20 )
%  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wrn = '';
if ( isnumeric( xupp ) )
   [ s1, s2 ] = size( xupp );
   if ( s1 == 1 && s2 == 1 )
      xupper  = xupp * ones( n, 1 );
   elseif ( min( s1, s2 ) ~= 1  || max( s1, s2 ) ~= n )
       wrn = ' BFO warning: wrong size of input for parameter xupper. Default used.';
       if ( verbose )
          disp( wrn )
       end
       xupper = myinf*ones( n, 1 );
   else
      if ( s2 > s1 )      % make sure the lower bound is a column vector 
         xupper = min( [ xupp ; myinf*ones( 1, n ) ] )';      
      else
         xupper = min( [ xupp'; myinf*ones( 1, n ) ] )';     
      end
   end
else
   wrn = ' BFO warning: wrong type of input for parameter xupper. Default used.';
   if ( verbose )
      disp( wrn )
   end
   xupper = myinf*ones( n, 1 );
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ xscale, user_scl, wrn ] = bfo_verify_xscale( xsc, n, solve, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies the user-supplied variable scaling.

%  INPUT:

%  xsc     : the unverified scaling vector
%  n       : the problem's dimension
%  solve   : true if minimization is requested
%  verbose : the current verbosity level

%  OUTPUT:

%  xscale  : the verifies scaling vector
%  user_scl: true if user-scaling can be used
%  wrn     : a possible warning

%  PROGRAMMING: Ph. Toint, September 2016 (this version 6 II 2016)

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wrn      = '';
user_scl = 0;

if ( isnumeric( xsc ) )

   %  If xscale = [], this essentially means that no scaling is necessary.

   if ( isempty( xsc ) )
      xscale = [];
      return
   end

   %  xscale is not empty.

   xtry = abs( xsc );
   [ s1, s2 ] = size( xtry );
   if ( s1 == 1 && s2 == 1 )
      if ( solve )
         user_scl = 2;       % remember that scales are specified by the user
                             % but also to ignore it for discrete variables
         xscale   = xtry * ones( n, 1 );
      end
   elseif ( min( s1, s2 ) ~= 1 )
      wrn = ' BFO warning: wrong size of input for parameter xscale. Default used.';
      if ( verbose )
         disp( wrn )
      end
      xscale = ones( n, 1 );
   elseif ( ( max( s1, s2 ) ~= n ) && solve )
      wrn = ' BFO warning: wrong size of input for parameter xscale. Default used.';
      if ( verbose )
         disp( wrn )
      end
      xscale = ones( n, 1 );
   else
      user_scl = 1;       % remember that scales are specified by the user
      if ( s1 < s2 )      % make sure the scaling is a column vector 
          xscale = xtry';
      else
          xscale = xtry;
      end
   end

else
   wrn = ' BFO warning: wrong type of input for parameter xscale. Default used.';
   if ( verbose )
      disp( wrn )
   end
   xscale = ones( n, 1 );
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ xtype, msg, wrn ] = bfo_verify_xtype( xtyp, n, solve, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies the user-supplied variable's types.

%  INPUT:

%  xupp    : the user-supplied variable types
%  n       : the number of variables
%  verbose : the current verbosity level

%  OUTPUT:

%  xtype   : the verified variable's type
%  msg     : a possible error message
%  wrn     : a possible warning

%  PROGRAMMING: Ph. Toint, December 2014. (This version 10 X 2017)

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xtype = '';
wrn   = '';
msg   = '';

if ( ischar( xtyp ) )
   xtok  = 1;
   for ii = 1:length( xtyp )
      if ( ~ismember( xtyp( ii ),                                                          ...
           { 'c', 'i', 'j', 'f', 's', 'w', 'x', 'y', 'z', 'r', 'd', 'k' } ) )
         xtok = 0;
         break
      end
   end
   if ( xtok )
      if ( solve )  
         if ( length( xtyp ) == 1 )
            xtype( 1:n ) = xtyp;
         elseif ( length( xtyp ) ~= n )
            msg = [ ' BFO error: xtype has length ', int2str( length( xtyp ) ),            ...
                    ' instead of ', int2str( n ),'. Terminating.' ];
            if ( verbose )
               disp( msg )
            end
         else
             xtype = xtyp;
         end
      else
         xtype = xtyp;
      end
   else
      wrn = ' BFO warning: wrong type of input for parameter xtype. Default used.';
      if ( verbose )
         disp( wrn )
      end
      xtype(1:n) = 'c';
   end
else
   wrn = ' BFO warning: wrong type of input for parameter xtype. Default used.';
   if ( verbose )
      disp( wrn )
   end
   xtype(1:n) = 'c';
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [ cat_states, cat_dictionnary, msg ] =                                       ...
               bfo_verify_cat_states( catst, n, xtype, verbose, cat_dictionnary )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies the user-supplied cat_states cell value.  Note that the categorical dictionnary
%  may be updated by thus routine as it is normal for the user to specify values of
%  categorical variables different from those at the starting point.

%  INPUT:

%  catst          : a value cell containing the user-supplied list of possible
%                   values for each variable
%  n              : the problem's dimension
%  xtype          : true if minimization is requested
%  verbose        : the current verbosity level
%  cat_dictionnary: the current categorical dictionnary

%  OUTPUT:

%  cat_states     : the verified list of categorical values
%  msg            : an error message, if relevant ('' if no error)

%  PROGRAMMING: Ph. Toint, August 2016. (This version 14 IX 2016)

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cat_states = {};
msg        = '';

if ( iscell( catst ) )
	 
   %  Check there is something specified for each variable.

   lcatstates = length( catst );
   if ( lcatstates ~= n && lcatstates ~= 0 )
      lcatstates = length( catst{ 1 } );
      if ( lcatstates ~= n && lcatstates ~= 0 )
         msg = ' BFO error: categorical states misspecified. Terminating.';
         if ( verbose )
            disp( msg )
         end
         return
      end
   end

   %  Verify the state(s) specified for each variable: states for categorical
   %  variables are strings or cell arrays of strings, states for discrete and
   %  continuous variables are ''. States for fixed, waiting or frozen variables
   %  are irrelevant.

   if ( lcatstates )
      for ii = 1:n
 
         cii   = catst{ ii };
	 errii = 0;
	
	 switch ( xtype( ii ) )
	
	 case { 'c', 'i' }
	
            errii = length( cii );
	   
	 case { 's' }

            lcii = length( cii );
	    if ( iscell( cii ) )
	       for iii = 1: lcii
	          if ( ischar( cii{iii} ) )
	             cat_dictionnary = union( cat_dictionnary, cii{iii} );
	          else
		     errii = 1;
	          end
	       end
	    elseif ( ischar( cii ) && lcii )
	       cat_dictionnary = union( cat_dictionnary, cii );
	    else
	       errii= 1;
	    end
	 otherwise
 	 end
	
	 if ( errii )
            msg = [ ' BFO error: misspecified categorical state for variable ',            ...
	             int2str( ii ), ' of type ', xtype(ii), '. Terminating.'];
            if ( verbose )
               disp( msg )
            end
            return
	 end
      end
      cat_states = catst;
   end
else
   msg = ' BFO error: categorical states misspecified. Terminating.';
   if ( verbose )
      disp( msg )
   end
   return
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ cn_name, msg ] = bfo_verify_cat_neighbours( catn, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies the user-supplied dynamical categorical neighbours routine.

%  INPUT:

%  catn   : the user-supplied idenifier for the categorical neighbourhood function
%  verbose: the current verbosity level


%  OUTPUT:

%  cn_name: the verified string containing the name of the dynamical neighbourhood function
%  msg    : an error message if relevant ('' if no error)


%  PROGRAMMING: Ph. Toint, August 2016 (This version 21 IX 2016).

%  DEPENDENCIES: bfo_exist_function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

msg = '';
if ( ischar( catn ) )
   cn_name = catn;
   if ( isempty( cn_name ) )
      return
   end
   if ( length( cn_name ) > 3 && strcmp( cn_name(1:4), 'bfo_' ) )
      msg = [ ' BFO error: the name of the cat-neighbours function starts with ''bfo_''.', ...
              ' Terminating' ];
      if ( verbose )
         disp( msg )
      end
      return
   end
   if ( ~bfo_exist_function( cn_name ) )
      msg = [ ' BFO error: m-file for cat-neighbours function ', cn_name, ' not found.',   ...
              ' Terminating. '];
      if ( verbose )
         disp( msg )
      end
      return
   end
elseif ( isa( catn, 'function_handle' ) )
   cn_name = func2str(  catn );
   if ( length( cn_name ) > 3 && strcmp( cn_name(1:4), 'bfo_' ) )
      msg = [ ' BFO error: the name of the cat-neighbours function starts with ''bfo_''.', ...
              ' Terminating' ];
      if ( verbose )
         disp( msg )
      end
      return
   end
   if ( ~bfo_exist_function( cn_name ) )
      msg = [ ' BFO error: m-file for variable bounds function ', cn_name, ' not found.',  ...
              ' Terminating. '];
      if ( verbose )
         disp( msg )
      end
      return
   end
else
   msg = ' BFO warning: wrong type of input for cat-neighbours. Terminating.';
   if ( verbose )
      disp( msg )
   end
   return
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ latbasis, msg ] = bfo_verify_lattice_basis( latb, verbose, level )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies the user-supplied lattice basis.

%  INPUT:

%  latb    : the user-supplied unverified lattice basis
%  verbose : the current verbosity level
%  level   : the current level

%  OUTPUT:

%  latbasis: the verified lattice basis
%  msg     : an error message if relevant ('' if no error)

%  PROGRAMMING: Ph. Toint, August 2016 (This version 14 IX 2016).

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

latbasis = [];
msg      = '';
if ( isnumeric( latb ) )
   latbasis  = latb;
else
   msg = ' BFO error: wrong type of input for parameter lattice-basis. Terminating.';
   if ( verbose )
      disp( msg )
   end
   return
end

%  Check that the lattice basis matrix is square.

[ nrlatb, nclatb ] = size( latbasis );
if ( nrlatb ~= nclatb )
   msg = ' BFO error: wrong dimension for the lattice-basis. Terminating.';
   if ( verbose )
      disp( msg )
   end
   return
end

%  Check that the lattice basis is sufficiently linearly independent.

if ( level == 1 )
   if ( min ( abs( eig( latbasis ) ) ) < 10^(-12) )
      msg = ' BFO error: lattice-basis is not numerically linearly independent. Terminating.';
      if ( verbose )
         disp( msg )
      end
      return
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ vb_name, msg, wrn ] = bfo_verify_variable_bounds( vbn, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies the user-supplied function specifying variable bounds for multilevel problems.

%  INPUT:

%  vbn    : the user-supplied idenifier for the variable bounds function
%  verbose: the current verbosity level

%  OUTPUT:

%  vb_name: the verified string containing the name of the variable bounds function
%  msg    : an error message if relevant ('' if no error)
%  wrn    : a possible warning

%  PROGRAMMING: Ph. Toint, August 2016 (This version 21 IX 2016).

%  DEPENDENCIES: bfo_exist_function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wrn = {};
msg = '';
if ( isempty( vbn ) )
   vb_name = '';
   return
end
if ( ischar( vbn ) )
   vb_name = vbn;
   if ( length( vb_name ) > 3 && strcmp( vb_name( 1:4 ), 'bfo_' ) )
      msg = [' BFO error: the name of the variable-bounds function starts with ''bfo_''.', ...
              ' Terminating.' ];
      if ( verbose )
         disp( msg )
      end
      return
   end
   if ( bfo_exist_function( vb_name ) )
      variable_bounds = str2func( vb_name );
   else
      msg = [ ' BFO error: m-file for variable bounds function ', vb_name, ' not found.',  ...
              ' Terminating. '];
      if ( verbose )
         disp( msg )
      end
      variable_bounds = '';
      return
   end
elseif ( isa( vbn, 'function_handle' ) )
   %vb_name = func2str(  varargin{ i+1 } );
   vb_name = func2str(  vbn ); %marghemod
   if ( length( vb_name ) > 3 && strcmp( vb_name( 1:4 ), 'bfo_' ) )
      msg = [' BFO error: the name of the variable-bounds function starts with ''bfo_''.', ...
              ' Terminating.' ];
      if ( verbose )
         disp( msg )
      end
      return
   end
   if ( bfo_exist_function( vb_name ) )
      variable_bounds = vbn;
   else
      msg = [ ' BFO error: m-file for variable bounds function ', vb_name,                 ...
             ' not found. Terminating. '];
      if ( verbose )
          disp( msg )
      end
      variable_bounds = '';
      return
   end
else
   wrn{ end+1 }=' BFO warning: wrong type of input for variable-bounds. Default bounds used.';
   if ( verbose )
      disp( wrn{ end } )
   end
   variable_bounds = '';
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ xlevel, msg, wrn ] = bfo_verify_xlevel( xlv, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies the user-supplied level description.

%  INPUT:

%  xlv     : the user-supplied level description
%  verbose : the current verbosity level

%  OUTPUT:

%  xlevel  : the verified level's description
%  msg     : an error message if relevant ('' if no error)
%  wrn     : a possible warning

%  PROGRAMMING: Ph. Toint, December 2014. (This version 14 IX 2016)

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

msg = '';
wrn = {};

if ( isempty( xlv ) )
   xlevel = [];
   return
end
if ( isnumeric( xlv ) )
   if ( length( xlv ) < 2 )
      msg = ' BFO error: to few variables for multilevel. Terminating.';
      if ( verbose )
          disp( msg )
      end
      return
   end
   xlevel = abs( round( xlv ) );
   if ( min( xlevel ) ~= 1 )
      msg = ' BFO error: lowest level different from 1. Terminating.';
      if ( verbose )
          disp( msg )
      end
      return
   end
else
   wrn{ end+1 } = ' BFO warning:  wrong input for xlevel parameter. Ignored.';
   if ( verbose )
      disp( wrn{ end } )
   end
   xlevel = [];
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ max_or_min, maximize, wrn ] =                                                   ...
                         bfo_verify_maxmin( maxmin, verbose, old_maxmin )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies the user-supplied request for maximization/minimization.  It also compares
%  it to its value at a previous call (for restarting purposes).


%  INPUT:

%  maxmin    : the user-supplied max_or_min request
%  verbose   : the current verbosity level
%  old_maxmin: the value of the max_or_min variable at a previous call

%  OUTPUT:

%  max_or_min: the verified max_or_min request
%  maximize  : true if the problem is a maximization
%  wrn       : a possible warning

%  PROGRAMMING: Ph. Toint, December 2014. (This version 14 IX 2016)

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wrn = '';
if ( ischar( maxmin ) )
   if ( ~strcmp( '', old_maxmin ) && ~strcmp( maxmin, old_maxmin ) ) 
      wrn = ' BFO warning: inconsistent max-or-min at restart. Using saved value.';
      if ( verbose )
         disp( wrn{ end } )
      end
      maxmin = old_maxmin;
   end
   if ( size( maxmin, 2 ) ~= 3 )
      wrn = [ ' BFO warning: badly specified choice of minimization or',                   ...
              ' maximization. Default (min-max) used.' ];
      if ( verbose )
         disp( wrn )
      end
      max_or_min = 'min';
      maximize   = 0;
      return
   end
   max_or_min = maxmin;
   if ( strcmp( maxmin( 1,: ), 'max' ) )
      maximize = 1;
   elseif ( strcmp( maxmin( 1,: ), 'min' ) )
      maximize = 0;
   else
      wrn = [ ' BFO warning:  unknown choice of minimization or maximization.',            ...
                       ' Default (min) used.' ];
      if ( verbose )
         disp( wrn{ end } )
      end
      max_or_min = 'min';
      maximize   = 0;
   end
else
   wrn=[ ' BFO error:  unknown choice of minimization or maximization.' Default (min) used.'];
   if ( verbose )
      disp( wrn{ end } ) 
   end
   max_or_min = 'min';
   maximize   = 0;
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ x0, n, msg ] = bfo_verify_x0( x0, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies the user-supplied starting point for optimization. The function returns
%  the verified starting point and the problem's (total) dimension.

%  INPUT:

%  x0     : the user-supplied unverified starting point
%  verbose: the current verbosity level

%  OUTPUT:

%  x0     : the verified starting point
%  n      : the problem's (total) dimension
%  msg    : an error message if relevant ('' if no error)

%  PROGRAMMING: Ph. Toint, August 2016 (This version 28 XII 2016).

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

msg = '';
if ( isnumeric( x0 ) )
   [ nl, nc ] = size( x0 );
   if ( min( nl, nc ) ~= 1 )
      msg = ' BFO error: second argument is not a valid starting point. Terminating.';
      if ( verbose )
         disp( msg )
      end
      n = 0;
   else
      n = max( nl, nc );                              % the dimension of the space
      if ( nl == 1 && nc > 1 )                        % make sure the starting point is 
           x0 = x0';                                  % a column vector
      end
   end
elseif ( iscell( x0 ) ) 
   n   = length( x0 );                                % the dimension of the space
else
   msg = ' BFO error: second argument is not a valid starting point. Terminating.';
   if ( verbose )
      disp( msg )
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ eldom, msg, not_in_eldom ] = bfo_verify_eldom( eldom, n, p, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verify the plausibility of the domain information for coordinate partially separable
%  problems.  Possibly return a list of unseen variables.

%  INPUT:

%  eldom  : a cell array containing the domain defining vectors
%  n      : the number of variables
%  p      : the number of element functions
%  verbose: the current verbosity level

%  OUTPUT:

%  eldom  : the verified eldom
%  msg    : a possible error message. '' if all is fine.
%  not_in_eldom : the list of variables in [1:n] not occurring in eldom.

%  PROGRAMMING : Ph. Toint, November 2016 (This version 19 I 2018).

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

not_in_eldom = [];
msg          = '';

if ( ~iscell( eldom ) )
   msg = ' BFO error: wrong type of input for parameter eldom.  Terminating.';
   if ( verbose )
      disp( msg )
   end
   return 
end
leldom = length( eldom );
if ( leldom ~= p )
   msg = [ ' BFO error: eldom has a length ', int2str( leldom ),                           ...
           ' not equal to the number of element functions ', int2str( p ),'.  Terminating.' ];  
   if ( verbose )
      disp( msg )
   end
   return 
end
unseen = ones( n, 1 );
for iel = 1:leldom
   if ( ~isnumeric( eldom{ iel } ) )
      msg =[ ' BFO error: wrong type of input for eldom{', int2str( iel ), '}. Terminating.'];
      if ( verbose )
         disp( msg )
      end
      return 
   end
   leldiel = length( eldom{ iel } );
   if ( leldiel < 1  || leldiel > n )
      msg = [ ' BFO error: eldom{', int2str( iel ), '} has a wrong length. Terminating.' ];
      if ( verbose )
         disp( msg )
      end
      return 
   end
   for j = 1:leldiel
      iv = eldom{ iel }( j );
      if (  ~ismember( iv, [1:n] ) )
         msg = [ ' BFO error: impossible specification for eldom{', int2str( iel ),        ...
                 '}. Terminating.'];
         if ( verbose )
            disp( msg )
         end
         return 
      end
      unseen( iv ) = 0;
   end
end

if ( sum( unseen ) )
   not_in_eldom = find( unseen > 0 );
end
clear unseen

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ prob, n, shfname, user_scl, maximize, multilevel,                               ...
           cat_dictionnary, msg, wrn, not_in_eldom ] =                                     ...
         bfo_verify_prob( prob, iprob, verbose, myinf, old_maxmin, cat_dictionnary )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies a user-supplied optimization problem in condensed form. The routine returns
%  the verified prob struct as well as all global indicators derived from its content but
%  not included in it.


%  INPUT:

%  prob           : the unverified user supplied prob struct (hopefully)
%  iprob          : 0 for the objective function, i for the i-th training problem
%  verbose        : the current verbosity level
%  myinf          : the value of infinity for lower and upper bounds
%  old_maxmin     : the value of max_or_min at previous call (used on restart)
%  cat_dictionnary: the current categorical dictionnary

%  OUTPUT:

%  prob           : the verified prob struc
%  n              : the problem's (total) dimension
%  shfname        : the objective function's short name
%  user_scl       : true if user scaling can be used
%  maximize       : true if maximization is requested
%  multilevel     : true if the problem is multilevel
%  cat_dictionnary: the updated categorical dictionnary
%  msg            : an error message, if relevant ('' if no error)
%  wrn            : a possible warning
%  not_in_eldom   : the indeces of the variables unseen in eldom

%  PROGRAMMING    : Ph. Toint, August 2016 (this version 1 XI 2016)

%  DEPENDENCIES   : bfo_verify_objf, bfo_verify_x0, bfo_verify_xlower,
%                   bfo_verify_xupper, bfo_verify_xtype, bfo_verify_xscale,
%                   bfo_verify_maxmin, bfo_verify_xlevel, bfo_verify_variable_bounds,
%                   bfo_verify_cat_states, bfo_verify_cat_neighbours, bfo_verify_eldom

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wrn         = {};
msg         = '';
objf_handle = '';
objf_name    = '';
shfname      = '';
maximize     = 0;
user_scl     = 0;
multilevel   = 0;
n            = 0;
not_in_eldom = [];

if ( verbose >= 4 )
   fprintf( ' Verifying problem %d: ', iprob )
end

if ( isfield( prob, 'objf' ) )
   if ( verbose >= 4 )
      fprintf( '    objf' )
   end
   [ prob.objf, objf_name, shfname, msg ] = bfo_verify_objf( prob.objf, verbose );
   if ( ~isempty( msg ) )
      return
   end
else
   msg = [ ' BFO error: no objective function specified for problem ', int2str( iprob ), ...
           ' Terminating. '];
   if ( verbose )
      disp( msg )
   end
   return
end

if ( isfield( prob, 'x0' ) )
   if ( verbose >= 4 )
      fprintf( ', x0' )
   end
   [ prob.x0, n, msg ] = bfo_verify_x0( prob.x0, verbose );
   if ( ~isempty( msg ) )
      return
   end
else
   msg = [ ' BFO error: no starting point specified for problem ', int2str( iprob ),     ...
           ' Terminating. '];
   if ( verbose )
      disp( msg )
   end
   return
end

if ( isfield( prob, 'eldom' ) )
   if ( verbose >= 4 )
      fprintf( ', eldom' )
   end
   [ eldom, msg, not_in_eldom ] = bfo_verify_eldom( prob.eldom, n, length( prob.objf ),    ...
                                                    verbose );
   if ( ~isempty( msg ) )
      return
   end
end

if ( isfield( prob, 'xlower' ) )
   if ( verbose >= 4 )
      fprintf( ', xlower' )
   end
   [ prob.xlower, wrnf ] = bfo_verify_xlower( prob.xlower, n, verbose, myinf );
   if ( ~isempty( wrnf ) )
      wrn{ end+1 } = wrnf;
   end
end
   
if ( isfield( prob, 'xupper' ) )
   if ( verbose >= 4 )
      fprintf( ', xupper' )
   end
   [ prob.xupper, wrnf ] = bfo_verify_xupper( prob.xupper, n, verbose, myinf );
   if ( ~isempty( wrnf ) )
      wrn{ end+1 } = wrnf;
   end
end
   
if ( isfield( prob, 'xtype' ) )
   if ( verbose >= 4 )
     fprintf( ', xtype' )
   end
   [ prob.xtype, msgf, wrnf ] = bfo_verify_xtype( prob.xtype, n, 1, verbose );
   if ( ~isempty( msgf ) )
      msg = msgf;
      return
   end
   if ( ~isempty( wrnf ) )
      wrn{ end+1 } = wrnf;
   end
end
   
if ( isfield( prob, 'xscale' ) )
   if ( verbose >= 4 )
      fprintf( ', xscale' )
   end
   [ prob.xscale, user_scl, wrnf ] = bfo_verify_xscale( prob.xscale, n, 1, verbose );
   if ( ~isempty( wrnf ) )
      wrn{ end+1 } = wrnf;
   end
end

if ( isfield( prob, 'max_or_min' ) )
   if ( verbose >= 4 )
      fprintf( ', max_or_min' )
   end
   [ prob.max_or_min, maximize, wrnf ] =                                                   ...
                   bfo_verify_maxmin( prob.max_or_min, verbose, old_maxmin );
   if ( ~isempty( wrnf ) )
      wrn{ end+1 } = wrnf;
   end
end

if ( isfield( prob, 'xlevel' ) )
   if ( verbose >= 4 )
      fprintf( ', xlevel' )
   end
   [ prob.xlevel, msg, wrnt ] = bfo_verify_xlevel( prob.xlevel, verbose );
   if ( ~isempty( msg ) )
      return
   end
   if ( ~isempty( wrnt ) )
      wrn{ end+1 } = wrnt;
   else
      multilevel = 1;
   end
end

if ( isfield( prob, 'variable_bounds' ) )
   if ( verbose >= 4 )
      fprintf( ', variable_bounds' )
   end
   [ vb_name, msg, wrnf ] = bfo_verify_variable_bounds( vbn, verbose );
   if ( ~isempty( wrnf ) )
      wrn{ end+1 } = wrnf;
   end
   if ( ~isempty( msg ) )
      return
   end
end

if ( isfield( prob, 'cat_states' ) )
   if ( verbose >= 4 )
      fprintf( ', cat_states' )
   end
   [ prob.cat_states, cat_dictionnary, msg ] =                                             ...
       bfo_verify_cat_states( prob.cat_states, n, prob.xtype, verbose, cat_dictionnary );
   if ( ~isempty( msg ) )
      return
   end
end

if ( isfield( prob, 'cat_neighbours' ) )
   if ( verbose >= 10 )
      fprintf( ', cat_neighbours' )
   end
   [ prob.cat_neighbours, msg ] = bfo_verify_cat_neighbours( prob.cat_neighbours, verbose );
   if ( ~isempty( msg ) )
      return
   end
end
   if ( verbose >= 4 )
      fprintf( '\n' )
   end
   
return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function bfo_print_prob( fid, prob )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Prints an optimization problem in condensed form to a file.

%  INPUT:

%  fid : the file id
%  prob: the problem to print

%  PROGRAMMING: Ph. Toint, August 2016 (this version 13 IX 2016)

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lprob = length( fieldnames(prob) );
if ( ischar( prob.objf ) )
   fprintf( fid, ' %18s ', prob.objf );
else 
   fprintf( fid, ' %18s ', func2str( prob.objf ) );
end
n = length( prob.x0 );
fprintf( fid, '%4d ', n );
fprintf( fid, '%+.12e ', prob.x0( 1:n ) );
fprintf( fid, '%2d ', lprob-2 );
if ( isfield( prob, 'xlower' ) )
   fprintf( fid, '1 ' );
   fprintf( fid, '%+.12e ', prob.xlower( 1:n ) );
end
if ( isfield( prob, 'xupper' ) )
   fprintf( fid, '2 ' );
   fprintf( fid, '%+.12e ', prob.xupper( 1:n ) );
end
if ( isfield( prob, 'xscale' ) )
   fprintf( fid, '3 ' );
   fprintf( fid, '%+.12e ', prob.xscale( 1:n ) );
end
if ( isfield( prob, 'xtype' ) )
   fprintf( fid, '4 ' );
   fprintf( fid, '%1s', prob.xtype( 1:n ) );
   fprintf( fid, ' ' );
end
if ( isfield( prob, 'max_or_min' ) )
   lmaxmin = size( prob.max_or_min, 1 );
   fprintf( fid, '5 %2d ', lmaxmin );
   for j = 1:lmaxmin
      fprintf( fid, '%3s ', prob.max_or_min( j,1:3 ) );
   end
end
if ( isfield( prob, 'xlevel' ) )
   fprintf( fid, '6 ' );
   fprintf( fid, '%2d ', prob.xlevel( 1:n ) );
end
if ( isfield( prob, 'variable_bounds' ) )
   fprintf( fid, '7 ' );
   fprintf( fid, '%18s ', prob.variable_bounds );
end
if ( isfield( prob, 'lattice_basis' ) )
   fprintf( fid, '8 ' );
   for j = 1:n
       fprintf( fid, '%+.12e ', prob.lattice_basis( j, 1:n ) );
   end
end
if ( isfield( prob, 'cat_states' ) )
   fprintf( fid, '9 ' );
   for j = 1:n
      lc = length( prob.cat_states{ j } );
      fprintf( fid, '%2d ', lc );
      if ( lc > 0 )
         fprintf( fid, '%18s ', prob.cat_states{ j }{ 1:lc } );
      end
   end
end
if ( isfield( prob, 'cat_neighbours' ) )
   fprintf( fid, '10 ' );
   fprintf( fid, '%18s ', prob.cat_neighbours );
end
fprintf( fid, '\n' );

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function prob = bfo_read_prob( fid )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Reads an optimization problem in condensed form from a file.


%  INPUT:

%  fid : the file id

%  OUTPUT:

%  prob: the read problem

%  PROGRAMMING: Ph. Toint, August 2016 (this version 13 IX 2016)

%  DEPENDENCIES: ~

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

objf    = str2func( fscanf( fid, '%s', 1 ) );
prob    = struct( 'objf', objf );
n       = fscanf( fid, '%d', 1 );
prob.x0 = fscanf( fid, '%e', n );
nof     = fscanf( fid, '%d', 1 );
for jf  = 1:nof
   id = fscanf( fid, '%d', 1 );
   switch( id )
   case 1
      prob.xlower = fscanf( fid, '%e', n );
   case 2
      prob.xupper = fscanf( fid, '%e', n );
   case 3
      prob.xscale = fscanf( fid, '%e', n );
   case 4
      prob.xtype  = fscanf( fid, '%s', 1 );
   case 5
      lmm = fscanf( fid, '%d', 1 );
      prob.max_or_min = [];
      for j = 1:lmm
         mm = fscanf( fid, '%s', 1 );
 	 prob.max_or_min = [ prob.max_or_min; mm ];
      end
   case 6
      prob.xlevel = fscanf( fid, '%d', n );
   case 7
      prob.variable_bounds = fscanf( fid, '%s', 1 );
   case 8
      prob.lattice_basis = zeros( n, n );
      for j = 1:n
         prob.lattice_basis( j, 1:n ) = fscanf( fid, '%e', n );
      end
   case 9
      prob_cat_states = cell( n );
      for i = 1:n
         lci = fscanf( fid, '%d', 1 );
	 prob_cat_states{ i } = cell( lci );
	 if ( lci > 0 )
	    for j = 1:lci
               prob_cat_states{ i }{ j } = fscanf( fid, '%s', 1 );
	    end
	 else
	    prob.cat_states{ i } = '';
	 end
      end
   case 10
      prob.cat_neighbours = fscanf( fid, '%s', 1 );
   end  
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function bfo_print_x( indent, name, x, xscale, verbose, varargin )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Prints a vector of variables, taking into account the possible non-trivial scale 
%  of continuous variables and the verbosity index.


%  INPUT:

%  indent   : a string (typically containing indentation blanks) to be
%             printed at the beginning of the line
%  name     : the name of the vector to be 'summary-printed'
%  x        : the vector to be 'summary-printed' 
%  xscale   : the transformation from scaled to unscaled continuous variables
%  verbose  : the current verbosity index
%  varargin : an optional argument list.  If present, it must consist of
%             xtype           : the variable's types
%             cat_dictionnary : the current categorical dictionnary
%             x0ref           : the reference vector state for default values
%             This allows categorical variables to appear with their string value.

%  PROGRAMMING: Ph. Toint, February 2015. (This version 16 IX 2016)

%  DEPENDENCIES: bfo_print_summary_vector, bfo_print_vector

%  TEST
%  bfo_print_x( '   ', 'vec', [1:n], 1, [1:n], 5 )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

iscaled = ~isempty( xscale );
if ( verbose == 3 )
   if ( ~isempty( varargin ) && ~isnumeric( varargin{ 3 } ) )
      if ( iscaled )
         bfo_print_summary_vector( indent, name, xscale.*x, varargin{ 1 }, varargin{ 2 },...
	                           varargin{ 3 } )
      else
         bfo_print_summary_vector( indent, name, x, varargin{ 1 }, varargin{ 2 },          ...
	                           varargin{ 3 } )
      end
   else
      if ( iscaled )
         bfo_print_summary_vector( indent, name, xscale.*x )
      else
         bfo_print_summary_vector( indent, name, x )
      end
   end
elseif ( verbose > 3 )
   if ( ~isempty( varargin ) )
      if ( iscaled )
         bfo_print_vector( indent, name, xscale.*x, varargin{ 1 }, varargin{ 2 },        ...
	                   varargin{ 3 } )
         if ( verbose >= 10 )
            bfo_print_vector( indent, [ 'scaled ', name ], x, varargin{ 1 }, varargin{ 2 },...
	                      varargin{ 3 } )
         end
      else
         bfo_print_vector( indent, name, x, varargin{ 1 }, varargin{ 2 }, varargin{ 3 } )
      end
   else
      if ( iscaled )
         bfo_print_vector( indent, name, xscale.*x )
         if ( verbose >= 10 )
            bfo_print_vector( indent, [ 'scaled ', name ], x )
         end
      else
         bfo_print_vector( indent, name, x )
      end
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function bfo_is_reserved = bfo_is_reserved( str )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies that the string str does not match any other BFO keyword.
%
%
%  INPUT:
%
%  str         : a string whose 'non-reserved' character must be verified
%
%
%  OUTPUT
%
%  is_reserved : true if the string str coincides with a BFO-reserved string.
%
%
%  PROGRAMMING: Ph. Toint, February 2015. (This version 1 XI 2016)
%
%  DEPENDENCIES: -
%
%  TEST
%  bfo_is_reserved( 'xupper' )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bfo_is_reserved = ( length( str ) > 3  && strcmp( str( 1:4 ), 'bfo_' )  ) &&               ...
                    ~ismember( str, { 'bfo_average_perf', 'bfo_robust_perf',               ...
                                      'bfo_dpprofile_perf' } );

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
         actl( end+1 ) = i;
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
         actu( end+1 ) = i;
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
if ( ~isempty( actl ) )
   xfeas( actl ) = xlower( actl );
end
if ( ~isempty( actu ) )
   xfeas( actl ) = xupper( actl );
end

alpha = norm( xfeas - x );

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ icont, idisc, icate, ifixd, ixed, iyed, ized, iwait, ired, ided, iked,          ...
           ncont, ndisc, ncate, nfixd, nxed, nyed, nzed, nwait, nred, nded, nked,          ...
	   sacc, Q ] = bfo_switch_context( xtype, npoll, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Updates various quantities derived from the variable's types, after a change in
%  optimization context.

%  INPUT :

%  xtype  : the variable's types
%  verbose: the current verbosity level

%  OUTPUT:

%  icont       : the indices of the continuous variables
%  idisc       : the indices of the integer variables
%  icate       : the indices of the categorical variables
%  ifixd       : the indices of the fixed variables
%  ixed        : the indices of the frozen continuous variables
%  iyed        : the indices of the frozen discrete variables
%  ized        : the indices of the frozen categorical variables
%  iwait       : the indices of the waiting variables
%  ired        : the indices of the dcat-eactivated continuous variables
%  ided        : the indices of the cat-deactivated integer variables
%  iked        : the indices of the cat-deactivated categorical variables
%  ncont       : the number of continuous variables
%  ndisc       : the number of integer variables
%  ncate       : the number of categorical variables
%  nfixd       : the number of fixed variables
%  nxed        : the number of frozen continuous variables
%  nyed        : the number of frozen discrete variables
%  nzed        : the number of frozen categorical variables
%  nwait       : the number of waiting variables
%  nred        : the number of cat-deactivated continuous variables
%  nded        : the number of cat-deactivated integer variables
%  nked        : the number of cat-deactivated categorical variables
%  sacc        : the accumulated step in continuous variables
%  Q           : a new orthonormal basis of dimension icont
%  npoll       : the maximum number of othonormal polling directions

%  PROGRAMMING: Ph. Toint, September 2016 (This version 18 IX 2016).

%  DEPENDENCIES: bfo_print_index_vector

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

icont   = find( xtype == 'c' );
idisc   = find( xtype == 'i' );
icate   = find( xtype == 's' );
ifixd   = find( xtype == 'f' );
ixed    = find( xtype == 'x' );
iyed    = find( xtype == 'y' );
ized    = find( xtype == 'z' );
iwait   = find( xtype == 'w' );
ired    = find( xtype == 'r' );
ided    = find( xtype == 'd');
iked    = find( xtype == 'k');
ncont   = length( icont );
ndisc   = length( idisc );
ncate   = length( icate );
nxed    = length( ixed );
nyed    = length( iyed );
nzed    = length( ized );
nwait   = length( iwait );
nfixd   = length( ifixd );
nred    = length( ired  );
nded    = length( ided  );
nked    = length( iked  );
sacc    = [];
if ( ncont > 0 )
   Q    = eye( ncont, min( ncont, npoll ) );
else
   Q    = 1;
end

if ( verbose >= 4 )
   disp( ' New optimization context: ' )
   bfo_print_index_vector( ' ', 'icont', icont );
   bfo_print_index_vector( ' ', 'idisc', idisc );
   bfo_print_index_vector( ' ', 'icate', icate );
   bfo_print_index_vector( ' ', 'ifixd', ifixd );
   bfo_print_index_vector( ' ', 'ixed' , ixed  );
   bfo_print_index_vector( ' ', 'iyed' , iyed  );
   bfo_print_index_vector( ' ', 'ized' , ized  );
   bfo_print_index_vector( ' ', 'iwait', iwait );
   bfo_print_index_vector( ' ', 'ired ', ired  );
   bfo_print_index_vector( ' ', 'ided ', ided  );
   bfo_print_index_vector( ' ', 'iked ', iked  );
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [ f_hist, x_hist, neval, el_hist, ev_hist ] =                                     ...
                   bfo_ehistupd( f_hist, x_hist, l_hist, neval, fx, x, categorical, varargin )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Maintains the global evaluation history as well as the elementwise history
%  for problems in sum form. This is called after an evaluation at x (yielding fx).
%  The variable argument list is specified (with all 4 arguments) when a sum-form
%  problem is considered.


%  INPUT:

%  f_hist     : the evaluation history for the complete objective function
%  x_hist     : the corresponding evaluation points
%  ev_hist    : the corresponding evaluation effort (for sum-form objectives)
%  l_hist     : the length of the maintained history (the last l_hist evaluations are
%               remembered)
%  neval      : the number of complete function evaluations so far. If the problem is in
%               sum form, neval is a vector of length equal to f_hist, and whose i-th entry
%               gives the number of complete function evaluations for obtaining the function
%               value given by f_hist(i)
%  fx         : the complete function value at x
%  x          : the current point x
%  categorical: true if categorical variables are present
%  el_hist    : varargin{ 1 }: a cell containing the elementwise structures, themselves
%                              containing the element evaluation history
%  fi         : varargin{ 2 }: the vector of element function values at x
%  eldom      : varargin{ 3 }: the indices of the variables defining the element domains
%                              ({} if all elements involve all variables)
%  nevali     : varargin{ 4 }: the number of element functions evaluated

%  OUTPUT
 
%  f_hist     : the updated f_hist
%  x_hist     : the updated x_hist
%  neval      : the updated neval
%  el_hist    : the updated el_hist
%  ev_hist    : the updated ev_hist

%  PROGRAMMING: Ph. Toint October 2016 (This version 1 XI 2016).

%  DEPENDENCIES: ~

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Determine if the function is called for a sum-form problem from the number 
%  of arguments and interpret these argments if this is the case.

sum_form = ( nargin > 7 );
if ( sum_form )
   el_hist          = varargin{ 1 };
   fi               = varargin{ 2 };
   eldom            = varargin{ 3 };
   nevali           = varargin{ 4 };
   ev_hist          = varargin{ 5 };
   n_elements       = length( fi );
   explicit_domains = length( eldom );
end

%  Maintain the global evaluation history.
%  1) maintain the list of the last l_hist point where the objective function was evaluated.

if ( categorical )
   lxh = length( x_hist );
   n   = length( x );
   if ( lxh < l_hist )
      x_hist{ lxh + 1 } = x;
   else
      x_hist = { x_hist{ 2:l_hist } x };
   end
else
   [ n, lxh ] = size( x_hist );
   n = max( n, length( x ) );
   if ( lxh < l_hist )
      x_hist( 1:n, lxh + 1 )  = x;
   else
      x_hist( 1:n, 1:l_hist ) = [ x_hist(1:n,2:l_hist) x ];
   end
end

%  2) maintain the complete list of evaluated objective function values as well
%     as the computational effort needed 

f_hist( end+1 ) = fx;
if ( sum_form )
   neval   = neval + nevali / n_elements;
   ev_hist( end+1 ) = neval;
else
   neval   = neval + 1;
   ev_hist = [];
end

%  Maintain the element-wise evaluation history for the last l_hist evaluation points.

if ( sum_form )
   el_hist          = varargin{ 1 };
   fi               = varargin{ 2 };
   eldom            = varargin{ 3 };
   ev_hist          = varargin{ 5 };
   n_elements       = length( fi );
   explicit_domains = length( eldom );
   for iel = 1:nevali
      histiel = el_hist{ iel };                                % history for elt iel
      lxelh   = length( histiel.xel );
      if ( categorical )
	 if ( explicit_domains )                               % explicit domains
            nel = length( eldom{ iel } );
            if ( lxelh < l_hist  || l_hist == 1 )              % in the beginning
               el_hist{ iel }.fel( lxelh+1 ) = fi( iel );
               el_hist{ iel }.xel{ lxelh+1}  = {{ x{ 1 }{ eldom{ iel } } }};
            else                                               % after l_hist evals
	       el_hist{ iel }.fel( 1:l_hist ) = [ histiel.fel( 2:l_hist ) fi(iel) ];
               el_hist{ iel }.xel = [ histiel.xel(1:nel,2:l_hist)  x( eldom{ iel } ) ];
            end
	 else                                                  % implicit domains
            if ( lxelh < l_hist  || l_hist == 1 )              % in the beginning
	       el_hist{ iel }.fel( lxelh+1 )       = fi( iel );
               el_hist{ iel }.xel( 1:n, lxelh+1 )  = x;
            else                                               % after l_hist evals
	       el_hist{ iel }.fel( 1:l_hist ) = [ histiel.fel( 2:l_hist ) fi(iel) ];
               el_hist{ iel }.xel = { histiel.xel(1:n,2:l_hist)  x };
            end
         end
      else
         [ nel, lxelh ] = size( histiel.xel );
	 if ( explicit_domains )                               % explicit domains
            nel = max( nel, length( eldom{ iel } ) );
            if ( lxelh < l_hist  || l_hist == 1 )              % in the beginning
	       el_hist{ iel }.fel( lxelh+1 )         = fi( iel );
               el_hist{ iel }.xel( 1:nel, lxelh+1 )  = x( eldom{ iel } );
            else                                               % after l_hist evals
	       el_hist{ iel }.fel( 1:l_hist ) = [ histiel.fel( 2:l_hist ) fi(iel) ];
               el_hist{ iel }.xel( 1:nel, 1:l_hist ) =                                  ...
	                           [ histiel.xel(1:nel,2:l_hist)  x( eldom{ iel } ) ];
            end
	 else                                                  % implicit domains
            if ( lxelh < l_hist  || l_hist == 1 )              % in the beginning
	       el_hist{ iel }.fel( lxelh+1 )       = fi( iel );
               el_hist{ iel }.xel( 1:n, lxelh+1 )  = x;
            else                                               % after l_hist evals
	       el_hist{ iel }.fel( 1:l_hist ) = [ histiel.fel( 2:l_hist ) fi(iel) ];
               el_hist{ iel }.xel( 1:n, 1:l_hist ) = [ histiel.xel(1:n,2:l_hist)  x ];
            end
         end
      end
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function s_hist = bfo_shistupd( s_hist, idc, xbest, fbest, xincr, xtype, cat_dictionnary,  ...
                              x0ref, cn_name, opt_context )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Updates the history of the minimization by avoiding repetitive information
%   for identical subspaces.


%   INPUT :

%   s_hist         : a struct containing the current subspace history information
%   idc            : the indices of the discrete and categorical variables
%   xbest          : the current best values of the variables
%   fbest          : the best objective value
%   xincr          : the current grid increments
%   xtype          : the type of thaevariables
%   cat_dictionnary: the current categorical value dictionnary
%   x0ref          : the reference vector for vector state values
%   cn_name        : the name of the dynamical neighbourhood function
%   opt_context    : the current optimization context

%   OUTPUT :

%   s_hist         : the updated history

%   PROGRAMMING: Ph. Toint, January 2010 (this version: 26 VIII 2016).

%   DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%  Get the length of the subspace list and the number of variables.

nh  = length( s_hist  );
n   = length( xbest );

%  Cellify xbest if there are categorical variables.

categorical = ~isnumeric( x0ref );
if ( categorical )
   xbest = bfo_cellify( xbest, xtype, cat_dictionnary, x0ref );
end

%  Search for the corresponding subspace in the current list.

found = 0;
for j = 1:nh
   if ( categorical )
      been_there = 1;
      for jj = idc
         if ( categorical )
	    if ( ~strcmp( s_hist( j ).x{ jj }, xbest{ jj } ) )
	       been_there = 0;
	       break
	    end
	 else
            if ( abs( s_hist( j ).x{ jj } - xbest{ jj } ) >= eps )
               been_there = 0;
               break
	    end
	 end
      end
   else      %   integer variables only
      been_there = ( norm( s_hist( j ).x( idc ) - xbest( idc ) ) < eps );
   end

   %  The subspace has been explored already.  Replace the best point in it
   %  by the current one if it improves the objective function.
   
   if ( been_there )
      found = 1;
      if ( fbest <= s_hist( j ).fx )
         s_hist( j ).x     = xbest;
         s_hist( j ).fx    = fbest;
         s_hist( j ).xincr = xincr;
	 if ( categorical  || length( cn_name ) )
            s_hist( j ).context = opt_context;
	 end
      end
   end
end

%  Unexplored subspace: store the current best point in it.

if ( ~found )
   s_hist( nh+1 ).x     = xbest;
   s_hist( nh+1 ).fx    = fbest;
   s_hist( nh+1 ).xincr = xincr;
   if ( categorical  || length( cn_name') )
      s_hist( nh+1 ).context = opt_context;
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
 
found = 0;                                     % not found yet

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

found  =  ( exist( [ fname, '.m' ] ) == 2  ||                                              ...
            ( length( fname ) > 3 && strcmp( fname(1:4), 'bfo_' ) ) );

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [ verbosity, verbose, user_verbosity, wrn ] = bfo_handle_verbosity( verbinput )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Verifies the verbosity input and returns the verified vector of verbosities and verbose
%  which corresponds to the first component of this vector.

%  INPUT:

%  verbinput     : a cell of strings whose components specify the verbosities
%                  at different levels (in case of multilevel optimization), or
%                  a string specifying the verbosity (single level optimization)

%  OUTPUT:

%  verbosity     : the verified version of verbinput if the verification was successful,
%                  the default value ('low') otherwise
%  verbose       : the numeric value corresponding of the first component of verbosity
%  user_verbosity: true iff the verification wa successful
%  wrn           : a warning in case of missspecification
  
%   PROGRAMMING  : Ph. Toint, December 2014 (this version: 27 II 2018).

%   DEPENDENCIES : -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

user_verbosity = 0;
wrn            = '';
verbosity      = 'low';
verbose        = 2;

% Multiple verbosity specification

if ( iscell( verbinput ) )
   cverbosity     = verbinput;
   user_verbosity = 1;
   for ii = length( cverbosity ):-1:1
      if ( ischar( cverbosity{ ii } ) )
         verbose = bfo_get_verbosity( cverbosity{ ii } );
         if ( verbose < 0 )
             user_verbosity = 0;
             wrn = ' BFO warning: unknown verbosity level. Default used.';
             disp( wrn )
         end
      else
         user_verbosity = 0;
         wrn = ' BFO warning: verbosity level is not a string. Default used.';
         disp( wrn )
      end
   end
   if ( user_verbosity )
      verbosity = cverbosity;
   else
      verbose    = 2;  % default
   end

% Verbosity specification as a single string

elseif ( ischar( verbinput ) )
   verbose = bfo_get_verbosity( verbinput );
   if ( verbose >= 0 )
      verbosity      = verbinput;
      user_verbosity = 1;
   else
      wrn = ' BFO warning: unknown verbosity level. Default used.';
      disp( wrn )
   end
end

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


%   PROGRAMMING: Ph. Toint, December 2014 (this version: 27 II 2018).

%   DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch ( verbosity )
case 'silent'
   verbint = 0;
case 'minimal'
   verbint = 1;
case 'low'
   verbint = 2;
case 'medium'
   verbint = 3;
case 'high'
   verbint = 4;
case 'debug'
   verbint = 10;
otherwise
   verbint = -1;
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function  bfo_print_banner( this_version )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Prints the BFO banner with properly centered line containing the version number.

%  INPUT:

%  this_version: the version number

%  PROGRAMMING: Ph. Toint 27 IX 2016 (This version 27 IX 2016).

%  DEPENDENCIES: -

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vtext    = [ '(c)  Ph. Toint, M. Porcelli, 2018  (', this_version, ')' ];
lvtext   = length( vtext );
is       = 4 + floor( ( 56 - lvtext ) / 2 );
vline    = '  *                                                        *\n';
vline( is:is+lvtext-1 ) = vtext;

fprintf( '\n')
%                    1         2         3         4         5         6
%           123456789012345678901234567890123456789012345678901234567890
fprintf(   '  **********************************************************\n')
fprintf(   '  *                                                        *\n')
fprintf(   '  *   BFO: brute-force optimization without derivatives    *\n')
fprintf(   '  *                                                        *\n')
fprintf(                               vline                               )
fprintf(   '  *                                                        *\n')
fprintf(   '  **********************************************************\n')
fprintf( '\n')

return 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [ val, fi, nevali ] = bfo_sum_objf( x, objf, elset, eldom, varargin )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Compute the value at x of a function in sum form, i.e. a function defined as the sum of
%  a collection of element functions.

%  INPUT:

%  x       : the point at which the objective function must be evaluated
%  objf    : a cell containing the element function handles
%  elset   : the indeces of the elements in objf whose sum defines the objective function
%  eldom   : a cell containing the list of variables defining the element domains
%            (each element function is assumed to depend on all variables if eldom = {})
%  fbound  : varargin{ 1 }: the acceptable upper bound on the function value
%  maximize: varargin{ 2 }: 1 if optimization is maximization, 0 if it is minimization


%  OUTPUT:

%  val     : the value of the sum-form function at x
%  nevalx  : the (possibly fractional) number of complete function evaluation necessary
%  fi      : the values of each evaluated element function (NaN if not evaluated)
%  nevali  : the number of evaluated element functions (in objf)

%  PROGRAMMING: Ph. Toint, October 2016 (This version 11 XII 2016)

%  DEPENDENCIES: ~

%  NOTE: if withbound, assumes f_i bounded below by zero.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = length( eldom );
if ( p == 0 )
   p = length( objf );
end
val     = 0;
nevali  = 0;
fi      = NaN * ones( 1, p );

%  Loop on the elements.

for i = 1:p
   if ( ~isempty( elset ) )
      elsi = elset( i );
   else
      elsi = i;
   end
   if ( ~isempty( eldom ) )            % element domains are explicit
      if ( iscell( x ) )
         fi( i ) = objf{ i }( elsi, { x{ 1 }( eldom{ i } ) } );
      else
         fi( i ) = objf{ i }( elsi, x( eldom{ i } ) );
      end
   else                              % each domain is the set of all variables
      fi( i ) = objf{ i }( elsi, x );
   end
   nevali = nevali + 1;
   val    = val + fi( i );

   %  Terminate evaluation if the user-supplied bound has been reached.
   %  Note that this assumes that f_i(x_i) >= 0 for all i=1:n_elements.
   
   if ( nargin > 4 )
      if ( ~varargin{ 2 } && val > varargin{ 1 } )
         return
      end
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function  [ options, wrn, verbosity, training, train ] = bfo_read_options_file( filename )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Reads the options file filename and construct an options list consisting of pairs
%  (keyword, value), which can then be parsed by the standard parsing process in BFO.


%  INPUT :

%  filename : the name of the user-supplied options file


%  OUTPUT:

%  options  : a cell containing the (keyword, values) pairs
%  wrn      : a possible warning
%  verbosity: a verbosity string, if found, '' otherwise.
%  training : true is if the 'training' keyword was read
%  train    : true if the 'training' keyword was read with value 'train'.

%  PROGRAMMING: Ph. Toint, December 2016 (This version 9 XII 2016)

%  DEPENDENCIES: bfo_interpret.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

options   = {};
verbosity = '';
training  = 0;
train     = 0;
wrn       = {};

fid = fopen( filename, 'r' );
if ( fid == -1 )
   wrn{ 1 } = [ ' BFO warning: cannot open file ', filename, '. Ignoring options file.' ];
else
   wrn          = {};
   continuation = 0;
   nline        = 0;

   %   As long as the file is not finished...

   while ( ~feof( fid ) )

      %  ... read a line.

      line   = fgets( fid );
      line   = line( 1:end-1 );  %  strip the \n
      nline  = nline + 1;
 
      %  Retain only lines which are non-empty and not starting 
      %  with a comment character.

      if ( ~isempty( line ) )
         poscom = find( line == '%', 1, 'first' );
         if ( ~length( poscom )  || poscom > 1 )

            %  Ignore the part of the line beyond the first comment character.

            if ( ~isempty( poscom ) )
               line = line( 1:poscom-1 );
            end
	 
            %  Ignore blank lines.
	 
            if ( ~isempty( line ) && ~isempty( find( line ~= ' ' ) ) )

               %  Strip leading and final blanks.

               line = strtrim( line );

               %  The line is a continuation from the previous one(s): 
               %  add its content to the current running value.

               if ( continuation > 0 )
                  if ( strcmp( line( end-2:end), '...' ) )
                     value       = [ value, line( 1: end-3 )];
                     continuation = 1;
                  else
	             value       = [ value, line ];
                     continuation = 0;
                  end

               %  This is a line containing a new option.

               else

                  errv = 0;

                  %  Find the keyword.

                  postext   = find( line ~= ' ', 1, 'first' );
                  line      = line( postext:end );
                  posblank  = find( line == ' ', 1, 'first' );
                  keyword   = line( 1: posblank - 1 );
                  line      = strtrim( line( posblank+1:end ) );

                  %  Find the value.
                  %  There is a value string, possibly only partial.

                  if ( ~isempty( line ) )
                     if ( length( line ) > 3 && strcmp( line( end-2:end ), '...' ) )
                        value = line( 1: end-3 );
                        continuation = 1;
                     else
                        value = line;
                        continuation = 0;
                     end

                  %  There is no value: ignore the option.
               
                  else
                     wrn{ end+1 } = [ ' BFO warning: syntax error on line ',               ...
                                      int2str( nline ), ' of options file ', filename,     ...
                                      '. Ignoring line.' ];
                     continuation = -1;
                  end
               end

               %  If the option is completed, insert it in the final option cell.

               if ( continuation == 0 )
                  [ value, errv ] = bfo_interpret( value );
                  if ( errv  )
                     wrn{ end+1 } = [ ' BFO warning: syntax error on line ',               ...
                                      int2str( nline ), ' of options file ', filename,     ...
                                      '. Ignoring line.' ];
                  else

                     %  Check for verbosity, training and (illegal) recursivity.

                     if ( strcmp( keyword, 'verbosity' ) )
                        verbosity = value;
                     elseif ( strcmp( keyword, 'options-file' ) )
                           wrn{ end+1 } = [ ' BFO warning: recursive options file',        ...
                                            ' detected and ignored.' ];
                     else
                        options{ end+1 } = keyword;
                        options{ end+1 } = value;
                        if ( strcmp( keyword, 'training-mode' ) )
                           training = 1;
                           train    =  strcmp( value, 'train' );
                        end
                     end
                  end
	       end
            end
         end
      end
   end
   fclose( fid );
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function  [ res, err ] = bfo_interpret( data )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Interprets a string and return the corresponding Matlab structure (numeric, string, cell
%  or value array cell).

%  INPUT :

%  data : the string to interpret


%  OUTPUT:

%  res  : the interpreted structure


%  PROGRAMMING: Ph. Toint, December 2016 (This version 9 XII 2016)

%  DEPENDENCIES: bfo_pop_item

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Remove all blanks for the input string.

data( find( data == ' ' ) ) = [];

%  Consider non-empty strings.

ldata = length( data );
if ( ldata )
   err   = 0;
   res   = {};

   %  Special treatment for value array cells: detect the double braces and replace them
   %  by % (which we know is not used) to define a specific starting/closing character.

   double = ( ldata > 4 && strcmp( data( 1:2 ), '{{' ) );
   if ( double )
      if ( strcmp( data( end-1:end ), '}}' ) )
         data = [ '%', data( 3:end-2 ), '%' ];
      else
         err = 1;
         return
      end
   end

   %  Determine the type of item to pop.

   d  = data( 1 );

   %  As long as the input string is nor completely analyzed, ...

   while ( ldata )

      %   ... for ecah type of starting character, first pop an item from the string
      %   and the interpret it. 

      switch( d )

      case '['
         if ( ldata < 2 )
            res{ end+1 } = data;
            err = 1;
         elseif ( ldata == 2 )
            res{ end+1 } = [];
            return
         else
            [ item, data, err ] = bfo_pop_item( data, ']' );
            if ( err )
               return
            end
            res{ end+1 }   = str2num( ['[',item,']'] );
      end

      case '%'
         if ( ldata < 2 )
            res{ end+1 } = data;
            err = 1;
         elseif ( ldata == 2 )
            res{ end+1 } = {};
            return
         else
            [ item, data, err ] = bfo_pop_item( data, '%' );
            if ( err )
               return
            end
            [ item, err ]  = bfo_interpret( item );
            if ( err )
               return
            end
            res{ end+1 }   = {item};
         end

      case '{'
         if ( ldata < 2 )
            res{ end+1 } = data;
            err = 1;
         elseif ( ldata == 2 )
            res{ end+1 } = {};
            return
         else
            [ item, data, err ] = bfo_pop_item( data, '}' );
            if ( err )
               return
            end
            [ item, err ]  = bfo_interpret( item );
            if ( err )
               return
            end
            res{ end+1 }   = item;
         end

      case ''''
         if ( ldata < 2 )
            res{ end+1 } = data;
            err = 1;
         elseif ( ldata == 2 )
            res{ end+1 } = '';
            return
         else
            [ item, data, err ] = bfo_pop_item( data, '''' );
            if ( err )
               return
            end
            res{ end+1 } = item;
         end

      otherwise
         [ item, data, err ] = bfo_pop_item( data, ',' );
         if ( err )
            return
         end
         res{ end+1 }   = str2num( item );
      end

      %  Determine the start of the remaining data, discounting the , if necessary.
      %  Also determine the type of the next item to pop, if any.

      ldata = length( data );
      if ( ldata )
         d = data( 1 );
         if ( d == ',' )
            data  = data( 2:end );
            d     = data( 1 );
            ldata = ldata-1;
         else
            data = data( 1:end );
         end
      end
   end

   %   If the string contained only one item, the cell structure is unnecessary.

   if ( length( res ) == 1 )
      res = res{ 1 };
   end

%  The empty input string

else
   res = '';
   err = 1;
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function  [ item, str, err ] = bfo_pop_item( str, cl )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Pops (in item) an item from the beginning of a string str, the item being defined 
%  by its termination character cl.  It also returns the unused part of the input string.

%  INPUT :

%  str  : the string from which the item should be popped
%  cl   : the character defining the end of the item

%  OUTPUT:

%  item : the popped item
%  str  : the remaining part of the input string
%  err  : >0 if an error occurred, 0 otherwise.

%  PROGRAMMING: Ph. Toint, December 2016 (This version 10 XII 2016)

%  DEPENDENCIES: ~

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

err = 0;

%  Define the starting character (op), the starting number of starting character (nop)
%  and the index from which str must be searched.

switch( cl )
case ','
  op  = ' ';
  nop = 0;
  is  = 1;
case '%',
  op  = '%';
  nop = 1;
  is  = 2;
case '}',
  op  = '{';
  nop = 1;
  is  = 2;
case ']'
  op  = '[';
  nop = 1;
  is  = 2;
case ''''
  op  = '''';
  nop = 1;
  is  = 2;
end
lstr = length( str );

%  Loop on the string str searching for the closing character.

for i = is:lstr
   si = str( i );
   if ( si == cl )
      nop = nop - 1;
      if ( nop == 0  || cl == ',' ) 
         item = str( is:i-1 );
         str  = str( i+1:end );
         return
      end
   elseif ( si == op )
      nop = nop + 1;
   end
   if ( i == lstr )
      if ( cl == ',' )
         item = str( is:end );
         str  = '';
      else
         err = 1;
         item = {};
         return
      end
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ alpha, beta, gamma, delta, eta, zeta, inertia, searchtype, stype, rseed, iota,  ...
           kappa, lambda, mu ] = bfo_default_algorithmic_parameters( mixed_integer );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Defines the BFO algorithmic parameters, typically using values obtained by
%  training.

%  INPUT:

%  mixed_integer: true iff there are discrete or categorical variables variables

%  OUTPUT:

%  alpha     : the grid expansion factor at successful iterations (>= 1)
%  beta      : a fraction ( in (0,1) ) defining the shrinking ratio between
%              successive grids for the continuous variables
%  gamma     : the maximum factor ( >= 1 ) by which the initial user-supplied
%              grid for continuous variables may be expanded
%  eta       : a fraction ( > 0 ) defining the improvement in objective function
%              deemed sufficient to stop polling the remaining variables, this
%              decrease being computed as eta times the squared mesh-size.
%  zeta      : a factor (>=1) by which the grid size is expanded when a
%              particular level (in multilevel use) is re-explored after a
%              previous optimization.
%  inertia   : the number of iterations used for averaging the steps in the
%              continuous variables, the basis for these variables being
%              computed for the next iteration as an orthonormal basis whose
%              first element is the (normalized) average step.
%              NOTE: inertia = 0 disables the averaging process.
%  searchtype: an string defining the strategy to use for exploring the tree
%              of possible values for the discrete variables.  Possible values
%              are:
%              'breadth-first' : all subspaces corresponding to interesting 
%                                values of the discrete variables are explored 
%                                before grid refinement
%              'depth-first'   : grid refinement is performed as soon as possible
%              'none'          : no recursion
%  stype     : an integer corresponding to searchtype:
%              0 : 'breadth-first'
%              1 : 'depth-first'
%              -1: 'none'
%  rseed     : a positive integer specifying the seed for the random
%              number generator rng( seed, 'twister') which is used to
%              initialize random sequences at the beginning of execution.
%              Random numbers are used for the choice of alternative basis
%              vectors for the continuous variables, both when refining 
%              the grid and when checking termination.
%  iota      : a power at least equal to 1 to which the stepsize shrinking factor
%              is raised after unsuccessful coordinate partially separable search
%  kappa     : the bracket expansion factor in min1d (case without quadratic interpolation)
%  lambda    : the min bracket expansion factor in min1d (case with quadratic interpolation)
%  mu        : the max bracket expansion factor in min1d (case with quadratic interpolation)

%  DEPENDENCIES : ~

%  PROGRAMMING: Ph. Toint, February 2018. (This version 29 II 2018)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%              alpha    beta   gamma    delta   eta     zeta  inertia  stype  rseed  iota   kappa lambda mu 

if ( mixed_integer )
   pars    = [ 2.0000, 0.3135, 5.0000, 3.6030, 0.0001, 1.5000,  10  ,    1  ,   91 , 1.2550   2     0.1  50 ];
else
   pars    = [ 1.4248, 0.1997, 2.3599, 1.0368, 0.0001, 1.5000,  11  ,    1  ,   91 , 1.2550,  2     0.1  50 ];
end

alpha   = pars( 1 );
beta    = pars( 2 ); 
gamma   = pars( 3 );
delta   = pars( 4 );
eta     = pars( 5 );
zeta    = pars( 6 );
inertia = pars( 7 );
stype   = pars( 8 );
rseed   = pars( 9 );
iota    = pars( 10 );
kappa   = pars( 11 );
lambda  = pars( 12 );
mu      = pars( 13 );

switch( stype )
case -1
   searchtype = 'none';
case 0
   searchtype = 'breadth-first';
case 1
   searchtype = 'depth-first';
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function  tp = bfo_cutest_data( cutest_name )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Auxiliary function to retrieve data of a CUTEst problem where the variables with even
%  index is constrained to be integer and lower/upper bounds are rounded accordingly.

%  WARNING: a CUTEst MATLAB interface is assumed to be currently installed! 
%  The *.SIF problem files are assumed to be located in $MASTSIF 

%  MARGHE: specify what is $MYARCH  ??

%  INPUT:

%  cutest_name: the name of the CUTEst problem

%  OUTPUT:

%  tp         : a struct containing the the corresponding condensed CUTEst test problem.

%  DEPENDENCIES : cutest_setup

%  PROGRAMMING: M. Porcelli, May 2010. (This version 25 XI 2015)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Decode the CUTEST problem

st = (['!runcutest -A $MYARCH -p matlab -D $MASTSIF/' cutest_name '> /dev/null']);
eval( st )

%  Setup the problem data by calling the appropriate CUTEst tool.

prob = cutest_setup();

%  Create the struct for the training problem, remembering that CUTEst problems 
%  are minimizations.

tp = struct( 'objf', @cutest_obj, 'x0', prob.x, 'xlower', prob.bl,                         ...
             'xupper', prob.bu, 'xtype', 'c', 'max_or_min', 'min' );

%  The lower and upper bound

tp.xlower( find( tp.xlower == -1e20 ) ) = -Inf;
tp.xupper( find( tp.xupper ==  1e20 ) ) =  Inf;

if(0)%D

%  Create the variable's type, while possibly setting even variables to integer
%  and verifying the feasibility of the starting point wrt the bounds.

for i = 2:length( tp.x0 )
   if ( mod( i, 2 ) == 0 )         % all variable with even indices are integer 
      tp.xtype( end+1 ) = 'i';
      tp.x0( i )        = round( tp.x0( i ) );
      tp.xlower( i )    = ceil ( tp.xlower( i ) );
      tp.xupper( i )    = floor( tp.xupper( i ) );
   else
      tp.xtype = [ tp.xtype, 'c' ];
   end
   tp.x0( i ) = max( tp.xlower( i ), min( tp.x0( i ), tp.xupper( i ) ) );
end

end%D

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
