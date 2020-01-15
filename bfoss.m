%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ xsearch, fsearch, nevalss, exc, x_hist, f_hist, el_hist ] =                     ...
                    bfoss( level, f, xbest, max_or_min, xincr, x_hist, f_hist, xtype,      ...
		           xlower, xupper, lattice_basis, varargin )
                                                                                             
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%                                                                      %%%%%%%%%%%%
%%%%%%%%%%%%                                 BFOSS                                %%%%%%%%%%%%
%%%%%%%%%%%%                                                                      %%%%%%%%%%%%
%%%%%%%%%%%%              A BFO-compatible library of Search-Step functions       %%%%%%%%%%%%
%%%%%%%%%%%%                                                                      %%%%%%%%%%%%
%%%%%%%%%%%%                      M. Porcelli and Ph. L. Toint                    %%%%%%%%%%%%
%%%%%%%%%%%%                                                                      %%%%%%%%%%%%
%%%%%%%%%%%%                        Version 1.0   (c) 2020                        %%%%%%%%%%%%
%%%%%%%%%%%%                                                                      %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%  1. BFOSS, a BFO compatible library of search-step functions
%
%    This file provides a library of search-step functions to be used with the BFO
%    derivative-free optimization package. The current version of BFOSS includes
%     - unstructured multivariate polynomial interpolation model of degree at most
%       linear (for C^1 functions) or quadratic (for C^2 functions)
%     - partially separable multivariate polynomial interpolation model of degree at 
%       most linear (for C^1 functions) or quadratic (for C^2 functions),
%    from which a search step is then computed using a standard trust-region mechanism.
%
%    The computation of the trial point xsearch and associated function value fsearch uses
%    information on past function values and associated points (as computed by BFO or by
%    previous calls to the library). This information is contained in the vectors f_hist
%    (the past f-values) and x_hist (the associated points).  Further information on the
%    problem is also provided in that input arguments specify the variables' types and
%    their bounds (if any).  For discrete variables, the basis of the considered lattice
%    is also supplied on input.  Finally, element-wise structure, function values and
%    points are supplied in the case where the function is partially separable, allowing
%    the library to exploit this commonly occuring structure.
%
%    NOTE: If the objective function is in elemental form, it ust use the (default) sum
%          composition function for BFOSS to be applicable.
%
%    A description of BFOSS and preliminary numerical results are available in Porcelli
%    and Toint (2017b).  BFO itself is described in Porcelli and Toint (2017a).
%
%%  2. Multivariate interpolation models
%
%    When interpolation models are used, BFOSS uses the past function values and points
%    to create an "interpolation set", which is used in turn to compute associated
%    Lagrange polynomials. A linear combination of these polynomials is constructed
%    to provide an interpolating polynomial IN THE CONTINUOUS VARIABLES, which can
%    then to be minimized to find a trial point. (Note that discrete variables are
%    considered fixed).
%
%    This mechanism requires that Lagrange polynomials can be computed on the tentative
%    interpolation set.  This in turn imposes that the geometry of the points in this set
%    is "poised" enough (meaning that each point brings additional information on the
%    function). This is measured by computing the maximum absolute value of the Lagrange
%    polynomials is the neighbourhood.  If this value is small enough, the improvement
%    in "poisedness" which would be obtained by exchanging a current interpolation point
%    by the best possible candidate is relatively modest, and the set is the declared
%    poised. However, for this value to be meaningfull, the Lagrange polynomials themselves
%    must be reasonably accurate.  As they are computed by solving a linear system
%    (possibly in the generalized sense), this requires the system matrix to be not
%    too ill-conditioned.  The construction of the interpolating polynomial thus proceeds
%    in four phases:
%    1) a tentative interpolation set is defined using past points/values as much as
%       possible (in the limits of availability and for an at most quadratic model),
%    2) this set is corrected to ensure that the matrix defining the Lagrange polynomials
%       is not too ill-conditioned,
%    3) it is possibly further improved by performing exchanges until the improvement
%       in poisedness becomes moderate enough,
%    4) the same (reasonably conditioned) matrix is then used to define the linear
%       combination of the Lagrange polynomials which interpolates function values at the
%       interpolation points.
%    Once available, the polynomial interpolant may be used as a model of the objective
%    function and its value minimized to provide the trial step.
%    Note that the degree of the interpolating polynomials (and associated Lagrange
%    polynomials) may vary depending on how data points are chosen and how many elements
%    in the monomial basis of the polynomials are considered. This later number may vary
%    between 2 (the minimal non-constant mode) to ((n+1)*(n+2))/2 (the full quadratic).
%
%    The use of multivariate interpolation models was pioneered by Winfield and, most
%    proeminently, by Powell in the 1970's. The user is refered to Conn, Scheinberg
%    and Vicente (2009) for a good introduction to this topic and its use in derivative-free
%    methods. This book also contains an excellent bibliography.
%
%    Importantly, the commonly occuring structure of partially separable functions
%    (see Griewank and Toint (1982)) can be exploited to great advantage in model-based
%    derivative-free methods. The idea (see Colson and Toint (2005)) is simply to
%    construct a model whose structure mimics that of the original function, thereby
%    supplying considerable additional information for its construction. A separate model
%    is typically constructed for each "element function" of the considered partially
%    separable decomposition, and these models are then assembled to form a global
%    structured model.  This is what BFOSS does, applying the techniques to compute an
%    interpolation model to each element of the objective by restricting the analysis to
%    its associated "element domain" (very often a small dimensional subspace).
%
%%  3. The trust_region mechanism
%
%    When a model of the objective function is available (an interpolating polynomial),
%    this model can the be minimized in a region around the current iterate where it
%    is believed to be valid.  This "trust-region" is a ball centered at the current
%    best point and whose radius is updated from one call of BFOSS to the next (some
%    information is saved by BFOSS in a few persistent variables).  The trust-region
%    mechanism used here is classical (see Conn, Gould and Toint (2000) for an extensive
%    coverage of these methods.
%
%   4. References
%
%    A. R. Conn, N. I. M. Gould and Ph. L. Toint, "Trust Region Methods", MPS-SIAM Series
%    on Optimization, vol.1, SIAM, Philadelphia, 2000.
%
%    A. R. Conn, K. Scheinberg and L. N. Vicente, "Introduction to Derivative-free 
%    Optimization",  MPS-SIAM Series on Optimization, SIAM, Philadelphia, 2009.
%
%    B. Colson and Ph. L. Toint, "Optimizing Partially Separable Functions Without 
%    Derivatives", Optimization Methods and Software, vol. 20(4-5), pp. 493-508, 2005.
%
%    A. Griewank and Ph. L. Toint, "On the unconstrained optimization of partially 
%    separable functions", in "Nonlinear Optimization 1981" (M. J. D. Powell, ed.)
%    Academic Press, London, pp. 301--312, 1982.

%    M. Porcelli and Ph. L. Toint, "BFO, a trainable derivative-free Brute Force
%    Optimizer for nonlinear bound-constrained optimization and equilibrium computations
%    with continuous and discrete variables", Transactions of the AMS on Mathematical
%    Software, vol. 44(1), 2017a.
%
%    M. Porcelli and Ph. L. Toint, "Global and local information in structured derivative free
%    optimization with BFO", in preparation, 2020.
%
%%  5. Arguments
%
%   INPUT:
%
%      level          : the level from which BFO call the search-step function.  If negative
%                       this signals that return from BFO to the user is imminent and that
%                       cleanup internal to BFOSS must be performed (no output value is
%                       expected in this case).
%      f              : the handle to the objective function.
%      xbest          : the current best point
%      max_or_min     : 'min' or 'max' depending on whether minimization or maximization
%      xincr          : the current mesh size needed to compute the current TR radius
%      x_hist         : an (n x min(nfcalls,l_hist)) array whose columns
%                       contain the  min(nfcalls,l_hist)) points at which f(x) has been
%                       evaluated last.
%      f_hist         : an array of length nfcalls containing the function values
%                       associated  with the columns of x_hist.
%      xlower         : the vector of lower bounds.
%      xupper         : the vector of upper bounds.
%      indfree        : a vector of current idices of free variables.
%      nevalss        : a scalar for the number of function evaluations in the search step
%                       function, computed so far.
%      Delta          : a scalar for the current trust-region radius
%      verbosity      : a string for the level of verbosity 
%      varargin       : a (possibly empty) list of arguments.
%                       For the interpolation models, the first argument in varargin
%                       must be the element-wise history (el_hist), a cell of structs
%                       produced by BFO, when the problem is patially separable. All
%                       following arguments are specified by a (keyword, value) pair.
%                       The keyword is a string and the associated value immediately
%                       follows the keyword in varargin. Possible keywords and associated
%                       values are as follows.
%         model-type  : the type of model to use (for now, only 'interpolation')
%         model-mode  : the technique used to define the relation between the number
%                       of data points and the number of monomial basis elements for
%                       constructing the model:
%                       'subbasis': the number of elements of the monomial basis is equal
%                                   the number of interpolation points (monomials are
%                                   ordered as follows: constant term, linear terms,
%                                   diagonal quadratic terms, and finally off-diagonal
%                                   quadratic terms by increasing subdiagonals.
%                       'minell2' : the number of elements of the monomail basis is
%                                   always equal to the number needed for a full quadratic,
%                                   the (possibly underdetermined) linear system being then
%                                   solved in the least-squares sense.
%                       Default: 'subbasis'
%         minimum-model-degree: the minimum degree of the polynomial model:
%                       'minimal'  : at least two basis monomials are used,
%                       'linear'   : the constant and the n linear monomials are used,
%                       'diagonal' : idem + the diagonal quadratic terms,
%                       'quadratic': idem + the off-diagonal quadratic terms.
%                       Default: 'minimal'.
%         maximum-model-degree: the maximal model degree at any iteration:
%                       'linear'   : the constant and the n linear monomials are used,
%                       'diagonal' : idem + the diagonal quadratic terms,
%                       'quadratic': idem + the off-diagonal quadratic terms.
%                       Default: 'quadratic'.
%         conditioning-limit:  a real number >= 1 specifying the maximum admissible
%                       conditioning for the interpolation matrix
%                       Default: 1e14.
%         poisedness-limit : a positive real number specifying an upper bound on the Lagrange
%                       polynomial in the trust region for declaring the interpolation poised
%                       Default: 100.
%         neighbourhood-limit: a real number >= 1 specifying the multiple of the trust
%                       region radius beyond which points aare considered too far to be
%                       relevant for the current interpolation model.
%                       Default: 10.
%         far-point-acceptability: a real (>1) specifyin the minimum improvement in poisedness
%                       required for including points in the history in the interpolation set,
%                       Default: 100.
%         TR-init-radius: a positive real specifying the initial trust-region radius,
%                       or a negative number for an automatic choice (not always optimal),
%                       Default: -1 (automatic)
%         TR-min-radius: a positive real specifying the trust-region radius under which no
%                       search step is computed,
%                       Default: 1e-8
%         TR-max-radius: a positive real specifying the maximal trust-region radius,
%                       Default: 100.
%         TR-accuracy : a positive real specifying the accuracy at which the trust-region
%                       constraint must be satisfied,
%                       Default: 1e-3.
%         TR-eta1     : the threshold on rho for successful iterations
%                       Default: 0.05.
%         TR-eta2     : the threshold on rho for very successful iterations
%                       Default: 0.9.
%         TR-alpha1   : the TR expansion factor at very successful iterations
%                       Default: 2.5.
%         TR-alpha2   : the TR contraction factor at unsuccessful iterations
%                       Default: 0.25.
%         TR-solver-step: the solver for the trust-region subproblem defining the
%                       search step. Possible choices are
%                       'more-sorensen' : a bound-constrained variant of the Moré-Sorensen
%                                         TR algorithm
%                       'ptcg'          : a projected truncated conjugate gradient method
%                       Default: 'more-sorensen'
%         TR-solver-Lagrange: the solver for the trust-region subproblems involving the
%                       Lagrange polynomials. Possible choices are
%                       'more-sorensen' : a bound-constrained variant of the Moré-Sorensen
%                                         TR algorithm
%                       'ptcg'          : a projected truncated conjugate gradient method
%                       Default: 'more-sorensen'
%         TR_PTCG_rel_acc: the relative accuracy on the projected gradient to terminate
%                       the projected truncated CG
%                       'Default: 1e-5.
%         TR_PTCG_abs_acc: the absolute accuracy on the projected gradient to terminate
%                       the projected truncated CG
%                       'Default: 1e-10.
%         TR_PTCG_maxcgits : the maximum number of CG iterations allowed in the TR problem
%                       solution, relative to dimension.
%                       Default: 3.
%         termination-order: the order of the critical point requested on the model
%                       for the search-step function to indicate termination to BFO:
%                       '1rst' : checks that the model's gradient norm is below epsilon
%                       '2nd'  : additionnaly checks that
%                                for minimization: the smallest eigenvalue of the models'
%                                                  Hessian is above -sqrt( epsilon )
%                                for maximization: the largest eigenvalue of the models'
%                                                  Hessian is below sqrt( epsilon );
%                       Using '1rst' results in (usually marginally) faster termination, at
%                       the expense of accuracy.
%                       Default: '2nd'.
%         epsilon     : the accuracy used for model-based termination
%                       Default: 1e-5.
%         min-step-length: the  minimum length of the step for computing  the trial
%                       function value,
%                       Default: 1e-10.
%         verbosity   : a string specifying the verbosity of the search-step finding process
%                       Possible values are 'silent', 'low', 'medium' and 'debug' .
%                       Default: 'low'.
%
%  OUTPUT:
%
%    xsearch : an array of length n containing the point returned by the 
%              user as a tentative improved iterate,
%    fsearch : the associated objective function value f(xsearch)
%    nevalss : the number of function evaluations performed internally by the
%              search-step function.
%    exc     : the search-step exit code.
%              exc =  3: seemingly at optimum: the search-step function has determined
%                        that the base point appears to be optimal (xsearch and fsearch
%                        are to be ignored in this case);
%              exc =  2: normal successful return: a search-step xsearch and associated
%                        function value fsearch has been found, which produces significant
%                        objective function improvement;
%              exc =  1: normal unusuccessful return: a  search-step xsearch and associated
%                        fsearch have been calculated, but do not produce significant
%                        objective function improvement;
%              exc  = 0: the search step function has found a point xsearch and a value
%                        fsearch, but leaves to BFO the decision to use it or not;
%              exc = -1: the search-step function could not reliably compute a search step,
%                        and the returned xsearch and fsearch should be ignored;
%              exc = -2: the objective function returned a NaN or infinite value: the
%                        returned xsearch and fsearch should be ignored;
%              exc = -3: the user has requested termination: the returned xsearch and
%                        fsearch should be ignored.
%    x_hist  : is the updated array or cell array to which is now appended the set of
%              vectors at which the objective function has been evaluated
%              (xsearch = x_hist( 1:n,end)).
%    f_hist  : the updated array of function values (f_search = f_hist(end)).
%    el_hist : the cell array containing the updated structs (for all element function 
%              evaluations during the search-step computation).
%
%  6. Software issues
%
%   DEPENDENCIES : bfoss_search_step_interp_model, bfoss_search_step_interp_model_ps
%
%   PROGRAMMING  : M. Porcelli and Ph. Toint, March 2017 to 2020
%                  (this version 13 I 2020).
%
%%  RESTRICTIONS
%
%   In the present version, BFOSS does not support
%     - categorical variables
%     - functions defined with a 'with-bound' interface.
%   Note that discrete variables are considered as fixed for the computation
%   of the search step.
%
%%  CONDITIONS OF USE

%   *** Use at your own risk! No guarantee of any kind given or implied. ***

%   Copyright (c) 2020, Ph. Toint and M. Porcelli. All rights reserved.
%
%   Redistribution and use of the BFO package and the BFOSS library in source and
%   binary forms, with or without modification, are permitted provided that the
%   following conditions are met:
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Set default values for output arguments.

xsearch = [];
fsearch = [];
nevalss = -1;
exc     = -1;

%  Set default basic values.

model_type            = 'interpolation';  % for now the only model type avaialble
verbosity             = 'low';         % the verbosity of the search-step finding process

%  Now verify if the problem is in elemental form. Also define varg1, the position
%  (in varargin) of the keyword of the first ( keyword, value ) pair.

varg1 = 1;
if ( iscell( f ) && ~isempty( varargin ) )
   el_hist  = varargin{ 1 };
   elemental_form = 1;
   varg1    = 2;
end
if ( varg1 == 1 )
   elemental_form = 0;
   el_hist  = [];
end

%  Set the default algorithmic parameters depending on model type and elemental-form.

switch ( model_type )
case 'interpolation'
   [ model_mode, minimum_model_degree, maximum_model_degree, kappa_ill, poisedupper,       ...
     toofarlimit, farthr, term_order, epsilon, TR_init_radius, TR_min_radius,              ...
     TR_max_radius, TR_accuracy, TR_eta1, TR_eta2, TR_alpha1, TR_alpha2,  TR_solver_step,  ...
     TR_solver_Lagr, TR_PTCG_rel_acc, TR_PTCG_abs_acc, TR_PTCG_maxcgits,min_step_length ] =...
                                    bfoss_interpolation_default_parameters( elemental_form );

end

%  See if this is the termination call, in which case all input arguments except level
%  are meaningless.

if ( level >= 0 )

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %                                                                                         %
   %                          Interpret the optional arguments                               %
   %                                                                                         %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   %  Extract verbosity first, if specified.

   nargs = size( varargin, 2 );
   for i = varg1:nargs
      if ( ischar( varargin{i} ) && strcmp( varargin{i}, 'verbosity' ) )
         if ( i < nargs && ischar( varargin{i+1} ) )
            switch ( varargin{i+1} )
	    case 'silent'
	       verbosity = 'silent';
	    case 'low'
	       verbosity = 'low';
	    case 'medium'
               verbosity = 'medium';
	    case 'debug'
               verbosity = 'debug';
	    otherwise
               disp( ' BFOSS error: Unknow verbosity. Using default.' )
	    end
         else
            disp( ' BFOSS error: the argument list is ill-constructed. Terminating.' )
            nevalss = -1;
            return
         end
         break
      end
   end

   %  Process the rest of the variable argument list.

   for i = varg1:2:nargs
      wrongkey = '';
      if ( ischar( varargin{ i } ) && nargs  > i )
         switch ( varargin{ i } )

         %  The type of the model used
      
         case 'model-type'
            if ( ischar( varargin{ i+1 } ) )
	       switch( varargin{ i+1 } )
	       case 'interpolation'
	       otherwise
	          wrongkey = 'model-type';
	       end
	    else
	       wrongkey =  'model-type';
   	    end

         %  The model mode, if any.
      
         case 'model-mode'
            if ( ischar( varargin{ i+1 } ) )
	       switch( varargin{ i+1 } )
	       case 'subbasis'
	          model_mode = 'subbasis';
	       case 'minell2'
	          model_mode = 'minell2';
	       otherwise
	          wrongkey = 'model-mode';
	       end
	    else
	       wrongkey =  'model-mode';
	    end

         %  The minimum model degree (for interpolation)

         case 'minimum-model-degree'
            if ( ischar( varargin{ i+1 } ) )
	       switch( varargin{ i+1 } )
	       case 'minimal'
	          minimum_model_degree = 'minimal';
 	       case 'linear'
	          minimum_model_degree = 'linear';
	       case 'diagonal'
	          minimum_model_degree = 'diagonal';
	       case 'quadratic'
	          minimum_model_degree = 'quadratic';
	       otherwise
	          wrongkey = 'minimum-model-degree';
	       end
	    else
	       wrongkey = 'minimum-model-degree';
	    end

         %  The maximum model degree (for interpolation)

         case 'maximum-model-degree'
            if ( ischar( varargin{ i+1 } ) )
	       switch( varargin{ i+1 } )
 	       case 'linear'
	          maximum_model_degree = 'linear';
	       case 'diagonal'
	          maximum_model_degree = 'diagonal';
	       case 'quadratic'
	          maximum_model_degree = 'quadratic';
	       otherwise
	          wrongkey = 'max-model-degree';
	       end
	    else
	       wrongkey = 'max-model-degree';
	    end

         %  The upper bound on the interpolation matrix conditioning
      
         case 'conditioning-limit'
            if ( isnumeric( varargin{ i+1 } ) )
	       num = varargin{ i+1 };
	       if ( max( size( num) ) == 1 && num >= 1 )
	          kappa_ill = num;
	       else
	          wrongkey = 'conditioning-limit';
	       end
	    else
	       wrongkey = 'conditioning-limit';
	    end

         %  The upper bound on the Lagrange polynomial in the
         %  trust region for declaring interpolation poised
      
         case 'poisedness-limit'
            if ( isnumeric( varargin{ i+1 } ) )
	       num = varargin{ i+1 };
	       if ( max( size( num) ) == 1 && num >= 0 )
	          poisedupper = num;
	       else
	          wrongkey = 'poisedness-limit';
	       end
	    else
	       wrongkey = 'poisedness-limit';
	    end

         %  The multiple of the TR radius beyond which points are considered too far
	 %  to be included in the interpolation set
      
         case 'neighbourhood-limit'
            if ( isnumeric( varargin{ i+1 } ) )
	       num = varargin{ i+1 };
	       if ( max( size( num) ) == 1 && num >= 1 )
	          toofarlimit = num;
	       else
	          wrongkey = 'neighbourhood-limit';
	       end
	    else
	       wrongkey = 'neighbourhood-limit';
	    end

         %  The minimum improvement in poisedness required for
         %  including points in the history in the interpolation set

         case 'far-point-acceptability'
            if ( isnumeric( varargin{ i+1 } ) )
	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num >= 0 )
	          farthr = num;
	       else
	          wrongkey = 'far-point-acceptability';
	       end
	    else
	       wrongkey = 'far-point-acceptability';
	    end

         %  The order of the critical point for model-based termination

         case 'termination-order'
            if ( ischar( varargin{ i+1 } ) )
	       switch( varargin{ i+1 } )
	       case '1rst'
	          term_order = '1rst';
	       case '2nd'
                  term_order = '2nd';
	       otherwise
	          wrongkey = 'termination-order';
	       end
	    else
	       wrongkey =  'termination-order';
   	    end

         %  The initial trust-region radius
      
         case 'TR-init-radius'
            if ( isnumeric( varargin{ i+1 } ) )
	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num > 0 )
	          TR_init_radius = num;
	       else
	          wrongkey = 'TR-init-radius';
	       end
	    else
	       wrongkey = 'TR-init-radius';
	    end

         %  The minimal trust-region radius for modelling to take place.
      
         case 'TR-min-radius'
            if ( isnumeric( varargin{ i+1 } ) )
	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num > 0 )
	          TR_min_radius = num;
	       else
	          wrongkey = 'TR-min-radius';
	       end
	    else
	       wrongkey = 'TR-min-radius';
	    end

        %   The maximum allowed radius for the trust region
     
         case 'TR-max-radius'
            if ( isnumeric( varargin{ i+1 } ) )
	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num > 0 )
	          TR_max_radius = num;
	       else
	          wrongkey = 'TR-max-radius';
	       end
	    else
	       wrongkey = 'TR-max-radius';
   	    end

         %  The required accuracy on the trsut-region constraint
      
         case 'TR-accuracy'
            if ( isnumeric( varargin{ i+1 } ) )
	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num > 0 )
	          TR_accuracy = num;
	       else
	          wrongkey = 'TR-accuracy';
	       end
	    else
	       wrongkey = 'TR-accuracy';
	    end

         %  The TR threshold for successful iterations
      
         case 'TR-eta1'
            if ( isnumeric( varargin{ i+1 } ) )
	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num > 0 )
	          TR_eta1 = num;
	       else
	          wrongkey = 'TR-eta1';
	       end
	    else
	       wrongkey = 'TR-eta1';
	    end

         %  The TR threshold for very successful iterations
      
         case 'TR-eta2'
            if ( isnumeric( varargin{ i+1 } ) )
	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num > 0 )
	          TR_eta2 = num;
	       else
 	          wrongkey = 'TR-eta2';
	       end
	    else
	       wrongkey = 'TR-eta2';
	    end

         %  The TR expansion factor at very successful iterations
      
         case 'TR-alpha1'
            if ( isnumeric( varargin{ i+1 } ) )
	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num > 0 )
	          TR_alpha1 = num;
	       else
	          wrongkey = 'TR-alpha1';
	       end
	    else
	       wrongkey = 'TR-alpha1';
	    end

         %  The TR contraction factor at unsuccessful iterations
      
         case 'TR-alpha2'
            if ( isnumeric( varargin{ i+1 } ) )
	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num > 0 )
	          TR_alpha2 = num;
	       else
	          wrongkey = 'TR-alpha2';
	       end
	    else
	       wrongkey = 'TR-alpha2';
	    end

         %  The solver for the step TR subproblem
      
         case 'TR-solver-step'
            if ( ischar( varargin{ i+1 } ) )
	       switch ( varargin{ i+1 } )
	       case 'more-sorensen'
	          TR_solver_step = 'more-sorensen';
	       case 'ptcg'
	          TR_solver_step = 'ptcg';
	       otherwise
	          wrongkey = 'TR-solver-step';
	       end
	    else
	       wrongkey = 'TR-solver-step';
	    end

         %  The solver for the Lagrange TR subproblems
      
         case 'TR-solver-Lagrange'
            if ( ischar( varargin{ i+1 } ) )
	       switch ( varargin{ i+1 } )
	       case 'more-sorensen'
	          TR_solver_Lagr = 'more-sorensen';
	       case 'ptcg'
	          TR_solver_Lagr = 'ptcg';
	       otherwise
	          wrongkey = 'TR-solver-Lagrange';
	       end
	    else
	       wrongkey = 'TR-solver-Lagrange';
	    end

         %  The relative accuracy on the projected gradient for terminating the PTCG
      
         case 'TR-PTCG-relative-acuracy'
            if ( isnumeric( varargin{ i+1 } ) )
 	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num > 0 && num < 1 )
	          TR_PTCG_rel_acc = num;
	       else
	          wrongkey = 'TR-PTCG-relative-accuracy';
	       end
	    else
	       wrongkey = 'TR-PTCG-relative-accuracy';
	    end

         %  The absolute accuracy on the projected gradient for terminating the PTCG
      
         case 'TR-PTCG-absolute-acuracy'
            if ( isnumeric( varargin{ i+1 } ) )
 	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num > 0 && num < 1 )
	          TR_PTCG_abs_acc = num;
	       else
	          wrongkey = 'TR-PTCG-absolute-accuracy';
	       end
	    else
	       wrongkey = 'TR-PTCG-absolute-accuracy';
	    end

         %  The maximum number of CG iterations in PTCG, relative to problem size
      
         case 'TR-PTCG-maxcgits'
            if ( isnumeric( varargin{ i+1 } ) )
 	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num > 0 )
	          TR_PTCG_maxcgits = ceil( num );
	       else
	          wrongkey = 'TR-PTCG-maxcgits';
	       end
	    else
	       wrongkey = 'TR-PTCG-maxcgits';
	    end

         %  The minimum steplength for computing f at the trial point
      
         case 'min-step-length'
            if ( isnumeric( varargin{ i+1 } ) )
 	       num = varargin{ i+1 };
	       if ( max( size( num ) ) == 1 && num > 0 )
	          min_step_length = num;
	       else
	          wrongkey = 'min-step-length';
	       end
	    else
	       wrongkey = 'min-step-length';
	    end

         %  The process verbosity (already handled above)
      
         case 'verbosity'

         %  Wrong keyword
      
         otherwise
            if ( ~strcmp( verbosity, 'silent' ) )
               disp( [ ' BFOSS warning: Unknown ', int2str( i ) ,'-th keyword. Ignoring.' ] )
            end
         end
         if ( ~isempty( wrongkey ) )
            if ( ~strcmp( verbosity, 'silent' ) )
	       disp( [ ' BFOSS warning: Incorrect value for keyword ', wrongkey,           ...
	               '. Default used.' ] )
            end
         end
      else
         if ( ~strcmp( verbosity, 'silent' ) )
            disp( ' BFOSS error: the argument list is ill-constructed. Terminating.' )
         end
         nevalss = -1;
         return 
      end
   end
end

%  Verify that the minimal trust-region radius is large enough for the quadratic terms
%  in the interpolation matrix to be above underflow.

TR_min_radius = 10 * max( sqrt( eps ), TR_min_radius );

%  Consider the different model types (to be completed in the future)

switch( model_type )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                            %
%                              Interpolation models                                          %
%                                                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

case 'interpolation'

   %  Verify the coherence of the interpolation degrees.

   if ( strcmp( maximum_model_degree, 'linear' ) )
      if ( ismember( minimum_model_degree, { 'diagonal', 'quadratic' } ) )
         minimum_model_degree = 'linear';
      end
   elseif ( strcmp( maximum_model_degree, 'diagonal' ) )
      if ( strcmp( minimum_model_degree, 'quadratic' ) )
         minimum_model_degree = 'diagonal';
      end
   end	 
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %   The elemental-form case
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   if ( elemental_form )

      [ xsearch, fsearch, nevalss, exc, x_hist, f_hist, el_hist ] =                        ...
           bfoss_search_step_interp_model_sf( level, f, xbest, max_or_min, xincr, x_hist,  ...
	                            f_hist, xtype, xlower, xupper, el_hist, model_mode,    ...
				    minimum_model_degree, maximum_model_degree,            ...
                                    kappa_ill, toofarlimit, poisedupper, farthr,           ...
				    TR_init_radius, TR_min_radius, TR_max_radius,          ...
				    TR_accuracy, TR_eta1, TR_eta2, TR_alpha1,TR_alpha2,    ...
                                    TR_solver_step, TR_solver_Lagr, TR_PTCG_rel_acc,       ...
				    TR_PTCG_abs_acc, TR_PTCG_maxcgits, term_order, epsilon,...
				    min_step_length, verbosity );

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %   The non-partially-separable case
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   else

     [ xsearch, fsearch, nevalss, exc, x_hist, f_hist ] =                                  ...
           bfoss_search_step_interp_model( level, f, xbest, max_or_min, xincr, x_hist,     ...
	                                   f_hist, xtype, xlower, xupper, model_mode,      ...
				           minimum_model_degree, maximum_model_degree,     ...
					   kappa_ill, toofarlimit, poisedupper, farthr,    ...
					   TR_init_radius, TR_min_radius, TR_max_radius,   ...
				 	   TR_accuracy, TR_eta1, TR_eta2, TR_alpha1,       ...
					   TR_alpha2, TR_solver_step, TR_solver_Lagr,      ...
					   TR_PTCG_rel_acc, TR_PTCG_abs_acc,               ...
					   TR_PTCG_maxcgits, term_order, epsilon,          ...
					   min_step_length, verbosity );
           
   end
   
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function  [ xsearch, fsearch, nevalss, exc, x_hist, f_hist ] =                             ...
          bfoss_search_step_interp_model( level, f, xbest, max_or_min, xincr, x_hist,      ...
	                                  f_hist, xtype,  xlower, xupper, model_mode,      ...
				          minimum_model_degree, maximum_model_degree,      ...
					  kappa_ill, toofarlimit, poisedupper, farthr,     ...
					  TR_init_radius, TR_min_radius, TR_max_radius,    ...
					  TR_accuracy, TR_eta1, TR_eta2, TR_alpha1,        ...
					  TR_alpha2, TR_solver_step, TR_solver_Lagr,       ...
					  TR_PTCG_rel_acc, TR_PTCG_abs_acc,                ...
					  TR_PTCG_maxcgits, term_order, epsilon,           ...
					  min_step_length, verbosity )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Compute a tentative improved point from an interpolation model using available points,
%  by minimizing this model in a trust region whose radius is remembered from one call to
%  the next and updated.
%
%  INPUT:
%
%    level          : the level of the current optimization (in BFO),
%    f              : is the handle to the objective function,
%    xbest          : is the vector of the current best point,
%    max_or_min     : 'min' or 'max' depending on whether minimization or maximization
%    xincr          : the current mesh size needed to compute the current trust-region radius
%    x_hist         : x_hist is an (n x min(nfcalls,l_hist)) array whose columns contain the 
%                     min(nfcalls,l_hist)) points at which f(x) has been evaluated last
%    f_hist         : is an array of length nfcalls containing the function values associated 
%                     with the columns of x_hist,
%    xtype          : a string of length n, defining the type of the variables,
%    xlower         : is the vector of lower bounds,
%    xupper         : is the vector of upper bounds,
%    model_mode     : the technique used to define the relation between the number
%                     of data points and the number of monomial basis elements for
%                     constructing the model:
%                     'subbasis': the number of elements of the monomial basis is equal
%                                 the number of interpolation points (monomials are
%                                 ordered as follows: constant term, linear terms,
%                                 diagonal quadratic terms, and finally off-diagonal
%                                 quadratic terms by increasing subdiagonals.
%                     'minell2' : the number of elements of the monomail basis is
%                                 always equal to the number needed for a full quadratic,
%                                 the (possibly underdetermined) linear system being then
%                                 solved in the least-squares sense.
%    minimum_model_degree: the minimum degree of the model
%                     'minimal'  : at least two basis monomials are used,
%                     'linear'   : the constant and the n linear monomials are used
%                     'diagonal' : + the diagonal quadratic terms
%                     'quadratic': + the off-diagonal quadratic terms
%    maximum_model_degree : the maximum degree of the model:
%                     'linear'   : the constant and the n linear monomials are used
%                     'diagonal' : + the diagonal quadratic terms
%                     'quadratic': + the off-diagonal quadratic terms
%    kappa_ill      : a positive real number specifying the maximum consitioning allowed
%                     for the interpolation matrix
%    toofarlimit    : a real number larger than 1 specifying the multiple of the TR radius
%                     beyond which points are considered irrelevant for interpolation
%    poisedupper    : a positive real number specifying an upper bound on the Lagrange
%                     polynomial in the trust region for declaring the interpolation poised
%    farthr         : a real (>1) specifyin the minimum improvement in poisedness
%                     required for including points in the history in the interpolation set
%    TR_init_radius : the initial trust-region-radius, or a negative number if it should
%                     be chosen as norm( xincr, 'inf')
%    TR_min_radius  : a positive real specifying the trust-region radius under which no
%                     search step is computed
%    TR_max_radius  : a positive real specifying the maximal trust-region radius
%    TR_accuracy    : a positive real specifying the accuracy at which the trust-region
%                     constraint must be satisfied
%    TR_eta1        : the threshold on rho for successful iterations
%    TR_eta2        : the threshold on rho for very successful iterations
%    TR_alpha1      : the TR expansion factor at very successful iterations
%    TR_alpha2      : the TR contraction factor at unsuccessful iterations
%    TR_solver_step : the TR solver for computing the search step ('more-sorensen' or 'ptcg')
%    TR_solver_Lagr : the TR solver for optimizing the Lagrange polynomials
%                     ('more-sorensen' or 'ptcg')
%    TR_PTCG_rel_acc: the relative projected gradient accuracy for PTCG termination
%    TR_PTCG_abs_acc: the absolute projected gradient accuracy for PTCG termination
%    TR_PTCG_maxcgits: the maximum number of CG iterations in PTCG relative to problem size
%    term_order     : the order of the critical point requested on the model
%                     for the search-step function to indicate termination to BFO:
%                     '1rst' : checks that the model's gradient norm is below epsilon
%                     '2nd'  : additionnaly checks that the smallest eigenvalue of
%                              the models' Hessian is above -sqrt( epsilon );
%    epsilon        : the accuracy used for maodel-based termination
%    min_step_length: the  minimum length of the step for computing  the trial
%                     function value,
%    verbosity      : a string for the level of verbosity.  Possible values are
%                     'silent', 'low', 'medium', 'debug'.
%
%  OUTPUT:
%  
%    xsearch : is an array of length n containing the point returned by the 
%              user as a tentative improved iterate,
%    fsearch : the associated objective function value f(xsearch)
%    nevalss : the number of function evaluations performed internally by the
%              search-step function.
%    exc     : the search-step exit code.
%              exc =  3: seemingly at optimum: the search-step function has determined
%                        that the base point appears to be optimal (xsearch and fsearch
%                        are to be ignored in this case);
%              exc =  2: normal successful return: a search-step xsearch and associated
%                        function value fsearch has been found, which produces significant
%                        objective function improvement;
%              exc =  1: normal unusuccessful return: a  search-step xsearch and associated
%                        fsearch have been calculated, but do not produce significant
%                        objective function improvement;
%              exc  = 0: the search step function has found a point xsearch and a value
%                        fsearch, but leaves to BFO the decision to use it or not;
%              exc = -1: the search-step function could not reliably compute a search step,
%                        and the returned xsearch and fsearch should be ignored;
%              exc = -2: the objective function returned a NaN or infinite value: the
%                        returned xsearch and fsearch should be ignored;
%              exc = -3: the user has requested termination: the returned xsearch and
%                        fsearch should be ignored.
%    x_hist  : is the updated array or cell array to which is now appended the set of
%              vectors at which the objective function has been evaluated
%              (xsearch = x_hist( 1:n,end)).
%    f_hist  : is the updated array of function values (f_search = f_hist(end)).
%
%  DEPENDENCIES : bfoss_compute_interpolation_set, bfoss_gH_from_P, bfoss_solve_TR_MS_bc
%
%  PROGRAMMING: M. Porcelli and Ph. Toint , March to October 2017 (this version 2 XII 2017)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  The function maintains a trust-region radius per level (in the case of multilevel
%  optimization) in the vector tr_radius.  Because it has to be remembered from one
%  call to the next, it is defined as a persistent variable.  This also means that
%  that it has to be cleanup explicitly with the function, which is done when a call
%  is made with level <0.

persistent tr_radius

%  If, at the previous call, the model was excellent and at its minimizer and the
%  base point has not changed, there is no reason to re-model, and the search
%  step can be skipped. To allow this the ratio of achieved to predicted reduction,
%  the occurence of a useless model and the corresponding iterate are remembered.

persistent at_min
persistent xprev

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Internal option

TR_restart               =  1;   % Whether or not the trust-region is restarted when it
                                 % becomes too small (< TR_min_radius)

%  Initialize output variables.

xsearch  = [];
fsearchi = [];
nevalss  =  0;
fsearch  = [];
exc      = -1;

%  Handle the trust-region radius and the previous point (as persistent variables).

if ( isempty( tr_radius ) )
   if ( TR_init_radius <= 0 )
      tr_radius = norm( xincr, 'inf' );
   else
      tr_radius = TR_init_radius;
   end
end
if ( isempty( xprev ) )
   xprev{ 1 } = [];
end

%  If the function is called with level < 0, this indicates that
%  all persistent variables must be cleared for imminent exit.  

if ( level < 0 )
   if ( strcmp( verbosity, 'debug' ) )
      disp( ' BFOSS msg: cleaning up persistent variables (NOPS).' )
   end
   clear tr_radius;
   clear xprev;
   clear at_min
   return

%  Otherwise, define one trust-region radius and xprev per level.

elseif ( level > length( tr_radius ) )
   tr_radius    = [ tr_radius, 1];
   xprev{level} = [];
end

%  Set dimensions of the problem.

n     = size( x_hist, 1 );      % dimension of the space
lhist = length( f_hist );       % number of already evaluated points available

%  Exit if the trust-region radius is too small.

if ( tr_radius( level ) < TR_min_radius )
   Delta = norm( xincr, 'inf' );
   if ( TR_restart && Delta > TR_min_radius )
      if ( ismember( verbosity , { 'medium', 'debug'} ) )
         disp( ' BFOSS wrn: tr_radius is too small. Restarting trust region.' )
      end
      tr_radius( level ) = Delta;
      
      %  Make sure the last vector of x_hist is  xbest before truncating the history.

      if ( norm( xbest - x_hist(:,lhist ) ) > 1e-15 )
         found = 0;%D
         for ih = lhist-1:-1:1
            if ( norm( xbest - x_hist(:,ih) ) <= 1e-15 )
	       ftmp          = f_hist( ih );
	       x_hist(:,ih)  = x_hist(:,end);
	       f_hist( ih )  = f_hist( end );
	       x_hist(:,end) = xbest;
	       f_hist( end ) = ftmp;
    	       found         = 1;%D
	    end
         end
	 if ( ~found )%D
	    ' BASE POINT NOT FOUND'%D
	    keyboard%D
	 end%D
      end
      lhist = 1;
   else
      if ( ismember( verbosity , { 'medium', 'debug'} ) )
          disp( ' BFOSS wrn: tr_radius is too small. Terminating the search-step.' )
         disp( [ ' -------------------------------------------------------- ' ] )
      end
      exc = -1;
      return
   end
end

%  Exit if the the model was excellent at a previous call and the base
%  point has not changed.

if ( ~isempty( at_min         ) &&  at_min &&                                              ...
     ~isempty( xprev{ level } ) && norm( xprev{ level } - xbest ) <= eps )
   msg = ' BFOSS msg: Seemingly at minimizer. Terminating the search-step.';
   if ( ismember( verbosity , { 'medium', 'debug'} ) )
      disp( msg )
      disp( [ ' -------------------------------------------------------- ' ] )
   end
   exc = 3;
   return
end

%  Verify history (for debugging purposes).

if( strcmp( verbosity, 'debug' ) )
   dopause = 0;
   for jj  = 1:size( x_hist, 2 )
      error = abs( f( x_hist(:, jj ) ) - f_hist( jj ) );
      if ( error > 1e-14 )
         disp( [ ' BFOSS err: error of ', num2str( error ), ' for the f value of x_hist( ',...
	          int2str( jj ), ') on entry' ] )
         dopause = 1;
      end
   end
   if ( dopause )
       pause
   else
      disp( 'BFOSS dbg: x_hist and f_hist are coherent on entry.' )
   end
end

%  Retrieve the indices of continuous variables.

indfree = find( xtype == 'c' );

%  Compute the number of degrees of freedom.

nfree = length( indfree );
nfix  = n - nfree;

%  Terminate the search step without a new trial point if there are no free variables.
%  Otherwise reset constants if the number of degrees of freedom is not equal
%  to the dimension of the space.

if ( nfix > 0 )
   if ( nfree <= 0 )
      msg = ' BFOSS msg: no free variables. Terminating the search step.';
      if ( ismember( verbosity ,  { 'medium', 'debug' } ) )
         disp( msg );
         disp( [ ' -------------------------------------------------------- ' ] )
      end
      exc = -1;
      return 
   end
   n = nfree;
end

%  Compute the minimum and maximum degrees.

pquad = ( ( n + 1 ) * ( n + 2 ) ) / 2;
switch( minimum_model_degree )
case  'linear'
   min_model_degree = n + 1;
case  'diagonal'
   min_model_degree = 2 * n + 1;
case  'quadratic'
   min_model_degree = pquad;
case  'minimal'
   min_model_degree = 2;
otherwise
   if ( ismember( verbosity, { 'medium', 'debug' } ) )
     disp( ' BFOSS wrn: wrong value of input parameter minimum_model_degree. Default used.' )
   end
   min_model_degree = n + 1; %default
end

switch( maximum_model_degree )
case  'linear'
   max_model_degree = n + 1;
case  'diagonal'
   max_model_degree = 2 * n + 1;
case  'quadratic'
   max_model_degree = pquad;
otherwise
   if ( ismember( verbosity, { 'medium', 'debug' } ) )
     disp( ' BFOSS wrn: wrong value of input parameter maximum_model_degree. Default used.' )
   end
   max_model_degree = pquad; %default
end

%  Check the bounds and reduce the tr_radius if there is no sufficient space
%  between the bounds.

for j = 1:nfree

    % Check difference between bounds.
    
    temp( indfree( j ) ) = xupper( indfree( j ) ) - xlower( indfree(j) );
    if ( temp( indfree( j ) ) < 2 * tr_radius( level ) )
        tr_radius( level ) = temp( indfree(j)  ) / 2;
        if ( strcmp( verbosity , 'debug' ) )
            disp( [ ' BFOSS msg: Range between lower and upper bound of component ',       ...
                int2str( indfree( j ) ), ' is less than 2*Delta !! New Delta = ',          ...
                num2str( tr_radius( level ) ) ] );
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Build a poised sample set for continuous variables from the available points.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( ismember( verbosity, { 'medium', 'debug' } ) )
   disp( [ ' BFOSS msg: Number of free variables               = ', int2str( nfree )] );
   disp( [ ' BFOSS msg: Dof of the initial model               = ',                        ...
           int2str( min_model_degree ) ] );
   disp( [ ' BFOSS msg: Dof of the quadratic model             = ', int2str( pquad ) ]);
end
 
[ Y, fY, x_new, f_new, nevalss, exc, Minv, scale ] =                                       ...
     bfoss_compute_interpolation_set( f, xbest, max_or_min, x_hist(:,end-lhist+1:end),     ...
                                      f_hist(end-lhist+1:end), xlower, xupper,             ...
	                              model_mode, min_model_degree, max_model_degree,      ...
				      kappa_ill, toofarlimit, poisedupper, farthr,         ...
				      TR_accuracy, TR_solver_Lagr, TR_PTCG_rel_acc,        ...
				      TR_PTCG_abs_acc, TR_PTCG_maxcgits, indfree, nevalss, ...
				      tr_radius( level ), verbosity );
p1 = size( Y, 2 );

if ( ismember( verbosity, { 'medium', 'high', 'debug' } ) )
   disp( [ ' BFOSS msg: Number of interpolation points         = ', int2str( p1 ) ] );
end
                                        
%  Terminate the search-step function if a sufficiently poised sample set
%  cannot be built from the available data.

if ( exc < 0 )
   if ( ismember( verbosity , { 'medium', 'debug' } ) )
      disp( ' BFOSS err: interpolation set with points with infinite f-value')
      disp( ' or insufficient poisedness. Terminating search_step_interp_model.' );
   end
   return
end
                              
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Compute the associated polynomial interpolation models.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Compute the gradient and Hessian of the polynomial defining the interpolation model
%  (in the free continuous variables).

[ gfx, Hfx ] = bfoss_gH_from_P( ( Minv * fY' ).*scale, n );

%  Compute the norm of the projected gradient of the model and exit if
%  the base point optimizes the model.

at_min = 0;
normg  = 0;
for  i = 1:nfree
   gi  = gfx(i);
   if gi < 0
      gp(i) = -min( abs( xupper( i ) - xbest( i ) ), -gi );
   else
      gp(i) =  min( abs( xlower( i ) - xbest( i ) ),  gi );
   end
   normg = max( normg, abs( gp( i ) ) );
end
if ( normg <= epsilon )
   if ( strcmp( term_order, '1rst' ) || strcmp( max_model_degree, 'linear' )      ||       ...
        ( strcmp( max_or_min, 'min' ) && min( eig( Hfx ) ) >= - sqrt( epsilon ) ) ||       ...
        ( strcmp( max_or_min, 'max' ) && max( eig( Hfx ) ) <=   sqrt( epsilon ) )   )
      if ( ismember( verbosity , { 'low', 'medium', 'debug' } ) )
         disp( [' BFOSS msg: Total number of function evaluations   = ',                   ...
             num2str( nevalss ) ] )
         disp(  ' BFOSS msg: Seemingly at solution. Terminating the search step.' )
         disp( [ ' ------------------------------------------------------------ ' ] )
      end
      at_min  =  1;
      if ( p1 >= n+1 )
         exc  =  3;
      else
         exc  = -1;
      end
      return
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Compute the minimizer of the model over the intersection between the 2-norm 
%      trust-region and the box [xlower, xupper].
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Compute the box for the trust-region step.

ll = xlower( indfree ) - xbest( indfree );
uu = xupper( indfree ) - xbest( indfree );

%  Solve the trust-region subproblem, using either the Moré-Sorensen algorithm or
%  preconditioned truncated conjugate-gradient.

switch ( TR_solver_step )
case 'more-sorensen'
   farbounds = ( min( min( [ uu'; -ll' ] ) ) >= tr_radius( level ) );
   switch( max_or_min )
   case 'min'
      if ( farbounds )
         [ pstep, ~, msg, ~ ] =                                                            ...
	   bfoss_solve_TR_MS(    nfree,  gfx,  Hfx, tr_radius( level ), TR_accuracy );
         value =  gfx' * pstep + 0.5 * pstep' * Hfx * pstep;
      else
         [ pstep, value, msg ] =                                                           ...
           bfoss_solve_TR_MS_bc( nfree,  gfx,  Hfx, ll, uu, tr_radius( level ), TR_accuracy );
      end
case 'max'
      if ( farbounds )
         [ pstep, ~, msg, ~ ] =                                                            ...
	   bfoss_solve_TR_MS(    nfree, -gfx, -Hfx, tr_radius( level ), TR_accuracy );
         value = -gfx' * pstep - 0.5 * pstep' * Hfx * pstep;
      else
         [ pstep, value, msg ] =                                                           ...
           bfoss_solve_TR_MS_bc( nfree, -gfx, -Hfx, ll, uu, tr_radius( level ), TR_accuracy );
         value = -value;
      end
   end
case 'ptcg'
   uu = min( [ uu';  (tr_radius(level)/sqrt(nfree))*ones(1,nfree) ] )';
   ll = max( [ ll'; -(tr_radius(level)/sqrt(nfree))*ones(1,nfree) ] )';
   switch( max_or_min )
   case 'min'
      [ pstep , ~, cgits, value ] =                                                        ...
            bfoss_projected_tcg( nfree, gfx, Hfx, ll, uu,                                  ...
	                         TR_PTCG_maxcgits * nfree, TR_PTCG_rel_acc, TR_PTCG_abs_acc );
   case 'max'
      [ pstep , ~, cgits, value ] =                                                        ...
            bfoss_projected_tcg( nfree, -gfx, -Hfx, ll, uu,                                ...
	                         TR_PTCG_maxcgits * nfree, TR_PTCG_rel_acc, TR_PTCG_abs_acc );
      value = -value;
   end
end
norms = norm( pstep );

if ( strcmp( verbosity , 'debug' )   )        
   disp( [' BFOSS msg: ', msg ] )
end

%  Check if the norm of the computed step is negligible.

if ( norms <= min_step_length * norm( xbest( indfree ) ) )
   if ( ismember( verbosity , { 'medium', 'debug' } ) )
      disp( [ ' BFOSS msg: Total number of function evaluations   = ', num2str( nevalss ) ] )
      disp(   ' BFOSS msg: Negligible step. Terminating the search step.' )
      disp( [ ' -------------------------------------------------------- ' ] )
   end
   exc = -1;
   return
end

%  Compute the new point.

xsearch            = xbest; 
xsearch( indfree ) = xbest( indfree ) + pstep;

%  Check if xsearch is already in x_hist. 

for i = 1 : lhist
    if ( norm( xsearch - x_hist(:,i) )  < eps )
      
       if ( ismember( verbosity , { 'medium', 'debug' } )  ) 
          msg = ' BFOSS msg: xsearch has been already evaluated.';
          disp( msg );
       end
       fsearch = f_hist(i);
       
       % Update x_hist and f_hist
       
       x_hist = [ x_hist, x_new ];
       f_hist = [ f_hist, f_new ];
       return
       exc    = 0;
    end
end

%  Otherwise, evaluate f at xsearch and update the x_hist and f_hist variables.

fsearch = f( xsearch );   %  This is the call to the objective function
nevalss = nevalss + 1;

x_hist = [ x_hist, x_new, xsearch ];
f_hist = [ f_hist, f_new, fsearch ];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Update the trust-region radius.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rho = ( f(xbest) - fsearch + 1e-15 ) / ( 1e-15 - value );   % ared / prered

old_tr_radius  = tr_radius( level ) ;

xprev{ level } = xsearch;

if ( rho < TR_eta1 )                                        % 'unsuccessful tr'
   exc = 1;
%   if ( p1 < pquad )
%      tr_radius( level ) = max( eps, sqrt( TR_alpha2 ) * norms);
%   else
      tr_radius( level ) = max( eps, TR_alpha2 * norms );
%   end
elseif ( rho > TR_eta2 )                                    % 'successful tr'
   exc = 2;
   tr_radius( level ) = min( TR_alpha1 * tr_radius( level ), TR_max_radius );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Printing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( ismember( verbosity , { 'low', 'medium', 'debug' } ) )
    if ( strcmp( verbosity, 'low' ) )
       disp( ' ' );
    end
    disp( [ ' BFOSS msg: model(xsearch) = ', num2str(f(xbest)+value), ', f(xsearch) = ',   ...
	    num2str( fsearch ) ] )
    disp( [ ' BFOSS msg: Current trust-region radius            = ',                       ...
            num2str( old_tr_radius ) ] );
    disp( [ ' BFOSS msg: Achieved to predicted reduction (rho)  = ', num2str( rho ) ] );
    disp( [ ' BFOSS msg: New trust-region radius                = ',                       ...
            num2str( tr_radius( level) ) ] );
    disp( [ ' BFOSS msg:                ||step||                = ',                       ...
            num2str( norms ) ] );
    disp( [ ' BFOSS msg:           ||gradmodel||                = ',                       ...
            num2str( normg( level ) ) ] );
    disp( [ ' BFOSS msg: Total number of function evaluations   = ', num2str( nevalss ) ] )
    disp( [ ' ------------------------------------------------------------- ' ] )
end

%  Verify history (for debugging purposes).

if( strcmp( verbosity, 'debug' ) )
   dopause = 0;
   for jj  = 1:size( x_hist, 2 )
      error = abs( f( x_hist(:, jj ) ) - f_hist( jj ) );
      if ( error > 1e-14 )
         disp( [ ' BFOSS err: error of ', num2str( error ), ' for the f value of x_hist( ',...
	          int2str( jj ), ') on exit' ] )
         dopause = 1;
      end
   end
   if ( dopause )
       pause
   else
      disp( 'BFOSS dbg: x_hist and f_hist are coherent on exit.' )
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [ xsearch, fsearchi, nevalss, exc, x_hist, f_hist, el_hist ] =                    ...
         bfoss_search_step_interp_model_sf( level, f, xbest, max_or_min, xincr, x_hist,    ...
	                            f_hist, xtype, xlower, xupper, el_hist,                ...
				    model_mode, minimum_model_degree, maximum_model_degree,...
				    kappa_ill, toofarlimit, poisedupper, farthr,           ...
				    TR_init_radius, TR_min_radius, TR_max_radius,          ...
				    TR_accuracy, TR_eta1, TR_eta2, TR_alpha1, TR_alpha2,   ...
                                    TR_solver_step, TR_solver_Lagr, TR_PTCG_rel_acc,       ...
				    TR_PTCG_abs_acc, TR_PTCG_maxcgits, term_order, epsilon,...
				    min_step_length, verbosity )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Compute a tentative improved point from an interpolation model using available points,
%  by minimizing this model in a trust region whose radius is remembered from one call to
%  the next and updated.
%  This version exploits the elemental-form structure of the objective function to build
%  a model with the same structure.
%
%  INPUT:
%
%    level          : the level of the current optimization (in BFO),
%    f              : is the cell of handles of the element functions,
%    xbest          : is the vector of the current best point,
%    max_or_min     : 'min' or 'max' depending on whether minimization or maximization,
%    xincr          : the current mesh size needed to compute the current trust-region radius
%    x_hist         : x_hist is an (n x min(nfcalls,l_hist)) array whose columns contain the 
%                     min(nfcalls,l_hist)) points at which f(x) has been evaluated last
%    f_hist         : is an array of length nfcalls containing the function values associated 
%                     with the columns of x_hist,
%    xtype          : a string of length n, defining the type of the variables,
%    xlower         : is the vector of lower bounds,
%    xupper         : is the vector of upper bounds, 
%    el_hist        : is a cell array of length equal to length( prob ) 
%                    (in extensive formulation) or to length( prob.objf ) (in 
%                    (in condensed formulation), whose i-th entry is a struct
%                    with fields
%                    eldom: the indices of the variables occuring in the i-th
%                           element function
%                    xel  : a matrix/cell of vector states, whose columns/entries contain
%                           the last l_hist points (in the domain of the i-th element 
%                           function) at which this element function has been evaluated
%                    fel  : the corresponding values of the i-th element function
%                    xbest: the projection of the current best point of the domain of
%                           the i-th element, in vector form
%                    fbest: the corresponding value (the i-th element function evaluated 
%                           at xbest).
%    model_mode     : the technique used to define the relation between the number
%                     of data points and the number of monomial basis elements for
%                     constructing the model:
%                     'subbasis': the number of elements of the monomial basis is equal
%                                 the number of interpolation points (monomials are
%                                 ordered as follows: constant term, linear terms,
%                                 diagonal quadratic terms, and finally off-diagonal
%                                 quadratic terms by increasing subdiagonals.
%                     'minell2' : the number of elements of the monomail basis is
%                                 always equal to the number needed for a full quadratic,
%                                 the (possibly underdetermined) linear system being then
%                                 solved in the least-squares sense.
%    minimum_model_degree: the minimum degree of the polynomial model:
%                     'minimal'  : at least two basis monomials are used,
%                     'linear'   : the constant and the n linear monomials are used
%                     'diagonal' : + the diagonal quadratic terms
%                     'quadratic': + the off-diagonal quadratic terms
%    maximum_model_degree: the maximum degree of the model:
%                     'linear'   : the constant and the n linear monomials are used
%                     'diagonal' : + the diagonal quadratic terms
%                     'quadratic': + the off-diagonal quadratic terms
%    kappa_ill      : a positive real number specifying the maximum consitioning allowed
%                     for the interpolation matrix
%    toofarlimit    : a real number larger than 1 specifying the multiple of the TR radius
%                     beyond which points are considered irrelevant for interpolation
%    poisedupper    : a positive real number specifying an upper bound on the Lagrange
%                     polynomial in the trust region for declaring the interpolation poised
%    farthr         : a real (>1) specifyin the minimum improvement in poisedness
%                     required for including points in the history in the interpolation set
%    TR_init_radius : the initial trust-region-radius, or a negative number if it should
%                     be chosen as norm( xincr, 'inf')
%    TR_min_radius  : a positive real specifying the trust-region radius under which no
%                     search step is computed
%    TR_max_radius  : a positive real specifying the maximal trust-region radius
%    TR_accuracy    : a positive real specifying the accuracy at which the trust-region
%                     constraint must be satisfied
%    TR_eta1        : the threshold on rho for successful iterations
%    TR_eta2        : the threshold on rho for very successful iterations
%    TR_alpha1      : the TR expansion factor at very successful iterations
%    TR_alpha2      : the TR contraction factor at unsuccessful iterations
%    TR_solver_step : the TR solver for computing the search step ('more-sorensen' or 'ptcg')
%    TR_solver_Lagr : the TR solver for optimizing the Lagrange polynomials
%                     ('more-sorensen' or 'ptcg')
%    TR_PTCG_rel_acc: the relative projected gradient accuracy for PTCG termination
%    TR_PTCG_abs_acc: the absolute projected gradient accuracy for PTCG termination
%    TR_PTCG_maxcgits: the maximum number of CG iterations in PTCG relative to problem size
%    term_order     : the order of the critical point requested on the model
%                     for the search-step function to indicate termination to BFO:
%                     '1rst' : checks that the model's gradient norm is below epsilon
%                     '2nd'  : additionnaly checks that the smallest eigenvalue of
%                              the models' Hessian is above -sqrt( epsilon );
%    epsilon        : the accuracy used for maodel-based termination
%    min_step_length: the  minimum length of the step for computing  the trial
%                     function value,
%    verbosity      : a string for the level of verbosity.  Possible values are
%                     'silent', 'medium', 'debug'.
%
%  OUTPUT:
%  
%    xsearch : is an array or a vector state of length n containing the point returned by the 
%              user as a tentative improved iterate,
%    fsearchi: a vector containing the values of the element functions at xsearch, such that
%              the value of the complete objective function at xsearch is sum(fsearchi)
%    nevalss : the number of function evaluations performed internally by the search-step
%              function. 
%    exc     : the search-step exit code.
%              exc =  3: seemingly at optimum: the search-step function has determined
%                        that the base point appears to be optimal (xsearch and fsearch
%                        are to be ignored in this case);
%              exc =  2: normal successful return: a search-step xsearch and associated
%                        function value fsearch has been found, which produces significant
%                        objective function improvement;
%              exc =  1: normal unusuccessful return: a  search-step xsearch and associated
%                        fsearch have been calculated, but do not produce significant
%                        objective function improvement;
%              exc  = 0: the search step function has found a point xsearch and a value
%                        fsearch, but leaves to BFO the decision to use it or not;
%              exc = -1: the search-step function could not reliably compute a search step,
%                        and the returned xsearch and fsearch should be ignored;
%              exc = -2: the objective function returned a NaN or infinite value: the
%                        returned xsearch and fsearch should be ignored;
%              exc = -3: the user has requested termination: the returned xsearch and
%                        fsearch should be ignored.
%    x_hist  : is the updated array or cell array to which is now appended the set of
%              vectors at which the objective function has been evaluated
%              (xsearch = x_hist( 1:n,end)).
%    f_hist  : is the updated array of function values (f_search = f_hist(end)).
%    el_hist : is the cell array containing the updated structs (for all element function 
%              evaluations during the search-step computation).
%
%  DEPENDENCIES : bfoss_compute_interpolation_set, bfoss_repair_Y, bfoss_gH_from_P
%
%  PROGRAMMING: M. Porcelli and Ph. Toint, March to October 2017 (this version 2 XII 2017).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  The function maintains a trust-region radius per level (in the case of multilevel
%  optimization) in the vector tr_radius.  Because it has to be remembered from one
%  call to the next, it is defined as a persistent variable.  This also means that
%  that it has to be cleanup explicitly with the function, which is done when a call
%  is made with level <0.

persistent tr_radius

%  If, at the previous call, the model was excellent and at its minimizer and the
%  base point has not changed, there is no reason to re-model, and the search
%  step can be skipped. To allow this the ratio of achieved to predicted reduction,
%  the occurence of a useless model and the corresponding iterate are remembered.

persistent at_min
persistent xprev

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Internal option

TR_restart  =  1;   % Whether or not the trust-region is restarted when it
                    % becomes too small (< TR_min_radius)

%  Initialize output variables.

xsearch   = [];
fsearchi  = [];
nevalss   =  0;
exc       = -1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( ismember( verbosity , { 'medium', 'debug'} ) )
   disp( [ ' ------------------------------------------------------------- ' ] )
   disp( ' BFOSS '  )
end

%  Handle the trust-region radius and the previous point (as persistent variables).

if ( isempty( tr_radius ) )
   if ( TR_init_radius <= 0 )
      tr_radius = norm( xincr, 'inf' );
   else
      tr_radius = TR_init_radius;
   end
end
if ( isempty( xprev ) )
   xprev{ 1 } = Inf * xbest;
end

%  If the function is called with level < 0, this indicates that
%  all persistent variables must be cleared for imminent exit.  

if ( level < 0 )
   if ( strcmp( verbosity, 'debug' ) )
      disp( ' BFOSS msg: cleaning up persistent variables (PS).' )
   end
   clear tr_radius;
   clear xprev;
   clear at_min
   return

%  Otherwise, define one trust-region radius and xprev per level.

elseif ( level > length( tr_radius ) )
   tr_radius      = [tr_radius, 1];
   xprev{ level } = [];
end

%  Set dimensions of the problem.

N     = size( x_hist, 1 );      % dimension of the full space
ne    = length( el_hist );      % number of element functions
lhist = Inf;                    % number of available already evaluated points

%  Exit or restart the TR if the trust-region radius is too small.

if ( tr_radius( level ) < TR_min_radius )
   Delta = norm( xincr, 'inf' );
   if ( TR_restart && Delta > TR_min_radius )
      if ( ismember( verbosity , { 'medium', 'debug'} ) )
         disp( ' BFOSS wrn: tr_radius is too small. Restarting trust region.' )
      end
      tr_radius( level ) = Delta;

      %  Make sure, for each element iel, the last vector of el_hist{iel}.xel
      %  is the projection of xbest on the element's domain before truncating
      %  the history.

      for iel = 1:ne
         exbest = xbest( el_hist{iel}.eldom );    % the projection of xbest
	 lhiel  = size( el_hist{ iel }.xel, 2 );
	 if ( norm( exbest - el_hist{ iel }.xel(:,lhiel ) ) > 1e-15 )    % not the same
	    found = 0;%D
 	    for ih = lhiel-1:-1:1                 % find proj(xbest) in history
	       if ( norm( exbest - el_hist{ iel }.xel(:,ih) ) <= 1e-15 ) % found it, swap.
	          ftmp = el_hist{ iel }.fel(ih);
	          el_hist{ iel }.xel(:,ih)  = el_hist{ iel }.xel(:,end);
	          el_hist{ iel }.fel( ih )  = el_hist{ iel }.fel( end );
		  el_hist{ iel }.xel(:,end) = exbest;
		  el_hist{ iel }.fel( end ) = ftmp;
		  found = 1;%D
	       end
	    end
	    if ( ~found )%D
	       ' PROJECTED BASE POINT NOT FOUND'%D
	       keyboard%D
	    end%D
	 end
      end
      lhist = 1;
   else
      if ( ismember( verbosity , { 'medium', 'debug'} ) )
         disp( ' BFOSS wrn: tr_radius is too small. Terminating the search-step.' );
         disp( [ ' -------------------------------------------------------- ' ] )
      end
      exc = -1;
      return
   end
end

%  Exit if the the model was excellent at a previous call and the base
%  point has not changed.

if ( ~isempty( at_min )         &&  at_min &&                                              ...
     ~isempty( xprev{ level } ) && norm( xprev{ level } - xbest ) <= eps )
   if ( ismember( verbosity , { 'medium', 'debug'} ) )
      disp( ' BFOSS msg: Seemingly at optimizer. Terminating the search-step.' )
      disp( [ ' -------------------------------------------------------- ' ] )
   end
   exc =  3;
   return
end

%  Initialize the element-wise evaluation counts.

nevalssi = zeros( ne, 1 );

%  Retrieve the indices of continuous variables.

indfree = find( xtype == 'c' );
Nfree   = length( indfree );

%  Check the bounds and reduce the trust-region radius if there is no
%  sufficient space between the bounds.

for j = 1:Nfree

    %  Check difference between bounds.
    
    dist = xupper( indfree(j) ) - xlower( indfree(j) );
    if ( dist < 2 * tr_radius( level ) )
        tr_radius( level ) = dist / 2;
        if ( strcmp( verbosity , 'debug' ) )
            disp( [ ' BFOSS msg: Range between lower and upper bound of component ',       ...
                int2str( indfree( j ) ), ' is less than 2*Delta !! New Delta = ',          ...
                num2str( tr_radius( level ) ) ] );
        end
    end
end

%  Initialize the gradient and the Hessian of the global model.

g = zeros( N, 1 );
H = sparse( N, N );

%  Initialize the cells for new points and new values.

x_new = cell( ne, 1 );
f_new = cell( ne, 1 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                            %
%                                 Loop on all elements                                       %
%                                                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

all_linear = 1;          %  true if all element models are at least linear
for i = 1:ne

   % Retrieve information for the i-th subspace.

   lhisti   = min( lhist, length( el_hist{ i }.fel ) );
   ex_hist  = el_hist{ i }.xel(:,end-lhisti+1:end);  % the projected iterates' history
   ef_hist  = el_hist{ i }.fel(1:end-lhisti+1:end);  % the associated el. funct. values
   if ( isfield( el_hist{ i }, 'eldom' ) )
      eldom = el_hist{ i }.eldom;            % the domain
   else
      eldom = [ 1:N ];
   end
   exbest   = xbest( eldom );                % the projected best iterate
   n        = length( eldom );               % dimension of the subspace
   fi       = @( x )f{ 1 }{ i }( i, x );     % the handle for the i-th element function

   %  Verify the elementwise history (for debugging purposes).
 
   if ( strcmp( verbosity, 'debug' ) )
      dopause = 0;
      for jj = 1:length( ef_hist )
         vjj = f{1}{i}( i, ex_hist(:, jj ) );
         error = abs( vjj - ef_hist(jj) );
         if ( error > 1e-15 )
            disp( [ ' BFOSS err: error of ', num2str( error ),                             ...
	            ' for the f value of el_hist{', int2str( i ),'.x_hist( ',              ...
	            int2str( jj ), ') on entry' ] )
            dopause = 1;
         end
      end
      if ( dopause )
          pause
      else
%         disp( [ 'BFOSS dbg: x_hist and f_hist are coherent for element ', int2str( i ), ...
%                 ' on entry.' ] )
      end
   end

   %  Retrieve the continuous and inactive variables.

   icont = [];
   ii    = 1;
   for j = 1:n
      if ( xtype( eldom(j) ) == 'c' )
         icont( ii ) = j;
	 ii          = ii + 1;
      end
   end

   %  Compute the number of degrees of freedom in the subspace.
    
   nfree    = length( icont );
   nfix     = n - nfree;
   eindfree = icont;

   %  See if there are free variables if the space of i-th domain.
    
   allfixed = 0;
   if ( nfix > 0 )
      if ( nfree == 0 )  
         if ( ismember( verbosity , { 'medium', 'debug' } ) )
            disp( [ ' BFOSS wrn: no free variables in  subspace ', int2str( i ) ] )
         end
         nevalssi( i ) = 0;            
         allfixed      = 1;
      end
      n = nfree;
   end

   if ( ~allfixed )
        
      % Compute the degree of the model

      pquad = ( ( n + 1 ) * ( n + 2 ) ) / 2;
	
      switch ( minimum_model_degree )
      case 'minimal'
         min_model_degree = 2;
      case 'linear'
         min_model_degree = n + 1;
      case 'diagonal'
         min_model_degree = 2 * n + 1;
      case 'quadratic'
         min_model_degree = pquad;
      otherwise
         if ( ismember( verbosity, { 'medium', 'debug' } ) )
            disp( ' BFOSS wrn: wrong value of input parameter minimum_model_degree.' )
	    disp( ' Default used.' )
         end
         min_model_degree = n + 1; % Default
      end
      switch ( maximum_model_degree )
      case 'linear'
         max_model_degree = n + 1;
      case 'diagonal'
         max_model_degree = 2 * n + 1;
      case 'quadratic'
         max_model_degree = pquad;
      otherwise
         if ( ismember( verbosity, { 'medium', 'debug' } ) )
            disp( ' BFOSS wrn: wrong value of input parameter maximum_model_degree.' )
	    disp( ' Default used.' )
         end
         max_model_degree = pquad; % Default
      end

      if ( ismember( verbosity, { 'medium', 'debug' } ) )
         disp( ' -------------------------------------------------------- ' )
         disp( [ ' BFOSS msg: Model ', model_mode, ' for the element function ',           ...
	         int2str( i ) ] );
         disp( [ ' BFOSS msg: Variables:   ', mat2str( eldom' ) ] );
         disp( [ ' BFOSS msg: Dof of the initial model               = ',                  ...
	         int2str( min_model_degree ) ] );
         disp( [ ' BFOSS msg: Dof of the quadratic model             = ',                  ...
	         int2str( pquad ) ] );
      end
        
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %  Build a poised interpolation set for continuous variables
      %  from the available points.
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              

      [ Y, fY, x_new{ i }, f_new{ i }, nevalssi( i ), exc, Minv, scale ] =                 ...
	           bfoss_compute_interpolation_set( fi, exbest, max_or_min, ex_hist,       ...
		                           ef_hist, xlower( eldom ), xupper( eldom ),      ...
				           model_mode, min_model_degree, max_model_degree, ...
				           kappa_ill, toofarlimit, poisedupper, farthr,    ...
					   TR_accuracy, TR_solver_Lagr, TR_PTCG_rel_acc,   ...
					   TR_PTCG_abs_acc, TR_PTCG_maxcgits, eindfree,    ...
					   nevalss, tr_radius( level ), verbosity );
      p1            = size( Y, 2 );
      all_linear    = all_linear && ( p1 >= n+1 );

      if ( ismember( verbosity, { 'medium', 'high', 'debug' } ) )
         disp( [ ' BFOSS msg: Number of interpolation points         = ', int2str( p1 ) ] );
      end

      if ( exc == -1 || exc == -2 )
         if ( ismember( verbosity , { 'medium', 'debug' } ) )
            disp( [ ' BFOSS err: interpolation set with points with infinite ',            ...
               'f-value or insufficient poisedness. Terminating search_step_interp_model.'] );
         end
      elseif ( exc == -3 )
         if ( ismember( verbosity , { 'medium', 'debug' } ) )
            disp( [ ' BFOSS msg: User has required termination.'                           ...
	            ' Exiting search_step_interp_model.'] );
         end
         return
      else
	
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %  Compute the associated polynomial interpolation model.
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

         %  Compute the gradient and Hessian  of the polynomial defining the elementwise
         %  interpolation model.
	
         [ gfx, Hfx ] = bfoss_gH_from_P( ( Minv * fY' ) .* scale, n );

         %  Accumulate them in the overall gradient and Hessian.

         elicont               = eldom( icont );
         g( elicont  )         = g( elicont ) + gfx;
         H( elicont, elicont ) = H( elicont, elicont ) + Hfx;

      end  
   end
end

if ( ismember( verbosity, { 'medium', 'debug' } ) )
   disp( ' ------------------------------------------------------------- ' )
end

%  Compute the norm of the projected gradient of the model and exit if
%  the base point optimizes the model.

at_min = 0;
normg  = 0.0;
for  j = 1:Nfree
   i = indfree( j );
   gi  = g(i);
   if gi < 0
      gp(i) = -min( abs( xupper( i ) - xbest( i ) ), -gi );
   else
      gp(i) =  min( abs( xlower( i ) - xbest( i ) ),  gi );
   end
   normg = max( normg, abs( gp( i ) ) );
end
if ( normg <= epsilon )
   if ( strcmp( term_order, '1rst' ) || strcmp( maximum_model_degree, 'linear' )       ||  ...
        ( strcmp( max_or_min, 'min' ) && min( eigs( H, 1, 'SA' ) )>=-sqrt( epsilon ))  ||  ...
        ( strcmp( max_or_min, 'max' ) && min( eigs( H, 1, 'LA' ) )<= sqrt( epsilon ))    )
      if ( ismember( verbosity , { 'low', 'medium', 'debug' } ) )
         nevalss =  nevalss + sum( nevalssi ) / ne;     
         disp( ' -------------------------------------------------------- ' )
         disp( [' BFOSS msg: number of fevals = ', num2str( nevalss ) ] )
         disp(  ' BFOSS msg: at the optimizer. Terminating the search step.' )
         disp( ' -------------------------------------------------------- ' )
      end
      at_min  =  1;
      if ( all_linear )
         exc  =  3;
      else
         exc  = -1;
      end
      return
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Compute the minimizer of the model over the intersection between the 2-norm 
%      trust-region and the box [ xlower, xupper ].
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Compute the box for the trust-region step.

ll = xlower( indfree ) - xbest( indfree );
uu = xupper( indfree ) - xbest( indfree );

%  Solve the trust-region subproblem, using either the Moré-Sorensen algorithm or
%  preconditioned truncated conjugate-gradient.

switch ( TR_solver_step )
case 'more-sorensen'
   farbounds = ( min( min( [ uu'; -ll' ] ) ) >= tr_radius( level ) );
   switch( max_or_min )
   case 'min'
      if ( farbounds )
         [ pstep, ~, msg, ~ ] =                                                            ...
	   bfoss_solve_TR_MS(    Nfree,  g( indfree ),  H( indfree, indfree),              ...
	                         tr_radius( level ), TR_accuracy );
         value =  g( indfree )' * pstep + 0.5 * pstep' * H( indfree, indfree ) * pstep;
      else
         [ pstep, value, msg ] =                                                           ...
           bfoss_solve_TR_MS_bc( Nfree,  g( indfree ),  H( indfree, indfree ), ll, uu,     ...
	                         tr_radius( level ), TR_accuracy );
      end
   case 'max'
      if ( farbounds )
         [ pstep, ~, msg, ~ ] =                                                            ...
	   bfoss_solve_TR_MS(    Nfree, -g( indfree ), -H( indfree, indfree),              ...
	                         tr_radius( level ), TR_accuracy );
         value = -g( indfree )' * pstep - 0.5 * pstep' * H( indfree, indfree ) * pstep;
      else
         [ pstep, value, msg ] =                                                           ...
           bfoss_solve_TR_MS_bc( Nfree, -g( indfree ), -H( indfree, indfree ), ll, uu,     ...
	                         tr_radius( level ), TR_accuracy );
         value = - value;
      end
   end
case 'ptcg'
   uu = min( [ uu';  ( tr_radius( level ) / sqrt( Nfree ) ) * ones( 1, Nfree ) ] )';
   ll = max( [ ll'; -( tr_radius( level ) / sqrt( Nfree ) ) * ones( 1, Nfree ) ] )';
   switch( max_or_min )
   case 'min'
      [ pstep, msg, cgits, value ] =                                                       ...
            bfoss_projected_tcg( Nfree,  g(indfree),  H( indfree, indfree ), ll, uu,       ...
	                         TR_PTCG_maxcgits * Nfree, TR_PTCG_rel_acc, TR_PTCG_abs_acc );
   case 'max'
      [ pstep, msg, cgits, value ] =                                                       ...
            bfoss_projected_tcg( Nfree, -g(indfree), -H( indfree, indfree ), ll, uu,     ...
	                         TR_PTCG_maxcgits * Nfree, TR_PTCG_rel_acc, TR_PTCG_abs_acc );
      value = -value;
   end
end
norms = norm( pstep );

if ( strcmp( verbosity , 'debug' )   )        
   disp( [' BFOSS msg: ', msg ] )
end

%  Check if the norm of the computed step is negligible.

if ( norms <= min_step_length * norm( xbest( indfree ) ) )
   if ( ismember( verbosity , { 'medium', 'debug' } ) )
      disp( [ ' -------------------------------------------------------- ' ] )
      disp( [' BFOSS msg: Total number of function evaluations  = ', num2str( nevalss ) ] )
      disp(  ' BFOSS msg: Negligible step. Terminating the search step.' )
      disp( [ ' -------------------------------------------------------- ' ] )
   end
   exc = -1;
   return
end

%  Compute the new point.

xsearch            = xbest;
xsearch( indfree ) = xbest( indfree ) + pstep;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  For each element, check if the projection of the trial point on the domain subspace
%  is already in the history.  If it is, retrieve the associated element function value.
%  Otherwise compute this value by calling the i-th element function. Also accumulate
%  the global value of the objective function at xsearch and update elementwise and global
%  histories. Note that the history of best points and values is maintained by BFO, so
%  no action is needed here.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fbest   = 0;      %  the current best value
fsearch = 0;      %  the value at xsearch
xisnew  = 0;      %  is xsearch a new point?

for i = 1:ne

   %  Compute the projection of the trial point.

   eldom = el_hist{i}.eldom;
   xproj = xsearch( eldom );

   %  Is it in the history?

   exhist = el_hist{ i };
   index  = 0;
   for k = 1:size( exhist, 1 )
      if ( norm( xproj - exhist.xel( :, k ) ) < eps  )
 	 index = k;
         break
      end
   end

   %  Update the element history for the new points and values obtained
   %  during the interpolation set computation.

   if ( ~isempty( f_new{ i } ) )
      el_hist{ i }.xel = [ el_hist{ i }.xel x_new{ i } ];
      el_hist{ i }.fel = [ el_hist{ i }.fel f_new{ i } ];
   end
   
   %  If in history, retrieve the element value.

   if ( index )
      fsearchi( i ) = exhist.fel( index );

   %  Not in history: compute a new element value.
   
   else

      xisnew        = 1;
      fsearchi( i ) = f{1}{i}( i, xproj );
      nevalssi( i ) = nevalssi( i ) + 1;

      %  Include the new point and value in element history.

      el_hist{ i }.xel = [ el_hist{ i }.xel xproj ];
      el_hist{ i }.fel = [ el_hist{ i }.fel fsearchi( i )];

   end
   
   %  Construct the old and new complete function values.

   fbest   = fbest   + el_hist{ i }.fbest;
   fsearch = fsearch + fsearchi( i );

   % Verify the updated history (for debugging purposes).

   if ( strcmp( verbosity, 'debug' ) )
      dopause = 0;
      for jj = 1:length( ef_hist )
         vjj = f{1}{i}( i, ex_hist(:, jj ) );
         error = abs( vjj - ef_hist(jj) );
         if ( error > 1e-15 )
            disp( [ ' BFOSS err: error of ', num2str( error ),                             ...
                    ' for the f value of el_hist{', int2str( i ),'.x_hist( ',              ...
	              int2str( jj ), ') on exit.' ] )
            dopause = 1;
         end
      end
      if ( dopause )
          pause
      else
%         disp( [ 'BFOSS dbg: x_hist and f_hist are coherent for element ', int2str( i ),  ...
%                 ' on exit.' ] )
      end
   end

end

%  Update the total number of complete function evaluations
%  and the global history.

nevalss =  nevalss + sum( nevalssi ) / ne;
if ( xisnew )
   x_hist = [ x_hist xsearch ];
   f_hist = [ f_hist fsearch ];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Update the trust-region radius.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rho = ( fbest - fsearch + 1e-15 ) / ( 1e-15 - value );

old_tr_radius  = tr_radius( level ); 

xprev{ level } = xsearch;

if ( rho < TR_eta1 )                                       % 'unsuccessful tr'
   exc = 1;
   tr_radius( level ) = max( eps, TR_alpha2 * norms );
elseif ( rho > TR_eta2 )                                   % 'successful tr'
   exc = 2;
   tr_radius( level ) = min( max( TR_alpha1 * norms, tr_radius( level ) ), TR_max_radius );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Printing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( ismember( verbosity , { 'low', 'medium', 'debug' } ) )
   if ( strcmp( verbosity, 'low' ) )
      disp( ' ' )
   end
   disp( [ ' BFOSS msg: model(xsearch) = ', num2str( fbest + value ),', f( xsearch ) = ',  ...
           num2str( fsearch ) ] )
   disp( [ ' BFOSS msg: Current trust-region radius            = ',                        ...
           num2str( old_tr_radius ) ] );
   disp( [ ' BFOSS msg: Achieved to predicted reduction (rho)  = ', num2str( rho ) ] );
   disp( [ ' BFOSS msg: New trust-region radius                = ',                        ...
           num2str( tr_radius( level) ) ] );
   disp( [ ' BFOSS msg:                ||step||                = ', num2str( norms ) ] );
   disp( [ ' BFOSS msg:           ||gradmodel||                = ',                        ...
           num2str( normg( level ) ) ] );
   disp( [ ' BFOSS msg: Total number of function evaluations   = ', num2str( nevalss ) ] )
   disp( ' ------------------------------------------------------------- ' )
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [ Y, fY, x_new, f_new, nevalss, exc, Minv, scale ] =                              ...
         bfoss_compute_interpolation_set( f, xbest, max_or_min, x_hist, f_hist,            ...
		                          xlower, xupper, model_mode,                      ...
					  min_model_degree, max_model_degree,              ...
					  kappa_ill, toofarlimit, poisedupper, farthr,     ...
					  TR_accuracy, TR_solver_Lagr, TR_PTCG_rel_acc,    ...
					  TR_PTCG_abs_acc, TR_PTCG_maxcgits, indfree,      ...
					  nevalss, Delta, verbosity )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Given the set of available pre-evaluated points in (x_hist, f_hist), build a poised
%  set Y of interpolation points around xbest and corresponding function values.
%
%  INPUT:
%
%    f              : is the handle to the objective function
%    xbest          : the current best point
%    max_or_min     : 'min' or 'max' depending on whether minimization or maximization
%    x_hist         : x_hist is an (n x min(nfcalls,l_hist)) array whose columns contain the 
%                     min(nfcalls,l_hist)) points at which f(x) has been evaluated last
%    f_hist         : is an array of length nfcalls containing the function values associated 
%                     with the columns of x_hist
%    xlower         : is the vector of lower bounds on the optimization problem's variables
%    xupper         : is the vector of upper bounds  on the optimization problem's variables
%    model_mode     : the technique used to define the relation between the number
%                     of data points and the number of monomial basis elements for
%                     constructing the model:
%                     'subbasis': the number of elements of the monomial basis is equal
%                                 the number of interpolation points (monomials are
%                                 ordered as follows: constant term, linear terms,
%                                 diagonal quadratic terms, and finally off-diagonal
%                                 quadratic terms by increasing subdiagonals.
%                     'minell2' : the number of elements of the monomial basis is
%                                 always equal to the number needed for a full quadratic,
%                                 the (possibly underdetermined) linear system being then
%                                 solved in the least-squares sense.
%    min_model_degree : the minimum degree of the polynomial which should be
%                     maintained once it has been reached
%                     'minimal'  : at least two basis monomials are used,
%                     'linear'   : the constant and the n linear monomials are used
%                     'diagonal' : + the diagonal quadratic terms
%                     'quadratic': + the off-diagonal quadratic terms
%    max_model_degree : the maximum degree of the model:
%
%                     'linear'   : the constant and the n linear monomials are used
%                     'diagonal' : + the diagonal quadratic terms
%                     'quadratic': + the off-diagonal quadratic terms
%    kappa_ill      : a positive real number specifying the maximum consitioning allowed
%                     for the interpolation matrix
%    toofarlimit    : a real number larger than 1 specifying the multiple of the TR radius
%                     beyond which points are considered irrelevant for interpolation
%    poisedupper    : a positive real number specifying an upper bound on the Lagrange
%                     polynomial in the trust region for declaring the interpolation poised
%    farthr         : a real (>1) specifyin the minimum improvement in poisedness
%                     required for including points in the history in the interpolation set
%    TR_accuracy    : a positive real specifying the accuracy at which the trust-region
%                     constraint must be satisfied
%    TR_solver_Lagr : the TR solver ('more-sorensen' or 'ptcg') for min/maximizing the
%                     Lagrange polynomials
%    TR_PTCG_rel_acc: the relative projected gradient accuracy for PTCG termination
%    TR_PTCG_abs_acc: the absolute projected gradient accuracy for PTCG termination
%    TR_PTCG_maxcgits: the maximum number of CG iterations in PTCG relative to problem size
%    indfree        : is a vector of current indeces of free variables
%    nevalss        : is a scalar for the number of function evaluations in the search step
%                     function, computed so far
%    Delta          : is a scalar specifying the current trust-region radius
%    verbosity      : a string specifying the level of verbosity 
%    
%  OUTPUT:
%
%    Y              : is a array with the interpolation set
%    fY             : is a vector of length mode_degree with objective function value f(Y)
%    x_new          : is an array with new evaluated points in the sample set (by column)
%    f_new          : is a a vector with f-values at new evaluated points (f(xnew))
%    nevalss        : is a scalar with the number of new function evaluations performed 
%    exc            : is the exit condition:
%                     exc =  0: normal return: an interpolation set has been found
%                     exc = -1: no sufficiently poised set can be found
%                     exc = -2: f-value is NaN or meaningless Inf;
%                     exc = -3: termination requested by meaningfull Inf
%                               has bee reached
%    Minv           : the (possibly generalized) of the (possibly regularized)
%                     interpolation matrix
%    poised         : is the poisedness of the interpolation set Y
%    scale          : the current interpolation set scaling
%
%  DEPENDENCIES     : bfoss_build_Minv_of_Y, bfoss_gH_from_P, bfoss_TR_1d,
%                     bfoss_solve_TR_MS, bfoss_solve_TR_MS_bc, bfoss_projected_tcg,
%                     (bfoss_poisedness)
%
%  PROGRAMMING      : M. Porcelli and Ph. Toint, March to October 2017
%                     (this version 27 XII 2017).
%
%  CONDITIONS OF USE: Use at your own risk! No guarantee of any kind given.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


exc         = 0;             % initialize the exit condition to "normal return"

%  Define the verbosity level.

%verbose = 1;
verbose = strcmp( verbosity , 'debug' );

%  Set internal parameters

myinf   = 1e+25;

n       = length( indfree );   
lf      = xlower( indfree );
uf      = xupper( indfree );
lhist   = size(x_hist,2);
x_new   = [];
f_new   = [];
plin    = n + 1;
pquad   = ( plin * ( n + 2 ) ) / 2; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Compute a first interpolation set by taking close points first
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Take as many point as possible in the history while keeping the model at
%  most fully quadratic.

%D%if ( verbose )
%D%  fprintf( [ ' BFOSS msg:',                                                             ...
%D%             ' Compute the interpolation points from  x_hist(1:%d) \n'],lhist );
%D%end

%  Sort the history points by increasing distance from the base point xbest.

dist  = zeros( lhist, 1 ); 
for i = 1:lhist
    dist( i ) = norm( x_hist( :, i ) - xbest );
end
[ sortdist, idist ] = sort( dist );

%  Find the base point xbest in the history (and its associated function value)
%  and choose it as the first interpolation point.

Y    = zeros( n, max_model_degree );
fY   = zeros( max_model_degree );
Y    = x_hist( indfree, idist( 1 ) );    % the base point xbest
%D%if ( norm( Y(:,1) - xbest(indfree) ) > 1e-15 )%D
%D%   ' WRONG BASE'%D
%D%   keyboard%D
%D%end%D
fY   = f_hist( idist( 1 ) );             % that is fbest = f( xbest )
p1   = 1;
dist = zeros( max_model_degree, 1 );
if ( verbose && norm( Y(:,1)-xbest ) > 1e-15 )
  fprintf( ' BFOSS err: The best point is not in the history.  Increase l_hist.' );
end
   
%  Choose the remaining interpolation points as the closest to xbest, avoiding duplicates.
   
for i = 2:lhist
   if ( sortdist( i ) > toofarlimit * Delta )
      break;
   else
      itry      = idist ( i );
      ytry      = x_hist( indfree, itry );
      duplicate = 0;
      for j = 1:p1
         if (  norm( ytry - Y(:,j) ) < 1.e-14 )
            duplicate = 1;
	    break
         end
      end
      if ( ~duplicate )
         p1         = p1 + 1;
         Y( :,p1 )  = ytry;
         fY( p1 )   = f_hist( itry );
         dist( p1 ) = sortdist( i );
         if ( p1 >= min( lhist, max_model_degree ) )
            break
         end
      end
   end
end
Y    = Y( 1:n, 1:p1 );
fY   = fY( 1:p1 );
dist = dist( 1:p1 );
Yh   = x_hist( indfree, idist( p1+1:lhist ) );

%  If not enough points could be found in the history for the initial model,
%  complete the interpolation set by adding random points in the trust-region.

new_init_points = [];
if ( p1 < min_model_degree )
   if ( verbose )
      fprintf(' BFOSS msg: Completing the interpolation set with %d random points\n',      ...
     		min_model_degree - p1 );
   end
   for i = p1+1:min_model_degree
      Y( :, i ) = min ( max( lf, Y( :, 1 ) + Delta*rand( n, 1 ) ), uf );
      dist( i ) = norm( Y( :, i ) - Y( :, 1 ) );
   end
   new_init_points = [ p1+1:min_model_degree ];
   Yh              = [];
end
p1 = size( Y, 2);

%D%if ( verbose )
%D%   disp( [ ' BFOSS msg: Initial number of interpolation points = ', int2str( p1 ) ] )
%D%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Ensure the resulting interpolation matrix is well-conditioned enough for the
%  Lagrange polynomials to make sense. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[ Minv, rankloss, scale ] = bfoss_build_Minv_of_Y( Y, model_mode, kappa_ill );

%  If the matrix had to be regularized, rebuild the matrix progressively to identify
%  the offending points and skip them.

if ( rankloss  && p1 - rankloss < min_model_degree )
%D%   if ( verbose )
%D%      disp( [ ' BFOSS msg: rank drop by ', int2str( nredundant ) ] )
%D%   end
   redundant = [];
   Ytmp  = zeros( n, p1 );
   Ytmp  = Y( :,1 );
   nYtmp = 1;
   for ir = 2:p1
      [ Minv, rankloss, scale ] = bfoss_build_Minv_of_Y( [ Ytmp Y(:,ir) ], model_mode,     ...
                                                         kappa_ill);
      if ( rankloss )
         redundant(end+1) = ir;
      else
         nYtmp = nYtmp + 1;
         Ytmp(:,nYtmp) = Y(:,ir);
      end
   end
%D%   if ( verbose )
%D%      disp( [ ' BFOSS msg: Y rebuilt now uses ', int2str( size( Y, 2 ) ), ' points' ] )
%D%   end

   %  Replace the offending points by random points in the trust region, reevaluating
   %  the rank each time. Do this at most 5 times before giving up.

   for itry = 1:5
      for jr = redundant
         Y( :, jr )      = min ( max( lf, Y( :, 1 ) + rand( n, 1 )/scale(2) ), uf ); 
         new_init_points = union( new_init_points, jr );
	 dist( jr )      = norm( Y( :, jr ) - Y( :, 1 ) );
      end
      [ Minv, rankloss, scale ] = bfoss_build_Minv_of_Y( Y, model_mode, kappa_ill );
      if ( ~rankloss )
         break
      end
   end
   p1eff = p1 - rankloss;
else
   p1eff = p1;
end
p1 = size( Y, 2 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Find the best replacement in the extra (past) points for each
%     of the current points by decreasing order of distance to the base.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

replaced = [];
p2       = size( Yh, 2 );
if ( p2 > 0 )                            %  There are extra points.

   notused = ones( 1, p2 );              %  Extra points have not been used yet.

   %  Sort the distance of the interpolation points to the base point, this time
   %  in descending order.

    [ ~,  idist ] = sort( dist, 'descend' );
   
   for j = 1:p1-1     %  The upper limit p1-1 avoids considering the base point
                      %  for which the distance is zero and therefore last in idist.
    
      %  Determine the index of the next point in Y to be replaced.
      
      jmax = idist( j );

      %  Get the coefficients of the jmax-th Lagrange polynomial and find its gradient
      %  and Hessian.

      [ g, H ] = bfoss_gH_from_P( Minv(:,jmax), n );

      %  Find the best replacement of point jmax in Yh.

      improvement = 0;
      jrepl       = 0;

      for jyh = 1:p2

         %  Skip already used extra points and points outside the trust region.

         v = Yh( :, jyh ) - Y( :, 1 );
         if ( notused( jyh ) && norm( v ) <= Delta )
	 
            %  Evaluate the Lagrange polynomial L(jmax) at Yh(jyh)
	    %  and remember the best choice.     

            v           = v * scale(2);
            Ljmax_value = v' * ( g + 0.5 * ( H * v ) );
            if ( abs( Ljmax_value ) >  improvement )
	       improvement = abs( Ljmax_value );
	       jrepl       = jyh;
	    end
         end
      end
      
      % If the replacement is significant, perform it.

      if ( improvement > farthr )
%D%         if ( verbose )
%D%            disp([ ' replace Y(', int2str( jmax ), ') by Yh(', int2str( jrepl ),        ...
%D%                   ') to improve by a factor ', num2str( improvement ) ] )
%D%         end
         Y( :, jmax ) = Yh( :, jrepl );
         [ Minv, ~, scale ] = bfoss_build_Minv_of_Y( Y, model_mode, kappa_ill );
         replaced = union( replaced, jmax );

         notused( jrepl ) = 0;
      end
   end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%          Perform a loop over at most p1 possible optimal local improvements.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%  Perform (at most) p1 improvement loops.

for  k = 1:p1

   poised = 0;

   %  Define the scaled bounds and the trust-region size, depending on the solver chosen.
   %  If the MS algorithm is used, the TR is imposed in Euclidean norm and the bounds are
   %  kept independent.  If the PTCG is used, the trust-region is defined is suitably scaled
   %  infinity-norm, and the resulting bounds are incorporated in the relative bounds.
   %  Everything is scaled.

   switch ( TR_solver_Lagr )
   case 'more-sorensen'
      Ds        = scale(2) * Delta;
      lb        = scale(2) * ( lf - Y(:,1) );
      ub        = scale(2) * ( uf - Y(:,1) );
      farbounds = ( min( min( [ ub'; -lb' ] ) ) >= Ds );
   case 'ptcg'
      lb        = scale(2) * max( [ ( lf - Y(:,1) )'; -(Delta/sqrt(n))*ones(1,n) ] )';
      ub        = scale(2) * min( [ ( uf - Y(:,1) )';  (Delta/sqrt(n))*ones(1,n) ] )';
   end

   %  Loop on all the optimal replacements of all points in Y to find the best one.

   for j = 2:p1
   
      %  Get the j-th Lagrange polynomial's gradient and Hessian at the current iterate.

      [ g, H ] = bfoss_gH_from_P( Minv(:,j), n );

      %  Minimize this polynomial and its opposite.

      if ( n == 1 )
         [ pstep, pvalue ] = bfoss_TR_1d(  g,  H, lb, ub, Ds );
         [ mstep, mvalue ] = bfoss_TR_1d( -g, -H, lb, ub, Ds );
      else
         switch ( TR_solver_Lagr )
         case 'more-sorensen'
            if ( farbounds )
               pstep  = bfoss_solve_TR_MS( n,  g,  H, Ds, TR_accuracy );
               pvalue =   pstep' * ( g + 0.5 * ( H * pstep ) );
               mstep  = bfoss_solve_TR_MS( n, -g, -H, Ds, TR_accuracy );
               mvalue = - mstep' * ( g + 0.5 * ( H * mstep ) );
	    else
               [ pstep, pvalue ] = bfoss_solve_TR_MS_bc( n,  g,  H, lb, ub, Ds, TR_accuracy );
               [ mstep, mvalue ] = bfoss_solve_TR_MS_bc( n, -g, -H, lb, ub, Ds, TR_accuracy );
            end
         case 'ptcg'
            [ pstep, ~, ~, pvalue ] = bfoss_projected_tcg( n , g,  H, lb, ub,              ...
	                             TR_PTCG_maxcgits * n, TR_PTCG_rel_acc, TR_PTCG_abs_acc );
            [ mstep, ~, ~, mvalue ] = bfoss_projected_tcg( n, -g, -H, lb, ub,              ...
	                             TR_PTCG_maxcgits * n, TR_PTCG_rel_acc, TR_PTCG_abs_acc );
         end
      end

      %  Select the maximum in absolute value (remembering that both mvalue and pvalue
      %  are negative) and construct the corresponding unscaled interpolation point.

      if ( mvalue < pvalue )
         improvement = abs( mvalue );
         y           = Y(:,1) + mstep / scale(2);
      else
         improvement = abs( pvalue );
         y           = Y(:,1) + pstep / scale(2);
      end

      %  Remember the current polynomial value, index and replacement point if
      %  this is the best so far

      if ( improvement > poised )
         poised = improvement;
         jmax   = j;
         ymax   = y;
      end
   end

   %  Return if the new set is poised enough.

   if ( poised <= poisedupper )
%D%      if ( verbose )
%D%         disp(['--- successful improvement after round k = ' num2str( k ) '  ---'])
%D%      end
      break
   end

   %  Perform the best replacement.

%D%   if ( verbose )
%D%      disp( [ ' optimally replace Y(', int2str( jmax ), ') to improve by a factor ',    ...
%D%              num2str( improvement ) ] )
%D%   end

   Y(:,jmax) = ymax;
   [ Minv, ~, scale ] = bfoss_build_Minv_of_Y( Y, model_mode, kappa_ill );
   replaced = sort([replaced, jmax]);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Evaluate the function values for the new interpolation points not in the history.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

replaced  = sort([new_init_points, replaced]);            % the set of new points
if ( ~isempty( replaced ) )
   for ij = 1:length( replaced )
      j   = replaced( ij );
%D%      if ( j == 1 )                                       % Should never happen!
%D%         disp( ' BFOSS internal error: BASE POINT HAS BEEN REPLACED!' )
%D%         keyboard
%D%      end
      
      %  Check if the j-th point in Y has been already evaluated.
      
      in_xhist     = false;
      w            = xbest;
      w( indfree ) = Y( : , j );
      if ( lhist > p1 )
         for i = 1 : lhist
            if ( norm( w - x_hist (:, i ) ) < eps )
               in_xhist = true;
               fY(j)    = f_hist( i );
%D%               if ( verbose )
%D%                   disp( [ ' BFOSS msg: Point ', int2str( j ),                          ...
%D%		           ' in sample set has been already evaluated.' ])
%D%               end
               break
            end
         end
      end

      %  The j-th point in Y has not yet been evaluated.  Evaluate it and perform
      %  the tests to detect and interpret NaN or Inf values.
      
      if ( ~in_xhist  )
         fY( j ) = f( w );        %  This is the call to the true objective function!
	 nevalss = nevalss + 1;
               
         %  Check if the new function value is Inf or NaN.  If yes, exit with the
	 %  appropriate exit code.

         if ( isnan( fY( j ) ) )
            exc = -2;
	 elseif ( abs( fY( j ) ) > myinf )
	    if ( ( fY( j ) ) >  myinf && strcmp( max_or_min, 'max' ) ||                    ...
	         ( fY( j ) ) < -myinf && strcmp( max_or_min, 'min' )    )
	       exc = -3
	    else
               exc = -2;
            end
	 end

         %  Update the list of new points.

         if ( exc < -1 )
%D%            if ( verbose )
%D%               disp( ' BFOSS wrn: NaN or Inf function value!')
%D% 	          keyboard%D
%D%	    end
	 else
            x_new(:, end+1 ) = w;
            f_new(end+1)    = fY(j);
	 end
      end
   end
end

if (  ismember( verbosity , { 'medium', 'debug' } ) )   
   disp( [ ' BFOSS msg: Poisedness of the final Y              = ', num2str( poised ) ] )
   disp( [ ' BFOSS msg: New evaluated points                   = ', num2str( length(f_new))])
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [ Minv, rankloss, scale ] = bfoss_build_Minv_of_Y( Y, model_mode, kappa_ill )
		 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Computes the generalized inverse of the (possibly shifted) matrix containing
%  the polynomial expansion of the interpolation points. The matrix being factorized has 
%  columns containing (Y(:,j)-Y(:,1))/ scale, where scale is the max( norm(Y(:,j)-Y(:,1)).
%
%  INPUT:
%
%  Y           : a matrix whose columns contain the current interpolation 
%                points
%  model_mode  : kind of model to build (0,1,2,3)
%  kappa_ill   : threshold to declare a system matrix as ill-conditioned
%
%  OUTPUT:
%
%  Minv        : the generalized inverse of the (possibly shifted) interpolation
%                matrix associated with the interpolation set Y.
%                containing the polynomial expansion of the interpolation points,
%  rankloss    : the number of singular values that were set to zero
%  scale       : the model diagonal scaling.
%
%  PROGRAMMING: A. Troeltzsch, Ph. Toint, S. Gratton, 2009-2011 for the QR version,
%               Ph. Toint for the Minv version. (This version 9 XI 2017)
%
%  DEPENDENCIES: bfoss_evalZ
%
%  CONDITIONS OF USE: Use at your own risk! No guarantee of any kind given.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[ n, p1 ] = size( Y );

qmax = ( ( n + 1 ) * ( n + 2 ) ) /2;

%  Define the size of the polynomial

switch ( model_mode )

case 'subbasis'

   q = min( p1, qmax );

case 'minell2'

   q = qmax;

otherwise

   disp ( '  UNDEFINED MODEL !!! ' )
%  keyboard

end

%  If shifting is active, the matrix of the interpolation points is first
%  shifted and scaled, and the base point and scaling factors defined.

if ( p1 > 1 ) 
   scaleY = 0;
   base = Y(:,1 );
   for  i = 1:p1
       Y(:,i) = Y(:,i) - base;
       scaleY = max( scaleY, norm( Y(:,i) ) );
   end
   scale = [1, scaleY^(-1)*ones(1,min(n,q-1)), scaleY^(-2)*ones(1,q-n-1)]';
   Y = Y / scaleY; 

%  Otherwise, the base point and scaling factors are trivial.

else
   scale = ones( q, 1 );
end

%  Compute the interpolation matrix and its SVD.

[ U, S, V ] = svd( bfoss_evalZ( Y, q )' );

%  Regularize it if necessary.

Sigma     = diag( S(1:p1,1:p1 ) );
upperS    = max( 1, max( Sigma ) );
nonsing   = find( kappa_ill * Sigma >= upperS );
Siginv    = zeros(p1,1);
rankloss  = p1 - length( nonsing );
if ( rankloss )
   Siginv( nonsing ) = Sigma( nonsing ).^(-1);
else
   Siginv            = Sigma.^(-1);
end
Sinv = [ diag( Siginv); zeros( q-p1, p1 ) ];
Minv = V * Sinv * U';

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function Z = bfoss_evalZ( X, q )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Compute the matrix Z(X), where X is a matrix whose columns contains points
%  of the underlying space.  The vector Z(x) is, for a vector x, given by the
%  sequence of the values of the (at most quadratic) monomials taken at x.  
%  More specifically, these values are:
%  Z(x)(1)        : 1,
%  Z(x)(2:n+1)    : the linear terms x(1)... x(n),
%  Z(x)(n+2:2n+2) : the diagonal terms of the quadratic: x(1)^2 ... x(n)^2
%  Z(x)(2n+3,3n+2): the first subdiagonal of the quadratic: 
%                    x(1)*x(2) ... x(n-1)*x(n)
%  Z(x)(3n+3,4n+1): the second subdiagonal of the quadratic: 
%                    x(1)*x(3) ... x(n-2)*x(n)
%  etc.
%
%  INPUTS:
%
%  X          : the n x m matrix whose columns contains the points at which the monomials
%               should be evaluated.
%  q          : the number of monomials considered (q <= (n+1)*(n+2)/2))
%
%  OUTPUT:
%
%  Z  : the matrix Z(X), of size q x m.
%
%  PROGRAMMING: Ph. Toint, January 2009,odified by M. Porcelli, 2016
%               (this version 26 X 2017)
%               
%
%  DEPENDENCIES: -
%
%  CONDITIONS OF USE: Use at your own risk! No guarantee of any kind given.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[ n, m ] = size( X );             % [ dimension of the space, number of points in X ]
nlin     = min( n+1, q );         % number of constant and linear terms
nquad    = max( 0, q-nlin );      % number of quadratic terms
nlin     = nlin - 1;              % number of linear terms
Z        = zeros( q, m );

if ( q == 1 )
   Z = ones( 1, m );                       % constant terms
elseif ( q <= n+1 )
   Z = [ ones( 1, m ); X(1:nlin,1:m) ];    % constant and linear
else
   ndiag = min( n, nquad );
   Z     = [ ones( 1, m ); X(1:n,1:m); 0.5*X(1:ndiag,1:m).^2 ]; % same + diagonal
   nquad = nquad - ndiag;
   if ( nquad > 0 )
      for k = 1:n-1                        % the (i+1)-th subdiagonal
          nsd = min( n-k, nquad );
          if ( nsd > 0 )
             Z = [ Z; X(k+1:k+nsd,1:m).*X(1:nsd,1:m) ];
             nquad = nquad - nsd;
          end
          if ( nquad == 0 )
             break;
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




function [ g, H ] = bfoss_gH_from_P( P, n )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Computes the gradient of the polynomial P at x, where P is represented by
%  the row vector containing its coefficients for the successive monomials.
%  More specifically, these values are:
%  P(1)        : constant coefficient,
%  P(2:n+1)    : coefficients for the linear terms in x(1)... x(n),
%  P(n+2:2n+2) : coefficients for the squared terms in x(1)^2 ... x(n)^2
%  P(2n+3,3n+2): coefficients for the quadratic terms of the first subdiagonal: 
%                in x(1)*x(2) ... x(n-1)*x(n)
%  (3n+3,4n+1): coefficients for the quadratic terms of the second subdiagonal: 
%                in x(1)*x(3) ... x(n-2)*x(n)
%  etc.
%
%  INPUT:
%
%  P : a column vector contains the coefficients of the polynomial
%  n : the number of variables
%
%  OUTPUT:
%
%  g : the gradient of P at x
%  H : the Hessian of P at x
%
%  PROGRAMMING: Ph. Toint (this version 1 XII 2017).
%
%  DEPENDENCIES: -
%
%  CONDITIONS OF USE: Use at your own risk! No guarantee of any kind given.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p1 = length( P );

%  The gradient

ng        = min( n, p1-1 );
g         = zeros( n, 1 );
g( 1:ng ) = P( 2:ng+1 );

%  The Hessian

H        = zeros( n, n );
nquad    = p1 - n - 1;
if( nquad > 0 )

    % diagonal

    ndiag = min( nquad, n );
    H     = diag( [ P( n+2:n+1+ndiag ); zeros( n-ndiag, 1 )] );
    nquad = nquad - ndiag;

    % subdiagonals

    if ( nquad > 0 )
       k = 2*n+1;
       for i = 1:n-1
          nsd = min( n-i, nquad);
          if ( nsd > 0 )
              for j = 1:nsd
                H( i+j, j ) = P( k+j ); 
                H( j, i+j ) = P( k+j ); 
              end
              k = k + nsd;
              nquad = nquad - nsd;
          end
          if ( nquad == 0 )
             break;
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




function [ s, msg, cgits, value ] =                                                        ...
         bfoss_projected_tcg( n, g, H, lb, ub, cgitmax, eps_rel, eps_abs );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  A simple (bound-)projected truncated conjugate-gradient algorithm,
%  without preconditioning, for solving the bound-constrained quadratic
%  minimization problem.
%
%  INPUT:
%
%  g      : the gradient of the quadratic
%  H      : the (symmetric) Hessian of the quadratic
%  lb     : lower bounds on the variables
%  ub     : upper bounds on the variables
%  cgitmax: the maximum number of CG iterations
%  eps_rel: the relative accuracy on the gradient
%  eps_abs: the absolute accuracy on the gradient
%
%  OUTPUT:
%
%  s       : the approximate minimizer
%  msg     : an informative (?) message 
%  cgits   : the number of CG iterations required
%  value   : the value of the quadratic at its minimizer
%
%  DEPENDENCIES: ~
%
%  PROGRAMMING : Ph. Toint, 9 December 2005 (this version 2 XII 2017).
%
%  CONDITIONS OF USE: Use at your own risk! No guarantee of any kind given.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%D%verbose = 0;              % 0 (silent), 1, 2 (debug)

%  Initializations

value     = 0;
cgitstart = 1;

%  Define the active bounds.

xfree = ones( n, 1 );
nfree = n;
for i = 1:n
   if ( ( abs( lb( i ) ) <= eps && g( i ) > 0 ) || ( abs( ub( i ) ) <= eps && g( i ) < 0 ) )
      xfree( i ) = 0;
      nfree      = nfree - 1;
   end
end
ng = norm( g .* xfree );

%  Zero Hessian: the solution is given by the projection on the box of a
%  sufficiently large step along the negative gradient.

if ( ~norm( H, 'inf' ) )
   amax  = 2 * max( max( abs( lb ) ), max( abs( ub ) ) ) / ng;
   s     = max( lb, min( - amax * ( g .* xfree ), ub ) );
   cgits = 0;
   msg   = 'BFOSS PTCG: successful exit (null Hessian)';
%D%   if ( verbose )
%D%      disp( msg )
%D%   end
   value = g' * s;
   return
end

%  Define the effective accuracy on the gradient's norm.

eps_eff = max ( min( eps_rel, sqrt( ng ) ) * ng, eps_abs );

%  Initialize the iteration counter, the interior indicator and the step.

cgits   = 0;
s       = zeros( n, 1 );

%  Loop on successive sets of CG iterations (in different faces,
%  corresponding to successive active sets).

for iface = 1:n

%D%   if ( verbose > 0 )
%D%      disp(' ')
%D%      disp( [ ' BFOSS PTCG: (re)starting CG on face ', num2str( iface ),                ...
%D%              ' [ nfree = ',num2str( nfree  ),' ]' ] )
%D%      disp( '      It       value       ||Pg||    face  nfree ' )
%D%      fprintf( '      %2d  % .7e  %.2e  %5d  %5d\n', cgits, value, ng, iface, nfree )
%D%   end

   % Reinitialize the first direction to the free negative gradient in the current face.
  
   p = - g  .* xfree;
   if ( iface > 1 )
      ng = norm( p );
   end

   %  Nothing more to do in this face if the projected gradient
   %  (in p) is small enough.

   if ( ng  < eps_eff )
         msg = [ ' BFOSS PTCG: successful exit (converged)' ];
%D%      if ( verbose > 1 )
%D%         disp(msg);
%D%      end
      return
   end

   % CG loop

   for cgits = cgitstart:cgitmax

      %  Piecewise linesearch loop along the current (projected) CG direction.
      %  The loop j is on successive breakpoints.
      
      newfix = 0;
      Hp     = H * p;
      pHp    = p' * Hp;
      gp     = g' * p;

      for j  = 1:n

         %  Find the step to the first bound(s), ie the first breakpoint.

         minsig = 1.0e99;
         imin   = 0;
         for  i = 1:n
            if ( xfree( i ) )
               if ( p(i) > 0 )
                  sigbound = ( ub(i) - s(i) ) / p(i);
               elseif ( p(i) < 0 )
                  sigbound = ( lb(i) - s(i) ) / p(i);
               else
                  sigbound = 1.0e100;
               end
               if ( sigbound < minsig )
                  minsig = sigbound;
                  imin   = i;
               end
            end
         end

         %  Check if negative curvature encountered.  If not, see if the quadratic's
	 %  minimizer occurs before the first breakpoint.  If yes, ignore the breakpoint
	 %  by setting imin = 0.

         if ( pHp <= 0 )
            sigma = minsig;
         else
            alpha = - gp / pHp;
            if ( abs( alpha ) < minsig )
               sigma = alpha;
               imin  = 0;
            else
               sigma = minsig;
            end
         end
	 
         %  Compute the quadratic value at the breakpoint or at the model line minimizer.
	 %  Also define the step to be at the breakpoint and compute the associated
	 %  quadratic's gradient.

         value = value + sigma * gp + 0.5 * sigma^2 * pHp;
         s     = s + sigma * p;
         g     = g + sigma * Hp;

         %  The trial point is on the boundary of the current face.

         if ( imin > 0 )

            %  Move to the first bound exactly and update the activity indicators.

            nfb = nfree;
            for  i = 1:n
               if ( xfree( i ) )
                  if ( p( i ) > 0 && abs( s( i ) - ub( i ) ) <= eps )
                     s( i )     = ub( i );
                     xfree( i ) = 0;
                     nfree      = nfree - 1;
                  elseif ( p( i ) < 0 && abs( s( i ) - lb( i ) ) <= eps )
                     s( i )     = lb( i );
                     xfree( i ) = 0;
                     nfree      = nfree - 1;
                  end
               end
            end

            %  Terminate the routine if there are no free variable left.

            if ( ~nfree )
               msg = ' BFOSS PTCG: successful exit (all variables at their bounds)';
%D%               if ( verbose > 1 )
%D%                  disp( msg )
%D%               end
               return
            end
		
            %  Update p.

            nfix   = nfb - nfree;
            newfix = newfix + nfix;
            if ( newfix == 1 )
	       pimin = p( imin );
	       gp    = gp - g( imin ) * pimin;
               p( imin ) = 0;
            else
               p   = p .* xfree;
               gp  = g' * p;
            end
            
            %  Terminate the piecewise search if the slope is positive or zero
            %  (this may happen if no free variable is left).

            if ( gp >= 0 )
%D%               if ( verbose > 1 )
%D%                  disp([ '      !!! slope is positive or zero: gp=', num2str( gp ),     ...
%D%	                 ' but there are still free variables'])
%D%               end
               break
            end

            %  Update Hp and pHp. If only one bound has been hit (the most frequent case),
	    %  then the update is cheaper.
	    
            if ( newfix == 1 )
               Hp = Hp  - pimin * H( :, imin );
            else
               Hp = H  * p;
            end
            pHp = p' * Hp;

         %  The new point is inside the current face: exit the piecewise search.

         else
%D%            if ( verbose > 1 )
%D%                disp('      break if new point is inside the current face')
%D%            end
            break;
         end

%D%         if ( verbose > 0 )
%D%            disp(' ')
%D%            disp( [' ---- new face [ nfree = ',num2str(nfree),' ] in piecewise search' ] )
%D%         end

      end
      
      %  End of the piecewise linesearch: exit CG if new variables have been activated.

      if ( newfix > 0 )
%D%         if ( verbose > 1 )
%D%             disp( [ '      end of piecewise linesearch: exit if new variables',        ...
%D%	             ' have been activated' ] )
%D%         end
         break;
      end

      %  CG body (1): compute the new free gradient and its norm.

      gfree  = g .* xfree;
      ngfree = norm( gfree );

      %  Print out

%D%      if ( verbose > 0 )
%D%         fprintf( '      %2d  %.7e  %.2e  %5d  %5d\n', cgits, value, ngfree, iface, nfree )
%D%      end

      %  CG termination ?
      %  1) accuracy obtained
            
      if ( ngfree < eps_eff )
         msg = [ ' BFOSS PTCG: successful exit (converged)'];
%D%       if ( verbose > 1 )
%D%	     disp(msg)
%D%	  end
         return
      end

      %  2) too many CG iterations

      if ( cgits >= cgitmax )
         msg   = ' BFOSS PTCG: maximum number of iterations reached' ;
%D%         if ( verbose > 1 )
%D%	       disp(msg)
%D%	    end
         return
      end

      %  CG body (2): compute the new search direction.
      
      beta   = ( ngfree / ng )^2;
      p      = - gfree + beta * p;
      ng     = ngfree;
      
   end

   % Update the iteration counter for starting the next CG loop.

   cgitstart = cgits;

end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [ step, value ] = bfoss_TR_1d( g, H, lb, ub, Delta )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Solves the bounded trust-region problem in dimension one, i.e. 
%
%        min_s q(s) = g*s + (1/2)*H*s^2
%
%  subject to
%
%        max( lb -Delta ) <= s <= min( ub, Delta ).
%
%
%  INPUT :
%
%  g    : the gradient of the quadratic model q(s)
%  H    : the Hessian  of the quadratic model q(s)
%  lb   : the lower bound on the step
%  ub   : the upper bound on the step
%  Delta: the trust-region radius
%
%
%  OUTPUT:
%
%  step : the minimizer
%  value: the value of the quadratic model at the minimizer
%
%
%  DEPENDENCIES     : ~
%
%  PROGRAMMING      : Ph. Toint, December 2017 (this version 23 XII 2017).
%
%  CONDITIONS OF USE: Use at your own risk! No guarantee of any kind given.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( g < 0 )
   step  = min( ub, Delta );
   if ( H > 0 )
      step = min( step, -g/H );
   end
   value = step * ( g + 0.5 * H * step );
elseif ( g > 0 )
   step  = max( lb, -Delta );
   if ( H > 0 )
      step = max( step, -g/H );
   end
   value = step * ( g + 0.5 * H * step );
else
   if ( H >= 0 )
      step  = 0;
      value = 0;
   else
      spos = min( ub,  Delta );
      sneg = max( lb, -Delta );
      if ( spos >= -sneg )
         step = spos;
      else
         step = sneg;
      end
      value = 0.5 * H * step^2;
   end
end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [ s, value, msg ] = bfoss_solve_TR_MS_bc( n, g, H, lb, ub, Delta, eps_D )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  An implementation of exact trust-region minimization based on the
%  Moré-Sorensen algorithm subject to bound constraints.
%
%  INPUT: 
%
%  n        : the problem's dimension
%  g        : the model's gradient
%  H        : the model's Hessian
%  lb       : lower bounds on the step
%  ub       : upper bounds on the step
%  Delta    : the trust-region's radius
%  eps_D    : the accuracy required on the equation ||s|| = Delta for a
%             boundary solution
%
%  OUTPUT:
%
%  s        : the trust-region step
%  value    : the value of the model at the optimal solution
%  msg      : an information message
%
%  DEPENDENCIES: bfoss_solve_TR_MS
%
%  PROGRAMMING: A. Troeltzsch, S. Gratton, July 2009,
%               modified by Ph. Toint and M. Porcelli for BFOSS (this version 27 XII 2017).
%
%  CONDITIONS OF USE: Use at your own risk! No guarantee of any kind given.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%D%verbose = 0;
theta      = 1.0e-13;          % accuracy of the interval on lambda
eps_bound  = 1.0e-5;           % the max distance | bound - x | < eps_bound for 
                               % a boundary solution
                                 
%  Initialization 

%msg        = '';
%lambda     = 0;                % initial lambda
value      = 0;                % initial model value
Delta0     = Delta;
g0         = g;                % copy initial gradient
s          = zeros( n, 1);     % initial zero step
warning off

%  Fix active components.

ind_g_crit = find( ( abs( lb ) <= 1e-10 & g > 0 ) | ( ub <= 1e-10 & g < 0 ) );

if ( ~isempty( ind_g_crit ) )
    ind_active = ind_g_crit';
    ind_free   = setdiff( 1:n, ind_active );
    nfree      = length( ind_free );
else
   ind_free   = 1:n;
   ind_active = [];               % indeces of active bounds
   nfree      = n;                % nbr of inactive indeces
end

%  Loop until no free variables anymore.

j = 0;
while nfree > 0

   %  Loop until all active variables are detected and fixed.
    
   new_call_to_MS = 1;
   while ( new_call_to_MS == 1 )
      j = j + 1;
         
      %  Minimize system in the (possibly) reduced subspace

%D%      if ( verbose >= 1 )
%D%         disp(['(', num2str(j), ') ---- minimizing in the (sub)space of ',              ...
%D%	        num2str(length(ind_free)), ' variable(s)'] )
%D%      end
        
      %  Call unconstrained MS in (possibly) reduced subspace.

      [ s_deltaMS, lambda, msg, hardcase ] ...
	       = bfoss_solve_TR_MS( nfree, g(ind_free), H(ind_free,ind_free), Delta, eps_D );

      s_after_reduced_ms = s;
      s_after_reduced_ms( ind_free ) = s_after_reduced_ms( ind_free ) + s_deltaMS;

      %  Compute critical components which became active during the last MS iteration.
      
      ind_u_crit = find( ( ub( ind_free ) - s_after_reduced_ms( ind_free ) ) <= eps_bound &...
                           ub( ind_free ) <=  1e25 );
      ind_l_crit = find( ( s_after_reduced_ms( ind_free ) - lb( ind_free ) ) <= eps_bound &...
                           lb( ind_free ) >= -1e25 );
        
      %  Fix these new active components
        
      if ( ~isempty( ind_u_crit ) || ~isempty( ind_l_crit ) )
           
         ind_active = [ ind_active ind_free(ind_u_crit) ind_free(ind_l_crit) ];
         ind_free   = setdiff(1:n,ind_active);
         nfree      = length(ind_free);
%D%         if ( verbose )
%D%            disp( 'fixed one or more variables' )
%D%         end
            
         %  If no inactive variables anymore --> exit.
            
         if ( ~nfree )
            s =  s_after_reduced_ms;
            value = 0.5 * s' * H * s + s' * g0;
%D%            if ( verbose )
%D%               disp('no inactive variables anymore - return')
%D%            end
            return
         end
            
      else            
         new_call_to_MS = 0;
      end
   end
    
   %  Check if step is outside the bounds.
    
%D%   if ( verbose == 2 )
%D%      disp( 'check if step inside bounds' )
%D%   end

   out_of_ubound = find( (ub(ind_free)-s_after_reduced_ms(ind_free)) < 0 );
   out_of_lbound = find( (s_after_reduced_ms(ind_free)-lb(ind_free)) < 0 );
   isempty_out_of_ubound = isempty( out_of_ubound );
   isempty_out_of_lbound = isempty( out_of_lbound );
   
   out_of_ubound_init = out_of_ubound;
   out_of_lbound_init = out_of_lbound;

   if ( ~isempty_out_of_ubound || ~isempty_out_of_lbound )

      back_inside = 0;
      lambda0 = lambda;
%D%      if ( verbose == 2 )
%D%         disp( 'step outside bounds!' )
%D%         out_of_ubound
%D%         out_of_lbound
%D%         disp( [ 'lambda_0=' num2str(lambda0) ] )
%D%      end

      %  Set lower bound on lambda.

      lower = lambda;
        
      %  Compute upper bound on lambda (using the closest bound out of the hit bounds).
        
      gnorm = norm( g );
      if ( ~isempty_out_of_ubound  )
         delta_b = min( abs(ub(ind_free(out_of_ubound)) - s(ind_free(out_of_ubound))) );
      end
      if ( ~isempty_out_of_lbound  )
         delta_b = min( abs(lb(ind_free(out_of_lbound)) - s(ind_free(out_of_lbound))) );
         if ( ~isempty_out_of_ubound )
            delta_b  = min( min( abs(ub(ind_free(out_of_ubound)) -                         ...
	                                    s(ind_free(out_of_ubound))) ), delta_b);
         end
      end       
        
      goverD   = gnorm / delta_b;
      Hnorminf = norm( H, inf );
      if ( Hnorminf > 0 )        % because Octave generates a NaN for null matrices.
         HnormF   = norm( H, 'fro' );
      else
         HnormF   = 0;
      end   
      upper = max( 0, goverD + min( Hnorminf, HnormF ) );

      %  Compute active components

      ind_u_active = find(abs(ub(ind_free)-s_after_reduced_ms(ind_free)) <= eps_bound);
      ind_l_active = find(abs(s_after_reduced_ms(ind_free)-lb(ind_free)) <= eps_bound);

      %  Loop on successive trial values for lambda.

      i = 0;
      while ( ( isempty( ind_u_active )  && isempty( ind_l_active ) ) ||                   ...
              ( isempty_out_of_lbound    && isempty_out_of_ubound   )    )
         i = i + 1; 

         %  Print lambda value

         old_lambda = lambda;
         new_lambda = -1;
%D%         if ( verbose )
%D%            disp( [ ' bfoss_solve_TR_MS_bc (',int2str(i),'): lower = ',num2str( lower ),...
%D%                       ' lambda = ', num2str( lambda ) , ' upper = ',  num2str( upper ) ] )
%D%         end

         %  Factorize H + lambda * I.

         [ R, p ] = chol( H(ind_free,ind_free) + lambda * eye( nfree ) );

         %  Successful factorization 

         if ( ~( p || hardcase ) )

            s_deltaH  = - R \ ( R' \ g(ind_free) );
            s_duringH = s;
            s_duringH( ind_free ) = s_duringH( ind_free ) + s_deltaH;


            %  Find components which are at its bound and became active

            ind_u_crit  = find( ( ub( ind_free ) - s_duringH( ind_free ) ) <= eps_bound && ...
	                          ub( ind_free ) <= 1e-10 );
            ind_l_crit  = find( ( s_duringH( ind_free ) - lb( ind_free ) ) <= eps_bound && ...
	                          lb( ind_free ) >= -1e-10 );

            %  Set these new active components to zero for one iteration

            if ( ~isempty( ind_u_crit ) )
               s_deltaH( ind_u_crit )              = 0;
               s_duringH( ind_free( ind_u_crit ) ) = 0;
            end
            if ( ~isempty( ind_l_crit ) )
               s_deltaH( ind_l_crit )              = 0;
               s_duringH( ind_free( ind_l_crit ) ) = 0;
            end

            out_of_ubound = find( (ub(ind_free)-s_duringH(ind_free)) < 0.0 );
            out_of_lbound = find( (s_duringH(ind_free)-lb(ind_free)) < 0.0 );
            isempty_out_of_ubound = isempty( out_of_ubound );
            isempty_out_of_lbound = isempty( out_of_lbound );
	    
            % Find an appropriate bound for the next homotopy-step when using Newton's method.

%            if ( 1 )

               if ( ~isempty_out_of_ubound  || ~isempty_out_of_lbound )

                  %  OUTSIDE the bounds: find the furthest step component.
		  
                  outside = 1;
                  if ( ~isempty_out_of_ubound  )
                     [diff_b_u,ind_b_u] = max(abs(ub(ind_free(out_of_ubound)) -            ...
		                           s_duringH(ind_free(out_of_ubound))));
                     ind_b   = out_of_ubound( ind_b_u );
                     norms_b = abs( s_deltaH( ind_b ) );
		     indfr_b = ind_free( ind_b );
		     sdelt_b = ub( indf_b ) - s( indfr_b );
                     delta_b = abs(  sdelt_b );
                     sign_b  = sign( sdelt_b );
                     out_of_ubound_init = [ out_of_ubound_init; out_of_ubound ];
                  end
                  if ( ~isempty_out_of_lbound )
                     [diff_b_l,ind_b_l] = max(abs(s_duringH(ind_free(out_of_lbound)) -     ...
		                              lb(ind_free(out_of_lbound))));
                     ind_b   = out_of_lbound(ind_b_l);
                     norms_b = abs( s_deltaH( ind_b ) );
		     indfr_b = ind_free( ind_b );
		     sdelt_b = lb (indfr_b ) - s( indfr_b );
                     delta_b = abs(  sdelt_b );
                     sign_b  = sign( sdelt_b );
                     out_of_lbound_init = [ out_of_lbound_init; out_of_lbound ];
                  end
                  if ( ~isempty_out_of_ubound &&  ~isempty_out_of_lbound )
                     if ( diff_b_u > diff_b_l )
                        ind_b   = out_of_ubound( ind_b_u );
                        norms_b = abs( s_deltaH( ind_b ) );
		        indfr_b = ind_free( ind_b );
			sdelt_b = ub( indfr_b ) - s( indfr_b );
                        delta_b = abs(  sdelt_b );
                        sign_b  = sign( sdelt_b );
                     else
                        ind_b   = out_of_lbound( ind_b_l );
                        norms_b = abs(s_deltaH( ind_b ));
		        indfr_b = ind_free( ind_b );
			sdelt_b = lb( indfr_b ) - s( indfr_b );
                        delta_b = abs(  sdelt_b );
                        sign_b  = sign( sdelt_b );
                     end
                  end

               else
	       
                  %  INSIDE the bounds, but no step component active:
                  %  find the closest components to its bound from the
                  %  set of components which were initially outside.

                  outside = 0;
		  
		  isempty_out_of_ubound_init = isempty( out_of_ubound_init );
		  isempty_out_of_lbound_init = isempty( out_of_lbound_init );
		  
                  if ( ~isempty_out_of_ubound_init )
                     [diff_b_u,ind_b_u] = min(abs(ub(ind_free(out_of_ubound_init)) -       ...
		                            s_duringH(ind_free(out_of_ubound_init))));
                     ind_b   = out_of_ubound_init(ind_b_u);
                     norms_b = abs( s_deltaH( ind_b ) );
		     indfr_b = ind_free( ind_b );
		     sdelt_b = ub( indfr_b ) - s( indfr_b );
                     delta_b = abs(  sdelt_b );
                     sign_b  = sign( sdelt_b );
                  end
                  if ( ~isempty_out_of_lbound_init )
                     [diff_b_l,ind_b_l] = min(abs(s_duringH(ind_free(out_of_lbound_init))  ...
		                            -lb(ind_free(out_of_lbound_init))));
                     ind_b   = out_of_lbound_init(ind_b_l);
                     norms_b = abs( s_deltaH( ind_b ) );
		     indfr_b = ind_free( ind_b );
		     sdelt_b = lb( indfr_b ) - s( indfr_b );
                     delta_b = abs(  sdelt_b );
                     sign_b  = sign( sdelt_b );
                  end
                  if ( ~isempty_out_of_ubound_init && ~isempty_out_of_lbound_init )
                     if ( diff_b_u < diff_b_l )
                        ind_b   = out_of_ubound_init(ind_b_u);
                        norms_b = abs( s_deltaH( ind_b ) );
			indfr_b = ind_free( ind_b );
			sdelt_b = ub( indfr_b ) - s( indfr_b );
                        delta_b = abs(  sdelt_b );
                        sign_b  = sign( sdelt_b );
                     else
                        ind_b   = out_of_lbound_init(ind_b_l);
                        norms_b = abs( s_deltaH( ind_b ) );
                        indfr_b = ind_free( ind_b );
			sdelt_b = lb( indfr_b ) - s( indfr_b );
                        delta_b = abs(  sdelt_b );
                        sign_b  = sign( sdelt_b );
                     end
                  end
               end
%            end
                 
%D%            %  Iteration printout
%D%
%D%            if ( verbose )
%D%               lambda_save(i)  = lambda;
%D%               norms_b_save(i) = norms_b;
%D%
%D%               if ( ~outside )
%D%                  fprintf( 1,'%s%d%s %12.8e %s %12.8e %s\n',' bfoss_solve_TR_MS_bc (',i,...
%D%                     '): |s_i| = ', norms_b, '  |bound_i| = ', delta_b , '   s < bounds' )
%D%               else
%D%                  fprintf( 1, %s%d%s %12.8e %s %12.8e\n',' bfoss_solve_TR_MS_bc (', i,  ...
%D%                  '): |s_i| = ', norms_b, '  |bound_i| = ', delta_b )
%D%               end
%D%            end

            %  Test if step inside bounds +/- eps_bound.
                 
            out_of_uEpsbound = find( ( ub( ind_free )-s_duringH( ind_free ) ) < -eps_bound );
            out_of_lEpsbound = find( ( s_duringH( ind_free )-lb( ind_free ) ) < -eps_bound );

            if ( isempty( out_of_uEpsbound ) && isempty( out_of_lEpsbound ) )
                    
%D%               if ( verbose >= 2 )
%D%                  disp('all components inside the bounds + eps_bound')
%D%               end
                    
               back_inside = 1;

               % Check if at least one component active.

               ind_u_active = find( abs( ub( ind_free ) - s_duringH(ind_free)) <= eps_bound );
               ind_l_active = find( abs( s_duringH( ind_free ) - lb(ind_free)) <= eps_bound );

               if ( ~isempty( ind_u_active ) || ~isempty( ind_l_active ) )
%D%                  if ( verbose >= 2 )
%D%                     disp( [ 'all components inside the bounds + eps_bound, '           ...
%D%                     num2str( length( ind_u_active )+length( ind_l_active  ) )          ...
%D%                     ' component/s close to one of its bounds' ] )
%D%                  end

                  %  Compute the current step after the homotopy-step

                  s_afterH  = s;
                  s_afterH( ind_free ) = s_afterH( ind_free ) + s_deltaH;

                  %  Move active components to their bounds
                        
                  if ( ~isempty( ind_u_active ) )
                     s_afterH( ind_free( ind_u_active ) ) = ub( ind_free( ind_u_active ) );
                  end
                  if ( ~isempty( ind_l_active ) )
                     s_afterH( ind_free( ind_l_active ) ) = lb( ind_free( ind_l_active ) );
                  end

                  %  Define information message.
                        
                  msg = 'boundary solution';

                  break;
               end
            end

            %  Compute new lambda.
                 
            %  Reset bounds on lambda.

            if ( isempty_out_of_ubound && isempty_out_of_lbound )
               upper = lambda;
            else
               lower = lambda;
            end
                    
            %  Check the sign of the chosen step component.
                    
            es = s_deltaH( ind_b );
            
            if ( sign( es ) ~= sign_b )
                        
               %  If the step component has the wrong sign
               %  (other than the active bound): one bisection step
                        
               new_lambda  = 0.5 * ( lower + upper );
	       
            else
	    
	       %  Newton step

               w1 = R' \ [ zeros( ind_b-1, 1 ); 1; zeros( ind_b+1, 1 ) ];
               w2 = R' \ s_deltaH;

               new_lambda = lambda + ( ( norms_b - delta_b ) / delta_b ) *                 ...
	                             ( norms_b^2 / ( es*(w1'*w2) ) );
               if ( ~back_inside && upper <= new_lambda )
                  upper = 2 * new_lambda;
               end
            end

            %  Check new value of lambda wrt its bounds.

            theta_range = theta * ( upper - lower );
            if ( new_lambda > lower + theta_range && new_lambda <= upper - theta_range )
               lambda = new_lambda;
            else
               lambda = max( sqrt( lower * upper ), lower + theta_range );
            end

         else %  Unsuccessful factorization: compute new lambda.
                
%D%            if ( verbose )
%D%               disp('unsuccessful factorization')
%D%            end
            hardcase = 0;
            lower    = lambda;
            lambda   = 0.5 * ( lower + upper );
         end

         %  Return with error message after 100 iterations.

         if ( i >= 100 )
            s   = zeros( n, 1 );
            msg = 'Error6';
            disp('Error in bfoss_solve_TR_MS_bc: iteration limit in bc-MS exceeded!')
            value = 0;
            return
         end

      end % end of while-loop
            
   else
        
      % MS was successful inside the bounds.
        
%D%      if ( verbose >= 2 )
%D%         disp( 'step inside bounds!' ) 
%D%      end
        
      %  Define information message.

      msg = '(partly) interior solution';
               
      %  Update the step and model value.

      s     = s_after_reduced_ms;
      value = s' * ( g0 + 0.5 * ( H * s ) );
      break
   end    
       
   %  Update the step and model value.

   s     = s_afterH;
   Hs    = H * s,
   value = s' * ( g0 +  0.5 * Hs );
    
   %  Update trust-region radius.

   Delta = Delta0 - norm( s );
    
   %  Update gradient.
    
   g  = g0 + Hs;
    
   %  Update active bounds.
    
   ind_active = find( ( ub - s ) <= eps_bound || ( s - lb ) <= eps_bound )';
   ind_free   = setdiff( 1:n, ind_active);
   nfree      = length( ind_free );
    
   if ( nfree )
       
      %  Check first-order criticality.
       
      ng_reduced = norm( g( ind_free ), inf );
      if ( ng_reduced <= 1e-5 )
%D%         if ( verbose >= 2 )
%D%            disp('point first order critical - return')
%D%            ng_reduced
%D%         end
         return
      end
    
      %  Fix components which became active.
    
      ind_g_crit = find(( abs( lb(ind_free)) <= 1e-10 && g(ind_free) > 0 ) ||              ...
                             ( ub(ind_free)  <= 1e-10 && g(ind_free) < 0 )   );

      if ( ~isempty( ind_g_crit ) )
         ind_active = [ ind_active  ind_free( ind_g_crit ) ];
         ind_free   = setdiff( ind_free, ind_active );
         nfree      = length( ind_free );
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




function [ s, lambda, msg, hardcase ] = bfoss_solve_TR_MS( n, g, H, Delta, eps_D )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  A simple implementation of "exact" unconstrained trust-region minimization, based on the
%  Moré-Sorensen algorithm.

%  INPUT: 

%  n        : the problem's dimension
%  g        : the model's gradient
%  H        : the model's Hessian
%  Delta    : the trust-region's radius
%  eps_D    : the accuracy required on the equation ||s|| = Delta for a
%             boundary solution

%  OUTPUT:

%  s        : the trust-region step
%  lambda   : the Lagrange multiplier corresponding to the trust-region constraint
%  msg      : an information message
%  hardcase : true if the hard case occurred
%
%  DEPENDENCIES: -
%
%  PROGRAMMING: Ph. Toint and S. Gratton, April 2009.
%               (modified by Ph. Toint for BFOSS, this version 27 XII 2017)
%
%  CONDITIONS OF USE: Use at your own risk! No guarantee of any kind given.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%D%verbose  = 0;
theta    = 1.0e-13;        % accuracy of the interval on lambda
epsilon  = 1.0e-15;        % theshold under which the gradient is considered as zero
hardcase = 0;              % hard case
D        = [];

warning off

%D%if ( verbose )
%D%   disp( ' bfoss_solve_TR_MS : ============ enter' )
%D%end

%  Compute initial bounds on lambda.

gnorm    = norm( g );
Hnorminf = norm( H, inf );

%  Zero gradient

if ( gnorm < epsilon )
%D%   if ( verbose )
%D%      disp( ' BFOSS solve_TR_MS : ============ zero gradient:' )
%D%   end
   if ( ~Hnorminf )
      if ( issparse( H ) )
         [ V, D ] = eigs( H, 1, 'SA' );
      else
         [ V, D ] = eig( H );
      end
      [ mu, imu ] = min( diag(D) );
      if ( mu < 0 )
         s = Delta * V(:,imu);
      else
         if ( gnorm == 0 )
            s = zeros( n, 1 );
         else
            s = - Delta * ( g / gnorm );
         end
      end
      lambda = -mu;
   else
      if ( gnorm == 0 )
         s     = zeros( n, 1 );
      else
         s     = - Delta * ( g / gnorm );
      end
      lambda = 0;
   end

%  Nonzero gradient

else

   %  Zero Hessian

   goverD   = gnorm / Delta;
   if ( ~Hnorminf )
      s   = - g / goverD;
      msg = ' BFOSS solve_TR_MS : ============ null Hessian exit';
%D%      if ( verbose )
%D%         disp( msg )
%D%      end
      lambda   = 0;
      hardcase = 0;
      return
   end

   %  Compute initial bounds on lambda.

   HnormF   = norm( H, 'fro' );
   lower    = max( 0, goverD - min( Hnorminf, HnormF ) );
   upper    = max( 0, goverD + min( Hnorminf, HnormF ) );

   %  Compute the interval of acceptable step norms.

   Dlower   = ( 1 - eps_D ) * Delta;
   Dupper   = ( 1 + eps_D ) * Delta;

%D%   if ( verbose )
%D%      disp( ' BFOSS solve_TR_MS : ============ nonzero gradient:' )
%D%   end

   %  Compute initial lambda.

   if ( lower == 0 )
      lambda = 0;
   else
      lambda = max( sqrt( lower * upper ), lower + theta * ( upper - lower ) );
   end

   %  Loop on successive trial values for lambda.

   for i = 1:300             %  nitmax = 300
      new_lambda = -1;
%D%      if ( verbose )
%D%         disp( [ ' bfoss_solve_TR_MS (',int2str(i),'): lower = ',  num2str( lower ),    ...
%D%                    ' lambda = ', num2str( lambda ) , ' upper = ',  num2str( upper ) ] )
%D%      end

      %  Factorize H + lambda I (the costly part).

      [ R, error ] = chol( H + lambda * eye( n ) );

      %  Successful factorization

      if ( ~error )
         s      = - R \ ( R' \ g );
         norms  = norm( s );
%D%         if ( verbose )
%D%            disp( [ ' BFOSS solve_TR_MS (',int2str(i),'): ||s|| = ', num2str( norms ),  ...
%D%                    ' Delta  = ', num2str( Delta ) ] ) 
%D%         end

         %  Test for successful termination.

         if ( ( lambda <= epsilon && norms <= Dupper ) ||                                  ...
              ( norms >= Dlower && norms <= Dupper ) )
            if ( norms < ( 1 - eps_D ) * Delta )
               msg = 'interior solution';
            else
               msg = 'boundary solution';
            end
%D%            if ( verbose )
%D%               disp( ' BFOSS solve_TR_MS : ============ successful exit' )
%D%            end
            return;
         end

         %  Newton's iteration on the secular equation

         w          = R' \ s;
         new_lambda = lambda + ( ( norms - Delta ) / Delta ) * ( norms^2 / ( w' * w ) );

         %  Check feasibility wrt lambda interval.

         if ( norms > Dupper )
            lower = lambda;
         else
            upper = lambda;
         end
         theta_range = theta * ( upper - lower );
         if ( new_lambda > lower + theta_range && new_lambda < upper - theta_range )
            lambda = new_lambda;
         else
            lambda = max( sqrt( lower * upper ), lower + theta_range );
         end

      %  Unsuccessful factorization: take new lambda as the middle 
      %  of the allowed interval

      else
         lower  = lambda;
         lambda = 0.5 * ( lower + upper );
      end

      %  Terminate the loop if the lambda interval has shrunk to meaningless.

      if ( upper - lower < theta * max( 1, upper ) )
         break
      end
   end
end

%  The pseudo-hard case

%  Find eigen decomposition and the minimal eigenvalue.

if ( ~isempty( D ) )
   if ( issparse( H ) )
      [ V, D ] = eigs( H, 1, 'SA' );
   else
      [ V, D ] = eig( H );
   end
end
[ mu, imu ] = min( diag(D) );

%D%if ( verbose )
%D%   gamma   = abs(V(:,imu)'*g);
%D%   disp( [' BFOSS solve_TR_MS : ============ pseudo hard case: gamma = ',               ...
%D%            num2str(gamma), ' ||g|| = ', num2str(norm(g))] )
%D%end

%  Compute the critical step and its orthogonal complement along the
%  eigenvectors corresponding to the most negative eigenvalue.

if ( mu < 0 )
   D        = D - mu * eye(n);
   maxdiag  = max( diag( D ) );
   ii       = find( abs( diag( D ) ) < 1e-10 * maxdiag );
   if ( ~isempty( ii ) && length( ii ) < n )
      D(ii,ii)    = eye( length( ii ) );
      Dinv        = inv( D );
      Dinv(ii,ii) = 0;
      scri        = -V * Dinv * V' * g;
      nscri       = norm( scri );
   else
      scri  = zeros( n, 1 );
      nscri = 0;
   end

   if ( nscri <= Delta )
      root = roots( [ norm(V(:,imu))^2  2*V(:,imu)'*scri  nscri^2-Delta^2 ] );
      s    = scri + root(1)*V(:,imu);
   else
      s    = Delta * scri / nscri;
   end
   lambda  = -mu;
%D%   if ( verbose )
%D%      disp([ ' BFOSS solve_TR_MS : ============ ||scri|| = ', num2str( norm(scri) ),    ...
%D%             ' lambda = ', num2str( lambda) ])
%D%   end

else
   s = zeros( n, 1 );
end
hardcase = 1;

msg = ' BFOSS solve_TR_MS : ============ hard case exit';
%D%if ( verbose )
%D%   disp( msg )
%D%end

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [ model_mode, minimum_model_degree, maximum_model_degree, kappa_ill, poisedupper, ...
          toofarlimit, farthr, term_order, epsilon, TR_init_radius, TR_min_radius,         ...
	  TR_max_radius, TR_accuracy, TR_eta1, TR_eta2, TR_alpha1, TR_alpha2,              ...
	  TR_solver_step, TR_solver_Lagr, TR_PTCG_rel_acc, TR_PTCG_abs_acc,                ...
	  TR_PTCG_maxcgits, min_step_length ] =                                            ...
	                             bfoss_interpolation_default_parameters( elemental_form )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Provides the default algorithmic parameters for the use of interpolation models,
%  depending on whether the problem's objective function is in elemntal-form or not.
%  The proposed values result from training.
%
%  INPUT:
%
%    elemental_form : true iff the problem is partially separable.
%
%  OUTPUT:
%
%  The values of the parameters, see their description in the associated comments.
%
%  DEPENDENCIES: ~
%
%  Programming: M. Porcelli and Ph. Toint, February 2018 (this version 9 II 2018).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( elemental_form )

   model_mode           = 'subbasis';    % the technique used to define the relation
                                         % between the number of data points and the
			                 % number of monomial basis elements for constructing
			                 % the model
   minimum_model_degree = 'minimal';     % the minimum degree of the polynomial model
   maximum_model_degree = 'quadratic';   % the maximum degree of the model ever
   kappa_ill            = Inf;           % the maximal conditioning of the interpolation
                                         % matrix
   poisedupper          = 100;           % an upper bound on the Lagrange polynomial in the
                                         % trust region for declaring interpolation poised
   toofarlimit          = 1e25;          % the multiple of the TR radius beyond which points
                                         % are irrelevant
   farthr               = 126;           % the minimum improvement in poisedness required for
                                         % including points in the history in the interp. set
   term_order           = '2nd';         % the order of the model's critical point for
                                         % termination
   epsilon              = 1e-5;          % the accuracy of the approximate gradient at the
                                         % base point under which no TR minimization is done
   TR_init_radius       = -1;            % automatic choice of initial TR radius
   TR_min_radius        = 1e-8;          % the trust-region radius under which no search step
                                         % is computed
   TR_max_radius        = 100;           % the maximal trust-region radius
   TR_accuracy          = 1e-3;          % the accuracy at which the trust-region constraint
                                         % must be satisfied
   TR_eta1   = 4.595008798357e-02;       % threshold on rho for successful iterations
   TR_eta2   = 9.023384421407e-01;       % threshold on rho for verysuccessful iterations
   TR_alpha1 = 2.502355780913e+00;       % TR expansion factor at very successful iterations
   TR_alpha2 = 2.500000000000e-01;       % TR contraction factor at unsuccessful iterations
   min_step_length      = 1e-10;         % minimum length of the step for computing  the
                                         % trial function value
   TR_solver_step       = 'ptcg';        % the TR solver for computing the search step
   TR_solver_Lagr       = 'more-sorensen'; % the TR solver for the optimization of the
                                         % Lagrange polynomials
   TR_PTCG_rel_acc      = 1e-5;          % the PTCG relative projected gradient accuracy
   TR_PTCG_abs_acc      = 1e-10;         % the PTCG absolute projected gradient accuracy
   TR_PTCG_maxcgits     = 2;             % the maximum number of CG iterations in PTCG
                                         % relative to problem size

else %  not partially separable

   model_mode           = 'subbasis';    % the technique used to define the relation
                                         % between the number of data points and the
			                 % number of monomial basis elements for constructing
			                 % the model
   minimum_model_degree = 'minimal';     % the minimum degree of the polynomial model
   maximum_model_degree = 'quadratic';   % the maximum degree of the model ever
   kappa_ill            = 1e12;          % the maximal conditioning of the interpolation
                                         % matrix
   poisedupper          = 100;           % an upper bound on the Lagrange polynomial in the
                                         % trust region for declaring interpolation poised
   toofarlimit          = 1.e25;         % the multiple of the TR radius beyond which points
                                         % are irrelevant
   farthr               = 100;           % the minimum improvement in poisedness required for
                                         % including points in the history in the interp. set
   term_order           = '2nd';         % the order of the model's critical point for
                                         % termination
   epsilon              = 1e-5;          % the accuracy of the approximate gradient at the
                                         % base point under which no TR minimization is done
   TR_init_radius       = -1;            % automatic choice of initial TR radius
   TR_min_radius        = 1e-8;          % the trust-region radius under which no search step
                                         % is computed
   TR_max_radius        = 100;           % the maximal trust-region radius
   TR_accuracy          = 1e-3;          % the accuracy at which the trust-region constraint
                                         % must be satisfied
   TR_eta1   = 0.05;                     % threshold on rho for successful iterations
   TR_eta2   = 8.978700564391e-01;       % threshold on rho for verysuccessful iterations
   TR_alpha1 = 2.502455253486e+00;       % TR expansion factor at very successful iterations
   TR_alpha2 = 2.464215443130e-01;       % TR contraction factor at unsuccessful iterations
   min_step_length      = 1e-10;         % minimum length of the step for computing  the
                                         % trial function value
   TR_solver_step       = 'ptcg';        % the TR solver for computing the search step
   TR_solver_Lagr       = 'more-sorensen'; % the TR solver for the optimization of the
                                         % Lagrange polynomials
   TR_PTCG_rel_acc      = 1e-5;          % the PTCG relative projected gradient accuracy
   TR_PTCG_abs_acc      = 1e-10;         % the PTCG absolute projected gradient accuracy
   TR_PTCG_maxcgits     = 3;             % the maximum number of CG iterations in PTCG
                                         % relative to problem size
					 
end

return

end

