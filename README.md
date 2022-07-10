<p align="right">
  <img src="http://web.math.unifi.it/users/porcelli/images/BFO_squared.png">
</p>

# BFO 2.0

BFO is an open-source direct-search derivative-free MATLAB solver for bound-constrained mathematical optimization problems.  

Its purpose is to find a local minimizer when applied to problems of the form

    min f(x) 

where f is a function from R^n to R and the variables contained in the vector x are subject to bound constraints 

    l <= x <= u 

where these inequalities hold componentwise and where components of l and/or u may be infinite.  


As indicated above, using BFO does not require the provision of derivatives of the function f(x). Like all methods of this type it is most efficient on problems involving a relatively small number of variables (not much above 10).

## Features

BFO's remarkable features are:

- its ability to handle a mix of **continuous**, **discrete** and **categorical** variables,
- a novel **self-training option**, which may significantly improve the performance of the code on the user's class of problems, its capacity to handle **multilevel**, **min-max** and **equilibrium problems**,
- an extremely versatile and easy-to-use interface.
 BFO also provides a number of user-oriented features, such as
 checkpointing and restart, facilities for specifying variable scaling, thereby improving problem conditioning,
- tools allowing the user to specify a variety of  termination conditions,
- **BFOSS**, a library whose purpose is to compute interpolation-based search steps.

## New features of BFO Release 2.0
Release 2.0 of the Matlab BFO package is a major upgrade from Release 1 [BFO 1.01](https://sites.google.com/site/bfocode/home]) and includes several important new problem-oriented features.

- **Using coordinate partially-separable problem structure**. This ubiquitous problem structure can now be exploited by BFO, leading to very significant gains in performance (orders of magnitude) also allowing the use of BFO for large problems (several thousands of variables).

- **The BFOSS library of model-based search steps**.
Release 2.0 of BFO now also supplies BFOSS, a BFO-compatible library whose purpose is to compute interpolation-based serch steps. Various options are available, ranging from sublinear to fully quadratic models. BFOSS supports model building for objective functions given
in coordinate-partially-separable form. This combination provide even further significant performance improvements.

- **Categorical variables**. BFO now supports the use of categorical variables. Categorical variables are unconstrained non-numeric variables whose possible ’states’ are
defined by strings (such as ’blue’). These states are not implicitly ordered, as would be the case for integer or continuous variables. The user specifies the application-dependent neighbourhoods for categorical variables in two possible ways (static neighbourhoods and dynamic ones), allowing full application-dependent flexibility.

- **Performance and data profile training strategies**. Because BFO is a trainable package (meaning that its internal algorithmic constants can be trained/optimized by the user to optimize its performance on a specific problem class), it needs to define training strategies which allow to decide if a particular option is better than another. BFO Release 2.0 now includes two new training strategies: Performance
profiling and data profiling. These strategies use the eponymous tools for comparing and optimizing algorithmic performance.

BFO Release 2.0 also improves performance and stability upon Release 1.0 and corrects a few bugs.

BFO has been thoroughly tested and is the subject of continued development.
It can be downloaded free of charge.

## Documentation

Documentations of BFO and BFOSS are available from the (extensive) comments at the
beginning of the bfo.m and bfoss.m files, respectively.

## Example of usage

The driver programs test_bfo_examples.m and test_bfo_examples_with_bfoss.m contains examples of calls to BFO and to
the BFOSS library. Just select the test example in the scripts and run:
```
>> test_bfo_examples
```
or 
```
>> test_bfo_examples_with_bfoss
```

## Main references

Please refer to the following papers if you make use of any part of this code in your research:

* [M. Porcelli and Ph. L. Toint,
"BFO, a trainable derivative-free Brute Force Optimizer for 
nonlinear bound-constrained optimization and equilibrium 
computations with continuous and discrete variables", ACM Transactions on Mathematical Software, 44:1 (2017), Article 6.](https://dl.acm.org/doi/10.1145/3085592)

* [M. Porcelli and Ph. L. Toint,
"A note on using performance and data profiles for training algorithms", ACM Transactions on Mathematical Software, 45:2 (2019), Article 20.](https://dl.acm.org/doi/10.1145/3310362)

* [M. Porcelli and Ph. L. Toint,
Exploiting problem structure in derivative free optimization, ACM Transactions on Mathematical Software, 48:1 (2022), Article 6.]
(https://doi.org/10.1145/3474054)

## Authors 

[Margherita Porcelli](https://sites.google.com/view/margherita-porcelli/) (University of Bologna, Italy) and [Philippe L. Toint](http://perso.fundp.ac.be/~phtoint/toint.html) (University of Namur, Belgium)

## Contributors 

**BFOSS** benefits from contributions by Serge Gratton (ENSEEIHT, Toulouse, France) and Anke Troeltzsch (German Aerospace Center, DLR, Germany).

## License

## Conditions of use
Use at your own risk! No guarantee of any kind given or implied.

Copyright (c) 2020, Ph. Toint and M. Porcelli. All rights reserved.

Redistribution and use of the BFO package in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

## Disclaimer 
This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but nor limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright owner or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.  

