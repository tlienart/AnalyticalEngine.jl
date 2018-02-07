# TODO

This is an un-curated list of things that should be done.
Feel free to add points, please add a TAG to indicate how hard it is + how urgent it is.

## General organisation

* This package will draw from everywhere so probably good to have a decent way of listing packages that this package relies on + how to deal with the license.

## Meta Learning and Hyperparam tuning

* (**high priority, hard**) Think about how to construct meta learners in a way that is clean and easy and whether that forces early design choices. Same with hyperparameter tuning (especially with a model like a pipeline)
  - think about pipelines, autotuning etc and how that would work
  - have a look at sklearn code for pipeline
  - usecase1: `tuningof(variableselectionof(baselearner,varctrl),tunectrl)`
  - usecase2: `pipeline(preproc, trafo, baselearner)`
  - usecase3: `stack([learner1,learner2,learner3], lasso)` 

## Generalized Linear Regression

### Flux fitting

* (**high priority, medium**) Check Flux example for fitting Linear Regression, mimick it. Check with @mike that the gradient is computing in most efficient way or whether would be more efficient to have analytical gradient (e.g. of p-norm)

### Ridge

* (**low priority, easy**) Allow for element-wise penalty, will change to `X'X+Diagonal(λ)`
* (**low priority, easy**) Clean up conditions, see [in OnlineStats](https://github.com/joshday/OnlineStats.jl/blob/master/src/stats/linregbuilder.jl) what they're doing (posdef check + symmetric)

### Lasso

* (**high priority, medium**) Implement a FISTA backend (either that or look up what they do in SparseRegression and copy that)

## API thinking

### Ensemble Learning / Meta Learning

* (**medium priority, hard**) Think about how the API for meta-learners would work efficiently (parallel etc) this should be done with @fkiraly
