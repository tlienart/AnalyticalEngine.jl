# TODO

This is an un-curated list of things that should be done.
Feel free to add points, please add a TAG to indicate how hard it is + how urgent it is.

**Brainstorm**: [hackmd](https://hackmd.io/JwQ2CMDYGZwBgLQHY4FMBMCAsBjcWF8ATJBAVmAEYyAOSgMwskriA===#)

## General organisation

* This package will draw from everywhere so probably good to have a decent way of listing packages that this package relies on + how to deal with the license.

## Misc

* (**low priority, easy**) ability to copy models (e.g. `ols = LinearRegression()` and `ols_copy = copy(ols)`)

## Meta Learning and Hyperparam tuning

* (**high priority, hard**) Think about how to construct meta learners in a way that is clean and easy and whether that forces early design choices. Same with hyperparameter tuning (especially with a model like a pipeline)
  - think about pipelines, autotuning etc and how that would work
  - have a look at sklearn code for pipeline
  - usecase1: `tuningof(variableselectionof(baselearner,varctrl),tunectrl)`
  - usecase2: `pipeline(preproc, trafo, baselearner)`
  - usecase3: `stack([learner1,learner2,learner3], lasso)`

## Generalized Linear Regression

* (**medium priority, think**) with or without intercept, how to avoid copying data? `hcat(ones(n), X)` with intercept seems inefficient?
* (**low priority, think**) loss through "activation function" so for example `|y-f(Xb)|₂` for some f (generalised least square). Bayesian LogReg is sometimes that btw with `f=σ`.
* (**low priority, easy**) there are a bunch of arguments that appear for multiple short regression like `fit_intercept`, `avgloss` and `avgpenalty`. Maybe good to use `kwargs...` or something.
* (**low priority, think**) api consistency, use `average_loss` instead of `avgloss`? same for penalty.
* (**high priority, think**) corresponding to the hierarchy of types there is a hierarchy of solver. Currently we write solvers for specific leaf nodes but we should use generic solvers as backend. This should be expressed neatly in the parent `fit!` which, potentially, should delegate to either specific fit functions `fit_ols`, `fit_ridge`, `fit_lasso`, etc. as well as have generic solver `flux` (needs differentiable stuff), `optim.jl` etc where for example we consider higher order questions like `isdifferentiable` `hasprox` etc.

### Ridge

* (**low priority, easy**) Allow for element-wise penalty, will change to `X'X+Diagonal(λ)`
* (**low priority, easy**) Clean up conditions, see [in OnlineStats](https://github.com/joshday/OnlineStats.jl/blob/master/src/stats/linregbuilder.jl) what they're doing (posdef check + symmetric)
* (**low priority, medium**) If `X` is very sparse, the analytical solution `X' * X + λ * eye(p) \ (X' * y)` probably sucks. Should at least use `speye`. Also probably avoid computing `X' * X`...

### Lasso

* (**high priority, medium**) Implement a FISTA backend (either that or look up what they do in SparseRegression and copy that)

### Flux Fit

* (**low priority, easy**) let the user specify an initialisation
* (**medium priority, easy**) check that the objective function is suitable for FLUX (needs to be differentiable)
* (**medium priority, medium**) think about the scaling. E.g.: raw L2 norm vs MSE, this will impact step size + balance between loss and penalty but should be something the user should be able to set easily. See `avgloss` and `avgpenalty` probably need to give it more thought.
* (**medium priority, medium**) passing a function called `update!` will cause issues if there's an equal in there so `foo(update!=update!)` will cause issues but `foo(update! = update!)` will work. This would be poor API design. Needs thought, see with @MikeInnes. Might make more sense to pass what is iterated over:

```julia
# could pass step num etc..
step(p, g, δ) = p .- δ * g
```

and then within `fit_flux` something like

```julia
function update!(params)
  for param in params
    param.data .= step(param.data, param.grad, 0.1)
    param.grad .= 0
  end
end
```

(see current solution and discuss)

* currently only works with 1 and 2 norm.
