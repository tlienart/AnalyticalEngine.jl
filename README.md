# AnalyticalEngine

[![Build Status](https://travis-ci.org/tlienart/AnalyticalEngine.jl.svg?branch=master)](https://travis-ci.org/tlienart/AnalyticalEngine.jl)

[![codecov.io](http://codecov.io/github/tlienart/AnalyticalEngine.jl/coverage.svg?branch=master)](http://codecov.io/github/tlienart/AnalyticalEngine.jl?branch=master)

## Who's behind this

* Thibaut Lienart (Imperial College London)
* Mike Innes (Julia Computing)
* Sebastian Vollmer (University of Warwick, Alan Turing Institute)
* Franz Kiraly (University College London)
* Avik Sengupta (Julia Computing)
* Valentin Churavy (Massachusetts Institute of Technology)

## Aims and Milestones

### Milestones

* March 2018
  - [**working prototype**] ~~have a basic `GeneralizedLinearRegression` that works well and showcases the ideas + works with Flux~~
  - [**WIP**] have a basic pipeline `JuliaDB -> AnalyticalEngine`
  - have an interface with `DecisionTree.jl`
* August 2018
  - have a full pipeline `JuliaDB -> FeatEng -> AnalyticalEngine`
  - have a working framework for metalearning
  - have working tools for hyperparameter tuning (BayesianOpt, K-Folds, ...)

### Aims

* **Major aims**: offer a modern SkLearn-style package that can:
  - work efficiently with large databases (via JuliaDB)
  - work efficiently with different compute infrastructure (parallel, GPU, ...)
  - work with generic optimisation algorithms (via `Optim.jl`)
  - work with auto-diff algorithms (via `Flux.jl`)
  - offer extensible meta-learning
  - offer modern and extensible hyperparameter tuning (such as Bayesian opt)
  - be extended easily by researchers/users in such a way that the maths matches well with the code

### Inspiration

* [MLR in R](https://github.com/mlr-org/mlr) 
* [Sklearn in Python](https://github.com/scikit-learn/scikit-learn) as well as [contributions](https://github.com/scikit-learn-contrib) such as [sklearn pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) and [lightning](https://github.com/scikit-learn-contrib/lightning)
* [Sklearn in Julia](https://github.com/cstjean/ScikitLearn.jl) 
