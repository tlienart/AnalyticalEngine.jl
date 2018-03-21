# AnalyticalEngine

[![Build Status](https://travis-ci.org/tlienart/AnalyticalEngine.jl.svg?branch=master)](https://travis-ci.org/tlienart/AnalyticalEngine.jl)

[![codecov.io](http://codecov.io/github/tlienart/AnalyticalEngine.jl/coverage.svg?branch=master)](http://codecov.io/github/tlienart/AnalyticalEngine.jl?branch=master)

## Aims and Milestones

### Milestones

* March 2018
  - [**working prototype**] ~~have a basic `GeneralizedLinearRegression` that works well and showcases the ideas + works with Flux~~
  - [**WIP**] have a basic pipeline `JuliaDB -> AnalyticalEngine`
  - have an interface with `DecisionTree.jl`
  - [**WIP**] have a way to deal with hyperparameters that works well with meta-learning
* August 2018
  - have a full pipeline `JuliaDB -> FeatEng -> AnalyticalEngine`
  - have a working framework for metalearning
  - have working tools for hyperparameter tuning (BayesianOpt, K-Folds, ...)
* Longer term
  - In the spirit of [MLR](https://github.com/mlr-org/mrl) we'd like to interface with as many dedicated packages ("solvers") as possible and promote the creation and maintenance of those.
    - In a first phase we won't care too much about this, focusing on the general pipeline, hyperparameter management etc but eventually this will become the key focus once we have a strong central API.
    - There are a ton of packages implementing / re-implementing specific capabilities, hopefully the API will lead to the merging / concentration of packages solving generic tasks efficiently

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
* Caret

## Quick examples

### Simple Linear Regression

```julia
using AnalyticalEngine
import RDatasets: dataset

boston = dataset("MASS", "boston")
y = convert(Array, boston[:MedV])
X = convert(Array, boston[[:LStat, :PTRatio, :Dis]])

lr = LinearRegression()
fit!(lr, X, y)

RSS = sum((y - predict(lr, X)).^2)
TSS = sum((y-mean(y)).^2)
R2 = 1 - RSS/TSS # ~0.63
```
