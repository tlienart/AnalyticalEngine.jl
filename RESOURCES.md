# Resources

**Todo**: add status banner from actual repo + (if poss) num of stars, maybe table of the whole thing.

## Existing Libraries

* [H2O](https://github.com/h2oai/h2o-3)
* [MLR](https://github.com/mlr-org/mrl)
* [SkLearn](https://github.com/scikit-learn/scikit-learn)

### Benchmarks

* [benchm-ml](https://github.com/szilard/benchm-ml) and [GMB-perf](https://github.com/szilard/GBM-perf) that shows that LightGBM is now the best tool in town

## TO FIND

* ? clustering
* ? feature engineering
* ? stacking
* ? NLP
* ? cross validation (and variants)

## Data processing

* [JuliaDB](https://github.com/JuliaComputing/JuliaDB.jl)
    - [Tutorial on JuliaDB](https://github.com/piever/JuliaDBTutorial/blob/master/hflights.ipynb)
* [DataFrames](https://github.com/JuliaData/DataFrames.jl)
* [CSV](https://github.com/JuliaData/CSV.jl)
* [LabelUtils](https://github.com/JuliaML/MLLabelUtils.jl)
* [Preprocessing](https://github.com/JuliaML/MLPreprocessing.jl)
* [Transformations](https://github.com/JuliaML/Transformations.jl)

### Datasets

* [MLDataSets](https://github.com/JuliaML/MLDatasets.jl)

## Loss functions, penalties, metrics

_The current path is to re-implement a significant portion of it as it is a key element to defining an ML model. Of course large portions of the re-implemented version will be very close to existing implementations_

* [LossFunctions](https://github.com/JuliaML/LossFunctions.jl)
* [PenaltyFunctions](https://github.com/JuliaML/PenaltyFunctions.jl)
* [MLMetrics](https://github.com/JuliaML/MLMetrics.jl)

## Regression

* [Regression](https://github.com/lindahua/Regression.jl)
* [GLM](https://github.com/JuliaStats/GLM.jl)
* [SparseRegression](https://github.com/joshday/SparseRegression.jl)
* [Parallel Sparse Regression with ADMM](https://github.com/madeleineudell/ParallelSparseRegression.jl) now defunct
* [Linear Least Squares](https://github.com/davidlizeng/LinearLeastSquares.jl) now seems defunct
* [LsqFit](https://github.com/JuliaNLSolvers/LsqFit.jl)

## Classification

* [kNN](https://github.com/johnmyleswhite/kNN.jl)
* [SVM](https://github.com/JuliaStats/SVM.jl)
* [LIBSVM](https://github.com/mpastell/LIBSVM.jl)
* [KSVM](https://github.com/Evizero/KSVM.jl)

## Low Rank Models

* [Low Rank Models](https://github.com/madeleineudell/LowRankModels.jl)

## Meta

* [Learning Strategies](https://github.com/JuliaML/LearningStrategies.jl)
* [Iteration Managers](https://github.com/sglyon/IterationManagers.jl)
* [Gaussian processes](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Gaussian_Processes_Julia.ipynb)
* [MLBase](https://github.com/JuliaStats/MLBase.jl)
* [Bags of little bootstrap](https://gist.github.com/jiahao/7033758)
* [Orchestra.jl (ensembles)](https://github.com/svs14/Orchestra.jl) now defunct
* [GradientBoost.jl](https://github.com/svs14/GradientBoost.jl) now defunct

## Tree models

* [DecisionTree](https://github.com/bensadeghi/DecisionTree.jl)
    * [Google PLANET model](http://static.googleusercontent.com/media/research.google.com/en/us/pubs/archive/36296.pdf)
* [XGBoost](https://github.com/dmlc/XGBoost.jl)
* [LightGBM](https://github.com/Allardvm/LightGBM.jl)

## Optimisation

* [Optim](https://github.com/JuliaNLSolvers/Optim.jl)
* [Stochastic Optim](https://github.com/JuliaML/StochasticOptimization.jl)
* [Prox](https://github.com/JuliaML/Prox.jl)

### Specific solvers that can be looked into

* [Liblinear](https://github.com/cjlin1/liblinear) used by sklearn

## Differentiable programs

* [Flux](https://github.com/FluxML/Flux.jl)
* [KNet](https://github.com/denizyuret/Knet.jl)
