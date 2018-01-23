export
    SupervisedModel,
    RegressionModel, ClassificationModel


doc"""
    SupervisedModel

Any model trying to represent a relation `(x, y)` where `x` is some
input data and y is an output associated to x.
"""
abstract type SupervisedModel end

doc"""
    RegressionModel <: SupervisedModel

Any `SupervisedModel` where the output is on a continuous scale.
"""
abstract type RegressionModel <: SupervisedModel end

doc"""
    ClassificationModel <: SupervisedModel

Any `SupervisedModel` where the output is a class out of a finite set
of possible classes.
"""
abstract type ClassificationModel <: SupervisedModel end
