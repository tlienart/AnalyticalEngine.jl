# Questions

See also all comments in the code marked with XXX

## (Jan 2018) Avoiding useless copies in `fit_intercept` vs not

In a linear model it has to be specified whether the intercept should be fit or not.
This changes `X` to `hcat(ones(size(X, 1)), X)`.
It seems rather inefficient to do the latter (copy + creation of a new object?).
Any thoughts on how this could be done more efficiently?
For some algorithms it may work to not expose this and work separately for the intercept and the coefficients but I don't think that works for all.
