# Questions

See also all comments in the code marked with XXX

## Instances behaving like functions

Say you have

```julia
struct Foo
    x::Int
end

f = Foo(2.0)
```

is there a way to make instances like `f` behave like a function for example have `f(2.0)` correspond to `2.0+f.x` or something of the sorts?

## Avoiding useless copies in fit_intercept vs not

In a linear model it has to be specified whether the intercept should be fit or not. This changes `X` to `hcat(ones(size(X, 1)), X)`. It seems rather inefficient to do the latter (creation of a new object?). Any thoughts on how this could be done more efficiently?
