# TODO

## Generalized Linear Regression

### Ridge

* Allow for element-wise penalty, will change to `X'X+Diagonal(λ)`
* Clean up conditions, see [in OnlineStats](https://github.com/joshday/OnlineStats.jl/blob/master/src/stats/linregbuilder.jl) what they're doing (posdef check + symmetric)
