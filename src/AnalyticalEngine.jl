module AnalyticalEngine

struct UnimplementedException <: Exception end

export fit!, predict


include("types.jl")

## Loss functions and penalties

include("mlfun/types-loss-penalty.jl")
include("mlfun/loss+penalty-functions.jl")
include("mlfun/utils-loss-penalty.jl")


## SUPERVISED MODELS

include("supervised/glr.jl")     # generalized linear regression
include("supervised/glr_fit.jl")
include("supervised/glr_predict.jl")


end # AnalyticalEngine
