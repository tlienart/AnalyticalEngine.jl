#=
Utility functions for supervised models.
=#

# function to mutate the hyper-parameters of a model
function set!(model::SupervisedModel; kwargs...)
	# retrieve the symbols corresponding to hyperparameters
	hp = hyperparameters(model)

	for pair ∈ kwargs
		symbol = pair[1]
		@assert symbol ∈ hp "Unrecognised hyperparameter $symbol"
		value = pair[2]
		eval(:($model.$symbol = $value))
	end
	return model
end

# Setting without mutating -> copy then set!.
set(mod::SupervisedModel; kwargs...) =
	(modc=deepcopy(mod); set!(modc; kwargs...); modc)
