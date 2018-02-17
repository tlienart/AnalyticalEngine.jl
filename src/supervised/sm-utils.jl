#=
Utility functions for supervised models.
=#

# read a refvalue explicitly (to clarify code)
get(rv::Base.RefValue{T}) where T = rv.x

# function to mutate the hyper-parameters of a model
function set!(model::SupervisedModel, d::Dict)
	# retrieve the hyperparameters
	hp = hyperparameters(model)

	# check that the given dictionary matches the keys
	@assert issubset(keys(d), keys(hp)) "some keys don't match hyperparams"

	# go over the keyvalue pairs and set.
	# what this does is: access the references returned by the hyparparameters
	# and sets them (Pointer access) so mod is mutated in the end.
	for kv ∈ d
		# hp[kv[1]] is the symbol (e.g. :loss)
		# kv[2] is the value (e.g. a new loss)
		setindex!(hp[kv[1]], kv[2])
	end
	return model
end

# Setting without mutating -> copy then set!.
set(mod::SupervisedModel, d::Dict) = (modc=deepcopy(mod); set!(modc, d); modc)
