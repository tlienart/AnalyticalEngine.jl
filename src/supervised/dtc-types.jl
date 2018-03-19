export
    DecisionTreeClassification, DTC, DecisionTreeClassifier

mutable struct DecisionTreeClassification <: ClassificationModel
    criterion::String
    max_depth::Union{Void, Int}
    rng::Union{Void, AbstractRNG}
    tree::Union{Void, DecisionTree.LeafOrNode}
end

# Useful shortcut
const DTC = DecisionTreeClassification

# Constructor
function DecisionTreeClassification(;
    criterion="gini",
    max_depth=nothing,
    rng=nothing)

    DTC(criterion,
        max_depth,
        rng)
end

# Synonym for SkLearn people
DecisionTreeClassifier(; kwargs...) = DecisionTreeClassification(kwargs...)

# Hyperparameters
hyperparameters(dtc::DTC) =
    (:criterion, :maxdepth, :rng)
