export ObjectiveFunction,
    NoLoss, NoPenalty,
    AtomicLoss, AtomicPenalty,
    ScaledLoss, ScaledPenalty,
    CompositeLoss, CompositePenalty

import Base.+, Base.-, Base.*, Base./, Base.convert

# (x, y, θ) -> L(x, y, θ) but likely never expressed
abstract type ObjectiveFunction end

# (x, y) -> L(x, y)
abstract type Loss <: ObjectiveFunction end
struct NoLoss <: Loss end
abstract type AtomicLoss <: Loss end
mutable struct ScaledLoss{AL} <: Loss where AL <: AtomicLoss
    loss::AL
    scale::Real
end
mutable struct CompositeLoss <: Loss
    losses::Vector{ScaledLoss}
end

(sl::ScaledLoss)(x, y) = sl.scale .* sl.loss(x, y)
(cl::CompositeLoss)(x, y) = sum(loss(x, y) for loss ∈ cl.losses)

# θ -> P(θ)
abstract type Penalty <: ObjectiveFunction end
struct NoPenalty <: Loss end
abstract type AtomicPenalty <: Penalty end
mutable struct ScaledPenalty{AP} <: Penalty where AP <: AtomicPenalty
    penalty::AP
    scale::Real
end
mutable struct CompositePenalty <: Penalty
    penalties::Vector{ScaledPenalty}
end

(sp::ScaledPenalty)(θ) = sp.scale .* sp.penalty(θ)
(cl::CompositePenalty)(θ) = sum(loss_i(θ) for loss_i in cl.penalties)

# =====================================
# Composition of Loss/Penalty functions
# =====================================

const AL = AtomicLoss
const AP = AtomicPenalty
const SL = ScaledLoss
const CL = CompositeLoss
const SP = ScaledPenalty
const CP = CompositePenalty
const NL = NoLoss
const NP = NoPenalty
const NoCost = Union{NL, NP}

convert(::Type{T}, a::AL) where T <: Union{NL, SL, CL} = ScaledLoss(a, 1)
convert(::Type{T}, a::AP) where T <: Union{NP, SP, CP} = ScaledPenalty(a, 1)

*(::NoLoss, ::Real) = NoLoss()
*(::Real, ::NoLoss) = NoLoss()
+(::NoLoss, l::Loss) = l
+(l::Loss, ::NoLoss) = l

*(::NoPenalty, ::Real) = NoPenalty()
*(::Real, ::NoPenalty) = NoPenalty()
+(::NoPenalty, p::Penalty) = p
+(p::Penalty, ::NoPenalty) = p

+(a::AL, b::AL) = convert(ScaledLoss, a) + convert(ScaledLoss, b)
+(a::AL, b::Union{SL, CL}) = convert(ScaledLoss, a) + b
+(b::Union{SL, CL}, a::AL) = a + b
*(a::AL, c::Real) = ScaledLoss(a, c)
*(c::Real, a::AL) = a * c

+(a::AP, b::AP) = convert(ScaledPenalty, a) + convert(ScaledPenalty, b)
+(a::AP, b::Union{SP, CP}) = convert(ScaledPenalty, a) + b
+(b::Union{SP, CP}, a::AP) = a + b
*(a::AP, c::Real) = ScaledPenalty(a, c)
*(c::Real, a::AP) = a * c

+(a::SL, b::SL) = CL([a, b])
+(a::CL, b::CL) = CL(vcat(a.losses, b.losses))
+(a::SL, c::CL) = CL(vcat(c.losses, a))
+(c::CL, a::SL) = a + c
*(a::SL, c::Real) = SL(a.loss, c * a.scale)
*(c::Real, a::SL) = a * c
*(c::Real, a::CL) = CL(a.losses .* c)
*(a::CL, c::Real) = c * a

+(a::SP, b::SP) = CP([a, b])
+(a::CP, b::CP) = CP(vcat(a.penalties, b.penalties))
+(a::SP, c::CP) = CP(vcat(c.penalties, a))
+(c::CP, a::SP) = a + c
*(a::SP, c::Real) = SP(a.penalty, c * a.scale)
*(c::Real, a::SP) = a * c
*(c::Real, a::CP) = CP(a.penalties .* c)
*(a::CP, c::Real) = c * a

-(a::ObjectiveFunction, b::ObjectiveFunction) = a + (-1 * b)
/(a::ObjectiveFunction, c::Real) = a * (1 / c)
