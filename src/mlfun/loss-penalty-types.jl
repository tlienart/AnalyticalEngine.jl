export ObjectiveFunction,
    AtomicLoss, AtomicPenalty,
    ScaledLoss, ScaledPenalty,
    CompositeLoss, CompositePenalty

import Base.+, Base.-, Base.*, Base./

# (x, y, θ) -> L(x, y, θ) but likely never expressed
abstract type ObjectiveFunction end

# (x, y) -> L(x, y)
abstract type Loss <: ObjectiveFunction end
abstract type AtomicLoss <: Loss end
mutable struct ScaledLoss{AL} <: Loss where AL <: AtomicLoss
    loss::AL
    scale::Real
end
const SAL = Union{AtomicLoss, ScaledLoss}
mutable struct CompositeLoss <: Loss
    losses::Vector{SAL}
end

(sl::ScaledLoss)(x, y) = sl.scale .* sl.loss(x, y)
(cl::CompositeLoss)(x, y) = sum(loss_i(x, y) for loss_i in cl.losses)

*(c::Real, al::AtomicLoss) = ScaledLoss(al, c)
*(al::AtomicLoss, c::Real) = c * al
*(c::Real, sl::ScaledLoss) = ScaledLoss(sl.loss, sl.scale * c)
*(sl::ScaledLoss, c::Real) = c * sl

# θ -> P(θ)
abstract type Penalty <: ObjectiveFunction end
abstract type AtomicPenalty <: Penalty end
mutable struct ScaledPenalty{AP} <: Penalty where AP <: AtomicPenalty
    penalty::AP
    scale::Real
end
const SAP = Union{AtomicPenalty, ScaledPenalty}
mutable struct CompositePenalty <: Penalty
    penalties::Vector{SAP}
end

(sp::ScaledPenalty)(θ) = sp.scale .* sp.penalty(θ)
(cl::CompositePenalty)(θ) = sum(loss_i(θ) for loss_i in cl.penalties)

*(c::Real, ap::AtomicPenalty) = ScaledPenalty(ap, c)
*(ap::AtomicPenalty, c::Real) = c * ap
*(c::Real, sp::ScaledPenalty) = ScaledPenalty(sp.penalty, sp.scale * c)
*(sp::ScaledPenalty, c::Real) = c * sp

# =====================================
# Composition of Loss/Penalty functions
# =====================================

+(la::SAL, lb::SAL) = CompositeLoss([la, lb])
+(lc::CompositeLoss, la::SAL) = CompositeLoss(vcat(lc.losses, la))
+(la::SAL, lc::CompositeLoss) = lc + la
+(lca::CompositeLoss, lcb::CompositeLoss) =
    CompositeLoss(vcat(lca.losses, lcb.losses))

-(la::SAL, lb::SAL) = CompositeLoss([la, -1lb])
-(lc::CompositeLoss, la::SAL) = CompositeLoss(vcat(lc.losses, -1la))
-(la::SAL, lc::CompositeLoss) = la + (-1lc)
-(lca::CompositeLoss, lcb::CompositeLoss) =
    CompositeLoss(vcat(lca.losses, -1lcb.losses))

+(pa::SAP, pb::SAP) = CompositePenalty([pa, pb])
+(pc::CompositePenalty, pa::SAP) = CompositePenalty(vcat(pc.penalties, pa))
+(pa::SAP, pc::CompositePenalty) = pc + pa
+(pca::CompositePenalty, pcb::CompositePenalty) =
    CompositePenalty(vcat(pca.penalties, pcb.penalties))

-(pa::SAP, pb::SAP) = CompositePenalty([pa, -1pb])
-(pc::CompositePenalty, pa::SAP) = CompositePenalty(vcat(pc.penalties, -1pa))
-(pa::SAP, pc::CompositePenalty) = pa + (-1pc)
-(pca::CompositePenalty, pcb::CompositePenalty) =
    CompositePenalty(vcat(pca.penalties, -1pcb.penalties))

/(se::Union{SAL, SAP}, c::Real) = se * (1/c)

*(c::Real, lc::CompositeLoss) = CompositeLoss(c * lc.losses)
*(lc::CompositeLoss, c::Real) = CompositeLoss(lc.losses * c)

*(c::Real, pc::CompositePenalty) = CompositePenalty(c * pc.penalties)
*(pc::CompositePenalty, c::Real) = CompositePenalty(pc.penalties * c)

/(comp::Union{CompositeLoss, CompositePenalty}, c::Real) = comp * (1/c)
