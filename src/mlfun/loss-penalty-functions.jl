export
    # Loss / Penalty types
    LPDistLoss, LPPenalty,
    L1DistLoss, L1Penalty,
    L2DistLoss, L2Penalty,
    LogisticLoss,
    # Utilities
    isdifferentiable, getp

## NoLoss / NoPenalty properties

(l::NoLoss)(x, y) = 0
(p::NoPenalty)(θ) = 0

isdifferentiable(nc::NoCost) = true

# ====================================
## LP-based loss and penalty functions
# ====================================
"""
    LPDistLoss{p} <: AtomicLoss
"""
struct LPDistLoss{p} <: AtomicLoss where p <: Real end

"""
    LPPenalty{P} <: AtomicPenalty
"""
struct LPPenalty{p} <: AtomicPenalty where p <: Real end

## Useful Shortcuts

const L1DistLoss = LPDistLoss{1}
const L1Penalty = LPPenalty{1}
const L2DistLoss = LPDistLoss{2}
const L2Penalty = LPPenalty{2}
const LPCost{p} = Union{LPDistLoss{p}, LPPenalty{p}}

(l::LPDistLoss)(x, y) = lp(x .- y, getp(l))
(p::LPPenalty)(θ) = lp(θ, getp(p))

isdifferentiable(lpc::LPCost{p}) where p = (p > 1)

"""
    LogisticLoss <: AtomicLoss
"""
struct LogisticLoss <: AtomicLoss end

(::LogisticLoss)(x, y) = -sum(logsigmoid, x .* y)
