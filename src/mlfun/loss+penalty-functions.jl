export
    NoLoss, NoPenalty,
    LPDistLoss, LPPenalty,
    L1DistLoss, L1Penalty,
    L2DistLoss, L2Penalty

# ============================
## None based loss and penalty
# ============================
"""
    NoLoss <: AtomicLoss
"""
struct NoLoss <: AtomicLoss end
(l::NoLoss)(x, y) = 0

"""
    NoPenalty <: AtomicPenalty
"""
struct NoPenalty <: AtomicPenalty end
(p::NoPenalty)(θ) = 0

# ====================================
## LP-based loss and penalty functions
# ====================================

"""
    LPDistLoss{P} <: AtomicLoss
"""
struct LPDistLoss{P} <: AtomicLoss where P <: Real end

(l::LPDistLoss)(x, y) = norm(x .- y, getp(l))

"""
    LPPenalty{P} <: AtomicPenalty
"""
struct LPPenalty{P} <: AtomicPenalty where P <: Real end

(p::LPPenalty)(θ) = norm(θ, getp(p))

## Shortcuts

const L1DistLoss = LPDistLoss{1}
const L1Penalty = LPPenalty{1}
const L2DistLoss = LPDistLoss{2}
const L2Penalty = LPPenalty{2}
