# retrieve p in an L_p norm
getp{P}(l::Union{LPDistLoss{P}, LPPenalty{P}}) = P
