# retrieve p in an L_p norm
getp(l::Union{LPDistLoss{P}, LPPenalty{P}}) where P = P
