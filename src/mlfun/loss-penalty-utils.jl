function lp(v, p)
    (p==Inf) && return maximum(v)
    (p==2) && return sum(abs2.(v))
    (p>0) && return sum(abs.(v).^p)
    throw(DomainError())
end

getp(lpc::LPCost{p}) where p = p
