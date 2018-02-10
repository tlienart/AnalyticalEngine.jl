lp(v, p) = (p==Inf) ? maximum(v) : (
             (p==2) ? sum(abs2, v) : (
               (p>0) ? sum(abs.(v).^p) : throw(DomainError())))

getp(lpc::LPCost{P}) where P = P
