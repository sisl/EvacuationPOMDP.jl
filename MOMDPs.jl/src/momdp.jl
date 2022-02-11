abstract type MOMDP{Sv,Sh,A,O} <: POMDP{Tuple{Sv,Sh},A,O} end

"""
    transitionvisible(m::MOMDP, s, a)

Tᵥ(s,a,v′) = p(v′ | s, a) where s = (v,h)
"""
function transitionvisible end

"""
    transitionhidden(m::MOMDP, s, a, sp_visible)

Tₕ(s,a,v′,h′) = p(h′ | s, a, v′) where s = (v,h)
"""
function transitionhidden end

"""stateindex(problem::MOMDP, s)"""
POMDPs.stateindex(m::MOMDP, s) = findfirst(map(s′->s′ == s, POMDPs.states(m)))

"""visiblestateindex(problem::MOMDP, s)"""
visiblestateindex(m::MOMDP{Sv,Sh,A,O}, s::Tuple{Sv,Sh}) where {Sv,Sh,A,O} = findfirst(map(sᵥ′->sᵥ′ == visible(m, s), visiblestates(m)))
visiblestateindex(m::MOMDP, sᵥ::Sv) where {Sv,Sh,A,O} = findfirst(map(sᵥ′->sᵥ′ == sᵥ, visiblestates(m)))

"""hiddenstateindex(problem::MOMDP, s)"""
hiddenstateindex(m::MOMDP{Sv,Sh,A,O}, s::Tuple{Sv,Sh}) where {Sv,Sh,A,O} = findfirst(map(sₕ′->sₕ′ == hidden(s), hiddenstates(m)))
hiddenstateindex(m::MOMDP{Sv,Sh,A,O}, sₕ::Sh) where {Sv,Sh,A,O} = findfirst(map(sₕ′->sₕ′ == sₕ, hiddenstates(m)))

"""actionindex(problem::MOMDP, a)"""
POMDPs.actionindex(m::MOMDP, a) = findfirst(map(a′->a′ == a, POMDPs.actions(m)))

"""obsindex(problem::MOMDP, o)"""
POMDPs.obsindex(m::MOMDP, o) = findfirst(map(o′->o′ == o, POMDPs.observations(m)))
