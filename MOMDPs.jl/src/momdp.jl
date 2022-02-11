abstract type MOMDP{Sv, Sh, A, O} <: POMDP{Tuple{Sv,Sh}, A, O} end

## momdp.jl

"""discount(m::MOMDP)"""
function discount end

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

"""
    observation(m::MOMDP, statep)
    observation(m::MOMDP, action, statep)
    observation(m::MOMDP, state, action, statep)
"""
function observation end
observation(problem::MOMDP, a, sp) = observation(problem, sp)
observation(problem::MOMDP, s, a, sp) = observation(problem, a, sp)

"""
    reward(m::MOMDP, s, a)
Return the immediate reward for the s-a pair.
    reward(m::MOMDP, s, a, sp)
Return the immediate reward for the s-a-s′ triple.
    reward(m::MOMDP, s, a, sp, o)
"""
function reward end
reward(m::MOMDP, s, a, sp) = reward(m, s, a)
reward(m::MOMDP, s, a, sp, o) = reward(m, s, a, sp)

isterminal(problem::MOMDP, state) = false

"""initialstate(m::MOMDP)"""
function initialstate end

"""initialobs(m::MOMDP, s)"""
function initialobs end

"""stateindex(problem::MOMDP, s)"""
stateindex(m::MOMDP, s) = findfirst(map(s′->s′ == s, states(m)))

"""visiblestateindex(problem::MOMDP, s)"""
visiblestateindex(m::MOMDP{Sv,Sh,A,O}, s::Tuple{Sv,Sh}) where {Sv,Sh,A,O} = findfirst(map(sᵥ′->sᵥ′ == visible(m, s), visiblestates(m)))
visiblestateindex(m::MOMDP, sᵥ::Sv) where {Sv,Sh,A,O} = findfirst(map(sᵥ′->sᵥ′ == sᵥ, visiblestates(m)))

"""hiddenstateindex(problem::MOMDP, s)"""
hiddenstateindex(m::MOMDP{Sv,Sh,A,O}, s::Tuple{Sv,Sh}) where {Sv,Sh,A,O} = findfirst(map(sₕ′->sₕ′ == hidden(s), hiddenstates(m)))
hiddenstateindex(m::MOMDP{Sv,Sh,A,O}, sₕ::Sh) where {Sv,Sh,A,O} = findfirst(map(sₕ′->sₕ′ == sₕ, hiddenstates(m)))

"""actionindex(problem::MOMDP, a)"""
actionindex(m::MOMDP, a) = findfirst(map(a′->a′ == a, actions(m)))

"""obsindex(problem::MOMDP, o)"""
obsindex(m::MOMDP, o) = findfirst(map(o′->o′ == o, observations(m)))


## space.jl

"""states(problem::MOMDP)"""
function states end

"""visiblestates(problem::MOMDP)"""
function visiblestates end

"""hiddenstates(problem::MOMDP)"""
function hiddenstates end

"""visible(problem::MOMDP, s)"""
visible(s) = s[1]
visible(m::MOMDP, s) = visible(s)

"""hidden(problem::MOMDP, s)"""
hidden(s) = s[2]
hidden(m::MOMDP, s) = hidden(s)

"""actions(problem::MOMDP)"""
function actions end
actions(problem::MOMDP, state) = actions(problem)

"""observations(problem::MOMDP)"""
function observations end
observations(problem::MOMDP, state) = observations(problem)

## type_inference.jl

"""
```
type A <: MOMDP{Int, Float64, Bool, Bool} end
statetype(A) # returns Tuple{Int, Float64}
```
"""
statetype(t::Type) = statetype(supertype(t))
statetype(t::Type{MOMDP{Sv,Sh,A,O}}) where {Sv,Sh,A,O} = Tuple{Sv,Sh}
statetype(p::MOMDP) = statetype(typeof(p))

visiblestatetype(t::Type) = visiblestatetype(supertype(t))
visiblestatetype(t::Type{MOMDP{Sv,Sh,A,O}}) where {Sv,Sh,A,O} = Sv
visiblestatetype(p::MOMDP) = visiblestatetype(typeof(p))

hiddenstatetype(t::Type) = hiddenstatetype(supertype(t))
hiddenstatetype(t::Type{MOMDP{Sv,Sh,A,O}}) where {Sv,Sh,A,O} = Sh
hiddenstatetype(p::MOMDP) = hiddenstatetype(typeof(p))

"""
```
type A <: MOMDP{Bool, Bool, Int, Bool} end
actiontype(A) # returns Int
```
"""
actiontype(t::Type) = actiontype(supertype(t))
actiontype(t::Type{MOMDP{Sv,Sh,A,O}}) where {Sv,Sh,A,O} = A
actiontype(p::MOMDP) = actiontype(typeof(p))

"""
```
type A <: MOMDP{Bool, Bool, Bool, Int} end
obstype(A) # returns Int
```
"""
obstype(t::Type) = obstype(supertype(t))
obstype(t::Type{MOMDP{Sv,Sh,A,O}}) where {Sv,Sh,A,O} = O
obstype(p::MOMDP) = obstype(typeof(p))


## POMDPModelTools: ordered_spaces.jl
"""
    ordered_visible_states(mdp)    
Return an `AbstractVector` of states ordered according to `visiblestateindex(mdp, a)`.
`ordered_visible_states(mdp)` will always return a `AbstractVector{A}` `v` containing all of the visible states in `visiblestates(mdp)` in the order such that `visiblestateindex(mdp, v[i]) == i`. You may wish to override this for your problem for efficiency.
"""
ordered_visible_states(m::MOMDP) = POMDPModelTools.ordered_vector(visiblestatetype(typeof(m)), s->visiblestateindex(m,s), visiblestates(m), "visiblestate")

"""
    ordered_hidden_states(mdp)    
Return an `AbstractVector` of states ordered according to `hiddenstateindex(mdp, a)`.
`ordered_hidden_states(mdp)` will always return a `AbstractVector{A}` `v` containing all of the hidden states in `hiddenstates(mdp)` in the order such that `hiddenstateindex(mdp, v[i]) == i`. You may wish to override this for your problem for efficiency.
"""
ordered_hidden_states(m::MOMDP) = POMDPModelTools.ordered_vector(hiddenstatetype(typeof(m)), s->hiddenstateindex(m,s), hiddenstates(m), "hiddenstate")

"""
    ordered_actions(mdp)    
Return an `AbstractVector` of actions ordered according to `actionindex(mdp, a)`.
`ordered_actions(mdp)` will always return an `AbstractVector{A}` `v` containing all of the actions in `actions(mdp)` in the order such that `actionindex(mdp, v[i]) == i`. You may wish to override this for your problem for efficiency.
"""
ordered_actions(m::MOMDP) = POMDPModelTools.ordered_vector(actiontype(typeof(m)), a->actionindex(m,a), actions(m), "action")

"""
    ordered_states(mdp)    
Return an `AbstractVector` of states ordered according to `stateindex(mdp, a)`.
`ordered_states(mdp)` will always return a `AbstractVector{A}` `v` containing all of the states in `states(mdp)` in the order such that `stateindex(mdp, v[i]) == i`. You may wish to override this for your problem for efficiency.
"""
ordered_states(m::MOMDP) = POMDPModelTools.ordered_vector(statetype(typeof(m)), s->stateindex(m,s), states(m), "state")

"""
    ordered_observations(momdp)    
Return an `AbstractVector` of observations ordered according to `obsindex(momdp, a)`.
`ordered_observations(mdp)` will always return a `AbstractVector{A}` `v` containing all of the observations in `observations(momdp)` in the order such that `obsindex(momdp, v[i]) == i`. You may wish to override this for your problem for efficiency.
"""
ordered_observations(m::MOMDP) = POMDPModelTools.ordered_vector(obstype(typeof(m)), o->obsindex(m,o), observations(m), "observation")


## POMDPModelTools: obs_weight.jl
obs_weight(p, s, a, sp, o) = pdf(observation(p, s, a, sp), o)


## simulator.jl
function simulate end
