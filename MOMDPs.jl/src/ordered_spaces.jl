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
