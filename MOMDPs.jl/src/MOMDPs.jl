module MOMDPs

using POMDPs
using Distributions
using POMDPModelTools
using Random

export
    MOMDP,
    transitionvisible,
    transitionhidden,
    visiblestateindex,
    hiddenstateindex,
    visiblestates,
    hiddenstates,
    visible,
    hidden,
    visiblestatetype,
    hiddenstatetype,
    ordered_visible_states,
    ordered_hidden_states,
    obs_weight

include("momdp.jl")
include("space.jl")
include("type_inference.jl")
include("ordered_spaces.jl")

end # module
