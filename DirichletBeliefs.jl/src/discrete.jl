# Goals: minimize calls to ordered_hidden_states (allocates memory)

# needs MOMDPs for hiddenstateindex in pdf(b, s)
# needs list of ordered_hidden_states for rand(b)

"""
    DiscreteSubspaceBelief

A belief specified by a probability vector.

Normalization of `b` is assumed in some calculations (e.g. pdf), but it is only automatically enforced in `update(...)`, and a warning is given if normalized incorrectly in `DiscreteSubspaceBelief(momdp, b)`.

# Constructor
    DiscreteSubspaceBelief(momdp, b::Vector{Float64}; check::Bool=true)

# Fields 
- `momdp` : the POMDP problem  
- `hidden_state_list` : a vector of ordered hidden states
- `b` : the probability vector 
"""
mutable struct DiscreteSubspaceBelief{M<:MOMDP, Sh}
    momdp::M
    hidden_state_list::Vector{Sh}       # vector of ordered states
    b::Vector{Float64}
    visiblestate::Any
end

function (b::DiscreteSubspaceBelief)(sv)
    b.visiblestate = sv
    return b
end

function DiscreteSubspaceBelief(momdp::MOMDP{Sv,Sh,A,O}, sv₀::Sv) where {Sv,Sh,A,O}
    b = uniform_belief(momdp)
    b(sv₀)
    return b
end

DiscreteSubspaceBelief(momdp::M) where M <: MOMDP = uniform_belief(momdp)

function DiscreteSubspaceBelief(momdp, b::Vector{Float64}; check::Bool=true)
    if check
        if !isapprox(sum(b), 1.0, atol=0.001)
            @warn """
                  b in DiscreteSubspaceBelief(momdp, b) does not sum to 1.
 
                  To suppress this warning use `DiscreteSubspaceBelief(momdp, b, check=false)`
                  """
        end
        if !all(0.0 <= p <= 1.0 for p in b)
            @warn """
                  b in DiscreteSubspaceBelief(momdp, b) contains entries outside [0,1].
 
                  To suppress this warning use `DiscreteSubspaceBelief(momdp, b, check=false)`
                  """
        end
    end
    return DiscreteSubspaceBelief(momdp, ordered_hidden_states(momdp), b, nothing)
end


"""
     uniform_belief(momdp)

Return a DiscreteSubspaceBelief with equal probability for each state.
"""
function uniform_belief(momdp::MOMDP)
    hidden_state_list = ordered_hidden_states(momdp)
    ns = length(hidden_state_list)
    return DiscreteSubspaceBelief(momdp, hidden_state_list, ones(ns) / ns, nothing)
end

pdf(b::DiscreteSubspaceBelief, s) = b.b[hiddenstateindex(b.momdp, s)]

function Random.rand(rng::Random.AbstractRNG, b::DiscreteSubspaceBelief)
    i = sample(rng, Weights(b.b))
    return (b.visiblestate, b.hidden_state_list[i])
end

function Base.fill!(b::DiscreteSubspaceBelief, x::Float64)
    fill!(b.b, x)
    return b
end

Base.length(b::DiscreteSubspaceBelief) = length(b.b)

support(b::DiscreteSubspaceBelief) = b.hidden_state_list

Statistics.mean(b::DiscreteSubspaceBelief) = sum(b.hidden_state_list.*b.b)/sum(b.b)
StatsBase.mode(b::DiscreteSubspaceBelief) = b.hidden_state_list[argmax(b.b)]

Base.:(==)(b1::DiscreteSubspaceBelief, b2::DiscreteSubspaceBelief) = b1.hidden_state_list == b2.hidden_state_list && b1.b == b2.b
Base.hash(b::DiscreteSubspaceBelief, h::UInt) = hash(b.b, hash(b.hidden_state_list, h))

"""
    DiscreteSubspaceUpdater

An updater type to update discrete belief using the discrete Bayesian filter.

# Constructor
    DiscreteSubspaceUpdater(momdp::MOMDP)

# Fields
- `momdp <: MOMDP`
"""
mutable struct DiscreteSubspaceUpdater{M<:MOMDP} <: Updater
    momdp::M
end

uniform_belief(up::DiscreteSubspaceUpdater) = uniform_belief(up.momdp)

function initialize_belief(up::DiscreteSubspaceUpdater, dist::Categorical)
    hidden_state_list = ordered_hidden_states(up.momdp)
    ns = length(hidden_state_list)
    b = zeros(ns)
    belief = DiscreteSubspaceBelief(up.momdp, hidden_state_list, b, nothing)
    for (sidx, s) in enumerate(support(dist))
        # sidx = hiddenstateindex(up.momdp, s)
        belief.b[sidx] = pdf(dist, s)
    end
    return belief
end

function update(up::DiscreteSubspaceUpdater, b::DiscreteSubspaceBelief, a, o)
    momdp = up.momdp
    hidden_state_space = b.hidden_state_list
    bp = zeros(length(hidden_state_space))
    sᵥ = visible(momdp, o)

    # TODO: Bake in the population belief update here.
    p_population = [1.1479e-5, 0.573952, 0.346954, 0.0705961, 0.00848645]

    # for (shi, sh) in enumerate(hidden_state_space)
    #     bp[shi] = p_population[shi] * obs_weight(momdp, sh, a, sh, o) # NOTE: s=sh, s′=sh
    # end

    # bp[end] = 1
    # bp = p_population .* map(sh->obs_weight(momdp, sh, a, sh, o), hidden_state_space)
    # bp = map(sh->obs_weight(momdp, sh, a, sh, o), hidden_state_space)

    for (shi, sh) in enumerate(hidden_state_space)
        if pdf(b, sh) > 0.0
            T = transitionhidden(momdp, sh, a, o)

            for (sp, tp) in weighted_iterator(T)
                spi = hiddenstateindex(momdp, sp)
                op = obs_weight(momdp, sh, a, sp, o)
                # bp[spi] += op * tp * b.b[shi]
                bp[spi] += op * tp * p_population[shi]
            end
        end
    end

    bp_sum = sum(bp)

    if bp_sum == 0.0
        error("""
              Failed discrete belief update: new probabilities sum to zero.

              b = $b
              a = $a
              o = $o

              Failed discrete belief update: new probabilities sum to zero.
              """)
    end

    # Normalize
    bp ./= bp_sum


    return DiscreteSubspaceBelief(momdp, b.hidden_state_list, bp, sᵥ)
end

update(up::DiscreteSubspaceUpdater, b::Categorical, a, o) = update(up, initialize_belief(up, b), a, o)