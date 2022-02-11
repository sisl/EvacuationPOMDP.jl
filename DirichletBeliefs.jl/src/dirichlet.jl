mutable struct DirichletBelief{P<:POMDP, S}
    pomdp::P
    state_list::Vector{S}
    b::Dirichlet
end

DirichletBelief(pomdp::P) where P <: POMDP = uniform_belief(pomdp)

function DirichletBelief(pomdp::P, α::Vector{<:Real}) where P <: POMDP
    state_list = ordered_states(pomdp)
    if length(α) != length(state_list)
        error("The Dirichlet α length ($(length(α))) does not match the number of states ($(length(state_list))).\n
               If this is intended, see `DirichletSubspaceBelief` and `DirichletSubspaceUpdater`.")
    end
    α = convert(Vector{Float64}, α)
    b = Dirichlet(α)
    return DirichletBelief(pomdp, state_list, b)
end

function uniform_belief(pomdp::P) where P <: POMDP
    state_list = ordered_states(pomdp)
    α = ones(length(state_list))
    return DirichletBelief(pomdp, α)
end

pdf(b::DirichletBelief, s) = normalize(b.b.alpha, 1)[stateindex(b.pomdp, s)]

function Random.rand(rng::Random.AbstractRNG, b::DirichletBelief)
    i = sample(rng, Weights(normalize(b.b.alpha, 1)))
    return b.state_list[i]
end

Random.rand(b::DirichletBelief) = rand(Random.GLOBAL_RNG, b)
Random.rand(rng::Random.AbstractRNG, b::DirichletBelief, n::Int) = [rand(rng, b) for _ in 1:n]
Random.rand(b::DirichletBelief, n::Int) = rand(Random.GLOBAL_RNG, b, n)

function Base.fill!(b::DirichletBelief, x::Real)
    fill!(b.b.alpha, x)
    return b
end

Base.length(b::DirichletBelief) = length(b.b.alpha)

support(b::DirichletBelief) = b.state_list

Statistics.mean(b::DirichletBelief) = mean(b.b)
StatsBase.mode(b::DirichletBelief) = mode(b.b)

Base.:(==)(b1::DirichletBelief, b2::DirichletBelief) = b1.state_list == b2.state_list && b1.b == b2.b
Base.hash(b::DirichletBelief, h::UInt) = hash(b.b, hash(b.state_list, h))


"""
    DirichletUpdater
"""
mutable struct DirichletUpdater{P<:POMDP} <: Updater
    pomdp::P
end

uniform_belief(up::DirichletUpdater) = uniform_belief(up.pomdp)

function initialize_belief(up::DirichletUpdater, distr::Categorical)
    belief = DirichletBelief(up.pomdp)
    for s in support(distr)
        sidx = stateindex(up.pomdp, s)
        belief.b.alpha[sidx] = 100*pdf(distr, s)
    end
    return belief
end

function update(up::DirichletUpdater, b::DirichletBelief, a, o)
    pomdp = up.pomdp
    state_space = b.state_list
    α = b.b.alpha
    α′ = copy(α)

    for (sᵢ, s) in enumerate(state_space)
        if pdf(b, s) > 0
            T = transition(pomdp, s, a)

            for (sp, tp) in weighted_iterator(T)
                spᵢ = stateindex(pomdp, sp)
                op = obs_weight(pomdp, s, a, sp, o)
                α′[spᵢ] += op*tp
            end
        end
    end

    return DirichletBelief(pomdp, state_space, Dirichlet(α′))
end

update(up::DirichletUpdater, distr::Categorical, a, o) = update(up, initialize_belief(up, distr), a, o)



##################################################
## Dirichlet subspace belief and updater
##################################################

mutable struct DirichletSubspaceBelief{M<:MOMDP, Sh, S}
    momdp::M
    hidden_state_list::Vector{Sh}
    state_list::Vector{S}
    b::Dirichlet
    visiblestate::Any
end

function (b::DirichletSubspaceBelief)(sv)
    b.visiblestate = sv
    return b
end

DirichletSubspaceBelief(momdp::M) where M <: MOMDP = uniform_subspace_belief(momdp)

function DirichletSubspaceBelief(momdp::M, α::Vector{<:Real}) where M <: MOMDP
    hidden_state_list = ordered_hidden_states(momdp)
    state_list = MOMDPs.ordered_states(momdp)
    α = convert(Vector{Float64}, α)
    b = Dirichlet(α)
    return DirichletSubspaceBelief(momdp, hidden_state_list, state_list, b, nothing)
end

function uniform_subspace_belief(momdp::M) where M <: MOMDP
    hidden_state_list = ordered_hidden_states(momdp)
    α = ones(length(hidden_state_list))
    return DirichletSubspaceBelief(momdp, α)
end

pdf(b::DirichletSubspaceBelief, s) = normalize(b.b.alpha, 1)[MOMDPs.hiddenstateindex(b.momdp, s)]

function Random.rand(rng::Random.AbstractRNG, b::DirichletSubspaceBelief)
    i = sample(rng, Weights(normalize(b.b.alpha, 1)))
    return (b.visiblestate, b.hidden_state_list[i])
end

Random.rand(b::DirichletSubspaceBelief) = rand(Random.GLOBAL_RNG, b)
Random.rand(rng::Random.AbstractRNG, b::DirichletSubspaceBelief, n::Int) = [rand(rng, b) for _ in 1:n]
Random.rand(b::DirichletSubspaceBelief, n::Int) = rand(Random.GLOBAL_RNG, b, n)

function Base.fill!(b::DirichletSubspaceBelief, x::Real)
    fill!(b.b.alpha, x)
    return b
end

Base.length(b::DirichletSubspaceBelief) = length(b.b.alpha)

support(b::DirichletSubspaceBelief) = b.hidden_state_list

Statistics.mean(b::DirichletSubspaceBelief) = mean(b.b)
StatsBase.mode(b::DirichletSubspaceBelief) = mode(b.b)

Base.:(==)(b1::DirichletSubspaceBelief, b2::DirichletSubspaceBelief) = b1.hidden_state_list == b2.hidden_state_list && b1.b == b2.b
Base.hash(b::DirichletSubspaceBelief, h::UInt) = hash(b.b, hash(b.hidden_state_list, h))


"""
    DirichletSubspaceUpdater

Dirichlet belief updater for cases where you only track a belief over a subset of the state space.
e.g., your state is represented as a vector `s = [a,b,c]` and you only track a belief over `c`.
"""
@with_kw mutable struct DirichletSubspaceUpdater{M<:MOMDP} <: Updater
    momdp::M
end

uniform_belief(up::DirichletSubspaceUpdater) = uniform_belief(up.momdp)

function initialize_belief(up::DirichletSubspaceUpdater, distr::Categorical)
    α = 100*distr.p
    return DirichletSubspaceBelief(up.momdp, α)
end

function update(up::DirichletSubspaceUpdater, b::DirichletSubspaceBelief, a, o)
    momdp = up.momdp
    hidden_state_space = b.hidden_state_list
    state_space = b.state_list
    α = b.b.alpha
    α′ = copy(α)
    sᵥ = b.visiblestate

    for sₕ in hidden_state_space
        if pdf(b, sₕ) > 0
            s = (sᵥ,sₕ)
            T = MOMDPs.transitionhidden(momdp, sₕ, a, o)

            for (sp, tp) in weighted_iterator(T)
                spᵢ = MOMDPs.hiddenstateindex(momdp, sp)
                op = MOMDPs.obs_weight(momdp, s, a, sp, o)
                α′[spᵢ] += op*tp
            end
        end
    end

    return DirichletSubspaceBelief(momdp, hidden_state_space, state_space, Dirichlet(α′), sᵥ)
end

update(up::DirichletSubspaceUpdater, distr::Categorical, a, o) = update(up, initialize_belief(up, distr), a, o)
