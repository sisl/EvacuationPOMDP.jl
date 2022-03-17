##################################################
## Baseline policies
##################################################

struct AcceptAllPolicy <: Policy end

function POMDPs.action(::AcceptAllPolicy, s::MDPState)
    return ACCEPT
end


##################################################


struct AMCITsPolicy <: Policy end

function POMDPs.action(::AMCITsPolicy, s::MDPState)
    return s.v == AMCIT  ? ACCEPT : REJECT
end


##################################################


struct SIVAMCITsPolicy <: Policy end

function POMDPs.action(::SIVAMCITsPolicy, s::MDPState)
    return (s.v == AMCIT || s.v == SIV) ? ACCEPT : REJECT
end

function POMDPs.action(::SIVAMCITsPolicy, s::POMDPState)
    sh = hidden(s)
    return (sh.v == AMCIT || sh.v == SIV) ? ACCEPT : REJECT
end


##################################################


@with_kw mutable struct AfterThresholdAMCITsPolicy <: Policy
    threshold = 20
    mdp_policy
end

function POMDPs.action(policy::AfterThresholdAMCITsPolicy, s::MDPState)
    if s.t <= policy.threshold
        return s.v == AMCIT ? ACCEPT : REJECT
    else
        return action(policy.mdp_policy, s)
    end
end


##################################################


@with_kw struct BeforeThresholdAMCITsPolicy <: Policy
    threshold = 20
    mdp_policy
end

function POMDPs.action(policy::BeforeThresholdAMCITsPolicy, s::MDPState)
    if s.t >= policy.threshold
        return s.v == AMCIT ? ACCEPT : REJECT
    else
        return action(policy.mdp_policy, s)
    end
end


##################################################

struct RandomBaselinePolicy <: Policy end

function POMDPs.action(::RandomBaselinePolicy, s::MDPState)
    return rand() < 0.5 ? ACCEPT : REJECT
end