##################################################
## State space
##################################################
@enum VisaStatus ISIS VulAfghan P1P2Afghan SIV AMCIT NULL

struct MDPState
    c::Int # chairs remaining 
    t::Int # time remaining 
    f::Int # family size 
    v::VisaStatus # visa status 
end

struct VisibleState
    c::Int
    t::Int
    f::Int
end

struct HiddenState
    v::VisaStatus
end

const POMDPState = Tuple{VisibleState, HiddenState}
const AbstractState = Union{MDPState, POMDPState}


##################################################
## Actions
##################################################
@enum Action REJECT ACCEPT # the possible actions are whether accept or reject a family at the gate


##################################################
## Observations
##################################################
@enum VisaDocument begin
    ISIS_indicator
    VulAfghan_document
    P1P2Afghan_document
    SIV_document
    AMCIT_document
    NULL_document
end


struct Observation
    c::Int
    t::Int
    f::Int
    vdoc::VisaDocument
end


##################################################
## Environment parameters
##################################################
@with_kw struct EvacuationParameters
    # average family size in afghanistan is 9. Couldn't find distribution.
    family_sizes::Vector{Int} = collect(1:13)

    # should have probabilities adding to 1. Largest density around 8
    family_prob = [0.3005, 0.191, 0.0118, 0.0137, 0.033, 0.0564, 0.0895, 0.1011, 0.0909, 0.0625, 0.0328, 0.0139, 0.0029]

    num_ISIS = 20 # assuming not that many are showing up at the airport
    num_Vulnerable_Afghan = 1000000
    num_P1P2_Afghan = 604500
    num_SIV = 123000
    num_AMCIT = 14786 # this includes family members of AMCITs to be evacuated

    reward_ISIS = -500
    reward_Vulnerable_Afghan = -3 # Question for Thomas: should it be positive? based on our conversation it seems like they were trying to accept people who might be vulnerable, such as women and children.
    reward_P1P2_Afghan = 1
    reward_SIV = 5
    reward_AMCIT = 20

    num_total_airport = num_AMCIT + num_SIV + num_P1P2_Afghan + num_Vulnerable_Afghan + num_ISIS

    visa_status::Vector{VisaStatus} = [ISIS, VulAfghan, P1P2Afghan, SIV, AMCIT]

    visa_status_lookup::Dict = Dict(
        ISIS => reward_ISIS,
        VulAfghan => reward_Vulnerable_Afghan,
        P1P2Afghan => reward_P1P2_Afghan,
        SIV => reward_SIV,
        AMCIT => reward_AMCIT)

    visa_prob = normalize([
        num_ISIS/num_total_airport,
        num_Vulnerable_Afghan/num_total_airport,
        num_P1P2_Afghan/num_total_airport,
        num_SIV/num_total_airport,
        num_AMCIT/num_total_airport], 1)

    visa_count = [num_ISIS,
                  num_Vulnerable_Afghan,
                  num_P1P2_Afghan,
                  num_SIV,
                  num_AMCIT]

    #for simplicity for now, "vulnerable afghan = afghan"
    v_stringtoint = Dict(
        "ISIS-K"=>reward_ISIS,
        # "two SIVs as good as one Taliban...US gov much more risk avers"
        "VulAfghan"=>reward_Vulnerable_Afghan,
        # anyone who doesn't have a visa....0 or negative.
        # even beat up over ...everyone directly served the us effort in
        # some way either through developer contractors or military
        "P1/P2 Afghan"=>reward_P1P2_Afghan,
        "SIV"=>reward_SIV,
        "AMCIT"=>reward_AMCIT)

     v_inttostring = Dict(
        reward_ISIS=>"ISIS-K",
        reward_Vulnerable_Afghan=>"Vul. Afghan",
        reward_P1P2_Afghan=>"P1/P2 Afghan",
        reward_SIV=>"SIV",
        reward_AMCIT=>"AMCIT")

    capacity::Int = 120 # keeping these both as integers of 20 for now.
    time::Int = 120
    size::Tuple{Int, Int} = (length(visa_status), length(family_sizes)) # grid size
    p_transition::Real = 0.8 # this is uncertainty we integrated in our model that described the likelihood of transitioning onto the airplane given they are let into the airport
    null_state::MDPState = MDPState(-1, -1, -1 ,NULL)
    accept_prob = [p_transition, 1-p_transition]
    reject_prob = [1.0]
end


@with_kw struct ClaimModel
    p_amcit = normalize([0, 0, 0, 0.0, 1.0], 1)
    p_siv = normalize([0, 0, 0, 0.99, 0.01], 1)
    p_p1p2 = normalize([0, 0, 0.95, 0.04, 0.01], 1)
    p_afghan = normalize([0, 0.94, 0.05, 0.009, 0.001], 1)
    p_isis = normalize([0.9, 0.09, 0.005, 0.005, 0], 1)
end


##################################################
## MDP and POMDP types
##################################################
mutable struct EvacuationMDP <: MDP{MDPState, Action}
    params::EvacuationParameters
    stateindices::Dict
end

function EvacuationMDP(params::EvacuationParameters=EvacuationParameters())
    mdp = EvacuationMDP(params, Dict())
    fill_state_inds!(mdp)
    return mdp
end


@with_kw mutable struct EvacuationPOMDPType <: MOMDP{VisibleState, HiddenState, Action, Observation}
    params::EvacuationParameters = EvacuationParameters()
    claims::ClaimModel = ClaimModel()
    visa_count = ones(length(params.visa_count)) # uniform prior updated as a Dirichlet, used in 𝑇(s′ | s, a)
    null_state = POMDPState((VisibleState(-1,-1,-1), HiddenState(NULL)))
    documentation = [ISIS_indicator, VulAfghan_document, P1P2Afghan_document, SIV_document, AMCIT_document]
    isnoisy = true
end


##################################################
## Action space
##################################################
POMDPs.actions(mdp::Union{EvacuationMDP, EvacuationPOMDPType}) = [REJECT, ACCEPT]


##################################################
## Termination
##################################################
# only valid if room for the family [assuming would not separate even though might] and if time is available to enter the airport
validtime(s::Union{MDPState,VisibleState}) = s.t > 0
validcapacity(s::Union{MDPState,VisibleState}) = s.c > 0


##################################################
## Reward function
##################################################
function R(params::EvacuationParameters, c::Int, t::Int, f::Int, v::VisaStatus, a::Action)
    # reward is just the visa status times family size i think! 
    if t ≤ 0 || c ≤ 0 # TODO: isterminal
        return -abs(c) # penalize overflow and underflow.
    elseif a == ACCEPT
        return params.visa_status_lookup[v]*f
    else
        return -sqrt(params.time-t) # 0
        # return -c/(t+1)
    end
end


##################################################
## Discount factor
##################################################
POMDPs.discount(mdp::Union{EvacuationMDP, EvacuationPOMDPType}) = 0.95


##################################################
## State-level shared functions (used primarily in transition)
##################################################
newstate(::Type{MDPState}, c, f, t, v) = MDPState(c, f, t, v)
newstate(::Type{POMDPState}, c, f, t, v) = POMDPState((VisibleState(c, f, t), HiddenState(v)))

getcapacity(s::MDPState) = s.c
getcapacity(s::POMDPState) = visible(s).c
gettime(s::MDPState) = s.t
gettime(s::POMDPState) = visible(s).t
getfamilysize(s::POMDPState) = visible(s).f
getfamilysize(s::MDPState) = s.f
getfamilysize(s::POMDPState) = visible(s).f
getstatus(s::MDPState) = s.v
getstatus(s::POMDPState) = hidden(s).v
