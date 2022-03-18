"""
# Observations
- how to model "observations"?
- some sort of documentation?
- Actual SIV approval number, or SIV processing number.
- IDEAS: (based on convos...should ask Thomas Billingsley)
- US Citizen: actual passport, lost passport but say they have one, picture of documents 
- On a list/check identity against a list 
- 'Take this kid' they say kid's mother is an AMCIT 
- SIV full verification 
- SIV applicant number 
- Signal (e.g. pic of pinapple) rely on to signal because of its relative obscurity (verification happens elsewhwew and need a quick way for a marine to not have to sift through a oacker but look at a token that gives them a good sense...takes a tenth of the time to verify)
"""
function POMDPs.observations(pomdp::EvacuationPOMDPType)
    params = pomdp.params
    return [Observation(c,t,f,vdoc)
        for c in 0:params.capacity
            for t in 0:params.time
                for f in params.family_sizes
                    for vdoc in documentation]
end


function POMDPs.reward(pomdp::EvacuationPOMDPType, s::POMDPState, a::Action)
    return R(pomdp.params, visible(s).c, visible(s).t, visible(s).f, hidden(s).v, a)
end


function MOMDPs.visiblestates(pomdp::EvacuationPOMDPType)
    params = pomdp.params
    min_capacity = 1-length(params.family_prob)
    𝒮ᵥ = [VisibleState(c,t,f)
        for c in min_capacity:params.capacity
            for t in 0:params.time
                for f in params.family_sizes]
    return 𝒮ᵥ
end


function MOMDPs.hiddenstates(pomdp::EvacuationPOMDPType)
    hidden_visa_statuses = [ISIS, VulAfghan, P1P2Afghan, SIV, AMCIT]
    𝒮ₕ = [HiddenState(v) for v in hidden_visa_statuses]
    return 𝒮ₕ
end


function POMDPs.states(pomdp::EvacuationPOMDPType)
    𝒮ᵥ = visiblestates(pomdp)
    𝒮ₕ = hiddenstates(pomdp)
    𝒮_pomdp::Vector{POMDPState} = [(v,h) for v in 𝒮ᵥ for h in 𝒮ₕ]
    push!(𝒮_pomdp, pomdp.null_state)
    return 𝒮_pomdp
end


function POMDPs.initialstate(pomdp::EvacuationPOMDPType)
    params = pomdp.params
    c_initial = params.capacity
    t_initial = params.time
    S = []
    for f in params.family_sizes
        for v in params.visa_status
            s = POMDPState((VisibleState(c_initial, t_initial, f), HiddenState(v)))
            push!(S, s)
        end
    end
    return SparseCat(S, normalize(ones(length(S)), 1))
end


validtime(s::POMDPState) = validtime(visible(s))
validcapacity(s::POMDPState) = validcapacity(visible(s))


function MOMDPs.transitionhidden(pomdp::EvacuationPOMDPType, sh::HiddenState, a::Action, o=missing)
    hiddenstates = ordered_hidden_states(pomdp)
    p = normalize(pomdp.visa_count, 1)
    sᵢ = hiddenstateindex(pomdp, sh)
    p[sᵢ] = 1
    normalize!(p, 1)
    return SparseCat(hiddenstates, p) # T(h′ | s, a, v′)
end


function POMDPs.isterminal(pomdp::EvacuationPOMDPType, s::POMDPState)
    # return !validtime(s) || !validcapacity(s) ||
    return s == pomdp.null_state
end


function likelihood(pomdp::EvacuationPOMDPType, params::EvacuationParameters, v::Int)
    p = normalize(pomdp.visa_count, 1)
    return p[v]
end


POMDPs.updater(pomdp::EvacuationPOMDPType) = DiscreteSubspaceUpdater(pomdp)


function POMDPModelTools.obs_weight(p::EvacuationPOMDPType, sh, a, shp, o)
    return pdf(observation(p, sh, a, shp), o.vdoc)
end


function MOMDPs.visible(pomdp::EvacuationPOMDPType, o::Observation)
    return VisibleState(o.c, o.t, o.f)
end


function POMDPs.observation(pomdp::EvacuationPOMDPType, s::POMDPState, a::Action, sp::POMDPState)
    documentation = pomdp.documentation
    state = sp # NOTE
    sv = visible(state)
    sh = hidden(state)
    sₕ_idx = hiddenstateindex(pomdp, state)
    if isnothing(sₕ_idx) # null state
        return Deterministic(
            Observation(sv.c, sv.t, sv.f, NULL_document))
    else
        if pomdp.individual_uncertainty
            if sh.v == AMCIT
                p = copy(pomdp.claims.p_amcit)
            elseif sh.v == SIV
                p = copy(pomdp.claims.p_siv)
            elseif sh.v == P1P2Afghan
                p = copy(pomdp.claims.p_p1p2)
            elseif sh.v == VulAfghan
                p = copy(pomdp.claims.p_afghan)
            elseif sh.v == ISIS
                p = copy(pomdp.claims.p_isis)
            else
                error("No case for $(sh.v).")
            end
        else
            p = zeros(length(documentation)) # NOISELESS
            p[sₕ_idx] = 1
        end
        obs = [Observation(sv.c, sv.t, sv.f, vdoc) for vdoc in documentation]

        # Handle null case
        push!(obs, Observation(sv.c, sv.t, sv.f, NULL_document))
        push!(p, 1e-100)

        normalize!(p, 1)
        return SparseCat(obs, p)
    end
end


function POMDPs.observation(pomdp::EvacuationPOMDPType, sh::HiddenState, a::Action, shp::HiddenState)
    documentation = pomdp.documentation
    state = shp # NOTE
    sₕ_idx = hiddenstateindex(pomdp, state)
    if isnothing(sₕ_idx) # null state
        return Deterministic(NULL_document)
    else
        if pomdp.individual_uncertainty
            if state.v == AMCIT
                p = copy(pomdp.claims.p_amcit)
            elseif state.v == SIV
                p = copy(pomdp.claims.p_siv)
            elseif state.v == P1P2Afghan
                p = copy(pomdp.claims.p_p1p2)
            elseif state.v == VulAfghan
                p = copy(pomdp.claims.p_afghan)
            elseif state.v == ISIS
                p = copy(pomdp.claims.p_isis)
            end
        else
            p = zeros(length(documentation)) # NOISELESS
            p[sₕ_idx] = 1
        end
        obs = copy(documentation)

        # Handle null case
        push!(obs, NULL_document)
        push!(p, 1e-100)

        normalize!(p, 1)
        return SparseCat(obs, p)
    end
end


function reset_population_belief!(pomdp::EvacuationPOMDPType)
    if pomdp.population_uncertainty
        pomdp.visa_count = ones(length(pomdp.visa_count))
    else
        pomdp.visa_count = deepcopy(pomdp.params.visa_count)
    end
end
