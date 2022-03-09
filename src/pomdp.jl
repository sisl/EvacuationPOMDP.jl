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
    return Deterministic(POMDPState((VisibleState(params.capacity, params.time, 3), HiddenState(AMCIT))))
end


validtime(s::POMDPState) = validtime(visible(s))
validcapacity(s::POMDPState) = validcapacity(visible(s))


# TODO: Reuse MDP version.
function POMDPs.transition(pomdp::EvacuationPOMDPType, s::POMDPState, a::Action)
    params = pomdp.params
    sv = visible(s)
    sh = hidden(s)
    next_states = POMDPState[]
    probabilities = Float64[]

    if !validtime(s) || !validcapacity(s)
        push!(next_states, pomdp.null_state)
        push!(probabilities, 1)
    else
        if a == ACCEPT
            # check if valid capacity
            visa_status = sh.v
            next_state_accept = POMDPState((VisibleState(sv.c - sv.f, sv.t - 1, 1), HiddenState(visa_status)))
            next_state_reject = POMDPState((VisibleState(sv.c, sv.t - 1, 1), HiddenState(visa_status)))

            if !validcapacity(next_state_accept)
                # no room for full family, so we make prob. 0 to accept and 1 reject
                probabilities = [1, 0]
                next_states = [next_state_accept, next_state_reject]
            else
                prob = params.accept_prob
                for f in 1:length(params.family_sizes)
                    for v in 1:length(params.visa_status)
                        # if get on plane
                        family_size = params.family_sizes[f]
                        visa_status = params.visa_status[v]
                        svp_accept = VisibleState(sv.c-sv.f, sv.t-1, family_size)
                        shp_accept = HiddenState(visa_status)
                        sp_accept = POMDPState((svp_accept, shp_accept))
                        push!(next_states, sp_accept)
                        visa_prob = likelihood(pomdp, v)
                        family_prob = params.family_prob[f]
                        push!(probabilities, prob[1] * visa_prob * family_prob)

                        # if not
                        svp_reject = VisibleState(sv.c, sv.t-1, family_size)
                        shp_reject = HiddenState(visa_status)
                        sp_reject = POMDPState((svp_reject, shp_reject))
                        push!(next_states, sp_reject)
                        push!(probabilities, prob[2] * visa_prob * family_prob)
                    end
                end
            end
        else # if reject
            for f in 1:length(params.family_sizes)
                for v in 1:length(params.visa_status)
                    svp = VisibleState(sv.c, sv.t-1, params.family_sizes[f])
                    shp = HiddenState(params.visa_status[v])
                    sp = POMDPState((svp, shp))
                    push!(next_states, sp)
                    push!(probabilities, params.reject_prob[1] *
                        likelihood(pomdp, v) * params.family_prob[f])
                end
            end
        end
    end
    normalize!(probabilities, 1)
    return SparseCat(next_states, probabilities)
end


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


function likelihood(pomdp::EvacuationPOMDPType, v::Int)
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
        if pomdp.isnoisy
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
        if pomdp.isnoisy
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

