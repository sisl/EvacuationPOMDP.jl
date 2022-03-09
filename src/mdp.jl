function POMDPs.states(mdp::EvacuationMDP)
    params::EvacuationParameters = mdp.params

    # The state space S for the evacuation problem is the set of all combinations 
    𝒮 = []
    # capacity ends at 0 
    min_capacity = 1-length(params.family_prob)
    for c in min_capacity:params.capacity
        # time ends at 0
        for t in 0:params.time
            # family size here we should have the ACTUAL family sizes
            for f in params.family_sizes
                # actual visa statuses
                for v in params.visa_status
                    new = MDPState(c, t, f, v) 
                    push!(𝒮, new)
                end
            end
        end
    end
    push!(𝒮, params.null_state)

    number_states = (params.capacity+1-min_capacity) * (params.time+1) * size(params.family_sizes)[1] * size(params.visa_status)[1] + 1
    @assert number_states == length(𝒮)

    return 𝒮
end


function fill_state_inds!(mdp::EvacuationMDP)
    mdp.stateindices = Dict(s=>i for (i,s) in enumerate(states(mdp)))
end

POMDPs.stateindex(mdp::EvacuationMDP, s::MDPState) = mdp.stateindices[s]
POMDPs.actionindex(mdp::EvacuationMDP, a::Action) = Int(a) + 1


function POMDPs.transition(mdp::EvacuationMDP, s::MDPState, a::Action)
    params = mdp.params
    next_states = MDPState[]
    probabilities = Float64[] 
    
    if !validtime(s) || !validcapacity(s)
        push!(next_states, params.null_state)
        push!(probabilities, 1) # double check 
    else
        if a == ACCEPT
            # check if valid capacity
            visa_status = s.v
            next_state_accept = MDPState(s.c - s.f, s.t - 1, 1, visa_status)
            next_state_reject = MDPState(s.c, s.t - 1, 1, visa_status)
            if !validcapacity(next_state_accept)
                # no room for full family, so we make prob. 0 to accept and 1 reject
                probabilities = [1,0]
                next_states = [next_state_accept, next_state_reject]
            else
                prob = params.accept_prob
                for f in 1:length(params.family_sizes)
                    for v in 1:length(params.visa_status)
                        # if get on plane
                        family_size = params.family_sizes[f]
                        visa_status = params.visa_status[v]
                        sp_accept = MDPState(s.c-s.f, s.t-1, family_size, visa_status)
                        push!(next_states, sp_accept)
                        visa_prob = params.visa_prob[v]
                        family_prob = params.family_prob[f]
                        push!(probabilities, prob[1] * visa_prob * family_prob)

                        # if not
                        sp_reject = MDPState(s.c, s.t-1, family_size, visa_status)
                        push!(next_states, sp_reject)
                        push!(probabilities, prob[2] * visa_prob * family_prob)
                    end
                end
            end
        else # if reject     
            for f in 1:length(params.family_sizes)
                for v in 1:length(params.visa_status)
                    sp = MDPState(s.c, s.t-1, params.family_sizes[f], params.visa_status[v])
                    push!(next_states, sp)
                    push!(probabilities, params.reject_prob[1] *
                        params.visa_prob[v] * params.family_prob[f])
                end
            end  
        end
    end
    normalize!(probabilities, 1)
    return SparseCat(next_states, probabilities)
end


function POMDPs.reward(mdp::EvacuationMDP, s::MDPState, a::Action)
    return R(mdp.params, s.c, s.t, s.f, s.v, a)
end


function POMDPs.initialstate(mdp::EvacuationMDP)
    params = mdp.params
    c_initial = params.capacity
    t_initial = params.time
    S = []
    for f in params.family_sizes
        for v in params.visa_status
            s = MDPState(c_initial, t_initial, f, v)
            push!(S, s)
        end
    end
    return SparseCat(S, normalize(ones(length(S)), 1))
end


function POMDPs.isterminal(mdp::EvacuationMDP, s::MDPState)
    # return !validtime(s) || !validcapacity(s) ||
    return s == mdp.params.null_state
end
