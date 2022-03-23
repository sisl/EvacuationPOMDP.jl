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


function likelihood(mdp::EvacuationMDP, params::EvacuationParameters, v::Int)
    return params.visa_prior[v]
end


function POMDPs.transition(m::M, s::S, a::Action; input_status::Union{Nothing,VisaStatus}=nothing, input_family_size::Union{Nothing, Int}=nothing, made_it_through::Union{Nothing,Bool}=nothing) where {M <: Union{EvacuationMDP, EvacuationPOMDPType}, S <: Union{MDPState, POMDPState}}
    params = m.params
    null_state = typeof(m) == EvacuationMDP ? params.null_state : m.null_state
    next_states = S[]
    probabilities = Float64[]
    
    if !validtime(s) || !validcapacity(s)
        push!(next_states, null_state)
        push!(probabilities, 1)
    else
        if a == ACCEPT
            # check if valid capacity
            visa_status = getstatus(s)
            next_state_accept = newstate(S, getcapacity(s) - getfamilysize(s), gettime(s) - 1, 1, visa_status)
            next_state_reject = newstate(S, getcapacity(s), gettime(s) - 1, 1, visa_status)

            if !validcapacity(next_state_accept)
                # no room for full family, so we make prob. 0 to accept and 1 reject
                probabilities = [1, 0]
                next_states = [next_state_accept, next_state_reject]
            else
                prob = params.accept_prob

                function append_next_state_accept!(family_size::Int, visa_status::VisaStatus, f::Int, v::Int)
                    visa_prob = likelihood(m, params, v)
                    family_prob = params.family_prob[f]
                    if isnothing(made_it_through) || made_it_through == true
                        sp_accept = newstate(S, getcapacity(s)-getfamilysize(s), gettime(s)-1, family_size, visa_status)
                        push!(next_states, sp_accept)
                        push!(probabilities, prob[1] * visa_prob * family_prob)
                    end

                    if isnothing(made_it_through) || made_it_through == false
                        # if not
                        sp_reject = newstate(S, getcapacity(s), gettime(s)-1, family_size, visa_status)
                        push!(next_states, sp_reject)
                        push!(probabilities, prob[2] * visa_prob * family_prob)
                    end
                end

                if !isnothing(input_status) && !isnothing(input_family_size)
                    # bypassing transition function (used for deterministic population trajectories)
                    f = input_family_size # array starts at 1, goes to max (i.e., already an index)
                    v = Int(input_status) + 1 # zero-based enum to one-based index
                    append_next_state_accept!(input_family_size, input_status, f, v)
                else
                    for f in 1:length(params.family_sizes)
                        for v in 1:length(params.visa_status)
                            # if get on plane
                            family_size = params.family_sizes[f]
                            visa_status = params.visa_status[v]
                            append_next_state_accept!(family_size, visa_status, f, v)
                        end
                    end
                end
            end
        else # if reject
            function append_next_state_reject!(family_size::Int, visa_status::VisaStatus, f::Int, v::Int)
                sp = newstate(S, getcapacity(s), gettime(s)-1, family_size, visa_status)
                push!(next_states, sp)
                visa_prob = likelihood(m, params, v)
                push!(probabilities, params.reject_prob[1] * visa_prob * params.family_prob[f])
            end

            if !isnothing(input_status) && !isnothing(input_family_size)
                f = input_family_size
                v = Int(input_status) + 1
                append_next_state_reject!(input_family_size, input_status, f, v)
            else
                for f in 1:length(params.family_sizes)
                    for v in 1:length(params.visa_status)
                        family_size = params.family_sizes[f]
                        visa_status = params.visa_status[v]
                        append_next_state_reject!(family_size, visa_status, f, v)
                    end
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
