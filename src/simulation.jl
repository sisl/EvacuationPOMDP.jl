"""Given a history of a simulation, return metrics."""
function get_metrics(params, history)
    total_accepted_people = 0
    total_accepted_families = 0
    total_rejected_people = 0
    total_rejected_families = 0
    total_reward = 0.0
    # Initialize visa_statuses dictionary
    visa_statuses = params.visa_status
    visa_dict_accepts = Dict{VisaStatus, Int}()
    for v in visa_statuses
        visa_dict_accepts[v] = 0
    end

    visa_dict_rejects = Dict{VisaStatus, Int}()
    for v in visa_statuses
        visa_dict_rejects[v] = 0
    end

    # MDPState(c, t, f, v)
    for (s, a, r, sp) in eachstep(history, "(s, a, r, sp)")
        # only counting the s not sp so as not to double count
        if typeof(s) <: POMDPState
            f = visible(s).f
            v = hidden(s).v
        else
            f = s.f
            v = s.v
        end

        if a==ACCEPT
            total_accepted_people += f
            total_accepted_families += 1
            visa_dict_accepts[v] += f
        else # action is reject
            total_rejected_people += f
            total_rejected_families +=1
            visa_dict_rejects[v] += f
        end

        total_reward += r
       # println("reward $r received when state $sp was reached after action $a was taken in state $s")
    end
    @assert total_accepted_people == sum(values(visa_dict_accepts))
    visa_dict_accepts_rate = merge(/, convert(Dict{VisaStatus, Float64}, visa_dict_accepts), merge(+, visa_dict_accepts, visa_dict_rejects))
    return total_accepted_people, total_rejected_people, total_accepted_families, total_reward, visa_dict_accepts, visa_dict_rejects, visa_dict_accepts_rate
end


"""Given a policy and mdp simulate a rollout and return the history."""
function simulation(mdp, policy)
    hr = HistoryRecorder()
    history = simulate(hr, mdp, policy)
    return history
end


function simulation(pomdp::POMDP, policy)
    up = updater(pomdp)
    initial_state = rand(initialstate(pomdp))
    initial_obs = rand(observation(pomdp, initial_state, REJECT, initial_state))
    prior_belief = initialize_belief(up, pomdp.params.visa_count, initial_state)
    initial_belief = update(up, prior_belief, REJECT, initial_obs)

    hr = HistoryRecorder()
    history = simulate(hr, pomdp, policy, up, initial_belief, initial_state)
    return history
end


""" Takes in a list and takes their rounded mean and standard deviation """
function mean_std(list_simulated_values, people::Bool)
    if people
        # round to whole people trunc
        mean_list = trunc(Int, mean(list_simulated_values))
        std_list = trunc(Int, std(list_simulated_values))
#        std_list = round(std(list_simulated_values); digits=2)
    else
        mean_list = round(mean(list_simulated_values); digits=2)
        std_list = round(std(list_simulated_values); digits=2)
    end
    return mean_list, std_list
end


"""Simulate n times and get the mean and std of n rollouts of a policy."""
function simulations(policy, str_policy, mdp, n_sims) # n is number of times to run
    people = false

    list_total_accepted_people = []
    list_total_rejected_people = []
    list_total_accepted_families = []
    list_total_reward = []
    list_visa_dict_accepts = []
    list_visa_dict_rejects = []
    list_visa_rate_accepts = []

    for i in 1:n_sims
        Random.seed!(i) # Determinism.
        history = simulation(mdp, policy) # do 1 simulation and get the history
        total_accepted_people, total_rejected_people, total_accepted_families, total_reward, visa_dict_accepts, visa_dict_rejects, visa_dict_accepts_rate = get_metrics(mdp.params, history)
        push!(list_total_accepted_people, total_accepted_people)
        push!(list_total_rejected_people, total_rejected_people)
        push!(list_total_accepted_families, total_accepted_families)
        push!(list_total_reward, total_reward)
        push!(list_visa_dict_accepts, visa_dict_accepts)
        push!(list_visa_dict_rejects, visa_dict_rejects)
        push!(list_visa_rate_accepts, visa_dict_accepts_rate)
    end


    mean_std_reward = mean_std(list_total_reward, false)
    mean_std_total_accepted_people = mean_std(list_total_accepted_people, true)
    mean_std_percent_accepted_people = mean_std(100 * list_total_accepted_people ./ (list_total_accepted_people + list_total_rejected_people), false)

    # calculate average ppl by visa type
    base_dict = Dict{VisaStatus, Int64}() #1=>0)
    # add together total people over sims
    for dict in list_visa_dict_accepts
        base_dict = merge(counter(base_dict), counter(dict)).map
    end

    # calculate average (rejected) ppl by visa type
    base_dict_reject = Dict{VisaStatus, Int64}() #1=>0)
    # add together total people over sims
    for dict in list_visa_dict_rejects
        base_dict_reject = merge(counter(base_dict_reject), counter(dict)).map
    end

    base_dict_rate = Dict{VisaStatus, Vector{Float64}}()
    for dict in list_visa_rate_accepts
        base_dict_rate = merge(vcat, base_dict_rate, dict)
    end

    if str_policy == "AMCITs" || str_policy == "POMDP"
        println()
        print("Policy\t\t\t\t&\t\tReward\t\t\t&\t\tTotal Accepted (\\% accepted)\t\t&\t")
        for visa_status in sort(collect(keys(base_dict)), rev=true)
            st_visa_status = string(visa_status)
            # just show what order the stats are showing up
            tabs = length(st_visa_status) <= 5 ? "\t\t\t" : "\t\t\t\t"
            print("$st_visa_status$tabs")
            visa_status == ISIS ? print("\\\\") : print("&\t")
        end
    end
    println()

    if length(str_policy) <= 6
        tabs_policy = "\t\t\t\t"
    elseif length(str_policy) <= 10
        tabs_policy = "\t\t\t"
    else
        tabs_policy = "\t\t"
    end

    print("$str_policy$tabs_policy&\t\t\$$(mean_std_reward[1]) \\pm $(mean_std_reward[2])\$\t&\t\t\$$(mean_std_total_accepted_people[1]) \\pm $(mean_std_total_accepted_people[2]) \\;($(mean_std_percent_accepted_people[1])\\pm$(mean_std_percent_accepted_people[2])\\%)\$\t\t&\t")
    for visa_status in sort(collect(keys(base_dict)), rev=true)
        st_visa_status = string(visa_status)
        expectation_acc = trunc(Int, base_dict[visa_status]/n_sims)
        expectation_rej = trunc(Int, base_dict_reject[visa_status]/n_sims)
        percent_acc_μ = round(100*mean(filter(!isnan, base_dict_rate[visa_status])), digits=2)
        percent_acc_σ = round(100*std(filter(!isnan, base_dict_rate[visa_status])), digits=2)
        percent_acc_μ = isnan(percent_acc_μ) ? "---" : trailing_zeros(percent_acc_μ)
        percent_acc_σ = isnan(percent_acc_σ) ? "---" : trailing_zeros(percent_acc_σ)
        print("\$$(expectation_acc)\\;($percent_acc_μ \\pm $percent_acc_σ\\%)\$\t")
        visa_status == ISIS ? print("\t\\\\") : print("&\t")
    end
    println()
end


function experiments(n_sims, mdp::EvacuationMDP, mdp_policy)
    # policies and n_sims can probably be put in our params function as a list. here for now.
    dict_policies = Dict(
        "MDP"=>mdp_policy,
        "AcceptAll"=>AcceptAllPolicy(),
        "AMCITs"=>AMCITsPolicy(),
        "SIV-AMCITs"=>SIVAMCITsPolicy(),
        "AfterThresholdAMCITs"=>AfterThresholdAMCITsPolicy(mdp_policy=mdp_policy),
        "BeforeThresholdAMCITs"=>BeforeThresholdAMCITsPolicy(mdp_policy=mdp_policy),
    )

    # for now don't report the std here. would need to keep track of every single value....

    for str_policy in sort(collect(keys(dict_policies)))
        xmdp = contains(str_policy, "pomdp") ? pomdp : mdp
        list_visa_dict_accepts = simulations(dict_policies[str_policy], str_policy, xmdp, n_sims)
    end
end


function experiments(n_sims, pomdp::EvacuationPOMDPType, pomdp_policy)
    simulations(pomdp_policy, "POMDP", pomdp, n_sims)
end


num_decimals(x) = findfirst('.', x) |> z -> z === nothing ? -1 : length(x) - z

trailing_zeros(x::Real, d=2) = trailing_zeros(string(x), d)
function trailing_zeros(x::String, d=2)
    n = num_decimals(x)
    if n == -1
        dot = "."
        n = 0
    else
        dot = ""
    end
    return string(x, dot, "0"^(d-n))
end