"""Given a history of a simulation, return metrics."""
function get_metrics(params, history)
    total_accepted_people = 0
    total_accepted_families = 0
    total_rejected_people = 0
    total_rejected_families = 0
    total_reward = 0.0
    reward_over_time = []
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

    if typeof(history) <: SimHistory
        iterator = eachstep(history, "(s, a, r, sp, t)")
    else
        iterator = history
    end

    # MDPState(c, t, f, v)
    for (t,(s, a, r, sp, truth)) in enumerate(iterator)
        # only counting the s not sp so as not to double count
        if typeof(s) <: POMDPState
            f = visible(s).f
            # v = hidden(s).v
        else
            f = s.f
            # v = s.v
        end
        v = truth

        if a == ACCEPT
            total_accepted_people += f
            total_accepted_families += 1
            visa_dict_accepts[v] += f
        else # action is reject
            total_rejected_people += f
            total_rejected_families +=1
            visa_dict_rejects[v] += f
        end

        total_reward += r
        push!(reward_over_time, r)
       # println("reward $r received when state $sp was reached after action $a was taken in state $s")
    end
    @assert total_accepted_people == sum(values(visa_dict_accepts))
    visa_dict_accepts_rate = merge(/, convert(Dict{VisaStatus, Float64}, visa_dict_accepts), merge(+, visa_dict_accepts, visa_dict_rejects))
    return total_accepted_people, total_rejected_people, total_accepted_families, total_reward, reward_over_time, visa_dict_accepts, visa_dict_rejects, visa_dict_accepts_rate
end

function obs2state(vdoc::VisaDocument)
    return VisaStatus(Int(vdoc))
end

function manual_simulate(mdp::EvacuationMDP, policy, population)
    first_fam = popfirst!(population)
    _s = rand(initialstate(mdp))
    status = obs2state(first_fam.obs_status)
    true_status = first_fam.true_status
    s = MDPState(_s.c, _s.t, first_fam.family_size, status)
    s_true = MDPState(_s.c, _s.t, first_fam.family_size, true_status)
    τ = []
    while !isterminal(mdp, s)
        a = action(policy, s)
        r = reward(mdp, s_true, a; as_obs=false)
        sp_true_status, sp_obs_status, family_size, made_it_through = popfirst!(population)
        status = obs2state(sp_obs_status)
        sp = rand(transition(mdp, s, a; input_status=status, input_family_size=family_size, made_it_through=made_it_through))
        push!(τ, (s=s, a=a, r=r, sp=sp, truth=true_status))
        s = sp
        true_status = sp_true_status
        s_true = MDPState(s.c, s.t, s.f, true_status)
    end
    return τ
end


function manual_simulate(pomdp::EvacuationPOMDPType, policy, up::Updater, population; include_all=false)
    first_fam = popfirst!(population)
    reset_population_belief!(pomdp)
    _s = rand(initialstate(pomdp))
    status = obs2state(first_fam.obs_status)
    true_status = first_fam.true_status
    s = newstate(POMDPState, visible(_s).c, visible(_s).t, first_fam.family_size, status)
    s_true = newstate(POMDPState, visible(s).c, visible(s).t, visible(s).f, true_status)
    o0 = rand(observation(pomdp, s, REJECT, s))
    prior_belief = initialize_belief(up, pomdp.visa_count, s)
    b = update(up, prior_belief, REJECT, o0)
    τ = []

    while !isterminal(pomdp, s)
        a = action(policy, b)
        r = reward(pomdp, s_true, a)
        sp_true_status, sp_obs_status, family_size, made_it_through = popfirst!(population)
        status = obs2state(sp_obs_status)
        sp = rand(transition(pomdp, s, a; input_status=status, input_family_size=family_size, made_it_through=made_it_through))
        o = rand(observation(pomdp, s, a, sp))
        bp = update(up, b, a, o)
        if include_all
            push!(τ, (s=s, a=a, r=r, sp=sp, b=b, o=o, o0=o0, truth=true_status))
        else
            push!(τ, (s=s, a=a, r=r, sp=sp, truth=true_status))
        end
        s = sp
        true_status = sp_true_status
        s_true = newstate(POMDPState, visible(s).c, visible(s).t, visible(s).f, true_status)
        b = bp
    end
    return τ
end


"""Given a policy and mdp simulate a rollout and return the history."""
function simulation(mdp::MDP, policy, population=nothing)
    if isnothing(population)
        hr = HistoryRecorder()
        history = simulate(hr, mdp, policy)
    else
        history = manual_simulate(mdp, policy, population)
    end
    return history
end


function simulation(pomdp::POMDP, policy, population=nothing)
    up = updater(pomdp)

    if isnothing(population)
        initial_state = rand(initialstate(pomdp))
        initial_obs = rand(observation(pomdp, initial_state, REJECT, initial_state))
        reset_population_belief!(pomdp)
        prior_belief = initialize_belief(up, pomdp.visa_count, initial_state)
        initial_belief = update(up, prior_belief, REJECT, initial_obs)
        hr = HistoryRecorder()
        history = simulate(hr, pomdp, policy, up, initial_belief, initial_state)
    else
        history = manual_simulate(pomdp, policy, up, population)
    end
    return history
end


""" Takes in a list and takes their rounded mean and standard deviation """
function mean_se(list_simulated_values, people::Bool, n_sims)
    rootn = sqrt(n_sims)
    if people
        # round to whole people trunc
        mean_list = trunc(Int, mean(list_simulated_values))
        se_list = trunc(Int, std(list_simulated_values)/rootn)
#        se_list = round(std(list_simulated_values); digits=2)
    else
        mean_list = round(mean(list_simulated_values); digits=2)
        se_list = round(std(list_simulated_values)/rootn; digits=2)
    end
    return mean_list, se_list
end


"""Simulate n times and get the mean and std of n rollouts of a policy."""
function simulations(policy, str_policy, mdp, n_sims) # n is number of times to run
    people = false

    list_total_accepted_people = []
    list_total_rejected_people = []
    list_total_accepted_families = []
    list_total_reward = []
    list_reward_over_time = []
    list_visa_dict_accepts = []
    list_visa_dict_rejects = []
    list_visa_rate_accepts = []

    populations = read_population_json()
    for i in 1:n_sims
        Random.seed!(i) # Determinism.
        population = copy(populations[i])
        history = simulation(mdp, policy, population) # do 1 simulation and get the history
        total_accepted_people, total_rejected_people, total_accepted_families, total_reward, reward_over_time, visa_dict_accepts, visa_dict_rejects, visa_dict_accepts_rate = get_metrics(mdp.params, history)
        push!(list_total_accepted_people, total_accepted_people)
        push!(list_total_rejected_people, total_rejected_people)
        push!(list_total_accepted_families, total_accepted_families)
        push!(list_total_reward, total_reward)
        push!(list_reward_over_time, reward_over_time)
        push!(list_visa_dict_accepts, visa_dict_accepts)
        push!(list_visa_dict_rejects, visa_dict_rejects)
        push!(list_visa_rate_accepts, visa_dict_accepts_rate)
    end

    round_people_to_ints = false # NOTE: Changing to allow "decimal" people.

    mean_se_reward = mean_se(list_total_reward, false, n_sims)
    mean_se_total_accepted_people = mean_se(list_total_accepted_people, false, n_sims)
    mean_se_total_people = mean_se(list_total_accepted_people + list_total_rejected_people, false, n_sims)
    mean_se_percent_accepted_people = mean_se(100 * list_total_accepted_people ./ (list_total_accepted_people + list_total_rejected_people), false, n_sims)

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

    base_dict_total = Dict{VisaStatus, Vector{Float64}}()
    for dict in list_visa_rate_accepts
        base_dict_rate = merge(vcat, base_dict_rate, dict)
    end

    if str_policy == "AMCITs" || str_policy == "POMDP"
        println()
        print("Policy\t\t\t\t&\t\tReward\t\t\t&\t\tAccepted/Total\t\t\t\t\t&\t")
        for visa_status in sort(collect(keys(base_dict)), rev=true)
            st_visa_status = string(visa_status)
            # just show what order the stats are showing up
            tabs = length(st_visa_status) <= 5 ? "\t\t" : "\t\t\t"
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

    # print("$str_policy$tabs_policy&\t\t\$$(mean_se_reward[1]) \\pm $(mean_se_reward[2])\$\t&\t\t\$$(mean_se_total_accepted_people[1]) \\pm $(mean_se_total_accepted_people[2]) \\;($(mean_se_percent_accepted_people[1])\\pm$(mean_se_percent_accepted_people[2])\\%)\$\t\t&\t")
    print("$str_policy$tabs_policy&\t\t\$$(mean_se_reward[1]) \\pm $(mean_se_reward[2])\$\t&\t\t\$($(mean_se_total_accepted_people[1]) \\pm $(mean_se_total_accepted_people[2]))/$(mean_se_total_people[1])\$\t\t&\t")
    for visa_status in sort(collect(keys(base_dict)), rev=true)
        st_visa_status = string(visa_status)
        expectation_acc = round(base_dict[visa_status]/n_sims, digits=2)
        expectation_rej = round(base_dict_reject[visa_status]/n_sims, digits=2)
        expectation_total = round((base_dict[visa_status]+base_dict_reject[visa_status])/n_sims, digits=2)
        if round_people_to_ints
            expectation_acc = trunc(Int, expectation_acc)
            expectation_rej = trunc(Int, expectation_rej)
            expectation_total = trunc(Int, expectation_total)
        end
        # percent_acc_μ = round(100*mean(filter(!isnan, base_dict_rate[visa_status])), digits=2)
        # percent_acc_σ = round(100*std(filter(!isnan, base_dict_rate[visa_status])), digits=2)
        # percent_acc_μ = isnan(percent_acc_μ) ? "---" : trailing_zeros(percent_acc_μ)
        # percent_acc_σ = isnan(percent_acc_σ) ? "---" : trailing_zeros(percent_acc_σ)
        # print("\$$(expectation_acc)\\;($percent_acc_μ \\pm $percent_acc_σ\\%)\$\t")
        expectation_acc = expectation_acc == 0.0 ? 0 : expectation_acc # remove .0 in zero decimal
        print("\$$expectation_acc/$expectation_total\$\t")
        visa_status == ISIS ? print("\t\\\\") : print("&\t")
    end
    println()

    data = Dict(
        str_policy=>Dict(
            "list_total_accepted_people" => list_total_accepted_people,
            "list_total_rejected_people" => list_total_rejected_people,
            "list_total_accepted_families" => list_total_accepted_families,
            "list_total_reward" => list_total_reward,
            "list_reward_over_time" => list_reward_over_time,
            "list_visa_dict_accepts" => list_visa_dict_accepts,
            "list_visa_dict_rejects" => list_visa_dict_rejects,
            "list_visa_rate_accepts" => list_visa_rate_accepts,
            "visa_accept" => base_dict,
            "visa_reject" => base_dict_reject,
            "visa_rate" => base_dict_rate,
            "visa_total" => base_dict_total
        )
    )
    return data
end


function experiments(n_sims, mdp::EvacuationMDP, mdp_policy)
    # policies and n_sims can probably be put in our params function as a list. here for now.
    dict_policies = Dict(
        "Level I"=>mdp_policy,
        "Random"=>RandomBaselinePolicy(),
        "AcceptAll"=>AcceptAllPolicy(),
        "AMCITs"=>AMCITsPolicy(),
        "SIV-AMCITs"=>SIVAMCITsPolicy(),
        "SIV-AMCITs-P1P2"=>SIVAMCITsP1P2Policy(),
        "Non-ISIS"=>NonISISPolicy(),
        "AfterThresholdAMCITs"=>AfterThresholdAMCITsPolicy(mdp_policy=mdp_policy),
        "BeforeThresholdAMCITs"=>BeforeThresholdAMCITsPolicy(mdp_policy=mdp_policy),
    )

    aggdata = Dict()
    for str_policy in sort(collect(keys(dict_policies)))
        data = simulations(dict_policies[str_policy], str_policy, mdp, n_sims)
        aggdata = merge(aggdata, data)
    end

    return aggdata
end


function experiments(n_sims, pomdp::EvacuationPOMDPType, pomdp_policy, str_policy)
    simulations(pomdp_policy, str_policy, pomdp, n_sims)
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


function generate_population_trajectories(seed=47)
    params = EvacuationParameters()
    family_distribution = SparseCat(params.family_sizes, params.family_prob)

    Random.seed!(seed)
    N = 1000
    pops = Vector(undef, N)
    tmp_pomdp = EvacuationPOMDPType()
    for i in 1:N
        pops[i] = []
        dir = Dirichlet(params.visa_prior_count)
        pop_distribution = SparseCat(params.visa_status, rand(dir))
        for t in 0:params.time+1
            true_status = rand(pop_distribution)
            family_size = rand(family_distribution)
            s = newstate(POMDPState, 1, 1, family_size, true_status)
            obs = rand(observation(tmp_pomdp, s, ACCEPT, s))
            obs_status = obs.vdoc
            made_it_through = rand() < params.p_transition
            sample = (true_status=true_status, obs_status=obs_status, family_size=family_size,
                      made_it_through=made_it_through)
            push!(pops[i], sample)
        end
    end
    open("population.json", "w+") do f
        write(f, JSON.json(pops))
    end
    num_isis_true = length(findall(map(pop->ISIS in map(p->p.true_status, pop), pops)))
    num_isis_obs = length(findall(map(pop->ISIS_indicator in map(p->p.obs_status, pop), pops)))
    @show num_isis_true, num_isis_obs
    return pops
end


function read_population_json()
    open("population.json", "r") do f
        pops = JSON.parse(read(f, String))
        [map(p->(true_status=eval(Meta.parse(p["true_status"])), obs_status=eval(Meta.parse(p["obs_status"])), family_size=p["family_size"], made_it_through=p["made_it_through"]), pop) for pop in pops]
    end
end
