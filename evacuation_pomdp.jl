### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ 30028219-7ea5-4b78-b0df-b3b98b25ee65
using PlutoUI

# ╔═╡ e2772773-4058-4a1b-9f99-d197942086d5
begin
	using POMDPs
	using DiscreteValueIteration
	using POMDPPolicies
	using POMDPModelTools #for sparse cat 
	using Parameters
	using Random
	using Plots; default(fontfamily="Computer Modern", framestyle=:box)
	using QuickPOMDPs
	using Distributions 
	using LinearAlgebra
	using POMDPSimulators
	using Measures
	using DataStructures
	using ColorSchemes
end

# ╔═╡ 41d0fe72-b83d-4cf8-bc91-44b0e8f561eb
TableOfContents()

# ╔═╡ c0292f00-7f92-11ec-2f15-b7975a5e40a1
md"""
# Evacuation POMDP
"""

# ╔═╡ fe9dfc73-f502-4d13-a019-8160a6af4617
Random.seed!(0xC0FFEE)

# ╔═╡ ffbb36c6-8e69-46d1-b132-845564de0ae2
md"""
## Environment Parameters
"""

# ╔═╡ a0a58020-32ed-42b4-ad9c-2d1c0357f3c3
md"""
## States
"""

# ╔═╡ 99790e08-5bb7-4611-8d4b-39db4d36ca83
struct State
    c::Int # chairs remaining 
    t::Int # time remaining 
    f::Int # family size 
    v::Int # visa status 
end

# ╔═╡ 8a899d1e-504b-4a12-aaaf-f5cf7949669a
@with_kw struct EvacuationParameters
    family_sizes::Vector{Int} = [1, 2, 3, 4, 5] 
    family_prob = [.1, .2, .3, .2, .2]
    visa_status::Vector{Int} = [-500, -3, 1, 5, 20] 
    # visa_status::Vector{Int} = [-2, -1, 0, 1, 2] # Note, this was uncommented.
    visa_prob = [.01, .46, .35, .13, .05] 
    # -2, -1, 0, 1, 2 for other reward structure. 
    v_stringtoint = Dict("ISIS-K"=>-500,
        # "two SIVs as good as one taliban...US gov much more risk avers"
		"Afghan"=>-3,
		# anyone who doesn't have a visa....0 or negative. 
		# even beat up over ...everyone directly served the us effort in
		# some way either through developer contracters or military
        "P1/P2 Afghan"=>1,
        "SIV"=>5,
        "AMCIT"=>20)

     v_inttostring = Dict(-500=>"ISIS-K",
        -3=>"Afghan",
        1=>"P1/P2 Afghan",
        5=>"SIV",
        20=>"AMCIT")
    
    capacity::Int = 120 # keeping these both as integers of 20 for now. 
    time::Int = 120
    size::Tuple{Int, Int} = (length(visa_status), length(family_sizes)) # grid size
    p_transition::Real = 0.8 # don't we always transition into this since time moves forward? I'm confused... 
    null_state::State = State(-1, -1, -1 ,-1) # is there someway to do this???
    accept_prob = [.80, .20]
    reject_prob = [1.0]
end

# ╔═╡ 6e7cf801-99d8-4bec-a036-3a0b2475a2eb
params = EvacuationParameters();

# ╔═╡ a9899239-df71-41a2-bc66-f08ee325ffe2
params.v_inttostring

# ╔═╡ e7b7253f-68c6-4026-9a86-37d016e189d2
begin
	# The state space S for the evacuation problem is the set of all combinations 
	𝒮 = []
	# capacity ends at 0 
	for c in 0:params.capacity
	    # time ends at 0
		for t in 0:params.time
			# family size here we should have the ACTUAL family sizes
	        for f in params.family_sizes
				# actual visa statuses
	            for v in params.visa_status
	                new = State(c, t, f, v) 
	                push!(𝒮, new)
	            end
	        end
	    end
	end
	push!(𝒮, params.null_state)
end

# ╔═╡ c4bb4cf3-c185-4e35-91fa-f1de57590002
number_states = (params.capacity+1) * (params.time+1) * size(params.family_sizes)[1] * size(params.visa_status)[1] + 1

# ╔═╡ b731de36-942b-4e63-8664-9bf97b43dcb4
md"""
> 280 seconds for `𝒮 = [𝒮; new]` vs. ~15 ms for `push!(𝒮, new)`
"""

# ╔═╡ 1b18e587-a645-4baa-b485-8fd05323d92e
length(𝒮)

# ╔═╡ 68a1a92f-b028-4041-86f0-3415d597c658
md"""
## Actions
"""

# ╔═╡ e6ab99cc-a698-4177-8651-10013c01bbaa
# the possible actions are whether accept or reject a family at the gate 
@enum Action REJECT ACCEPT

# ╔═╡ 3a121ae5-47a9-4bf8-8979-7c873c94dd4f
𝒜 = [REJECT, ACCEPT]

# ╔═╡ a4d22509-0bb4-4779-a9f1-5be28da4a748
# only inbounds if room for the family [assuming would not separate even though might]
# and if time is available to enter the airport 
validtime(s::State) = 0 < s.t

# ╔═╡ fca35d50-4bc8-41cc-bf67-54a8f15490de
validcapacity(s::State) = 0 ≤ s.c # maybe try back to 0

# ╔═╡ 63aa8075-9d3c-44a8-9442-eef2d14372a8
md"""
## Transition function
"""

# ╔═╡ d655a73f-d0b8-4735-8cd0-d03c7adeda29
#***** ENUMERATING OVER ALL STATES ******
function T(s::State, a::Action)
    next_states = State[]
    probabilities = Float64[] 
    
    if !validtime(s) 
        push!(next_states,params.null_state)
        push!(probabilities, 1) # double check 
    else
        if a == ACCEPT
			# check if valid capacity 
            next_state_accept = State(s.c - s.f, s.t - 1, 1, 1)
            if !validcapacity(next_state_accept) 
				# no room for full family, so we make prob. 0 to accept and 1 reject
				prob = [0,1]
            else
                prob = params.accept_prob
            end
            for f in 1:length(params.family_sizes)
                for v in 1:length(params.visa_status)
                    # if get on plan
					family_size = params.family_sizes[f]
					visa_status = params.visa_status[v]
                    push!(next_states, 
						State(s.c-s.f, s.t-1, family_size, visa_status))
					visa_prob = params.visa_prob[v]
					family_prob = params.family_prob[f]
                    push!(probabilities, prob[1] * visa_prob * family_prob)

					# if not
                    push!(next_states, State(s.c, s.t-1, family_size, visa_status))
                    push!(probabilities, prob[2] * visa_prob * family_prob)
                end
            end
		else # if reject     
            for f in 1:length(params.family_sizes)
                for v in 1:length(params.visa_status)
                    push!(next_states, State(s.c, s.t-1,
						params.family_sizes[f], params.visa_status[v]))
                    push!(probabilities, params.reject_prob[1] *
						params.visa_prob[v] * params.family_prob[f])
                end
            end  
        end
    end                
    normalize!(probabilities, 1)
    @assert sum(probabilities) ≈ 1
    return SparseCat(next_states, probabilities)
end

# ╔═╡ 5022c6f3-09f8-44bc-b41e-a86cbf8f787c
md"""
## Reward function
"""

# ╔═╡ 2dbe879c-ea53-40fb-a334-5cb8f254faf7
function R(s::State, a::Action)
    # reward is just the visa status times family size i think! 
    if a == ACCEPT
        return s.v*s.f
	else
		return 0
	end
end

# ╔═╡ dfb98cff-a786-4174-bc43-0fd22eec29bd
md"""
## Discount factor
"""

# ╔═╡ 463f720a-f10e-4419-b7fc-84e60b917b9a
γ = 0.95

# ╔═╡ 55a665bb-85a6-4eb8-bf5f-9ba4ac0783fb
md"""
## Termination
"""

# ╔═╡ f4b1ca44-9db9-48b8-89cb-0a4a86e022db
termination(s::State)= s == params.null_state # change to 1 or the other

# ╔═╡ 47a33688-6458-44a6-a5e5-3a6a220e9c39
md"""
## MDP formulation
"""

# ╔═╡ dda2b2be-b488-48c9-8475-2e4876ae517f
abstract type Evacuation <: MDP{State, Action} end

# ╔═╡ 889f89e8-61d2-4273-a63c-bcb6e6b0852d
begin
	c_initial = params.capacity
	t_initial = params.time
	f_initial = rand(params.family_sizes, 1)[1]
	v_initial = rand(params.visa_status, 1)[1]
	initial_state = State(c_initial, t_initial, f_initial, v_initial)
	statetype = typeof(initial_state)
	initialstate_array = [initial_state]
end;

# ╔═╡ 6bf5f2f2-b5d1-48a6-b811-f416bcfc9899
mdp = QuickMDP(Evacuation,
    states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ,
    initialstate = initialstate_array, 
    isterminal   = termination,
    render       = render,
    statetype    = statetype 
);

# ╔═╡ 5acdce32-b587-4ea2-8317-a753429fcd7b
md"""
## Solving MDP
"""

# ╔═╡ 4bfee8de-cc8d-436d-98ad-5f7002ece4cf
solver = ValueIterationSolver(max_iterations=30, belres=1e-6, verbose=true);

# ╔═╡ f7ae5cc2-29d0-42fd-8210-d95417e825b1
mdp_policy = solve(solver, mdp)  # look at this to see why it's not graphing anymore

# ╔═╡ f755070e-615b-4ca3-9c28-58170af21856
md"""
## Baseline policies
"""

# ╔═╡ a52c28f0-82cc-49e2-8bf2-611a5178a7fc
md"""
### Accept all
"""

# ╔═╡ d29420f6-e511-42aa-8728-5df57df8018b
struct AcceptAll <: Policy end

# ╔═╡ 2926975b-310d-462e-9f28-b747c960e0a8
# accept everyone until capacity is 0
function POMDPs.action(::AcceptAll, s::State)    # action(policy, state)
    return ACCEPT
end;

# ╔═╡ c3177a4e-02a2-444d-9f47-f3d5ef0a9819
AcceptAll_policy = AcceptAll()

# ╔═╡ d2370403-143e-46e4-a1a4-02a937e7b7e8
md"""
### AMCITS
"""

# ╔═╡ 7624961c-8e2d-4441-ba4a-43a7ed192ad4
struct AMCITS <: Policy end

# ╔═╡ cdd21e13-86ce-4856-add2-0203cb12591a
function POMDPs.action(::AMCITS, s::State)
    return (s.v == params.v_stringtoint["AMCIT"])  ? ACCEPT : REJECT
end;

# ╔═╡ c1d6bb79-bcff-4b3e-8011-314ef6f7a4e8
AMCITS_policy = AMCITS()

# ╔═╡ 9762a300-61d1-4a23-89de-18d3f7fedf7f
md"""
### SIV-AMCITS
"""

# ╔═╡ eec68b57-e50f-447b-b852-1130e10d4d23
struct SIV_AMCITS <: Policy end

# ╔═╡ 2aee5336-148c-4a95-9007-72d976d6459a
function POMDPs.action(::SIV_AMCITS, s::State)
    return (s.v == params.v_stringtoint["AMCIT"] ||
		s.v == params.v_stringtoint["SIV"]) ? ACCEPT : REJECT
end;

# ╔═╡ a5e729f2-f8f5-42d6-b540-16d1e62d2b91
SIV_AMCITS_policy = SIV_AMCITS()

# ╔═╡ 64c74a69-3945-4599-840a-be37f0ee7ab2
md"""
### After threshold AMCITS
"""

# ╔═╡ 99836668-cc71-4111-a6c2-914fac3cd44a
# if want to change this need to make it a mutable struct 
@with_kw struct AfterThresholdAMCITS <: Policy
    threshold = 20 # could define this in parameters 
end

# ╔═╡ 7d0c384e-cb4c-46a5-8bd2-52181dcb7d2a
function POMDPs.action(policy::AfterThresholdAMCITS, s::State)
    if s.t <= 20 #policy.threshold
        return s.v == params.v_stringtoint["AMCIT"] ? ACCEPT : REJECT
    else
        return action(mdp_policy, s)
    end
end

# ╔═╡ de0b6e81-6681-4ec2-9352-4adb6ee14cdb
AfterThresholdAMCITS_policy = AfterThresholdAMCITS()

# ╔═╡ 98f5c2d7-8340-4788-a6e7-34c561c9d656
md"""
### Before threshold AMCITS
"""

# ╔═╡ 135a2da9-39ab-4efe-a4a3-518b3519b5be
@with_kw struct BeforeThresholdAMCITS <: Policy
    threshold = 20
end

# ╔═╡ dc21b305-3aa8-479e-9582-59298a99f231
function POMDPs.action(policy::BeforeThresholdAMCITS, s::State)
    if s.t >= 20 #policy.threshold
        return s.v == params.v_stringtoint["AMCIT"] ? ACCEPT : REJECT
    else
        return action(mdp_policy, s)
    end
end

# ╔═╡ a3df3c80-32ee-45e5-8efa-743fc6670762
BeforeThresholdAMCITS_policy = BeforeThresholdAMCITS()

# ╔═╡ e02ff44c-86a5-4748-8533-d177abfc9262
#simulations(BeforeThresholdAMCITS_policy, mdp, 10)
# could play with changing this threshold

# ╔═╡ 55473de6-68f1-49b7-83d3-a1ab70f07185
md"""
## Simulation
"""

# ╔═╡ 88b6b527-d0be-41cc-a800-3e1179a8fdd9
"""Given a history of a simulation, return metrics."""
function get_metrics(history)
    total_accepted_people = 0
    total_accepted_families = 0 
    total_rejected_people = 0
    total_rejected_families = 0
    total_reward = 0.0
    # Initialize visa_statuses dictionary
    visa_statuses = params.visa_status
    visa_dict_accepts = Dict(0=>0) # make dictionaries accept just ints 
    for v in visa_statuses
        visa_dict_accepts[v] = 0
    end
    
    visa_dict_rejects = Dict()
    for v in visa_statuses
        visa_dict_rejects[v] = 0
    end

    # State(c, t, f, v)
    for (s, a, r, sp) in eachstep(history, "(s, a, r, sp)") 
        # only counting the s not sp so as not to double count 
        if a==ACCEPT
            total_accepted_people += s.f
            total_accepted_families += 1
            visa_dict_accepts[s.v] += 1
        else # action is reject 
            total_rejected_people += s.f
            total_rejected_families +=1
            visa_dict_rejects[s.v] += 1
        end

        total_reward += r
       # println("reward $r received when state $sp was reached after action $a was taken in state $s")    
    end
    return total_accepted_people, total_accepted_families, total_reward, visa_dict_accepts 
end

# ╔═╡ 9283a1f9-2f75-4835-9d71-c479cac7641e
"""Given a policy and mdp simulate a rollout and return the history."""
function simulation(policy, mdp)
    hr = HistoryRecorder()
    history = simulate(hr, mdp, policy)
    return history
end

# ╔═╡ c9b7288d-2d59-4008-b0a8-2853fa0baad2
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

# ╔═╡ 2d97fba8-1b5b-4e98-9b53-0efbf3aac8d4
"""Simulate n times and get the mean and std of n rollouts of a policy."""
function simulations(policy, str_policy, mdp, n_sims) # n is number of times to run 
    people = false
    
    list_total_accepted_people = []
    list_total_accepted_families = []
    list_total_reward = []
    list_visa_dict_accepts = []
    
    for i in 1:n_sims
        history = simulation(policy, mdp) # do 1 simulation and get the history 
        total_accepted_people, total_accepted_families, total_reward, visa_dict_accepts = get_metrics(history)
        push!(list_total_accepted_people, total_accepted_people)
        push!(list_total_accepted_families, total_accepted_families)
        push!(list_total_reward, total_reward)
        push!(list_visa_dict_accepts, visa_dict_accepts)
    end

    
    mean_std_reward = mean_std(list_total_reward, false)
    mean_std_total_accepted_people = mean_std(list_total_accepted_people, true)
    
    
    # calculate average ppl by visa type 
    base_dict = Dict{Int64, Int64}(1=>0)
    # add together total people over sims 
    for dict in list_visa_dict_accepts
        base_dict = merge(counter(base_dict), counter(dict)).map

    end

    # divide by num sims 
    for visa_status in keys(base_dict)
        base_dict[visa_status]= trunc(Int, base_dict[visa_status]/n_sims)
    end
    # print in latex form 
   # base_dict_sorted = sort(base_dict)
    
    if (str_policy == "AMCITS-policy")
        print("Policy        &     Reward        &  Num Accepted   &")
        for visa_status in sort(collect(keys(base_dict)), rev=true)
            if (visa_status != 0) # for some reason got a zero in there...) 
                st_visa_status = params.v_inttostring[visa_status]
				# just show what order the stats are showing up
				print("$st_visa_status &")
            end
        end
    end
    println()

    print("$str_policy & $(mean_std_reward[1]) \$\\pm\$ $(mean_std_reward[2]) & $(mean_std_total_accepted_people[1]) \$\\pm\$ $(mean_std_total_accepted_people[2]) & ")
    for visa_status in sort(collect(keys(base_dict)), rev=true)
        if (visa_status != 0) # for some reason got a zero in there...) 
            st_visa_status = params.v_inttostring[visa_status]
            print("$(base_dict[visa_status]) &   ")
        end
    end
    println()
end

# ╔═╡ 9d286c16-6256-4ad5-8ebd-d2b102c6562c
function experiments()
    # policies and n_sims can probably be put in our params function as a list. here for now. 
    n_sims = 1000
    dict_policies = Dict("mdp-policy"=>mdp_policy, 
        "AcceptAll-policy"=>AcceptAll_policy, 
        "AMCITS-policy"=>AMCITS_policy, 
        "SIV-AMCITS-policy"=>SIV_AMCITS_policy, 
        "AfterThresholdAMCITS-policy"=>AfterThresholdAMCITS_policy, 
        "BeforeThresholdAMCITS-policy"=>BeforeThresholdAMCITS_policy)
    
    # for now don't report the std here. would need to keep track of every single value....
    for str_policy in sort(collect(keys(dict_policies)))
        list_visa_dict_accepts = simulations(dict_policies[str_policy], str_policy, mdp, n_sims)
    end
end

# ╔═╡ 2a8a9dc0-f866-4cdb-81ee-4f4da1ee40e6
md"""
## Colorscheme
- [https://en.wikipedia.org/wiki/Flag\_of\_Afghanistan](https://en.wikipedia.org/wiki/Flag_of_Afghanistan)
"""

# ╔═╡ b0da5b04-afc3-47b6-8e21-329456c5d7e8
afghan_red = colorant"#be0000";

# ╔═╡ d343e7ac-edc6-469f-9f21-b6d7e42b3e3c
afghan_green = colorant"#007a36";

# ╔═╡ 87bf7bda-81d5-4325-9f4b-c8e8f6af243a
cmap = ColorScheme([afghan_red, afghan_green])

# ╔═╡ c07c44c9-866d-4e95-8c24-6a5f96b2a57c
md"""
## Visualizations
"""

# ╔═╡ 39237f4f-6c60-43a5-ab51-abd13f438b9b
"""
Pass in policy and chairs and time remaing. Spit out graph of family size versus visa status.
"""
function vis_time_step(policy, c, t)
    (v_size, f_size) = params.size  #visa, family 5, 5
    policyGraph = zeros(v_size, f_size) 
    
	visa_statuses = params.visa_status
	x = visa_statuses
	family_sizes = params.family_sizes
	y = family_sizes
        
    for f in 1:f_size
        for v in 1:v_size
            act = action(policy, State(c, t, family_sizes[f], visa_statuses[v])) 
            if act == ACCEPT
                policyGraph[v,f] = 500
            else
                policyGraph[v,f] = 0
            end
        end
    end
    
    z = policyGraph'

    timeVal = string(t)
    capVal = string(c)
    titleX = string("t: " * timeVal * " c: " * capVal)

    return heatmap(x, y, z, 
         # aspect_ratio = 1,
         legend = :none, 
         xlims = (params.visa_status[1], params.visa_status[length(params.visa_status)]),
         ylims = (params.family_sizes[1], params.family_sizes[length(params.family_sizes)]),         
         xlabel = "Visa Status",
         ylabel = "Family Size",
         title = titleX,
         xtickfont = font(6, "Courier"), 
         ytickfont = font(6, "Courier"),
         thickness_scaling = .9,
         color=cmap.colors,   
	)
end

# ╔═╡ 33b4e174-6cc8-4dc9-8d81-01408e1b31a4
md"""
# Dirichlet belief
"""

# ╔═╡ 621e43d5-39da-4fd5-8f4b-6cd5fb18d0d5
𝒟₀ = Dirichlet(Int.(params.visa_prob .* 100))

# ╔═╡ a9e85360-3add-42d9-82c6-fe66bf506811
# Plot Dirichlet distribution counts over each provided category.
function Plots.plot(𝒟::Dirichlet, categories::Vector, cmap; kwargs...)
	transposed = reshape(categories, (1, length(categories)))
	bar(
	    transposed,
	    𝒟.alpha',
	    labels = transposed,
	    bar_width = 1,
		c = [get(cmap, i/length(categories)) for i in 1:length(categories)]';
		kwargs...
	)
end

# ╔═╡ 6679eb9a-164d-4602-9d2f-0569b0018144
function vis_all(policy)
    total_time = params.time 
    total_capacity = params.capacity
    graph_per_n = 60
    heat_maps = []
    time_points = (total_time/graph_per_n) + 1 # to include 0 
    capacity_points = (total_capacity/graph_per_n) + 1 
    num_graphs = trunc(Int, time_points*capacity_points)
    
    for t in 0:total_time
        if t % graph_per_n == 0 
            for c in 0:total_capacity
                if c % graph_per_n == 0
                push!(heat_maps, vis_time_step(policy, c, t))
                end
            end
        end
    end 
    plot(heat_maps..., layout = num_graphs, margin = 1mm)
    #print(num_graphs)
    #plot(heat_maps[1], heat_maps[2], heat_maps[3],heat_maps[4],heat_maps[5],
    #    heat_maps[6],heat_maps[7],heat_maps[8],heat_maps[9],heat_maps[10],
    #    heat_maps[11],heat_maps[12],heat_maps[13],heat_maps[14],heat_maps[15],heat_maps[16],layout = 16, margin = 5mm)
end

# ╔═╡ 40430b14-07b2-40c6-95fe-9f60a5c6e75f
vis_all(mdp_policy)

# ╔═╡ 83a49887-dd56-4f84-b7ab-1bf07c87f491
cmap_bar = ColorScheme([afghan_red, colorant"lightgray", afghan_green])

# ╔═╡ ea0311f8-5752-41e3-9dab-a1e35d5e733b
visa_statuses = ["ISIS-K", "Afghan", "P1/P2 Afghan", "SIV", "AMCIT"] # NOTE: ordering

# ╔═╡ de3c5a5d-65e0-423e-9eec-7bd4ae61a70d
params.visa_prob

# ╔═╡ ff81b525-3d5e-433b-b9bb-110172b4364e
plot(𝒟₀, visa_statuses, cmap_bar, title="Dirichlet expert prior")

# ╔═╡ 4943cac7-d42e-47d9-adea-d2be4d25d391
md"""
## Updating Dirichlet belief
"""

# ╔═╡ d34f13c9-4de1-4ffc-88c6-87a95163861a
𝒟_true = Categorical([0.01, 0.50, 0.31, 0.10, 0.08]);

# ╔═╡ c9447ee0-2477-49ae-8ca5-45fb5578e26e
is_uniform_initial = true

# ╔═╡ dab370d8-5953-418b-95a2-3cb9db8b7ee4
if is_uniform_initial
	𝒟_belief = Dirichlet(ones(length(params.visa_status)))
else
	𝒟_belief = deepcopy(𝒟₀) # initial belief
end

# ╔═╡ 63f4d4e6-f92c-4a9d-beea-27ca15492647
begin
	b_p = plot(𝒟_belief, visa_statuses, cmap_bar, title="updated Dirichlet posterior")
	v′ = rand(𝒟_true) # sample a new visa status from the latent _true_ distribution
	𝒟_belief.alpha[v′] += 1 # update pseudocount
	b_p
end

# ╔═╡ 118b7027-3e5a-4036-8f5d-9e5c8818a642
begin
	v′ # trigger
	new_visa_probs = normalize(𝒟_belief.alpha ./ 𝒟_belief.alpha0, 1)
end

# ╔═╡ a3233be1-4002-4545-b306-9197a556c3f9
md"""
## Dirichlet belief updater
Model after: [https://github.com/JuliaPOMDP/BeliefUpdaters.jl/blob/master/src/discrete.jl](https://github.com/JuliaPOMDP/BeliefUpdaters.jl/blob/master/src/discrete.jl)

- `DirichletBelief{P<:POMDP, S}`
    - `uniform_belief`
    - `pdf`
    - `rand`
    - `support`
    - `mean`
    - `mode`
    - `==` and `hash`
- `DirichletUpdater{P<:POMDP} <: Updater`
    - `uniform_belief(up::DirichletUpdater)`
    - `initialize_belief(up::DirichletUpdater, distr::Any)`
    - `update(up::DirichletUpdater, b::DirichletBelief, a, o)`
"""

# ╔═╡ 4a3c62fb-ea78-41ac-92b5-9d55f5f05789
pdf(𝒟_belief, normalize([1,1,1,1,1], 1))

# ╔═╡ 958c44ef-9c64-484c-ae80-0a62f2142225
md"""
# POMDP formulation
"""

# ╔═╡ 257308b7-c3c5-43cc-98ba-8f8a8306fb61
md"""
## Observations
- how to model "observations"?
- some sort of documentation?
- Actual SIV approval number, or SIV processing number.
"""

# ╔═╡ a081e9f0-1ee6-4c28-9409-c3ce1e190084
@enum Observation ISIS Afghan P1P2Afghan SIV AMCIT # TODO: documented visa status?

# ╔═╡ 18a6f1e2-5ee2-40b2-8861-9e634a74de3a
md"""
## Observation function
The observation function models the likelihood of an observation $o$ given the next state $s'$, $O(o \mid s')$.

$$\begin{align*}
O(o_\text{ISIS} \mid s'_\text{ISIS}) &= x\%\\
O(o_\text{Afhgan} \mid s'_\text{Afghan}) &= x\%\\
O(o_\text{P1/P2 Afhgan} \mid s'_\text{P1/P2 Afghan}) &= x\%\\
O(o_\text{SIV} \mid s'_\text{SIV}) &= x\%\\
O(o_\text{AMCIT} \mid s'_\text{AMCIT}) &= x\%
\end{align*}$$
"""

# ╔═╡ b2aef310-148f-4386-a41e-ead3c32f6ca4
function O(a::Action, s′::State)
	if params.v_inttostring[s′] == "ISIS-K"
		# return SparseCat([])
	end
	# if s′ == HUNGRYₛ
	# 	return SparseCat([CRYINGₒ, QUIETₒ],
	# 		             [params.p_crying_when_hungry, 1-params.p_crying_when_hungry])
	# elseif s′ == FULLₛ
	# 	return SparseCat([CRYINGₒ, QUIETₒ],
	# 		             [params.p_crying_when_full, 1-params.p_crying_when_full])
	# end
end

# ╔═╡ edae2f7c-cd4c-42f6-a423-ad5f1f1bf5cd
md"""
> Note, may need to define `O(s, a, s′) = O(NULL, a, s′)`
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
DiscreteValueIteration = "4b033969-44f6-5439-a48b-c11fa3648068"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Measures = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
POMDPModelTools = "08074719-1b2a-587c-a292-00f91cc44415"
POMDPPolicies = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
POMDPSimulators = "e0d0a172-29c6-5d4e-96d0-f262df5d01fd"
POMDPs = "a93abf59-7444-517b-a68a-c42f96afdd7d"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuickPOMDPs = "8af83fb2-a731-493c-9049-9e19dbce6165"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
ColorSchemes = "~3.16.0"
DataStructures = "~0.18.11"
DiscreteValueIteration = "~0.4.5"
Distributions = "~0.25.41"
Measures = "~0.3.1"
POMDPModelTools = "~0.3.10"
POMDPPolicies = "~0.4.2"
POMDPSimulators = "~0.3.13"
POMDPs = "~0.9.3"
Parameters = "~0.12.3"
Plots = "~1.25.7"
PlutoUI = "~0.7.32"
QuickPOMDPs = "~0.2.13"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "f87e559f87a45bece9c9ed97458d3afe98b1ebb9"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.1.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BeliefUpdaters]]
deps = ["POMDPModelTools", "POMDPs", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "7d4f9d57116796ae3fc768d195386b0a42b4a58d"
uuid = "8bb6e9a1-7d73-552c-a44a-e5dc5634aac4"
version = "0.2.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "54fc4400de6e5c3e27be6047da2ef6ba355511f8"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "6b6f04f93710c71550ec7e16b650c1b9a612d0b6"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.16.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonRLInterface]]
deps = ["MacroTools"]
git-tree-sha1 = "21de56ebf28c262651e682f7fe614d44623dc087"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.1"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ae02104e835f219b8930c7664b8012c93475c340"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.2"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiscreteValueIteration]]
deps = ["POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPs", "Printf", "SparseArrays"]
git-tree-sha1 = "7ac002779617a7e1693ccdcc3a534f555b3ea61e"
uuid = "4b033969-44f6-5439-a48b-c11fa3648068"
version = "0.4.5"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "5863b0b10512ed4add2b5ec07e335dc6121065a5"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.41"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "4a740db447aae0fbeb3ee730de1afbb14ac798a1"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.63.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "aa22e1ee9e722f1da183eb33370df4c1aeb6c2cd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.63.1+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "22df5b96feef82434b07327e2d3c770a9b21e023"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LightGraphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "432428df5f360964040ed60418dd5601ecd240b6"
uuid = "093fc24a-ae57-5d10-9952-331d41423f4d"
version = "1.3.5"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "f755f36b19a5116bb580de457cda0c140153f283"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.6"

[[deps.NamedTupleTools]]
git-tree-sha1 = "63831dcea5e11db1c0925efe5ef5fc01d528c522"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.13.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "648107615c15d4e09f7eca16307bc821c1f718d8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.13+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[deps.POMDPLinter]]
deps = ["Logging"]
git-tree-sha1 = "cee5817d06f5e1a9054f3e1bbb50cbabae4cd5a5"
uuid = "f3bd98c0-eb40-45e2-9eb1-f2763262d755"
version = "0.1.1"

[[deps.POMDPModelTools]]
deps = ["CommonRLInterface", "Distributions", "LinearAlgebra", "POMDPLinter", "POMDPs", "Random", "SparseArrays", "Statistics", "Tricks", "UnicodePlots"]
git-tree-sha1 = "58ca1062c4c0e14f618ee1b3483eed38adec74b1"
uuid = "08074719-1b2a-587c-a292-00f91cc44415"
version = "0.3.10"

[[deps.POMDPPolicies]]
deps = ["BeliefUpdaters", "Distributions", "LinearAlgebra", "POMDPModelTools", "POMDPs", "Parameters", "Random", "SparseArrays", "StatsBase"]
git-tree-sha1 = "b9b6233a436aaa2829c3fd9dd6d51a9d2695cc30"
uuid = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
version = "0.4.2"

[[deps.POMDPSimulators]]
deps = ["BeliefUpdaters", "DataFrames", "Distributed", "NamedTupleTools", "POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPs", "ProgressMeter", "Random"]
git-tree-sha1 = "3861f0b2a33b38be6876180e46096115d2664dba"
uuid = "e0d0a172-29c6-5d4e-96d0-f262df5d01fd"
version = "0.3.13"

[[deps.POMDPTesting]]
deps = ["POMDPs", "Random"]
git-tree-sha1 = "6186037fc901d91703c0aa7ab10c145eeb6d0796"
uuid = "92e6a534-49c2-5324-9027-86e3c861ab81"
version = "0.2.5"

[[deps.POMDPs]]
deps = ["Distributions", "LightGraphs", "NamedTupleTools", "POMDPLinter", "Pkg", "Random", "Statistics"]
git-tree-sha1 = "3a8f6cf6a3b7b499ec4294f2eb2b16b9dc8a7513"
uuid = "a93abf59-7444-517b-a68a-c42f96afdd7d"
version = "0.9.3"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "92f91ba9e5941fc781fecf5494ac1da87bdac775"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "6f1b25e8ea06279b5689263cc538f51331d7ca17"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.1.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "7e4920a7d4323b8ffc3db184580598450bde8a8e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "ae6145ca68947569058866e443df69587acc1806"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.32"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.QuickPOMDPs]]
deps = ["BeliefUpdaters", "NamedTupleTools", "POMDPModelTools", "POMDPTesting", "POMDPs", "Random", "Tricks", "UUIDs"]
git-tree-sha1 = "f65b3fdaf87308dd87241f85f391abbdc0361962"
uuid = "8af83fb2-a731-493c-9049-9e19dbce6165"
version = "0.2.13"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "37c1631cb3cc36a535105e6d5557864c82cd8c2b"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e6bf188613555c78062842777b116905a9f9dd49"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "2884859916598f974858ff01df7dfc6c708dd895"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f35e1879a71cca95f4826a14cdbf0b9e253ed918"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.15"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "d21f2c564b21a202f4677c0fba5b5ee431058544"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnicodePlots]]
deps = ["Contour", "Crayons", "Dates", "SparseArrays", "StatsBase"]
git-tree-sha1 = "62595983da672758a96f89e07f7fd3735f16c18c"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "2.7.0"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═30028219-7ea5-4b78-b0df-b3b98b25ee65
# ╠═41d0fe72-b83d-4cf8-bc91-44b0e8f561eb
# ╟─c0292f00-7f92-11ec-2f15-b7975a5e40a1
# ╠═e2772773-4058-4a1b-9f99-d197942086d5
# ╠═fe9dfc73-f502-4d13-a019-8160a6af4617
# ╟─ffbb36c6-8e69-46d1-b132-845564de0ae2
# ╠═8a899d1e-504b-4a12-aaaf-f5cf7949669a
# ╠═6e7cf801-99d8-4bec-a036-3a0b2475a2eb
# ╠═a9899239-df71-41a2-bc66-f08ee325ffe2
# ╟─a0a58020-32ed-42b4-ad9c-2d1c0357f3c3
# ╠═99790e08-5bb7-4611-8d4b-39db4d36ca83
# ╠═e7b7253f-68c6-4026-9a86-37d016e189d2
# ╠═c4bb4cf3-c185-4e35-91fa-f1de57590002
# ╟─b731de36-942b-4e63-8664-9bf97b43dcb4
# ╠═1b18e587-a645-4baa-b485-8fd05323d92e
# ╟─68a1a92f-b028-4041-86f0-3415d597c658
# ╠═e6ab99cc-a698-4177-8651-10013c01bbaa
# ╠═3a121ae5-47a9-4bf8-8979-7c873c94dd4f
# ╠═a4d22509-0bb4-4779-a9f1-5be28da4a748
# ╠═fca35d50-4bc8-41cc-bf67-54a8f15490de
# ╟─63aa8075-9d3c-44a8-9442-eef2d14372a8
# ╠═d655a73f-d0b8-4735-8cd0-d03c7adeda29
# ╟─5022c6f3-09f8-44bc-b41e-a86cbf8f787c
# ╠═2dbe879c-ea53-40fb-a334-5cb8f254faf7
# ╟─dfb98cff-a786-4174-bc43-0fd22eec29bd
# ╠═463f720a-f10e-4419-b7fc-84e60b917b9a
# ╟─55a665bb-85a6-4eb8-bf5f-9ba4ac0783fb
# ╠═f4b1ca44-9db9-48b8-89cb-0a4a86e022db
# ╟─47a33688-6458-44a6-a5e5-3a6a220e9c39
# ╠═dda2b2be-b488-48c9-8475-2e4876ae517f
# ╠═889f89e8-61d2-4273-a63c-bcb6e6b0852d
# ╠═6bf5f2f2-b5d1-48a6-b811-f416bcfc9899
# ╟─5acdce32-b587-4ea2-8317-a753429fcd7b
# ╠═4bfee8de-cc8d-436d-98ad-5f7002ece4cf
# ╠═f7ae5cc2-29d0-42fd-8210-d95417e825b1
# ╟─f755070e-615b-4ca3-9c28-58170af21856
# ╟─a52c28f0-82cc-49e2-8bf2-611a5178a7fc
# ╠═d29420f6-e511-42aa-8728-5df57df8018b
# ╠═2926975b-310d-462e-9f28-b747c960e0a8
# ╠═c3177a4e-02a2-444d-9f47-f3d5ef0a9819
# ╟─d2370403-143e-46e4-a1a4-02a937e7b7e8
# ╠═7624961c-8e2d-4441-ba4a-43a7ed192ad4
# ╠═cdd21e13-86ce-4856-add2-0203cb12591a
# ╠═c1d6bb79-bcff-4b3e-8011-314ef6f7a4e8
# ╟─9762a300-61d1-4a23-89de-18d3f7fedf7f
# ╠═eec68b57-e50f-447b-b852-1130e10d4d23
# ╠═2aee5336-148c-4a95-9007-72d976d6459a
# ╠═a5e729f2-f8f5-42d6-b540-16d1e62d2b91
# ╟─64c74a69-3945-4599-840a-be37f0ee7ab2
# ╠═99836668-cc71-4111-a6c2-914fac3cd44a
# ╠═7d0c384e-cb4c-46a5-8bd2-52181dcb7d2a
# ╠═de0b6e81-6681-4ec2-9352-4adb6ee14cdb
# ╟─98f5c2d7-8340-4788-a6e7-34c561c9d656
# ╠═135a2da9-39ab-4efe-a4a3-518b3519b5be
# ╠═dc21b305-3aa8-479e-9582-59298a99f231
# ╠═a3df3c80-32ee-45e5-8efa-743fc6670762
# ╠═e02ff44c-86a5-4748-8533-d177abfc9262
# ╟─55473de6-68f1-49b7-83d3-a1ab70f07185
# ╠═88b6b527-d0be-41cc-a800-3e1179a8fdd9
# ╠═9283a1f9-2f75-4835-9d71-c479cac7641e
# ╠═c9b7288d-2d59-4008-b0a8-2853fa0baad2
# ╠═2d97fba8-1b5b-4e98-9b53-0efbf3aac8d4
# ╠═9d286c16-6256-4ad5-8ebd-d2b102c6562c
# ╟─2a8a9dc0-f866-4cdb-81ee-4f4da1ee40e6
# ╠═b0da5b04-afc3-47b6-8e21-329456c5d7e8
# ╠═d343e7ac-edc6-469f-9f21-b6d7e42b3e3c
# ╠═87bf7bda-81d5-4325-9f4b-c8e8f6af243a
# ╟─c07c44c9-866d-4e95-8c24-6a5f96b2a57c
# ╠═39237f4f-6c60-43a5-ab51-abd13f438b9b
# ╠═6679eb9a-164d-4602-9d2f-0569b0018144
# ╠═40430b14-07b2-40c6-95fe-9f60a5c6e75f
# ╟─33b4e174-6cc8-4dc9-8d81-01408e1b31a4
# ╠═621e43d5-39da-4fd5-8f4b-6cd5fb18d0d5
# ╠═a9e85360-3add-42d9-82c6-fe66bf506811
# ╠═83a49887-dd56-4f84-b7ab-1bf07c87f491
# ╠═ea0311f8-5752-41e3-9dab-a1e35d5e733b
# ╠═de3c5a5d-65e0-423e-9eec-7bd4ae61a70d
# ╠═ff81b525-3d5e-433b-b9bb-110172b4364e
# ╟─4943cac7-d42e-47d9-adea-d2be4d25d391
# ╠═d34f13c9-4de1-4ffc-88c6-87a95163861a
# ╠═c9447ee0-2477-49ae-8ca5-45fb5578e26e
# ╠═dab370d8-5953-418b-95a2-3cb9db8b7ee4
# ╠═63f4d4e6-f92c-4a9d-beea-27ca15492647
# ╠═118b7027-3e5a-4036-8f5d-9e5c8818a642
# ╠═a3233be1-4002-4545-b306-9197a556c3f9
# ╠═4a3c62fb-ea78-41ac-92b5-9d55f5f05789
# ╟─958c44ef-9c64-484c-ae80-0a62f2142225
# ╠═257308b7-c3c5-43cc-98ba-8f8a8306fb61
# ╠═a081e9f0-1ee6-4c28-9409-c3ce1e190084
# ╠═18a6f1e2-5ee2-40b2-8861-9e634a74de3a
# ╠═b2aef310-148f-4386-a41e-ead3c32f6ca4
# ╟─edae2f7c-cd4c-42f6-a423-ad5f1f1bf5cd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
