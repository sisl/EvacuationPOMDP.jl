### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ d79dcca8-071a-4682-9178-53044a39cc5b
import Pkg

# ╔═╡ af8cff8a-07d2-4d6b-9dc6-d7e3de839b17
Pkg.add("Plots")

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

# ╔═╡ ab3eb808-9faf-4e5f-abc9-158b9f42234f


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
	# Question about how this distribution should lool...
    family_sizes::Vector{Int} = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] 
	# should have probabilities adding to 1. Largest density around 8
    family_prob = [.05, .05, .05, .05, .05, .05, .05, .075, .3, .075, .05, .05, .05]
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

# ╔═╡ Cell order:
# ╠═30028219-7ea5-4b78-b0df-b3b98b25ee65
# ╠═41d0fe72-b83d-4cf8-bc91-44b0e8f561eb
# ╠═d79dcca8-071a-4682-9178-53044a39cc5b
# ╠═af8cff8a-07d2-4d6b-9dc6-d7e3de839b17
# ╠═ab3eb808-9faf-4e5f-abc9-158b9f42234f
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
