### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ e2772773-4058-4a1b-9f99-d197942086d5
begin
	# This causes Pluto to no longer automatically manage packages (so you'll need to install each of these in the Julia REPL)
	# But, this is needed to use the two local packages MOMDPs and DirichletBeliefs
	using Revise
	using Pkg
	Pkg.develop(path="..//MOMDPs.jl") 
	Pkg.develop(path="..//DirichletBeliefs.jl") 
	using MOMDPs
	using DirichletBeliefs

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

# ╔═╡ 30028219-7ea5-4b78-b0df-b3b98b25ee65
using PlutoUI

# ╔═╡ 66791136-7ebd-4006-b35c-d69a44a0d81f
using StatsBase

# ╔═╡ 70dffecd-3db5-45a9-8104-d82f30fdead2
using BSON

# ╔═╡ 2188b608-8797-404a-9ea4-806e7422431b
using LaTeXStrings

# ╔═╡ 05870354-856b-4342-8dce-00219b602342
using BasicPOMCP

# ╔═╡ d4d4de96-b8aa-484d-b594-afb48dd472bc
using D3Trees

# ╔═╡ ab91d45d-7ded-4386-88b3-a28d2ebe44ff
using TikzGraphs, Graphs

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

# ╔═╡ f75f136d-ad50-4076-b4b8-477fde8d4a53
@enum VisaStatus ISIS VulAfghan P1P2Afghan SIV AMCIT NULL

# ╔═╡ 1c6307a9-261a-462b-a92b-5bb2ff9a1a6f
fam_distr = MixtureModel([TruncatedNormal(1, 0.6, 1, 13), TruncatedNormal(8, 2, 1, 13)])

# ╔═╡ 4cb13bfc-c5c9-4cd2-9f68-e273487945d2
fam_samples = round.(Int, rand(fam_distr, 10000));

# ╔═╡ b7f36e41-8f88-446a-9cf8-ed6b56bf6079
begin
	histogram(fam_samples, bins=1:13, xticks=(1:13), label="family size", c=:gray, normalize=:pdf, ylims=(0, 0.31), xlabel="family size at gate", ylabel="likelihood")
	title!("family size distribution at gate")
	fam_plot = plot!(size=(500,300))
end

# ╔═╡ e4151c6a-06eb-4e66-ae4f-8e1fca4ffbcf
fam_fit = fit(Histogram, fam_samples)

# ╔═╡ e3ebaec0-0712-4c3a-bb04-ba6fcfe6a785
normalize(fam_fit.weights, 1)

# ╔═╡ bd76ffc5-b98f-4b82-903c-f9799733de23
round.(Int, rand(fam_distr, 10))

# ╔═╡ a0a58020-32ed-42b4-ad9c-2d1c0357f3c3
md"""
## States
"""

# ╔═╡ 99790e08-5bb7-4611-8d4b-39db4d36ca83
struct MDPState
    c::Int # chairs remaining 
    t::Int # time remaining 
    f::Int # family size 
    v::VisaStatus # visa status 
end

# ╔═╡ 8a899d1e-504b-4a12-aaaf-f5cf7949669a
@with_kw struct EvacuationParameters
	# average family size in afghanistan is 9. Couldn't find distribution.
    family_sizes::Vector{Int} = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] 
	# should have probabilities adding to 1. Largest density around 8
    # family_prob = [.05, .05, .05, .05, .05, .05, .05, .075, .3, .075, .05, .05, .05]
	family_prob = [0.3005, 0.191, 0.0118, 0.0137, 0.033, 0.0564, 0.0895, 0.1011, 0.0909, 0.0625, 0.0328, 0.0139, 0.0029]
	
	num_ISIS = 20 # assuming not that many are showing up at the airport 
	num_Vulnerable_Afghan = 1000000 
	num_P1P2_Afghan = 604500
	num_SIV = 123000
	num_AMCIT = 14786 # This includes family members of AMCITs to be evacuated 

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
		AMCIT => reward_AMCIT,
	)
	
	visa_prob = normalize([num_ISIS/num_total_airport,
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
    v_stringtoint = Dict("ISIS-K"=>-500,
        # "two SIVs as good as one taliban...US gov much more risk avers"
		"VulAfghan"=>-3,
		# anyone who doesn't have a visa....0 or negative. 
		# even beat up over ...everyone directly served the us effort in
		# some way either through developer contracters or military
        "P1/P2 Afghan"=>1,
        "SIV"=>5,
        "AMCIT"=>20)

     v_inttostring = Dict(-500=>"ISIS-K",
        -3=>"Vul. Afghan",
        1=>"P1/P2 Afghan",
        5=>"SIV",
        20=>"AMCIT")
    
    capacity::Int = 120 # keeping these both as integers of 20 for now. 
    time::Int = 120
    size::Tuple{Int, Int} = (length(visa_status), length(family_sizes)) # grid size
    p_transition::Real = 0.8 # don't we always transition into this since time moves forward? I'm confused... [this is uncertainty we integrated in our previous model that described the liklihood of transitioning onto the airplane given they are let into the airport. We could simplify it if you want.]
    null_state::MDPState = MDPState(-1, -1, -1 ,NULL) 
    accept_prob = [.80, .20]
    reject_prob = [1.0]
end

# ╔═╡ 6e7cf801-99d8-4bec-a036-3a0b2475a2eb
params = EvacuationParameters();

# ╔═╡ 4d99b27d-2369-49ca-b1d2-90bb07f61e2d
bar(params.family_prob)

# ╔═╡ b0f104b6-8aa8-4f38-9ab1-f8cc8f519e6c
Dirichlet(round.(Int, 100 * params.visa_prob) .+ 1)

# ╔═╡ a9899239-df71-41a2-bc66-f08ee325ffe2
params.v_inttostring

# ╔═╡ 3e453a9a-7682-49fc-bdd8-8e86658eefb5
params.visa_prob

# ╔═╡ 82808920-d4b2-4848-b368-9d89f93b66e3
md"""
> TODO: Make the `v` portion an index and not the value (then look up the value in the `reward` function). That way we _could_ have the same value associated to an given visa status.
"""

# ╔═╡ e7b7253f-68c6-4026-9a86-37d016e189d2
begin
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
end

# ╔═╡ c4bb4cf3-c185-4e35-91fa-f1de57590002
number_states = (params.capacity+1+min_capacity) * (params.time+1) * size(params.family_sizes)[1] * size(params.visa_status)[1] + 1

# ╔═╡ ce86b093-7e0e-4c70-9810-9acdb547c5e4
𝒮[1]

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

# ╔═╡ 63aa8075-9d3c-44a8-9442-eef2d14372a8
md"""
## Transition function
"""

# ╔═╡ 5022c6f3-09f8-44bc-b41e-a86cbf8f787c
md"""
## Reward function
"""

# ╔═╡ 2dbe879c-ea53-40fb-a334-5cb8f254faf7
R(s::MDPState, a::Action) = R(s.c, s.t, s.f, s.v, a)

# ╔═╡ 351d84fe-76a4-4995-b07f-6c5279ce629f
function R(c::Int, t::Int, f::Int, v::VisaStatus, a::Action)
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

# ╔═╡ b5164cb4-7e74-4536-9125-a0732a860690
R(10, 50, 1, AMCIT, REJECT)

# ╔═╡ 9abfcc56-ddbd-4670-8e24-b2472cf35676
@bind current_time Slider(0:120, default=120, show_value=true)

# ╔═╡ d6ddd6e0-efde-4af3-8885-ddf4f32bf163
time_penalty(c,t) = -c/(t+1) # sqrt?

# ╔═╡ 359069f8-4131-4346-bb75-9d941350b23c
@bind current_cap Slider(0:120, default=120, show_value=true)

# ╔═╡ 91217d58-7d5b-4559-ba0d-6f07e204ade7
time_penalty(current_cap, current_time)

# ╔═╡ cd445706-0002-45d3-b405-20b2206cde64
timecolor = cgrad(:blues, 0:120)

# ╔═╡ 5e90197e-0eae-47f4-86bd-ba618b3b1c93
get(timecolor, current_time/params.time)

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
termination(s::MDPState)= s == params.null_state # change to 1 or the other

# ╔═╡ 47a33688-6458-44a6-a5e5-3a6a220e9c39
md"""
## MDP formulation
"""

# ╔═╡ dda2b2be-b488-48c9-8475-2e4876ae517f
abstract type EvacuationMDP <: MDP{MDPState, Action} end

# ╔═╡ 889f89e8-61d2-4273-a63c-bcb6e6b0852d
begin
	c_initial = params.capacity
	t_initial = params.time
	f_initial = rand(params.family_sizes, 1)[1]
	v_initial = rand(params.visa_status, 1)[1]
	initial_state = MDPState(c_initial, t_initial, f_initial, v_initial)
	typeofstate = typeof(initial_state)
	initialstate_array = [initial_state]
end;

# ╔═╡ 5acdce32-b587-4ea2-8317-a753429fcd7b
md"""
## Solving MDP
"""

# ╔═╡ 4bfee8de-cc8d-436d-98ad-5f7002ece4cf
solver = ValueIterationSolver(max_iterations=30, belres=1e-6, verbose=true);

# ╔═╡ c9d38947-173d-4baf-9b3f-81bacf4d16cc
@bind reload_mdp_policy CheckBox(true)

# ╔═╡ fbd975c5-fc0b-43bd-86e0-7ed9c4efc0a5
md"""
> Loading takes about 1.5 minutes.
"""

# ╔═╡ 1e180b5a-34e4-4509-8eba-76445fbbc9ee
# if reload_mdp_policy
# 	mdp_policy = BSON.load("mdp_policy.bson", @__MODULE__)[:mdp_policy]
# else
# 	mdp_policy = solve(solver, mdp) # ≈ 577 seconds (~10 minutes)
# 	BSON.@save "mdp_policy.bson" mdp_policy # ≈ 77 seconds, 400 MB
# end

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
function POMDPs.action(::AcceptAll, s::MDPState)    # action(policy, state)
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
function POMDPs.action(::AMCITS, s::MDPState)
    return s.v == AMCIT  ? ACCEPT : REJECT
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
function POMDPs.action(::SIV_AMCITS, s::MDPState)
    return (s.v == AMCIT || s.v == SIV) ? ACCEPT : REJECT
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
function POMDPs.action(policy::AfterThresholdAMCITS, s::MDPState)
    if s.t <= 20 # policy.threshold
        return s.v == AMCIT ? ACCEPT : REJECT
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
function POMDPs.action(policy::BeforeThresholdAMCITS, s::MDPState)
    if s.t >= 20 # policy.threshold
        return s.v == AMCIT ? ACCEPT : REJECT
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

# ╔═╡ d1646d70-426d-4545-ba31-25a7712fe852
md"""
# Claim models
(individual observation models)
> TODO: Revisit "lying", "honesty"
"""

# ╔═╡ 3c353487-26a2-4c8d-85ee-61a8fd3f9339
params.visa_status_lookup

# ╔═╡ cbdedb5c-647c-45f1-8287-e25abbf9b2d0
p_amcit = normalize([0, 0, 0, 0.0, 1.0], 1)

# ╔═╡ 194dbc5b-80a6-49bb-abb6-3eb21d304f42
dot(p_amcit, [-500, -3, 1, 5, 20])

# ╔═╡ 9babaea3-5ed9-4349-ba5b-95f9549213eb
md"""
$O(obs \mid s)$
"""

# ╔═╡ 178629dc-b0c2-4e43-a5f0-935b554e4ff9
p_siv = normalize([0, 0, 0, 0.99, 0.01], 1)

# ╔═╡ bf2a07d8-002b-42b3-8483-7ba2533815d8
p_p1p2 = normalize([0, 0, 0.95, 0.04, 0.01], 1)

# ╔═╡ 7ae07ad8-8aca-426b-8b32-4eea6a9f422d
p_afghan = normalize([0, 0.9, 0.15, 0.009, 0.005], 1)

# ╔═╡ aee6d6f4-b65e-4775-883b-193424692334
p_isis = normalize([0.01, 0.94, 0.04, 0.009, 0.001], 1)

# ╔═╡ c07c44c9-866d-4e95-8c24-6a5f96b2a57c
md"""
## Visualizations
"""

# ╔═╡ 33b4e174-6cc8-4dc9-8d81-01408e1b31a4
md"""
# Dirichlet belief
"""

# ╔═╡ 621e43d5-39da-4fd5-8f4b-6cd5fb18d0d5
# turn probabilities into counts 
𝒟₀ = Dirichlet(round.(normalize(params.visa_prob, 1) .* 100) .+ 1)

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

# ╔═╡ 96dbe1e9-5e66-434b-9455-1e5972665f11
Plots.plot(0:0.1:15, x->pdf(fam_distr, x), lab=false)

# ╔═╡ 3a079763-a17c-4111-b59c-58f8d4391368
Plots.plot(map(t->time_penalty(current_cap,t), 120:-1:0),
		   xlabel="time left", ylims=(-120,10), label=false,
		   c=get(timecolor, current_cap/params.time))

# ╔═╡ c53d06e3-a3f3-446b-bd33-32317fdbbe08
begin
	Plots.plot()
	for c in 0:20:params.capacity
		Plots.plot!(map(t->time_penalty(c,t), 120:-1:0),
			        xlabel="time left", ylims=(-120,10), label=false,
			        c=get(timecolor, c/params.time))
	end
	Plots.plot!()
end

# ╔═╡ 83a49887-dd56-4f84-b7ab-1bf07c87f491
cmap_bar = ColorScheme([afghan_red, colorant"lightgray", afghan_green])

# ╔═╡ ea0311f8-5752-41e3-9dab-a1e35d5e733b
visa_statuses = ["ISIS-K", "Vul. Afghan", "P1/P2 Afghan", "SIV", "AMCIT"] # NOTE: ordering

# ╔═╡ 5405e0bd-5d5d-4133-a318-5087f72ec872
function plot_claims(p; kwargs...)
	categories = visa_statuses
	transposed = reshape(categories, (1, length(categories)))
	p = reshape(p, (1, length(p)))
	bar(
	    transposed,
	    p,
	    labels = transposed,
	    bar_width = 1,
		legend=:outertopright,
		margin = 3Plots.mm,
		ylims = (0, 1.1),
		size = (600,150),
		c = [get(cmap_bar, i/length(categories)) for i in 1:length(categories)]';
		kwargs...
	)
end

# ╔═╡ c40a65f6-6e2b-407d-b283-ac3e63bfddfd
plot_claims(p_amcit; title=L"P(v_\mathrm{obs} \mid v=\texttt{AMCIT})")

# ╔═╡ 427aaf58-f86d-4d7b-83a2-43f72a45d4fc
plot_claims(p_siv; title=L"P(v_\mathrm{obs} \mid v=\texttt{SIV})")

# ╔═╡ c4668fee-626b-4540-8960-8314550d16a4
plot_claims(p_p1p2; title=L"P(v_\mathrm{obs} \mid v=\texttt{P1/P2})")

# ╔═╡ 0620a248-8c5e-4ceb-bab7-1936ff3a6039
plot_claims(p_afghan; title=L"P(v_\mathrm{obs} \mid v=\texttt{Vul. Afghan})")

# ╔═╡ 16a3fe52-e9e4-4913-9f40-196b9f7b5a80
plot_claims(p_isis; title=L"P(v_\mathrm{obs} \mid v=\texttt{ISIS})")

# ╔═╡ de3c5a5d-65e0-423e-9eec-7bd4ae61a70d
params.visa_prob

# ╔═╡ ff81b525-3d5e-433b-b9bb-110172b4364e
Plots.plot(𝒟₀, visa_statuses, cmap_bar, title="Dirichlet expert prior")

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
# Updating pseudocounts 
# Could seed out belief with the expert prior or use a uniform prior
begin
	b_p = Plots.plot(𝒟_belief, visa_statuses, cmap_bar, title="updated Dirichlet posterior")
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

# ╔═╡ c5ab9206-d49b-4058-9ed0-0e3b7351a92b
md"""
## States
"""

# ╔═╡ 272f8ecb-c644-4faf-9425-c85fed301daa
struct VisibleState
	c::Int
	t::Int
	f::Int
end

# ╔═╡ a4d22509-0bb4-4779-a9f1-5be28da4a748
# only inbounds if room for the family [assuming would not separate even though might]
# and if time is available to enter the airport 
validtime(s::Union{MDPState,VisibleState}) = 0 < s.t

# ╔═╡ fca35d50-4bc8-41cc-bf67-54a8f15490de
validcapacity(s::Union{MDPState,VisibleState}) = 0 < s.c # maybe try back to 0

# ╔═╡ 029a416e-dce4-4027-ad8b-aaa04b87c4ab
struct HiddenState
	v::VisaStatus
end

# ╔═╡ 161bbfba-1c6b-4051-b5f1-11f85b8a869a
const POMDPState = Tuple{VisibleState, HiddenState}

# ╔═╡ caae0147-5917-4ece-a425-2b2b64ceded4
const AbstractState = Union{MDPState, POMDPState}

# ╔═╡ 0967fc7f-5a91-4b0a-9591-a899e47978a3
hidden_visa_statuses = [ISIS, VulAfghan, P1P2Afghan, SIV, AMCIT];

# ╔═╡ 9af2b018-72e4-42b7-a5c0-8cef3dcec675
visible_null_state = VisibleState(-1,-1,-1)

# ╔═╡ af19cfed-bec6-4870-bdf5-2a4b890e4107
hidden_null_state = HiddenState(NULL)

# ╔═╡ e55207d2-6bab-4d36-9dfe-1ae740c56854
null_pomdp_state = POMDPState((visible_null_state, hidden_null_state))

# ╔═╡ 261c4b96-429c-4f19-b8ab-dcee589fef91
begin
	𝒮ᵥ = [VisibleState(c,t,f)
		for c in 0:params.capacity
			for t in 0:params.time
				for f in params.family_sizes]
	𝒮ₕ = [HiddenState(v) for v in hidden_visa_statuses]
	𝒮_pomdp::Vector{POMDPState} = [(v,h) for v in 𝒮ᵥ for h in 𝒮ₕ]
	push!(𝒮_pomdp, null_pomdp_state)
end

# ╔═╡ 36a0e6ef-2f1a-4361-a017-a3b2163bc8a4
length(𝒮_pomdp)

# ╔═╡ d4fe4f99-f420-47af-8896-3ee31c635690
sindex = (pomdp, v) -> findfirst(params.visa_status .== v)

# ╔═╡ d532a9eb-3eda-49c7-b139-7846485c4610
md"""
## Actions
"""

# ╔═╡ 90573690-39ac-45d9-a2f7-e8947eab0be3
md"""
## Transition (TODO: reuse)
"""

# ╔═╡ 6d8cb27d-cbec-486f-b1b2-a1142f506edc
sterm1 = ((VisibleState(-1, 1, 1), HiddenState(AMCIT)))

# ╔═╡ 944936f1-cf1f-45ef-b016-64cda8197ec1
# function POMDPs.transition(pomdp::EvacuationPOMDP, sh::HiddenState, a::Action)
# 	return MOMDPs.transitionhidden(pomdp, sh, a)
# end

# ╔═╡ 73af03c1-4c9d-419c-a9a1-e5137d6846ad
md"""
## Particle debugging
"""

# ╔═╡ 257308b7-c3c5-43cc-98ba-8f8a8306fb61
md"""
## Observations
- how to model "observations"?
- some sort of documentation?
- Actual SIV approval number, or SIV processing number.
- IDEAS: (based on convos...should ask Thomas Billingsley)
- US Citizen: actual passport, lost passport but say they have one, picture of documents 
- On a list/check identity against a list 
- "Take this kid" they say kid's mother is an AMCIT 
- SIV full verification 
- SIV applicant number 
- Signal (e.g. pic of pinapple) rely on to signal because of its relative obscurity (verification happens elsewhwew and need a quick way for a marine to not have to sift through a oacker but look at a token that gives them a good sense...takes a tenth of the time to verify)

"""

# ╔═╡ a081e9f0-1ee6-4c28-9409-c3ce1e190084
@enum VisaDocument begin
	ISIS_indicator
	VulAfghan_document
	P1P2Afghan_document
	SIV_document
	AMCIT_document
	NULL_document
end

# ╔═╡ 79939adc-66c2-4f3f-8a94-cd38ff01af74
documentation = [ISIS_indicator, VulAfghan_document, P1P2Afghan_document, SIV_document, AMCIT_document]

# ╔═╡ 5628373b-b217-47c2-bc79-3d436c3c57b8
struct Observation
	c::Int
	t::Int
	f::Int
	vdoc::VisaDocument
end

# ╔═╡ be4772d2-8413-4cf3-8c36-a6cb76bbc59d
begin
	𝒪 = [Observation(c,t,f,vdoc)
			for c in 0:params.capacity
				for t in 0:params.time
					for f in params.family_sizes
						for vdoc in documentation]
	length(𝒪)
end

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
md"""
- then try a bunch of free online solvers. Solve with a few online solvers and compare.
- Each have their own drawbacks. 
- Solving MDP w/ value iteration guaranteed to give optimal...With perfect information vs. realistic partially observable.
"""

# ╔═╡ 40ab3b5e-6dff-4881-9919-45aad74b0c71
NOISY = false

# ╔═╡ 6829eb50-5531-4989-a712-329ba8cc1b1d
OLD_NOISE = false

# ╔═╡ ffc7c1cb-9e08-42ce-9b7c-dfd497125b6d
NOISE = 0.05

# ╔═╡ cae0c4b1-556a-432c-81cb-0564646d2a32
p_amcit

# ╔═╡ 2dbbb052-07d8-4de4-8566-7662682b7a45
p_siv

# ╔═╡ c49a6c2f-1390-4962-bc81-2a94c701c7a7
p_p1p2

# ╔═╡ e446e5b4-33aa-4aa9-a1d7-244ac8d9f8ab
p_afghan

# ╔═╡ 51bcdac2-6985-4422-9f6d-336e805ca492
p_isis

# ╔═╡ 56c51f5d-f632-4afe-8e10-32ef28160f48
begin
	sh_test = HiddenState(AMCIT)
	shp_test = HiddenState(P1P2Afghan)
	o_test = Observation(1,1,1,AMCIT_document)
end;

# ╔═╡ f5b3646a-071b-4e7e-8085-d33772cc26ca
# observation(pomdp, rand(𝒮ₕ), ACCEPT, rand(𝒮ₕ))

# ╔═╡ 65d3f8d8-770e-46a4-933c-d29928b337ed
o_test

# ╔═╡ f66f2579-d2d4-4a5b-9805-4922dbb99e9b
md"""
## POMDP type
"""

# ╔═╡ 810354f9-c28b-4d82-a05b-12fe2636022c
@with_kw mutable struct EvacuationPOMDP <: MOMDP{VisibleState, HiddenState, Action, Observation}
	visa_count = params.visa_count # updated as a Dirichlet, used in 𝑇(s′ | s, a)
end

# ╔═╡ d1abf4d0-a794-4146-abbb-ceaeae769a33
MOMDPs.visiblestates(pomdp::EvacuationPOMDP) = 𝒮ᵥ

# ╔═╡ 0266dfb4-788a-4c2f-870d-84810f73c9fb
MOMDPs.hiddenstates(pomdp::EvacuationPOMDP) = 𝒮ₕ

# ╔═╡ 5fd0ddce-3a1b-4bd0-89e1-7d328dd7212f
POMDPs.states(pomdp::EvacuationPOMDP) = 𝒮_pomdp

# ╔═╡ 1cf30637-81ee-491b-96d1-9996ed0d3347
POMDPs.initialstate(pomdp::EvacuationPOMDP) = Deterministic(POMDPState((VisibleState(params.capacity, params.time, 3), HiddenState(AMCIT))))

# ╔═╡ a9674df8-913b-402c-bcd0-b31f4f1f3fcc
POMDPs.actions(pomdp::EvacuationPOMDP) = [ACCEPT, REJECT]

# ╔═╡ 05bd12b0-3809-4858-bf57-d583b25e6815
function MOMDPs.transitionvisible(pomdp::EvacuationPOMDP, sv::VisibleState, a::Action, o=missing)
	visiblestates = ordered_visible_states(pomdp)
	p = ones(length(visiblestates))
	sᵢ = visiblestateindex(pomdp, sv)
	p[sᵢ] = 1
	normalize!(p, 1)
	return SparseCat(visiblestates, p)
	# return Distribution T(v′ | s, a)
end

# ╔═╡ 86f4190a-621c-4443-a92d-b6c3b57c51b4
function MOMDPs.transitionhidden(pomdp::EvacuationPOMDP, sh::HiddenState, a::Action, o=missing)
	hiddenstates = ordered_hidden_states(pomdp)
	noise = 0.05
	p = noise*ones(length(hiddenstates))
	sᵢ = hiddenstateindex(pomdp, sh)
	p[sᵢ] = 1
	normalize!(p, 1)
	return SparseCat(hiddenstates, p)
	# return Distribution T(h′ | s, a, v′)
end

# ╔═╡ ad638eef-1302-4ffd-9450-7bbc24654110
POMDPs.observations(pomdp::EvacuationPOMDP) = 𝒪

# ╔═╡ f00bbcde-bc67-4b66-b9db-2b46a6ae81e6
function POMDPs.observation(pomdp::EvacuationPOMDP,
	                        sh::HiddenState, a::Action, shp::HiddenState)
	global documentation
	state = shp # NOTE
	sₕ_idx = hiddenstateindex(pomdp, state)
	if isnothing(sₕ_idx) # null state
		return Deterministic(NULL_document)
	else
		if NOISY
			if OLD_NOISE
				p = NOISE * ones(length(documentation))
			else
				if state.v == AMCIT
					p = copy(p_amcit)
				elseif state.v == SIV
					p = copy(p_siv)
				elseif state.v == P1P2Afghan
					p = copy(p_p1p2)
				elseif state.v == VulAfghan
					p = copy(p_afghan)
				elseif state.v == ISIS
					p = copy(p_isis)
				end
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

# ╔═╡ 85d69241-3bac-4839-91eb-107583100336
function MOMDPs.visible(pomdp::EvacuationPOMDP, o::Observation)
	return VisibleState(o.c, o.t, o.f)
end

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
    return total_accepted_people, total_accepted_families, total_reward, visa_dict_accepts, visa_dict_rejects, visa_dict_accepts_rate
end

# ╔═╡ 8cf33e98-9c75-4208-a99b-db9758f34d15
validtime(s::POMDPState) = validtime(visible(s))

# ╔═╡ 09919642-7393-4c1f-bf5b-f1edb1f6c87c
validcapacity(s::POMDPState) = validcapacity(visible(s))

# ╔═╡ d655a73f-d0b8-4735-8cd0-d03c7adeda29
#***** ENUMERATING OVER ALL STATES ******
function T(s::MDPState, a::Action)
    next_states = AbstractState[]
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

# ╔═╡ 6bf5f2f2-b5d1-48a6-b811-f416bcfc9899
mdp = QuickMDP(EvacuationMDP,
    states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ,
    initialstate = initialstate_array, 
    isterminal   = termination,
    render       = render,
    statetype    = typeofstate 
);

# ╔═╡ 93dfa811-4280-4f32-8161-b64df94c4520
validcapacity(POMDPState((VisibleState(0, 1, 1), HiddenState(AMCIT))))

# ╔═╡ b755ee40-f5dc-4c65-bcd2-f6c5847a5f05
function POMDPs.observation(pomdp::EvacuationPOMDP,
	                        s::POMDPState, a::Action, sp::POMDPState)
	global documentation
	state = sp # NOTE
	sv = visible(state)
	sh = hidden(state)
	sₕ_idx = hiddenstateindex(pomdp, state)
	if isnothing(sₕ_idx) # null state
		return Deterministic(
			Observation(sv.c, sv.t, sv.f, NULL_document))
	else
		if NOISY
			if OLD_NOISE
				p = NOISE * ones(length(documentation))
			else
				if sh.v == AMCIT
					p = copy(p_amcit)
				elseif sh.v == SIV
					p = copy(p_siv)
				elseif sh.v == P1P2Afghan
					p = copy(p_p1p2)
				elseif sh.v == VulAfghan
					p = copy(p_afghan)
				elseif sh.v == ISIS
					p = copy(p_isis)
				else
					error("No case for $(sh.v).")
				end
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

# ╔═╡ 6cc9003e-63e0-47dc-85c0-5b69758daf72
function POMDPModelTools.obs_weight(p::EvacuationPOMDP, sh, a, shp, o)
	return pdf(observation(p, sh, a, shp), o.vdoc)
end

# ╔═╡ 61e6c2b8-387d-4275-bd2f-fede4fcf1835
function likelihood(pomdp::EvacuationPOMDP, v::Int)
	p = normalize(pomdp.visa_count, 1)
	return p[v]
end	

# ╔═╡ 0d06eb82-acdc-4956-8a9b-3a1261f4c00b
#***** ENUMERATING OVER ALL STATES ******
function POMDPs.transition(pomdp::EvacuationPOMDP, s::POMDPState, a::Action)
    sv = visible(s)
	sh = hidden(s)
	next_states = AbstractState[]
    probabilities = Float64[] 
    
    # if !validtime(sv) || !validcapacity(sv)
    #     push!(next_states, null_pomdp_state)
    #     push!(probabilities, 1) # double check 
    # else
	if a == ACCEPT
		# check if valid capacity
		visa_status = sh.v
		next_state_accept = POMDPState(
			(VisibleState(sv.c - sv.f, sv.t - 1, 1), HiddenState(visa_status)))
		next_state_reject = POMDPState(
			(VisibleState(sv.c, sv.t - 1, 1), HiddenState(visa_status)))

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
    normalize!(probabilities, 1)
    return SparseCat(next_states, probabilities)
end

# ╔═╡ 27bd873f-c7b1-4323-99e3-6f6be02eb8b5
pomdp = EvacuationPOMDP()

# ╔═╡ f7ce4e19-b186-49c2-8780-b9db1b90e8d3
sterm = rand(transition(pomdp, ((VisibleState(-1, 1, 1), HiddenState(AMCIT))), REJECT))

# ╔═╡ 8fceeff8-cd27-49b9-acc3-e7ebde859c20
transition(pomdp, POMDPState((VisibleState(1,120,2), HiddenState(AMCIT))), ACCEPT)

# ╔═╡ d112f66b-bd60-480c-9fc3-275b90206e6c
transitionhidden(pomdp, HiddenState(AMCIT), ACCEPT)

# ╔═╡ d1f69a0b-cc97-4b71-bd45-1b2dfca023da
pomdp.visa_count

# ╔═╡ cbc41677-d977-4c5e-a8cc-6216dc31ef9b
pomdp.visa_count[2]

# ╔═╡ 40c84bf1-88c8-47d3-9a5b-6f9308d883e8
observation(pomdp, sh_test, ACCEPT, shp_test)

# ╔═╡ 198b9497-28d3-41fa-9581-4cec05273a96
map(sh->obs_weight(pomdp, sh, ACCEPT, HiddenState(NULL), Observation(-1,-1,-1,NULL_document)), ordered_hidden_states(pomdp))

# ╔═╡ 66c974f1-8f58-4c4a-96fd-fd1ac7aa08aa
obs_weight(pomdp, sh_test, ACCEPT, shp_test, o_test)

# ╔═╡ 2b4fdacd-067b-4878-91c3-3f1cde1e2d97
observation(pomdp, rand(𝒮ₕ), ACCEPT, rand(𝒮ₕ))

# ╔═╡ 4083a989-f196-4b9a-abf2-8f7b34f09168
transition(pomdp, rand(𝒮_pomdp), ACCEPT)

# ╔═╡ 4f00e4e5-aaad-43de-9f22-e7bf683ecc15
md"""
Extract the visible state portion of the observation, and convert it to a `VisibleState`.
"""

# ╔═╡ 0682719b-e3f4-4d22-9b72-395cf33d8a3d
md"""
## Reward
"""

# ╔═╡ 34b1b547-c5df-49df-b58f-d41ff17b04a9
function POMDPs.reward(pomdp::EvacuationPOMDP, s::POMDPState, a::Action)
	return R(visible(s).c, visible(s).t, visible(s).f, hidden(s).v, a)
end

# ╔═╡ 8085e3eb-cd8b-4d57-bfe4-df35c0eae8c1
md"""
## Termination & Discount
"""

# ╔═╡ c4a4bc9e-b7d3-4817-9ed7-807da5f8fd57
function POMDPs.isterminal(::EvacuationPOMDP, s::POMDPState)
	# return s == null_pomdp_state # || !validtime(s) || !validcapacity(s)
	return !validtime(s) || !validcapacity(s)
end

# ╔═╡ e964e830-1de7-4f0a-ae5c-f728453f4df5
isterminal(pomdp, sterm1)

# ╔═╡ 2647feec-3748-4eb9-909b-79049b06f0e8
isterminal(pomdp, sterm)

# ╔═╡ fbc6ab5c-a2a1-4215-a412-7fe10e43117e
POMDPs.discount(::EvacuationPOMDP) = γ

# ╔═╡ 5629a62b-0532-4736-aa8b-e814192ed9c0
md"""
## Generative model
"""

# ╔═╡ 8aeff62d-6c27-4f6b-9b0d-8d18df1c2902
gen_state = rand(𝒮_pomdp)

# ╔═╡ 0c3d6891-74fc-4d39-a0d4-080d929677f8
@gen(:sp, :o, :r)(pomdp, gen_state, REJECT)

# ╔═╡ 96176ef1-2919-41a0-a206-bfe228195ad8
md"""
## Population beliefs
"""

# ╔═╡ d47c2f84-ca16-4337-80ef-3abd94a77f6a
prior_belief = Categorical([0.01, 0.50, 0.14, 0.20, 0.15]);

# ╔═╡ 321989f4-7457-42a2-86ea-c5289d323bd4
normalize(prior_belief.p .* map(sh->obs_weight(pomdp, sh, ACCEPT, sh, o_test), ordered_hidden_states(pomdp)), 1)

# ╔═╡ 453da136-bd07-4e7c-a47a-0bad8765eb7e
up = DirichletSubspaceUpdater(pomdp)

# ╔═╡ da134037-11fb-4f76-8381-e128a37d43eb
_b′ = initialize_belief(up, prior_belief)

# ╔═╡ c01e8357-c094-4373-ba49-faa149dc7191
begin
	b′ = initialize_belief(up, prior_belief)
	sv = VisibleState(params.capacity, params.time, 1)
	# b′(sv)
	for num_updates in 1:10
		vdoc = documentation[rand(𝒟_true)]
		o = Observation(sv.c, sv.t, sv.f, vdoc)
		b′ = update(up, b′, ACCEPT, o)
	end
	[mean(b′) b′.b.alpha]
end

# ╔═╡ 230fd9b3-837c-491e-85e6-d27be29618e3
bar_labels = map(sₕ->replace(replace(string(sₕ), r"Main.workspace#\d+\."=>""), "HiddenState"=>""), hiddenstates(pomdp))

# ╔═╡ b1f02e06-7131-4de3-9b40-b9d7e87ce99e
Plots.plot(b′.b, bar_labels, cmap_bar)

# ╔═╡ edae2f7c-cd4c-42f6-a423-ad5f1f1bf5cd
md"""
> Note, may need to define `O(s, a, s′) = O(NULL, a, s′)`
"""

# ╔═╡ 6349762b-1c5e-4b9b-b2eb-90573f19313e
md"""
## POMDP Solver
"""

# ╔═╡ 7d540d22-1d52-4eef-a942-b236668217b6
# updater(pomdp::EvacuationPOMDP) = DirichletSubspaceUpdater(pomdp)

# ╔═╡ 7123fa88-ee0e-462a-85e5-c6a7a485ca84
updater(pomdp::EvacuationPOMDP) = DiscreteSubspaceUpdater(pomdp)

# ╔═╡ 2a6aa36b-f022-42f6-abab-cbf564990dcd
rollout_estimator = FORollout(SIV_AMCITS())

# ╔═╡ 1d3d333b-0e6f-46ff-83d0-fe9af88c8c8f
pomcp_solver = POMCPSolver(max_depth=10, tree_queries=100,  estimate_value=rollout_estimator) # TODO: Change max_depth \approx 10

# ╔═╡ 4108e9ec-7562-4a9e-8fb7-525ddf53d268
function POMDPs.action(::SIV_AMCITS, s::POMDPState)
	sh = hidden(s)
    return (sh.v == AMCIT || sh.v == SIV) ? ACCEPT : REJECT
end;

# ╔═╡ 39237f4f-6c60-43a5-ab51-abd13f438b9b
"""
Pass in policy and chairs and time remaing. Spit out graph of family size versus visa status.
"""
function vis_time_step(policy, c, t)
    (v_size, f_size) = params.size  #visa, family 5, 5
    policyGraph = zeros(v_size, f_size) 
    
	# TODO: Revisit due to `params.visa_status` changed to @enum
	visa_statuses = params.visa_status
	x = Int.(visa_statuses)
	family_sizes = params.family_sizes
	y = family_sizes
        
    for f in 1:f_size
        for v in 1:v_size
            act = action(policy, MDPState(c, t, family_sizes[f], visa_statuses[v])) 
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
    titleX = string("\$t_{$timeVal}, c_{$capVal}\$")

    return heatmap(x, y, z, 
         # aspect_ratio = 1,
         legend = :none, 
         xlims = (x[1], x[end]),
         ylims = (params.family_sizes[1], params.family_sizes[end]),
         xlabel = "visa status",
         ylabel = "family size",
         title = titleX,
         xtickfont = font(6, "Courier"), 
         ytickfont = font(6, "Courier"),
         thickness_scaling = .9,
         color=cmap.colors,   
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
    
    for t in reverse(0:total_time)
        if t % graph_per_n == 0 
            for c in reverse(0:total_capacity)
                if c % graph_per_n == 0
					if c == 0
						c = 1
					end
					if t == 0
						t = 1
					end
					push!(heat_maps, vis_time_step(policy, c, t))
                end
            end
        end
    end 

	Plots.plot(heat_maps..., layout=num_graphs, margin=2mm)
end

# ╔═╡ 40430b14-07b2-40c6-95fe-9f60a5c6e75f
vis_all(mdp_policy)

# ╔═╡ e1be8687-43db-46e9-99ad-eff59c3c2985
pomcp_policy = solve(pomcp_solver, pomdp)

# ╔═╡ b618680a-cec0-4efb-981e-65d074b1eb0a
begin
	# sv₀ = visible(rand(initialstate(pomdp)))
	sv₀ = VisibleState(params.capacity, params.time, 1)
	# b₀ = initialize_belief(up, prior_belief)
	# b₀(sv₀)
	b₀ = DirichletSubspaceBelief(pomdp, sv₀)
end;

# ╔═╡ 4efe4359-7041-45c9-bedf-939a41954831
observation(pomdp, POMDPState((sv₀,sh_test)), ACCEPT, POMDPState((sv₀,shp_test)))

# ╔═╡ 37d8c716-128b-4871-a722-f94b867b1cfc
Plots.plot(b₀.b, bar_labels, cmap_bar)

# ╔═╡ fdb3db7f-7784-477f-ace2-b65df9031b41
md"""
## Individual beliefs
"""

# ╔═╡ f7d1d21a-0660-424b-939c-406433b28285
params.visa_prob

# ╔═╡ 545f10eb-9b00-4546-b49f-eb85ac195212
pomdp.visa_count

# ╔═╡ 7b748a7d-7583-4918-9e47-27c0d8f0b64b
individual_updater(pomdp::EvacuationPOMDP) = DiscreteSubspaceUpdater(pomdp)

# ╔═╡ 33beafb5-1fd2-4e0e-892b-1b5b9d2e0a77
up_indiv = DiscreteSubspaceUpdater(pomdp)

# ╔═╡ 033ecd38-b298-489b-8990-5521d4abfb85
begin
	b₀_indiv = initialize_belief(up_indiv, prior_belief, params.visa_count)
	b₀_indiv(sv₀)
	# b₀ = DirichletSubspaceBelief(pomdp, sv₀)
end;

# ╔═╡ d92fd437-f33a-4334-9a50-24cdccdb2a30
function simulation(policy, pomdp::POMDP)
	updater = individual_updater(pomdp)
	initial_state = rand(initialstate(pomdp))
	initial_obs = rand(observation(pomdp, initial_state, REJECT, initial_state))
	initial_belief = update(updater, b₀_indiv, REJECT, initial_obs)

	hr = HistoryRecorder()
	history = simulate(hr, pomdp, policy, updater, initial_belief, initial_state)
	return history
end

# ╔═╡ 2d97fba8-1b5b-4e98-9b53-0efbf3aac8d4
"""Simulate n times and get the mean and std of n rollouts of a policy."""
function simulations(policy, str_policy, mdp, n_sims) # n is number of times to run 
    people = false
    
    list_total_accepted_people = []
    list_total_accepted_families = []
    list_total_reward = []
    list_visa_dict_accepts = []
    list_visa_dict_rejects = []
    list_visa_rate_accepts = []

    for i in 1:n_sims
        history = simulation(policy, mdp) # do 1 simulation and get the history 
        total_accepted_people, total_accepted_families, total_reward, visa_dict_accepts, visa_dict_rejects, visa_dict_accepts_rate = get_metrics(history)
        push!(list_total_accepted_people, total_accepted_people)
        push!(list_total_accepted_families, total_accepted_families)
        push!(list_total_reward, total_reward)
        push!(list_visa_dict_accepts, visa_dict_accepts)
        push!(list_visa_dict_rejects, visa_dict_rejects)
        push!(list_visa_rate_accepts, visa_dict_accepts_rate)
    end

    
    mean_std_reward = mean_std(list_total_reward, false)
    mean_std_total_accepted_people = mean_std(list_total_accepted_people, true)
    
    
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
    
    # divide by num sims 
    # for visa_status in keys(base_dict)
    #     base_dict[visa_status] = trunc(Int, base_dict[visa_status]/n_sims)
    # end

    # divide by num sims 
    # for visa_status in keys(base_dict_reject)
    #     base_dict_reject[visa_status] =
    #         trunc(Int, base_dict_reject[visa_status]/n_sims)
    # end

    base_dict_rate = Dict{VisaStatus, Vector{Float64}}()
    for dict in list_visa_rate_accepts
        base_dict_rate = merge(vcat, base_dict_rate, dict)
    end
    
    # print in latex form 
   # base_dict_sorted = sort(base_dict)
    
    # if (str_policy == "AMCITS-policy")
        print("Policy        &     Reward        &  Total Accepted (\\% accepted)  &")
        for visa_status in sort(collect(keys(base_dict)), rev=true)
            if (visa_status != 0) # for some reason got a zero in there...) 
                st_visa_status = string(visa_status)
                # just show what order the stats are showing up
                print("$st_visa_status &")
            end
        end
    # end
    println()

    print("$str_policy & \$$(mean_std_reward[1]) \\pm $(mean_std_reward[2])\$ & \$$(mean_std_total_accepted_people[1]) \\pm $(mean_std_total_accepted_people[2])\$ & ")
    for visa_status in sort(collect(keys(base_dict)), rev=true)
        if (visa_status != 0) # for some reason got a zero in there...) 
            st_visa_status = string(visa_status)
            expectation_acc = trunc(Int, base_dict[visa_status]/n_sims)
            expectation_rej = trunc(Int, base_dict_reject[visa_status]/n_sims)
            percent_acc_μ = 100*round(mean(filter(!isnan, base_dict_rate[visa_status])), digits=5)
            percent_acc_σ = 100*round(std(filter(!isnan, base_dict_rate[visa_status])), digits=5)
            print("\$$(expectation_acc) ($percent_acc_μ \\pm $percent_acc_σ\\%)\$ &   ")
        end
    end
    println()
end

# ╔═╡ 9d286c16-6256-4ad5-8ebd-d2b102c6562c
function experiments(n_sims=1000)
    # policies and n_sims can probably be put in our params function as a list. here for now. 
    dict_policies = Dict(
		"pomdp-policy"=>pomcp_policy,
		# "mdp-policy"=>mdp_policy, 
        # "AcceptAll-policy"=>AcceptAll_policy, 
        # "AMCITS-policy"=>AMCITS_policy, 
        # "SIV-AMCITS-policy"=>SIV_AMCITS_policy, 
        # "AfterThresholdAMCITS-policy"=>AfterThresholdAMCITS_policy, 
        # "BeforeThresholdAMCITS-policy"=>BeforeThresholdAMCITS_policy,
	)
    
    # for now don't report the std here. would need to keep track of every single value....
	
    for str_policy in sort(collect(keys(dict_policies)))
		xmdp = contains(str_policy, "pomdp") ? pomdp : mdp

		list_visa_dict_accepts =
			simulations(dict_policies[str_policy], str_policy, xmdp, n_sims)
    end
end

# ╔═╡ 7a301ad6-e31f-4d20-b527-a26573473c0e
begin
	Random.seed!(1)

	updater2 = individual_updater(pomdp)
	initial_state2 = rand(initialstate(pomdp))
	initial_obs2 = rand(observation(pomdp, initial_state2, REJECT, initial_state2))
	initial_belief2 = update(updater2, b₀_indiv, REJECT, initial_obs2)

	hist = simulate(HistoryRecorder(max_steps=10), pomdp, pomcp_policy,
		updater2,
		initial_belief2,
		initial_state2
	)
end

# ╔═╡ d608e18d-cfab-4d8c-ab4a-28c123b0092e
transition(pomdp, hist[5].s, hist[5].a)

# ╔═╡ 5cacf770-5ec6-4ce5-a181-b1d9aecdd627
b₀_indiv

# ╔═╡ 85a03af8-4734-48ee-8c6d-a8905e1f2feb
begin
	Random.seed!(0)
	online_action, info = action_info(pomcp_policy, b₀_indiv, tree_in_info=true)
	online_action
end

# ╔═╡ 7fdc98bb-0c88-434f-99ed-e7876df3c6e2
D3Tree(info[:tree], init_expand=1)

# ╔═╡ 7538f1b7-0c90-481a-ba73-430510db8d4c
begin
	b_indiv = DiscreteSubspaceBelief(pomdp, sv₀)
end

# ╔═╡ d9f7475d-afb2-4012-9483-07cca631db98
plot_claims(b_indiv.b)

# ╔═╡ 2ce923be-ed33-47fa-83ee-a798632cb305
normalize(b_indiv.b .* normalize(b₀.b.alpha, 1), 1) |> plot_claims

# ╔═╡ 3a7066a4-4273-4ac7-8bfe-63e7d54f0f5b
md"""
### Updating individual belief
"""

# ╔═╡ 40a6caa8-7f10-4816-ba65-bb4c49053775
begin
	b′_indiv = initialize_belief(up_indiv, prior_belief, params.visa_count)
	sv_indiv = VisibleState(params.capacity, params.time, 1)
	b′_indiv(sv_indiv)
	@info "Initial" plot_claims(b′_indiv.b)
	s_indiv = POMDPState((sv_indiv, HiddenState(NULL)))
	a_indiv = ACCEPT
	for num_updates in 1:20
		global s_indiv, b′_indiv
		spacing = "\n"^2num_updates
		# vdoc = documentation[rand(𝒟_true)]
		# o = Observation(sv.c, sv.t, sv.f, vdoc)
		sp = rand(transition(pomdp, s_indiv, a_indiv))
		if isterminal(pomdp, sp)
			@warn "Terminal!"
			break
		end
		o = rand(observation(pomdp, s_indiv, a_indiv, sp))
		b′_indiv = update(up_indiv, b′_indiv, a_indiv, o)
		@info spacing hidden(sp).v o.vdoc plot_claims(b′_indiv.b)
		s_indiv = sp
	end
	# [mean(b′_indiv) b′_indiv.b]
	plot_claims(b′_indiv.b)
end

# ╔═╡ f1914e6d-8dd2-4412-af20-93530ef0d030
md"""
# Visa probability distribuion
"""

# ╔═╡ 0d1c09ae-e27d-4d9c-84f4-13c4eca12b43
plot_claims(params.visa_prob; title="Visa probability distribution")

# ╔═╡ 56d742f1-4fb7-4afe-8674-f343d6672364
params.visa_prob

# ╔═╡ f429f2b4-959b-4ed2-bd49-6ba961ba2382
md"""
# Simulations
"""

# ╔═╡ ed5c5dfc-883b-44be-9fe1-4d054ee94312
params.visa_status_lookup

# ╔═╡ e655ae93-24ab-4305-a774-9306f585c6cd
macro seeprints(expr)
	quote
		stdout_bk = stdout
		rd, wr = redirect_stdout()
		$expr
		redirect_stdout(stdout_bk)
		close(wr)
		read(rd, String) |> Text
	end
end

# ╔═╡ d0407ec6-3566-409c-a53a-7b9e0501c4ad
@seeprints experiments(50)

# ╔═╡ 98a0e66f-5df2-4d43-a4ae-e685d8f32fce
argmax([0.04, 0.04, 0.83, 0.04, 0.04])

# ╔═╡ a1024190-ba85-4974-9666-d7df52660cff
initial_obs = [Observation(-1,-1,-1,NULL_document)]

# ╔═╡ 1c2788f6-1f60-4841-8b0e-4a9429d593ec
trajectory = Tuple[]

# ╔═╡ 6ba39de1-73dd-48ab-a075-da9937248187
@seeprints begin

	initial_belief = b₀_indiv # b₀
	belief_updater = individual_updater(pomdp) # updater
	s₀ = rand(initialstate(pomdp))
	initial_belief(visible(s₀))
	o₀ = rand(observation(pomdp, s₀, REJECT, s₀))
	initial_belief = update(belief_updater, initial_belief, REJECT, o₀)

	# global variable hack (due to @seeprints)
	initial_obs[1] = o₀
	empty!(trajectory)
	
	for (t, (s, a, o, b, sp, r)) in enumerate(stepthrough(pomdp, pomcp_policy,
		                                                  belief_updater,
		                                                  initial_belief,
														  s₀,
		                                                  "s,a,o,b,sp,r",
		                                                  max_steps=130)) # 121
        @show t
		println("Capacity=$(visible(s).c), time remaining=$(visible(s).t)")
        println(hidden(s).v," of size ", visible(s).f)
        # @show a
		if a == ACCEPT
			println("\t——————ACCEPT—————— ✅")
			@info ACCEPT, t
		else
			println("\t(reject) ❌")
		end
        @show o
		@show round.(b.b, digits=3)
		@show VisaStatus(argmax(b.b) - 1) # 0-based enums
		if hidden(s).v != VisaStatus(argmax(b.b) - 1)
			@show "~~~~~ M I S M A T C H ~~~~~"
			@warn "mismatch ($t)" # Only happens in the noisy case due to sampling
		end
        # @show b.visiblestate
		# @show round.(b.b.alpha, digits=2)
        println(hidden(sp).v," of size ", visible(sp).f)
        # @show hidden(sp)
        @show r
		println("—"^20)
		push!(trajectory, (s,a,o,b,r))
	end
end

# ╔═╡ 090746e7-56d6-4127-be40-5f448596316a
@seeprints begin
	empty!(trajectory)
	
	for (t, (s, a, sp, r)) in enumerate(stepthrough(mdp, mdp_policy,
		                                                  "s,a,sp,r",
		                                                  max_steps=130)) # 121
        @show t
		println("Capacity=$(s.c), time remaining=$(s.t)")
  #       println(hidden(s).v," of size ", visible(s).f)
  #       # @show a
		# if a == ACCEPT
		# 	println("\t——————ACCEPT—————— ✅")
		# 	@info ACCEPT, t
		# else
		# 	println("\t(reject) ❌")
		# end
  #       @show o
		# @show round.(b.b, digits=3)
		# @show VisaStatus(argmax(b.b) - 1) # 0-based enums
		# if hidden(s).v != VisaStatus(argmax(b.b) - 1)
		# 	@show "~~~~~ M I S M A T C H ~~~~~"
		# 	@warn "mismatch ($t)" # Only happens in the noisy case due to sampling
		# end
  #       # @show b.visiblestate
		# # @show round.(b.b.alpha, digits=2)
  #       println(hidden(sp).v," of size ", visible(sp).f)
  #       # @show hidden(sp)
  #       @show r
		# println("—"^20)
		push!(trajectory, (s,a,r))
	end
end

# ╔═╡ 9caeb007-5d5a-4eeb-a818-ff0e13e2af2f
trajectory

# ╔═╡ 2e4ba16f-6961-4f27-8782-deabe71c4d87
trajectory[1:5]

# ╔═╡ 5e000528-bdae-43a7-bd58-a7e2fb3d80be
md"""
# Trajectory plotting
"""

# ╔═╡ c81f187c-caa1-4a6d-8b3f-a816c063e044
visa_statuses

# ╔═╡ eab90b66-88bc-466f-a3a7-5f9d323d4c29
visa_status_labels = ["ISIS", "VulAfghan", "P1/P2", "SIV", "AMCIT", ""]

# ╔═╡ 2dee938e-a737-4c1c-a946-5847129b1fd0
begin
	color_accept = "green!70!black"
	color_reject = "red!70!black"
end

# ╔═╡ 69e64da0-121d-4432-87e2-d9d3441725dd
begin
	g = DiGraph(length(trajectory)) # 9, 20
	node_styles = Dict()
	node_tags = fill("", nv(g))
	for i in 1:nv(g)-1
		local s, a, o, b, r, sh, sv
		try
			(s,a,_,b,r) = trajectory[i]
			if i > 1
				(_,_,o,_,_) = trajectory[i-1]
			else
				o = initial_obs[1]
			end
			sv = visible(s)
			sh = hidden(s)
		catch err
			(s,a,r) = trajectory[i]			
			sv = s
			sh = s
		end
		add_edge!(g, i, i+1)
		color = a == ACCEPT ? color_accept : color_reject
		rcolor = r <= 0 ? color_reject : color_accept
		try
			node_styles[i] =
			"circle, draw=black, fill=$color, minimum size=$(sv.f)mm,
			 label={[align=center]below:\$t_{$(params.time-i+1)}\$\\\\
			        {\\scriptsize\\color{$rcolor}\$($(round(r, digits=2)))\$}},
			 label={[align=center]above:$(visa_status_labels[Int(sh.v)+1])\\\\
			        {\\color{gray}($(visa_status_labels[Int(o.vdoc)+1]))}}"
			node_tags[i] = Int(o.vdoc) != Int(sh.v) ? "{\\color{white}x}" : ""
		catch err
			node_styles[i] =
			"circle, draw=black, fill=$color, minimum size=$(sv.f)mm,
			 label={[align=center]below:\$t_{$(params.time-i+1)}\$\\\\
			        {\\scriptsize\\color{$rcolor}\$($(round(r, digits=2)))\$}},
			 label={[align=center]above:$(visa_status_labels[Int(sh.v)+1])}"
			node_tags[i] = ""
		end
	end
	node_tags[nv(g)] = raw"\ldots"
	node_styles[nv(g)] = ""
	tp = TikzGraphs.plot(g, node_tags, node_styles=node_styles,
		            options="grow'=right, level distance=22mm, semithick, >=stealth'")
end

# ╔═╡ 3667a247-83c3-40ea-a610-03e6975ea45d
TikzGraphs.save(TikzGraphs.PDF("traj_mdp"), tp)

# ╔═╡ 9a142f50-9c0b-4d5a-807d-07d4992b5155
md"""
# Aggregate statistics
"""

# ╔═╡ d3ffbfb2-38ff-4003-a93b-0c804d90d8fc
params.visa_count

# ╔═╡ f1e42b93-d895-4706-8044-9863c65908e7
pomdp.visa_count

# ╔═╡ af8f6589-a0b7-491b-a3e1-562560ac16a8
sum(pomdp.visa_count - params.visa_count) - 1

# ╔═╡ 9757ae45-a980-42a5-9373-44aad5a34d41
hist2 = simulation(pomcp_policy, pomdp)

# ╔═╡ 82220030-f58b-46ea-b2ed-f08996047de2
Int.(hist2[1].b.counts)

# ╔═╡ 720ff25f-ad84-43d3-8275-4a399ac82406
md"""
# BSON testing
"""

# ╔═╡ d426766a-2481-4997-8561-d8e7640f3d9d
# test = Dict{String, Action}("state"=>ACCEPT)

# ╔═╡ 6b8149b8-79fc-4176-99d6-6045b2330f02
loaded = BSON.load("test.bson", @__MODULE__)[:test]

# ╔═╡ e538a0ba-f20b-460c-b758-99807d1e3f43
testmdp = QuickMDP(EvacuationMDP,
    states       = [𝒮[1], 𝒮[2]],
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ,
    initialstate = initialstate_array, 
    isterminal   = termination,
    render       = render,
    statetype    = typeofstate 
);

# ╔═╡ 85549c92-3328-4cfa-985f-c2cf908e6ef8
test = ValueIterationPolicy(testmdp, [0.1 0.2; 0.9 0.1])

# ╔═╡ 6856d64d-f752-45c6-adde-ad6d15cc3182
BSON.@save "test.bson" test

# ╔═╡ Cell order:
# ╠═30028219-7ea5-4b78-b0df-b3b98b25ee65
# ╠═41d0fe72-b83d-4cf8-bc91-44b0e8f561eb
# ╟─c0292f00-7f92-11ec-2f15-b7975a5e40a1
# ╠═e2772773-4058-4a1b-9f99-d197942086d5
# ╠═fe9dfc73-f502-4d13-a019-8160a6af4617
# ╠═caae0147-5917-4ece-a425-2b2b64ceded4
# ╟─ffbb36c6-8e69-46d1-b132-845564de0ae2
# ╠═f75f136d-ad50-4076-b4b8-477fde8d4a53
# ╠═4d99b27d-2369-49ca-b1d2-90bb07f61e2d
# ╠═1c6307a9-261a-462b-a92b-5bb2ff9a1a6f
# ╠═96dbe1e9-5e66-434b-9455-1e5972665f11
# ╠═4cb13bfc-c5c9-4cd2-9f68-e273487945d2
# ╠═b7f36e41-8f88-446a-9cf8-ed6b56bf6079
# ╠═66791136-7ebd-4006-b35c-d69a44a0d81f
# ╠═e4151c6a-06eb-4e66-ae4f-8e1fca4ffbcf
# ╠═e3ebaec0-0712-4c3a-bb04-ba6fcfe6a785
# ╠═bd76ffc5-b98f-4b82-903c-f9799733de23
# ╠═8a899d1e-504b-4a12-aaaf-f5cf7949669a
# ╠═6e7cf801-99d8-4bec-a036-3a0b2475a2eb
# ╠═b0f104b6-8aa8-4f38-9ab1-f8cc8f519e6c
# ╠═a9899239-df71-41a2-bc66-f08ee325ffe2
# ╠═3e453a9a-7682-49fc-bdd8-8e86658eefb5
# ╟─a0a58020-32ed-42b4-ad9c-2d1c0357f3c3
# ╠═99790e08-5bb7-4611-8d4b-39db4d36ca83
# ╟─82808920-d4b2-4848-b368-9d89f93b66e3
# ╠═e7b7253f-68c6-4026-9a86-37d016e189d2
# ╠═c4bb4cf3-c185-4e35-91fa-f1de57590002
# ╠═ce86b093-7e0e-4c70-9810-9acdb547c5e4
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
# ╠═351d84fe-76a4-4995-b07f-6c5279ce629f
# ╠═b5164cb4-7e74-4536-9125-a0732a860690
# ╠═9abfcc56-ddbd-4670-8e24-b2472cf35676
# ╠═91217d58-7d5b-4559-ba0d-6f07e204ade7
# ╠═d6ddd6e0-efde-4af3-8885-ddf4f32bf163
# ╠═359069f8-4131-4346-bb75-9d941350b23c
# ╠═3a079763-a17c-4111-b59c-58f8d4391368
# ╠═cd445706-0002-45d3-b405-20b2206cde64
# ╠═5e90197e-0eae-47f4-86bd-ba618b3b1c93
# ╠═c53d06e3-a3f3-446b-bd33-32317fdbbe08
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
# ╠═70dffecd-3db5-45a9-8104-d82f30fdead2
# ╠═c9d38947-173d-4baf-9b3f-81bacf4d16cc
# ╟─fbd975c5-fc0b-43bd-86e0-7ed9c4efc0a5
# ╠═1e180b5a-34e4-4509-8eba-76445fbbc9ee
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
# ╠═d92fd437-f33a-4334-9a50-24cdccdb2a30
# ╠═c9b7288d-2d59-4008-b0a8-2853fa0baad2
# ╠═2d97fba8-1b5b-4e98-9b53-0efbf3aac8d4
# ╠═9d286c16-6256-4ad5-8ebd-d2b102c6562c
# ╠═d0407ec6-3566-409c-a53a-7b9e0501c4ad
# ╟─2a8a9dc0-f866-4cdb-81ee-4f4da1ee40e6
# ╠═b0da5b04-afc3-47b6-8e21-329456c5d7e8
# ╠═d343e7ac-edc6-469f-9f21-b6d7e42b3e3c
# ╠═87bf7bda-81d5-4325-9f4b-c8e8f6af243a
# ╟─d1646d70-426d-4545-ba31-25a7712fe852
# ╠═2188b608-8797-404a-9ea4-806e7422431b
# ╠═3c353487-26a2-4c8d-85ee-61a8fd3f9339
# ╠═194dbc5b-80a6-49bb-abb6-3eb21d304f42
# ╠═cbdedb5c-647c-45f1-8287-e25abbf9b2d0
# ╟─9babaea3-5ed9-4349-ba5b-95f9549213eb
# ╠═c40a65f6-6e2b-407d-b283-ac3e63bfddfd
# ╠═178629dc-b0c2-4e43-a5f0-935b554e4ff9
# ╠═427aaf58-f86d-4d7b-83a2-43f72a45d4fc
# ╠═bf2a07d8-002b-42b3-8483-7ba2533815d8
# ╠═c4668fee-626b-4540-8960-8314550d16a4
# ╠═7ae07ad8-8aca-426b-8b32-4eea6a9f422d
# ╠═0620a248-8c5e-4ceb-bab7-1936ff3a6039
# ╠═aee6d6f4-b65e-4775-883b-193424692334
# ╠═16a3fe52-e9e4-4913-9f40-196b9f7b5a80
# ╠═5405e0bd-5d5d-4133-a318-5087f72ec872
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
# ╟─c5ab9206-d49b-4058-9ed0-0e3b7351a92b
# ╠═161bbfba-1c6b-4051-b5f1-11f85b8a869a
# ╠═272f8ecb-c644-4faf-9425-c85fed301daa
# ╠═029a416e-dce4-4027-ad8b-aaa04b87c4ab
# ╠═0967fc7f-5a91-4b0a-9591-a899e47978a3
# ╠═9af2b018-72e4-42b7-a5c0-8cef3dcec675
# ╠═af19cfed-bec6-4870-bdf5-2a4b890e4107
# ╠═e55207d2-6bab-4d36-9dfe-1ae740c56854
# ╠═261c4b96-429c-4f19-b8ab-dcee589fef91
# ╠═36a0e6ef-2f1a-4361-a017-a3b2163bc8a4
# ╠═d4fe4f99-f420-47af-8896-3ee31c635690
# ╠═d1abf4d0-a794-4146-abbb-ceaeae769a33
# ╠═0266dfb4-788a-4c2f-870d-84810f73c9fb
# ╠═5fd0ddce-3a1b-4bd0-89e1-7d328dd7212f
# ╠═1cf30637-81ee-491b-96d1-9996ed0d3347
# ╟─d532a9eb-3eda-49c7-b139-7846485c4610
# ╠═a9674df8-913b-402c-bcd0-b31f4f1f3fcc
# ╠═8cf33e98-9c75-4208-a99b-db9758f34d15
# ╠═09919642-7393-4c1f-bf5b-f1edb1f6c87c
# ╠═93dfa811-4280-4f32-8161-b64df94c4520
# ╟─90573690-39ac-45d9-a2f7-e8947eab0be3
# ╠═0d06eb82-acdc-4956-8a9b-3a1261f4c00b
# ╠═6d8cb27d-cbec-486f-b1b2-a1142f506edc
# ╠═e964e830-1de7-4f0a-ae5c-f728453f4df5
# ╠═f7ce4e19-b186-49c2-8780-b9db1b90e8d3
# ╠═2647feec-3748-4eb9-909b-79049b06f0e8
# ╠═8fceeff8-cd27-49b9-acc3-e7ebde859c20
# ╠═944936f1-cf1f-45ef-b016-64cda8197ec1
# ╠═05bd12b0-3809-4858-bf57-d583b25e6815
# ╠═86f4190a-621c-4443-a92d-b6c3b57c51b4
# ╠═d112f66b-bd60-480c-9fc3-275b90206e6c
# ╟─73af03c1-4c9d-419c-a9a1-e5137d6846ad
# ╠═d608e18d-cfab-4d8c-ab4a-28c123b0092e
# ╠═d1f69a0b-cc97-4b71-bd45-1b2dfca023da
# ╠═cbc41677-d977-4c5e-a8cc-6216dc31ef9b
# ╠═7a301ad6-e31f-4d20-b527-a26573473c0e
# ╠═5cacf770-5ec6-4ce5-a181-b1d9aecdd627
# ╟─257308b7-c3c5-43cc-98ba-8f8a8306fb61
# ╠═a081e9f0-1ee6-4c28-9409-c3ce1e190084
# ╠═79939adc-66c2-4f3f-8a94-cd38ff01af74
# ╠═5628373b-b217-47c2-bc79-3d436c3c57b8
# ╠═be4772d2-8413-4cf3-8c36-a6cb76bbc59d
# ╠═ad638eef-1302-4ffd-9450-7bbc24654110
# ╟─18a6f1e2-5ee2-40b2-8861-9e634a74de3a
# ╟─b2aef310-148f-4386-a41e-ead3c32f6ca4
# ╠═40ab3b5e-6dff-4881-9919-45aad74b0c71
# ╠═6829eb50-5531-4989-a712-329ba8cc1b1d
# ╠═ffc7c1cb-9e08-42ce-9b7c-dfd497125b6d
# ╠═b755ee40-f5dc-4c65-bcd2-f6c5847a5f05
# ╠═cae0c4b1-556a-432c-81cb-0564646d2a32
# ╠═2dbbb052-07d8-4de4-8566-7662682b7a45
# ╠═c49a6c2f-1390-4962-bc81-2a94c701c7a7
# ╠═e446e5b4-33aa-4aa9-a1d7-244ac8d9f8ab
# ╠═51bcdac2-6985-4422-9f6d-336e805ca492
# ╠═f00bbcde-bc67-4b66-b9db-2b46a6ae81e6
# ╠═6cc9003e-63e0-47dc-85c0-5b69758daf72
# ╠═56c51f5d-f632-4afe-8e10-32ef28160f48
# ╠═40c84bf1-88c8-47d3-9a5b-6f9308d883e8
# ╠═4efe4359-7041-45c9-bedf-939a41954831
# ╠═198b9497-28d3-41fa-9581-4cec05273a96
# ╠═66c974f1-8f58-4c4a-96fd-fd1ac7aa08aa
# ╠═321989f4-7457-42a2-86ea-c5289d323bd4
# ╠═f5b3646a-071b-4e7e-8085-d33772cc26ca
# ╠═85d69241-3bac-4839-91eb-107583100336
# ╠═65d3f8d8-770e-46a4-933c-d29928b337ed
# ╟─f66f2579-d2d4-4a5b-9805-4922dbb99e9b
# ╠═810354f9-c28b-4d82-a05b-12fe2636022c
# ╠═61e6c2b8-387d-4275-bd2f-fede4fcf1835
# ╠═27bd873f-c7b1-4323-99e3-6f6be02eb8b5
# ╠═2b4fdacd-067b-4878-91c3-3f1cde1e2d97
# ╠═4083a989-f196-4b9a-abf2-8f7b34f09168
# ╟─4f00e4e5-aaad-43de-9f22-e7bf683ecc15
# ╟─0682719b-e3f4-4d22-9b72-395cf33d8a3d
# ╠═34b1b547-c5df-49df-b58f-d41ff17b04a9
# ╟─8085e3eb-cd8b-4d57-bfe4-df35c0eae8c1
# ╠═c4a4bc9e-b7d3-4817-9ed7-807da5f8fd57
# ╠═fbc6ab5c-a2a1-4215-a412-7fe10e43117e
# ╟─5629a62b-0532-4736-aa8b-e814192ed9c0
# ╠═8aeff62d-6c27-4f6b-9b0d-8d18df1c2902
# ╠═0c3d6891-74fc-4d39-a0d4-080d929677f8
# ╟─96176ef1-2919-41a0-a206-bfe228195ad8
# ╠═d47c2f84-ca16-4337-80ef-3abd94a77f6a
# ╠═453da136-bd07-4e7c-a47a-0bad8765eb7e
# ╠═da134037-11fb-4f76-8381-e128a37d43eb
# ╠═c01e8357-c094-4373-ba49-faa149dc7191
# ╠═b1f02e06-7131-4de3-9b40-b9d7e87ce99e
# ╠═230fd9b3-837c-491e-85e6-d27be29618e3
# ╟─edae2f7c-cd4c-42f6-a423-ad5f1f1bf5cd
# ╟─6349762b-1c5e-4b9b-b2eb-90573f19313e
# ╠═05870354-856b-4342-8dce-00219b602342
# ╠═7d540d22-1d52-4eef-a942-b236668217b6
# ╠═7123fa88-ee0e-462a-85e5-c6a7a485ca84
# ╠═1d3d333b-0e6f-46ff-83d0-fe9af88c8c8f
# ╠═2a6aa36b-f022-42f6-abab-cbf564990dcd
# ╠═4108e9ec-7562-4a9e-8fb7-525ddf53d268
# ╠═e1be8687-43db-46e9-99ad-eff59c3c2985
# ╠═b618680a-cec0-4efb-981e-65d074b1eb0a
# ╠═033ecd38-b298-489b-8990-5521d4abfb85
# ╠═85a03af8-4734-48ee-8c6d-a8905e1f2feb
# ╠═d4d4de96-b8aa-484d-b594-afb48dd472bc
# ╠═7fdc98bb-0c88-434f-99ed-e7876df3c6e2
# ╠═37d8c716-128b-4871-a722-f94b867b1cfc
# ╟─fdb3db7f-7784-477f-ace2-b65df9031b41
# ╠═f7d1d21a-0660-424b-939c-406433b28285
# ╠═545f10eb-9b00-4546-b49f-eb85ac195212
# ╠═7b748a7d-7583-4918-9e47-27c0d8f0b64b
# ╠═33beafb5-1fd2-4e0e-892b-1b5b9d2e0a77
# ╠═7538f1b7-0c90-481a-ba73-430510db8d4c
# ╠═d9f7475d-afb2-4012-9483-07cca631db98
# ╠═2ce923be-ed33-47fa-83ee-a798632cb305
# ╟─3a7066a4-4273-4ac7-8bfe-63e7d54f0f5b
# ╠═40a6caa8-7f10-4816-ba65-bb4c49053775
# ╟─f1914e6d-8dd2-4412-af20-93530ef0d030
# ╠═0d1c09ae-e27d-4d9c-84f4-13c4eca12b43
# ╠═56d742f1-4fb7-4afe-8674-f343d6672364
# ╟─f429f2b4-959b-4ed2-bd49-6ba961ba2382
# ╠═ed5c5dfc-883b-44be-9fe1-4d054ee94312
# ╟─e655ae93-24ab-4305-a774-9306f585c6cd
# ╠═98a0e66f-5df2-4d43-a4ae-e685d8f32fce
# ╠═a1024190-ba85-4974-9666-d7df52660cff
# ╠═1c2788f6-1f60-4841-8b0e-4a9429d593ec
# ╠═6ba39de1-73dd-48ab-a075-da9937248187
# ╠═090746e7-56d6-4127-be40-5f448596316a
# ╠═9caeb007-5d5a-4eeb-a818-ff0e13e2af2f
# ╠═2e4ba16f-6961-4f27-8782-deabe71c4d87
# ╟─5e000528-bdae-43a7-bd58-a7e2fb3d80be
# ╠═c81f187c-caa1-4a6d-8b3f-a816c063e044
# ╠═eab90b66-88bc-466f-a3a7-5f9d323d4c29
# ╠═ab91d45d-7ded-4386-88b3-a28d2ebe44ff
# ╠═2dee938e-a737-4c1c-a946-5847129b1fd0
# ╠═69e64da0-121d-4432-87e2-d9d3441725dd
# ╠═3667a247-83c3-40ea-a610-03e6975ea45d
# ╟─9a142f50-9c0b-4d5a-807d-07d4992b5155
# ╠═d3ffbfb2-38ff-4003-a93b-0c804d90d8fc
# ╠═f1e42b93-d895-4706-8044-9863c65908e7
# ╠═af8f6589-a0b7-491b-a3e1-562560ac16a8
# ╠═82220030-f58b-46ea-b2ed-f08996047de2
# ╠═9757ae45-a980-42a5-9373-44aad5a34d41
# ╟─720ff25f-ad84-43d3-8275-4a399ac82406
# ╠═d426766a-2481-4997-8561-d8e7640f3d9d
# ╠═6b8149b8-79fc-4176-99d6-6045b2330f02
# ╠═6856d64d-f752-45c6-adde-ad6d15cc3182
# ╠═e538a0ba-f20b-460c-b758-99807d1e3f43
# ╠═85549c92-3328-4cfa-985f-c2cf908e6ef8
