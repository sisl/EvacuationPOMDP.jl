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
	Pkg.develop(path="..")
	Pkg.develop(path="..//MOMDPs.jl") 
	Pkg.develop(path="..//DirichletBeliefs.jl") 
	using EvacuationPOMDP
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

# ╔═╡ 70dffecd-3db5-45a9-8104-d82f30fdead2
using BSON

# ╔═╡ 05870354-856b-4342-8dce-00219b602342
using BasicPOMCP

# ╔═╡ d4d4de96-b8aa-484d-b594-afb48dd472bc
using D3Trees

# ╔═╡ 41d0fe72-b83d-4cf8-bc91-44b0e8f561eb
TableOfContents()

# ╔═╡ c0292f00-7f92-11ec-2f15-b7975a5e40a1
md"""
# Evacuation POMDP
"""

# ╔═╡ fe9dfc73-f502-4d13-a019-8160a6af4617
Random.seed!(0xC0FFEE)

# ╔═╡ 3771f6a0-3926-4234-b4b2-69ec8f96fa41
md"""
## Family size distribution
"""

# ╔═╡ 47a33688-6458-44a6-a5e5-3a6a220e9c39
md"""
## MDP formulation
"""

# ╔═╡ 6bf5f2f2-b5d1-48a6-b811-f416bcfc9899
mdp = EvacuationMDP()

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
> - **Running**: takes about 191 seconds
> - **Loading**: takes about 1.8 seconds, 73 MB
"""

# ╔═╡ 1e180b5a-34e4-4509-8eba-76445fbbc9ee
if reload_mdp_policy
	mdp_policy = BSON.load("mdp_policy.bson", @__MODULE__)[:mdp_policy]
else
	mdp_policy = solve(solver, mdp)
	BSON.@save "mdp_policy.bson" mdp_policy
end

# ╔═╡ 4f9b0d74-73d5-42e9-9ee6-178797ac023c
println()

# ╔═╡ 2d58015e-1e3d-4000-ae68-5d5222fa8101
level1data = experiments(1000, mdp, mdp_policy)

# ╔═╡ 8d98f5fc-9f8f-4cb2-a791-3e1a987ed56e
function annotate_policy!(k, x, y)
	pos = :left
	if k == "POMDP-lite"
		x, y = (x-24, y)
	elseif k == "Random"
		x, y = (x-2, y+25)
	else
		x, y = x < 120 ? (x-1, y+20) : (x, y)
	end	
	annotate!([(x+2, y, (k, 7, pos))])
end

# ╔═╡ 118ab216-0548-4ec4-b933-cedf78f08c55
BSON.@save "level1data.bson" level1data

# ╔═╡ a17792e3-d4fb-498c-bf8b-d25f9e9efc5c
mean_std(map(sum, level1data["Level I"]["list_reward_over_time"]), false)

# ╔═╡ dd15d7f4-a163-44d4-8f8f-7915a78ed4f6
function mismatch_mean(A)
    max_length = maximum(map(length, A))
    Z = [map(a->i <= length(a) ? a[i] : nothing, A) for i in 1:max_length]
    return map(mean, map(z->filter(!isnothing, z), Z))
end

# ╔═╡ 8d2b9590-8d76-4675-9b8a-e6a47cbccb8c
function mismatch_std(A)
    max_length = maximum(map(length, A))
    Z = [map(a->i <= length(a) ? a[i] : nothing, A) for i in 1:max_length]
    stds = map(std, map(z->filter(!isnothing, z), Z))
    return map(σ->isnan(σ) ? 0 : σ, stds)
end

# ╔═╡ 2b2ed9bd-e033-47ea-b24b-4789f28ab08c
md"""
# Deterministic population trajectories
"""

# ╔═╡ 0fe49a8e-4ee5-41b8-8802-8e61d997b252
begin
	Random.seed!(0)
	transition(mdp, rand(initialstate(mdp)), ACCEPT; input_family_size=10, input_status=AMCIT, made_it_through=false)
end

# ╔═╡ d1646d70-426d-4545-ba31-25a7712fe852
md"""
# Claim models
"""

# ╔═╡ 9babaea3-5ed9-4349-ba5b-95f9549213eb
begin
	# pgfplotsx()
	gr()
	claim_plot = plot_all_claims(EvacuationPOMDPType())
	# gr()
	claim_plot
end

# ╔═╡ 29798743-18cc-4c7b-af0d-ffb962a3eec9
savefig(claim_plot, "claim_plot.pdf")

# ╔═╡ 67d3b29c-4aeb-45df-a449-cd669590fb58
EvacuationPOMDPType().claims

# ╔═╡ 59c54092-e44e-422c-a29d-3fc457718190
sum([0.95, 0.045, 0.004, 0.001])

# ╔═╡ 470e32ce-57da-48b6-a27e-0899083a47a3
md"""
# MDP policy plot
"""

# ╔═╡ 40430b14-07b2-40c6-95fe-9f60a5c6e75f
begin
	# pgfplotsx()
	gr()
	policy_plot = vis_all(mdp.params, mdp_policy)
	# gr()
	policy_plot
end

# ╔═╡ 34b70a62-622e-41e3-a865-7cbd8b6a8fb5
savefig(policy_plot, "policy_plot.pdf")

# ╔═╡ 57c7d1f6-9920-464d-8b14-563fccd6878f
# vis_all(mdp.params, pomcp_policy) # requires action(policy, belief) NOTE `belief`.

# ╔═╡ 958c44ef-9c64-484c-ae80-0a62f2142225
md"""
# POMDP formulation
"""

# ╔═╡ 4a747878-429e-455f-8894-a92c22be4dcf
md"""
# Debugging
"""

# ╔═╡ 56c51f5d-f632-4afe-8e10-32ef28160f48
begin
	sh_test = HiddenState(AMCIT)
	shp_test = HiddenState(P1P2Afghan)
	o_test = Observation(1,1,1,AMCIT_document)
end;

# ╔═╡ f66f2579-d2d4-4a5b-9805-4922dbb99e9b
md"""
## POMDP type
"""

# ╔═╡ 27bd873f-c7b1-4323-99e3-6f6be02eb8b5
pomdp_level3 = EvacuationPOMDPType()

# ╔═╡ 3b468bb0-9e90-40b0-9a24-60fba6ca4fc5
fam_distr = plot_family_size_distribution(pomdp_level3.params.family_prob)

# ╔═╡ f9b11f1e-04b9-488c-9884-ab0bb77a945c
savefig(fam_distr, "fam_distr.pdf")

# ╔═╡ 67139013-44dc-4f83-afcc-eb3aaacb0eab
pomdp_level3.params.visa_prob

# ╔═╡ d112f66b-bd60-480c-9fc3-275b90206e6c
transitionhidden(pomdp_level3, HiddenState(AMCIT), ACCEPT)

# ╔═╡ 40c84bf1-88c8-47d3-9a5b-6f9308d883e8
observation(pomdp_level3, sh_test, ACCEPT, shp_test)

# ╔═╡ f094c09e-2f51-4b1b-81cb-a6832baccb92
normalize(pomdp_level3.visa_count, 1) |> plot_claims

# ╔═╡ 36c1fe73-2559-4f9e-95f3-3e79ed0c0264
pomdp_level3.visa_count

# ╔═╡ bb444390-3331-446d-b53a-c8fb91da8c05
md"""
## Population distribution (true)
"""

# ╔═╡ 7ba2489f-fc55-4002-bc05-e47876eba660
begin
	gr()
	plot_claims(normalize(pomdp_level3.params.visa_count, 1))
	plot!(size=(400,200), xtickfont=(6), bottom_margin=2Plots.mm)
	ylims!(0 ,0.7)
	title!("population distribution")
	xlabel!("priority status")
	pop_distr_plt = ylabel!("probability")
end

# ╔═╡ a015b077-2fc6-49b6-a99c-0069cf9dfd7b
savefig(pop_distr_plt, "pop_distr.pdf")

# ╔═╡ c08f9f26-46d0-405a-80d9-6b766af0bf44
md"""
## POMDP-lite type
"""

# ╔═╡ 4cb1fb17-523e-4eab-a9e7-7e1bf0b9edde
mdp_level2a = EvacuationPOMDPType(
	individual_uncertainty=false,
	population_uncertainty=true)

# ╔═╡ 06167243-febb-4c48-b088-efabacff64f5
pomdp_level2b = EvacuationPOMDPType(
	individual_uncertainty=true,
	population_uncertainty=false)

# ╔═╡ fded8908-5b32-4e81-9ef1-0fc4349303c9
observation(pomdp_level2b, sh_test, ACCEPT, shp_test)

# ╔═╡ 5629a62b-0532-4736-aa8b-e814192ed9c0
md"""
## Generative model
"""

# ╔═╡ 8aeff62d-6c27-4f6b-9b0d-8d18df1c2902
gen_state = rand(states(pomdp_level3))

# ╔═╡ 0c3d6891-74fc-4d39-a0d4-080d929677f8
@gen(:sp, :o, :r)(pomdp_level3, gen_state, REJECT)

# ╔═╡ 6349762b-1c5e-4b9b-b2eb-90573f19313e
md"""
## POMDP Solver
"""

# ╔═╡ 80aef6fb-6c23-41a9-a0e1-76d7a04c503a
function solve_pomdp(::Type{POMCPSolver}, pomdp::EvacuationPOMDPType)
	rollout_estimator = FORollout(SIVAMCITsP1P2Policy())

	pomcp_solver = POMCPSolver(
		max_depth=10,
		tree_queries=100,
		estimate_value=rollout_estimator)
	
	pomcp_policy = solve(pomcp_solver, pomdp)
	return pomcp_policy
end

# ╔═╡ 6fe49d6b-343a-4250-8744-a6ba734e51d0
pomcp_policy_2a = solve_pomdp(POMCPSolver, mdp_level2a);

# ╔═╡ b2f56210-7de8-4cf2-9f54-1982dde001d8
level2adata = experiments(1000, mdp_level2a, pomcp_policy_2a, "Level IIa")

# ╔═╡ e24fb800-57fb-4a7e-8c48-b349b661ae93
BSON.@save "level2adata.bson" level2adata

# ╔═╡ 10ed9198-8dd3-43e1-b311-d200fc934649
mean_std(map(sum, level2adata["Level IIa"]["list_reward_over_time"]), false)

# ╔═╡ fe0b9861-f14a-4ecd-bc3a-8cbf70badb97
pomcp_policy_2b = solve_pomdp(POMCPSolver, pomdp_level2b);

# ╔═╡ b7c4ae6e-148a-4bcc-8bf1-8c0b01ef54d2
level2bdata = experiments(1000, pomdp_level2b, pomcp_policy_2b, "Level IIb")

# ╔═╡ 595dc813-048c-415b-a98d-d943f8db9328
BSON.@save "level2bdata.bson" level2bdata

# ╔═╡ 9f7b2cf0-9765-4846-b510-4cd084a4b760
mean_std(map(sum, level2bdata["Level IIb"]["list_reward_over_time"]), false)

# ╔═╡ c9e3d5ee-f0d1-4bd3-9d9c-4ddeab2e87d3
pomcp_policy_3 = solve_pomdp(POMCPSolver, pomdp_level3)

# ╔═╡ d0407ec6-3566-409c-a53a-7b9e0501c4ad
level3data = experiments(1000, pomdp_level3, pomcp_policy_3, "Level III")

# ╔═╡ af739bac-2f7d-4c5e-93d4-2b1377509239
begin
	# plot = EvacuationPOMDP.plot
	# plot! = EvacuationPOMDP.plot!
	pgfplotsx()
	plot()
	hline!([0], c=:gray, lw=1, label=false)
	vline!([120-20+1], c=:gray, lw=0.5, label=false, style=:dash)
	annotate!([(120-20, -850, ("Threshold (AMCITs)", 7, :right, :gray))])
	policy_keys = ["Level I", "Level IIa", "Level IIb", "Level III", "AMCITs", "SIV-AMCITs",  "SIV-AMCITs-P1P2", "AcceptAll", "AfterThresholdAMCITs", "BeforeThresholdAMCITs", "Random"]
	# policy_keys = ["Level I", "Level IIa", "Level IIb", "Level III"]

	colors = [:magenta, :crimson, :red, :darkorange, :gold, :lightgreen, :green, :cyan, :blue, :purple, :black]
	for (i,k) in enumerate(policy_keys)
		style = occursin("Threshold", k) ? :dash : :solid
		style = k == "Random" ? :dot : style
		if k in keys(level1data)
			data = level1data[k]
		elseif k in keys(level2adata)
			data = level2adata[k]
		elseif k in keys(level2bdata)
			data = level2bdata[k]
		elseif k in keys(level3data)
			data = level3data[k]
		end
		μ_cumulative_reward = cumsum(mismatch_mean(data["list_reward_over_time"]))
		# σ_cumulative_reward = sqrt.(cumsum(mismatch_std(data["list_reward_over_time"])))
		plot!(μ_cumulative_reward,
			  # ribbon=σ_cumulative_reward,
			  fillalpha=0.5, lw=2, label=k, style=style, c=colors[i])
		annotate_policy!(k, length(μ_cumulative_reward), last(μ_cumulative_reward))
		scatter!([length(μ_cumulative_reward)], [last(μ_cumulative_reward)], label=false, c=colors[i], ms=3)
	end

	plot!(xlabel="simulation time", ylabel="mean cumulative reward", legend=:bottomleft, title="cumulative reward for each policy")
	agg_plot = xlims!(1, xlims()[2]+37)
	gr()
	agg_plot
end

# ╔═╡ 6d708221-9d27-4e14-a621-420128f2faa3
savefig(agg_plot, "agg_plot.tex")

# ╔═╡ c0ef8c17-bf5d-4a77-bcdb-30d43721c18d
BSON.@save "level3data.bson" level3data

# ╔═╡ bfa95d79-673d-4696-a786-b07ac811919e
mean_std(map(sum, level3data["Level III"]["list_reward_over_time"]), false)

# ╔═╡ 7a301ad6-e31f-4d20-b527-a26573473c0e
begin
	Random.seed!(1)
	simulation(pomdp_level3, pomcp_policy_3)
end

# ╔═╡ fdb3db7f-7784-477f-ace2-b65df9031b41
md"""
## Individual beliefs
"""

# ╔═╡ 33beafb5-1fd2-4e0e-892b-1b5b9d2e0a77
up = DiscreteSubspaceUpdater(pomdp_level3)

# ╔═╡ 85a03af8-4734-48ee-8c6d-a8905e1f2feb
begin
	Random.seed!(0)
	sv₀ = VisibleState(pomdp_level3.params.capacity, pomdp_level3.params.time, 1)
	b₀ = initialize_belief(up, pomdp_level3.visa_count, sv₀)
	online_action, info = action_info(pomcp_policy_3, b₀, tree_in_info=true)
	online_action
end

# ╔═╡ 4efe4359-7041-45c9-bedf-939a41954831
observation(pomdp_level3, POMDPState((sv₀,sh_test)), ACCEPT, POMDPState((sv₀,shp_test)))

# ╔═╡ 7fdc98bb-0c88-434f-99ed-e7876df3c6e2
D3Tree(info[:tree], init_expand=1)

# ╔═╡ 3a7066a4-4273-4ac7-8bfe-63e7d54f0f5b
md"""
### Updating individual belief
"""

# ╔═╡ 40a6caa8-7f10-4816-ba65-bb4c49053775
begin
	localpomdp = EvacuationPOMDPType()
	sv = VisibleState(localpomdp.params.capacity, localpomdp.params.time, 1)
	reset_population_belief!(localpomdp)
	b′ = initialize_belief(up, localpomdp.visa_count, sv)
	# b′(sv)

	@info "Initial" plot_claims(b′.b)
	s = POMDPState((sv, HiddenState(NULL)))
	a = ACCEPT
	for num_updates in 1:20
		global s, b′
		spacing = "\n"^2num_updates
		# vdoc = documentation[rand(𝒟_true)]
		# o = Observation(sv.c, sv.t, sv.f, vdoc)
		sp = rand(transition(localpomdp, s, a))
		if isterminal(localpomdp, sp)
			@warn "Terminal!"
			break
		end
		o = rand(observation(localpomdp, s, a, sp))
		b′ = update(up, b′, a, o)
		@info spacing hidden(sp).v o.vdoc plot_claims(b′.b) plot_claims(normalize(b′.counts,1))
		s = sp
	end
	# [mean(b′) b′.b]
	[plot_claims(b′.b), plot_claims(normalize(b′.counts,1))]
end

# ╔═╡ f1914e6d-8dd2-4412-af20-93530ef0d030
md"""
# Visa probability distribuion
"""

# ╔═╡ 0d1c09ae-e27d-4d9c-84f4-13c4eca12b43
plot_claims(pomdp_level3.params.visa_prob; text="Visa probability distribution")

# ╔═╡ 56d742f1-4fb7-4afe-8674-f343d6672364
pomdp_level3.params.visa_prob

# ╔═╡ 9f2b615d-4187-424f-adb5-f092c78faec5
plot_claims_tiny(pomdp_level3.params.visa_prob)

# ╔═╡ f429f2b4-959b-4ed2-bd49-6ba961ba2382
md"""
# Simulations
"""

# ╔═╡ 38b970ed-0fc2-44d4-bf0d-4ad5fa70d1ff
traj_seed = 3

# ╔═╡ 466f9de5-2df5-48a3-b481-7a45858410ae
begin
	Random.seed!(traj_seed)
	level1_trajectory = simulate_trajectory(mdp, mdp_policy; seed=traj_seed)
end

# ╔═╡ 7ff64ea4-7086-40ad-bd1b-faed62a4d7f2
begin
	Random.seed!(traj_seed)
	level2a_trajectory = simulate_trajectory(mdp_level2a, pomcp_policy_2a; 
											   seed=traj_seed)
end

# ╔═╡ bc26811d-9f22-46e5-97ef-5e2516dc1ec5
begin
	Random.seed!(traj_seed)
	level2b_trajectory = simulate_trajectory(pomdp_level2b, pomcp_policy_2b; 
											   seed=traj_seed)
end

# ╔═╡ ff841f9f-0603-4d05-a820-736a4cc62e3d
begin
	Random.seed!(traj_seed)
	level3_trajectory = simulate_trajectory(pomdp_level3, pomcp_policy_3; 
											   seed=traj_seed)
end

# ╔═╡ 3b3110a8-8322-404f-8ad9-f4361640666e
[plot_claims_tiny(τ[end-1].b) for τ in level3_trajectory]

# ╔═╡ 9c10c8d3-71a7-422b-b24e-8e2cb2997cca
map(t->t[end-1].b, level3_trajectory[end-2:end])

# ╔═╡ 5e000528-bdae-43a7-bd58-a7e2fb3d80be
md"""
# Trajectory plotting
"""

# ╔═╡ 2bb01d5f-e8f4-4986-ad6f-5a44daa464b9
N = 20

# ╔═╡ f2d91ff3-db6d-41eb-89cb-85711048922c
plot_trajectory(mdp, level1_trajectory, "traj_level1"; N=N)

# ╔═╡ 281a7c2d-b1ce-4b33-b227-f3b2a6237725
plot_trajectory(mdp_level2a, level2a_trajectory, "traj_level2a"; N=N, show_population=true, show_belief=false)

# ╔═╡ 6b77d60f-48cc-4c64-aaf8-d496965c087f
plot_trajectory(pomdp_level2b, level2b_trajectory, "traj_level2b"; N=N, show_population=false, show_belief=true)

# ╔═╡ edabc978-aaec-4a09-a234-4fa9caf2b58e
plot_trajectory(pomdp_level3, level3_trajectory, "traj_level3"; N=N, show_population=true, show_belief=true)

# ╔═╡ Cell order:
# ╠═30028219-7ea5-4b78-b0df-b3b98b25ee65
# ╠═41d0fe72-b83d-4cf8-bc91-44b0e8f561eb
# ╟─c0292f00-7f92-11ec-2f15-b7975a5e40a1
# ╠═e2772773-4058-4a1b-9f99-d197942086d5
# ╠═fe9dfc73-f502-4d13-a019-8160a6af4617
# ╟─3771f6a0-3926-4234-b4b2-69ec8f96fa41
# ╠═3b468bb0-9e90-40b0-9a24-60fba6ca4fc5
# ╠═f9b11f1e-04b9-488c-9884-ab0bb77a945c
# ╟─47a33688-6458-44a6-a5e5-3a6a220e9c39
# ╠═6bf5f2f2-b5d1-48a6-b811-f416bcfc9899
# ╟─5acdce32-b587-4ea2-8317-a753429fcd7b
# ╠═4bfee8de-cc8d-436d-98ad-5f7002ece4cf
# ╠═70dffecd-3db5-45a9-8104-d82f30fdead2
# ╠═c9d38947-173d-4baf-9b3f-81bacf4d16cc
# ╟─fbd975c5-fc0b-43bd-86e0-7ed9c4efc0a5
# ╠═1e180b5a-34e4-4509-8eba-76445fbbc9ee
# ╠═4f9b0d74-73d5-42e9-9ee6-178797ac023c
# ╠═2d58015e-1e3d-4000-ae68-5d5222fa8101
# ╠═af739bac-2f7d-4c5e-93d4-2b1377509239
# ╠═6d708221-9d27-4e14-a621-420128f2faa3
# ╠═8d98f5fc-9f8f-4cb2-a791-3e1a987ed56e
# ╠═b2f56210-7de8-4cf2-9f54-1982dde001d8
# ╠═b7c4ae6e-148a-4bcc-8bf1-8c0b01ef54d2
# ╠═d0407ec6-3566-409c-a53a-7b9e0501c4ad
# ╠═c0ef8c17-bf5d-4a77-bcdb-30d43721c18d
# ╠═e24fb800-57fb-4a7e-8c48-b349b661ae93
# ╠═595dc813-048c-415b-a98d-d943f8db9328
# ╠═118ab216-0548-4ec4-b933-cedf78f08c55
# ╠═a17792e3-d4fb-498c-bf8b-d25f9e9efc5c
# ╠═10ed9198-8dd3-43e1-b311-d200fc934649
# ╠═9f7b2cf0-9765-4846-b510-4cd084a4b760
# ╠═bfa95d79-673d-4696-a786-b07ac811919e
# ╠═dd15d7f4-a163-44d4-8f8f-7915a78ed4f6
# ╠═8d2b9590-8d76-4675-9b8a-e6a47cbccb8c
# ╟─2b2ed9bd-e033-47ea-b24b-4789f28ab08c
# ╠═0fe49a8e-4ee5-41b8-8802-8e61d997b252
# ╟─d1646d70-426d-4545-ba31-25a7712fe852
# ╠═67139013-44dc-4f83-afcc-eb3aaacb0eab
# ╠═9babaea3-5ed9-4349-ba5b-95f9549213eb
# ╠═29798743-18cc-4c7b-af0d-ffb962a3eec9
# ╠═67d3b29c-4aeb-45df-a449-cd669590fb58
# ╠═59c54092-e44e-422c-a29d-3fc457718190
# ╟─470e32ce-57da-48b6-a27e-0899083a47a3
# ╠═40430b14-07b2-40c6-95fe-9f60a5c6e75f
# ╠═34b70a62-622e-41e3-a865-7cbd8b6a8fb5
# ╠═57c7d1f6-9920-464d-8b14-563fccd6878f
# ╟─958c44ef-9c64-484c-ae80-0a62f2142225
# ╠═d112f66b-bd60-480c-9fc3-275b90206e6c
# ╟─4a747878-429e-455f-8894-a92c22be4dcf
# ╠═7a301ad6-e31f-4d20-b527-a26573473c0e
# ╠═56c51f5d-f632-4afe-8e10-32ef28160f48
# ╠═40c84bf1-88c8-47d3-9a5b-6f9308d883e8
# ╠═fded8908-5b32-4e81-9ef1-0fc4349303c9
# ╠═4efe4359-7041-45c9-bedf-939a41954831
# ╟─f66f2579-d2d4-4a5b-9805-4922dbb99e9b
# ╠═27bd873f-c7b1-4323-99e3-6f6be02eb8b5
# ╠═f094c09e-2f51-4b1b-81cb-a6832baccb92
# ╠═36c1fe73-2559-4f9e-95f3-3e79ed0c0264
# ╠═bb444390-3331-446d-b53a-c8fb91da8c05
# ╠═7ba2489f-fc55-4002-bc05-e47876eba660
# ╠═a015b077-2fc6-49b6-a99c-0069cf9dfd7b
# ╟─c08f9f26-46d0-405a-80d9-6b766af0bf44
# ╠═4cb1fb17-523e-4eab-a9e7-7e1bf0b9edde
# ╠═06167243-febb-4c48-b088-efabacff64f5
# ╠═6fe49d6b-343a-4250-8744-a6ba734e51d0
# ╠═fe0b9861-f14a-4ecd-bc3a-8cbf70badb97
# ╟─5629a62b-0532-4736-aa8b-e814192ed9c0
# ╠═8aeff62d-6c27-4f6b-9b0d-8d18df1c2902
# ╠═0c3d6891-74fc-4d39-a0d4-080d929677f8
# ╟─6349762b-1c5e-4b9b-b2eb-90573f19313e
# ╠═05870354-856b-4342-8dce-00219b602342
# ╠═80aef6fb-6c23-41a9-a0e1-76d7a04c503a
# ╠═c9e3d5ee-f0d1-4bd3-9d9c-4ddeab2e87d3
# ╠═85a03af8-4734-48ee-8c6d-a8905e1f2feb
# ╠═d4d4de96-b8aa-484d-b594-afb48dd472bc
# ╠═7fdc98bb-0c88-434f-99ed-e7876df3c6e2
# ╟─fdb3db7f-7784-477f-ace2-b65df9031b41
# ╠═33beafb5-1fd2-4e0e-892b-1b5b9d2e0a77
# ╟─3a7066a4-4273-4ac7-8bfe-63e7d54f0f5b
# ╠═40a6caa8-7f10-4816-ba65-bb4c49053775
# ╟─f1914e6d-8dd2-4412-af20-93530ef0d030
# ╠═0d1c09ae-e27d-4d9c-84f4-13c4eca12b43
# ╠═56d742f1-4fb7-4afe-8674-f343d6672364
# ╠═9f2b615d-4187-424f-adb5-f092c78faec5
# ╠═3b3110a8-8322-404f-8ad9-f4361640666e
# ╠═9c10c8d3-71a7-422b-b24e-8e2cb2997cca
# ╟─f429f2b4-959b-4ed2-bd49-6ba961ba2382
# ╠═38b970ed-0fc2-44d4-bf0d-4ad5fa70d1ff
# ╠═466f9de5-2df5-48a3-b481-7a45858410ae
# ╠═7ff64ea4-7086-40ad-bd1b-faed62a4d7f2
# ╠═bc26811d-9f22-46e5-97ef-5e2516dc1ec5
# ╠═ff841f9f-0603-4d05-a820-736a4cc62e3d
# ╟─5e000528-bdae-43a7-bd58-a7e2fb3d80be
# ╠═2bb01d5f-e8f4-4986-ad6f-5a44daa464b9
# ╠═f2d91ff3-db6d-41eb-89cb-85711048922c
# ╠═281a7c2d-b1ce-4b33-b227-f3b2a6237725
# ╠═6b77d60f-48cc-4c64-aaf8-d496965c087f
# ╠═edabc978-aaec-4a09-a234-4fa9caf2b58e
