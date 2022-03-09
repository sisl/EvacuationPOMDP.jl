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

# ╔═╡ 00e07de2-e92b-499d-b446-595a968b1ecc
using TikzGraphs

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
experiments(1000, mdp, mdp_policy)

# ╔═╡ a17792e3-d4fb-498c-bf8b-d25f9e9efc5c
with_terminal() do
	println("Hello!")
end

# ╔═╡ d1646d70-426d-4545-ba31-25a7712fe852
md"""
# Claim models
"""

# ╔═╡ 470e32ce-57da-48b6-a27e-0899083a47a3
md"""
# MDP policy plot
"""

# ╔═╡ 40430b14-07b2-40c6-95fe-9f60a5c6e75f
vis_all(mdp.params, mdp_policy)

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
pomdp = EvacuationPOMDPType()

# ╔═╡ 3b468bb0-9e90-40b0-9a24-60fba6ca4fc5
plot_family_size_distribution(pomdp.params.family_prob)

# ╔═╡ 9babaea3-5ed9-4349-ba5b-95f9549213eb
plot_all_claims(pomdp)

# ╔═╡ d112f66b-bd60-480c-9fc3-275b90206e6c
transitionhidden(pomdp, HiddenState(AMCIT), ACCEPT)

# ╔═╡ 40c84bf1-88c8-47d3-9a5b-6f9308d883e8
observation(pomdp, sh_test, ACCEPT, shp_test)

# ╔═╡ c08f9f26-46d0-405a-80d9-6b766af0bf44
md"""
## POMDP-lite type
"""

# ╔═╡ 4cb1fb17-523e-4eab-a9e7-7e1bf0b9edde
pomdplite = EvacuationPOMDPType(isnoisy=false)

# ╔═╡ fded8908-5b32-4e81-9ef1-0fc4349303c9
observation(pomdplite, sh_test, ACCEPT, shp_test)

# ╔═╡ 5629a62b-0532-4736-aa8b-e814192ed9c0
md"""
## Generative model
"""

# ╔═╡ 8aeff62d-6c27-4f6b-9b0d-8d18df1c2902
gen_state = rand(states(pomdp))

# ╔═╡ 0c3d6891-74fc-4d39-a0d4-080d929677f8
@gen(:sp, :o, :r)(pomdp, gen_state, REJECT)

# ╔═╡ 6349762b-1c5e-4b9b-b2eb-90573f19313e
md"""
## POMDP Solver
"""

# ╔═╡ 80aef6fb-6c23-41a9-a0e1-76d7a04c503a
function solve_pomdp(::Type{POMCPSolver}, pomdp::EvacuationPOMDPType)
	rollout_estimator = FORollout(SIVAMCITsPolicy())

	pomcp_solver = POMCPSolver(
		max_depth=10,
		tree_queries=100,
		estimate_value=rollout_estimator)
	
	pomcp_policy = solve(pomcp_solver, pomdp)
	return pomcp_policy
end

# ╔═╡ 6fe49d6b-343a-4250-8744-a6ba734e51d0
pomcp_policy_lite = solve_pomdp(POMCPSolver, pomdplite);

# ╔═╡ b2f56210-7de8-4cf2-9f54-1982dde001d8
experiments(1000, pomdplite, pomcp_policy_lite)

# ╔═╡ e1be8687-43db-46e9-99ad-eff59c3c2985
pomcp_policy = solve_pomdp(POMCPSolver, pomdp)

# ╔═╡ d0407ec6-3566-409c-a53a-7b9e0501c4ad
experiments(1000, pomdp, pomcp_policy)

# ╔═╡ 7a301ad6-e31f-4d20-b527-a26573473c0e
begin
	Random.seed!(1)
	simulation(pomdp, pomcp_policy)
end

# ╔═╡ fdb3db7f-7784-477f-ace2-b65df9031b41
md"""
## Individual beliefs
"""

# ╔═╡ 33beafb5-1fd2-4e0e-892b-1b5b9d2e0a77
up = DiscreteSubspaceUpdater(pomdp)

# ╔═╡ 85a03af8-4734-48ee-8c6d-a8905e1f2feb
begin
	Random.seed!(0)
	sv₀ = VisibleState(pomdp.params.capacity, pomdp.params.time, 1)
	b₀ = initialize_belief(up, pomdp.params.visa_count, sv₀)
	online_action, info = action_info(pomcp_policy, b₀, tree_in_info=true)
	online_action
end

# ╔═╡ 4efe4359-7041-45c9-bedf-939a41954831
observation(pomdp, POMDPState((sv₀,sh_test)), ACCEPT, POMDPState((sv₀,shp_test)))

# ╔═╡ 7fdc98bb-0c88-434f-99ed-e7876df3c6e2
D3Tree(info[:tree], init_expand=1)

# ╔═╡ 3a7066a4-4273-4ac7-8bfe-63e7d54f0f5b
md"""
### Updating individual belief
"""

# ╔═╡ 40a6caa8-7f10-4816-ba65-bb4c49053775
begin
	sv = VisibleState(pomdp.params.capacity, pomdp.params.time, 1)
	b′ = initialize_belief(up, pomdp.params.visa_count, sv)
	# b′(sv)

	@info "Initial" plot_claims(b′.b)
	s = POMDPState((sv, HiddenState(NULL)))
	a = ACCEPT
	for num_updates in 1:20
		global s, b′
		spacing = "\n"^2num_updates
		# vdoc = documentation[rand(𝒟_true)]
		# o = Observation(sv.c, sv.t, sv.f, vdoc)
		sp = rand(transition(pomdp, s, a))
		if isterminal(pomdp, sp)
			@warn "Terminal!"
			break
		end
		o = rand(observation(pomdp, s, a, sp))
		b′ = update(up, b′, a, o)
		@info spacing hidden(sp).v o.vdoc plot_claims(b′.b)
		s = sp
	end
	# [mean(b′) b′.b]
	plot_claims(b′.b)
end

# ╔═╡ f1914e6d-8dd2-4412-af20-93530ef0d030
md"""
# Visa probability distribuion
"""

# ╔═╡ 0d1c09ae-e27d-4d9c-84f4-13c4eca12b43
plot_claims(pomdp.params.visa_prob; text="Visa probability distribution")

# ╔═╡ 56d742f1-4fb7-4afe-8674-f343d6672364
pomdp.params.visa_prob

# ╔═╡ f429f2b4-959b-4ed2-bd49-6ba961ba2382
md"""
# Simulations
"""

# ╔═╡ ff841f9f-0603-4d05-a820-736a4cc62e3d
pomdp_trajectory = simulate_trajectory(pomdp, pomcp_policy)

# ╔═╡ 7ff64ea4-7086-40ad-bd1b-faed62a4d7f2
pomdplite_trajectory = simulate_trajectory(pomdplite, pomcp_policy_lite)

# ╔═╡ 466f9de5-2df5-48a3-b481-7a45858410ae
mdp_trajectory = simulate_trajectory(mdp, mdp_policy)

# ╔═╡ 5e000528-bdae-43a7-bd58-a7e2fb3d80be
md"""
# Trajectory plotting
"""

# ╔═╡ 69e64da0-121d-4432-87e2-d9d3441725dd
plot_trajectory(pomdp, pomdp_trajectory, "traj_pomdp")

# ╔═╡ 281a7c2d-b1ce-4b33-b227-f3b2a6237725
plot_trajectory(pomdplite, pomdplite_trajectory, "traj_pomdplite")

# ╔═╡ cacc6e08-b2c0-4ec3-a768-fa3eb8308789
plot_trajectory(mdp, mdp_trajectory, "traj_mdp")

# ╔═╡ 9a142f50-9c0b-4d5a-807d-07d4992b5155
md"""
# Aggregate statistics
"""

# ╔═╡ d3ffbfb2-38ff-4003-a93b-0c804d90d8fc
pomdp.params.visa_count

# ╔═╡ f1e42b93-d895-4706-8044-9863c65908e7
pomdp.visa_count

# ╔═╡ 9757ae45-a980-42a5-9373-44aad5a34d41
hist2 = simulation(pomdp, pomcp_policy)

# ╔═╡ Cell order:
# ╠═30028219-7ea5-4b78-b0df-b3b98b25ee65
# ╠═41d0fe72-b83d-4cf8-bc91-44b0e8f561eb
# ╟─c0292f00-7f92-11ec-2f15-b7975a5e40a1
# ╠═e2772773-4058-4a1b-9f99-d197942086d5
# ╠═fe9dfc73-f502-4d13-a019-8160a6af4617
# ╟─3771f6a0-3926-4234-b4b2-69ec8f96fa41
# ╠═3b468bb0-9e90-40b0-9a24-60fba6ca4fc5
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
# ╠═d0407ec6-3566-409c-a53a-7b9e0501c4ad
# ╠═b2f56210-7de8-4cf2-9f54-1982dde001d8
# ╠═a17792e3-d4fb-498c-bf8b-d25f9e9efc5c
# ╟─d1646d70-426d-4545-ba31-25a7712fe852
# ╠═9babaea3-5ed9-4349-ba5b-95f9549213eb
# ╟─470e32ce-57da-48b6-a27e-0899083a47a3
# ╠═40430b14-07b2-40c6-95fe-9f60a5c6e75f
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
# ╟─c08f9f26-46d0-405a-80d9-6b766af0bf44
# ╠═4cb1fb17-523e-4eab-a9e7-7e1bf0b9edde
# ╠═6fe49d6b-343a-4250-8744-a6ba734e51d0
# ╟─5629a62b-0532-4736-aa8b-e814192ed9c0
# ╠═8aeff62d-6c27-4f6b-9b0d-8d18df1c2902
# ╠═0c3d6891-74fc-4d39-a0d4-080d929677f8
# ╟─6349762b-1c5e-4b9b-b2eb-90573f19313e
# ╠═80aef6fb-6c23-41a9-a0e1-76d7a04c503a
# ╠═05870354-856b-4342-8dce-00219b602342
# ╠═e1be8687-43db-46e9-99ad-eff59c3c2985
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
# ╟─f429f2b4-959b-4ed2-bd49-6ba961ba2382
# ╠═ff841f9f-0603-4d05-a820-736a4cc62e3d
# ╠═7ff64ea4-7086-40ad-bd1b-faed62a4d7f2
# ╠═466f9de5-2df5-48a3-b481-7a45858410ae
# ╟─5e000528-bdae-43a7-bd58-a7e2fb3d80be
# ╠═00e07de2-e92b-499d-b446-595a968b1ecc
# ╠═69e64da0-121d-4432-87e2-d9d3441725dd
# ╠═281a7c2d-b1ce-4b33-b227-f3b2a6237725
# ╠═cacc6e08-b2c0-4ec3-a768-fa3eb8308789
# ╟─9a142f50-9c0b-4d5a-807d-07d4992b5155
# ╠═d3ffbfb2-38ff-4003-a93b-0c804d90d8fc
# ╠═f1e42b93-d895-4706-8044-9863c65908e7
# ╠═9757ae45-a980-42a5-9373-44aad5a34d41
