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
mdpdata = experiments(1000, mdp, mdp_policy)

# ╔═╡ d0407ec6-3566-409c-a53a-7b9e0501c4ad
# pomdpdata = experiments(1000, pomdp, pomcp_policy, "POMDP")

# ╔═╡ b2f56210-7de8-4cf2-9f54-1982dde001d8
# pomdplitedata = experiments(1000, pomdplite, pomcp_policy_lite, "POMDP-lite")

# ╔═╡ c0ef8c17-bf5d-4a77-bcdb-30d43721c18d
BSON.@save "pomdpdata.bson" pomdpdata

# ╔═╡ e24fb800-57fb-4a7e-8c48-b349b661ae93
BSON.@save "pomdplitedata.bson" pomdplitedata

# ╔═╡ 7b608351-17ef-4dd9-aa17-1c69bacd47d9
BSON.@save "mdpdata.bson" mdpdata

# ╔═╡ a17792e3-d4fb-498c-bf8b-d25f9e9efc5c
with_terminal() do
	println("Hello!")
end

# ╔═╡ dd15d7f4-a163-44d4-8f8f-7915a78ed4f6
function mismatch_mean(A)
    max_length = maximum(map(length, A))
    Z = [map(a->i <= length(a) ? a[i] : nothing, A) for i in 1:max_length]
    return map(mean, map(z->filter(!isnothing, z), Z))
end

# ╔═╡ af739bac-2f7d-4c5e-93d4-2b1377509239
begin
	plot = EvacuationPOMDP.plot
	plot! = EvacuationPOMDP.plot!
	plot()
	for k in keys(mdpdata)
		# k ∉ ["MDP", "Random"] && continue
		plot!(mismatch_mean(mdpdata[k]["list_reward_over_time"]),
			  # ribbon=mismatch_std(mdpdata[k]["list_reward_over_time"])/10,
			  fillalpha=0.5, label=k)
	end
	plot!(ylims=(-50, 20), 
		  xlabel="simulation time", ylabel="reward", legend=:bottomleft)
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

# ╔═╡ 470e32ce-57da-48b6-a27e-0899083a47a3
md"""
# MDP policy plot
"""

# ╔═╡ 40430b14-07b2-40c6-95fe-9f60a5c6e75f
vis_all(mdp.params, mdp_policy)

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
pomdp = EvacuationPOMDPType()

# ╔═╡ 3b468bb0-9e90-40b0-9a24-60fba6ca4fc5
plot_family_size_distribution(pomdp.params.family_prob)

# ╔═╡ 67139013-44dc-4f83-afcc-eb3aaacb0eab
pomdp.params.visa_prob

# ╔═╡ 9babaea3-5ed9-4349-ba5b-95f9549213eb
claim_plot = plot_all_claims(pomdp)

# ╔═╡ 29798743-18cc-4c7b-af0d-ffb962a3eec9
savefig(claim_plot, "claim_plot.pdf")

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
		tree_queries=1000,
		estimate_value=rollout_estimator)
	
	pomcp_policy = solve(pomcp_solver, pomdp)
	return pomcp_policy
end

# ╔═╡ 6fe49d6b-343a-4250-8744-a6ba734e51d0
pomcp_policy_lite = solve_pomdp(POMCPSolver, pomdplite);

# ╔═╡ e1be8687-43db-46e9-99ad-eff59c3c2985
pomcp_policy = solve_pomdp(POMCPSolver, pomdp)

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

# ╔═╡ 9f2b615d-4187-424f-adb5-f092c78faec5
plot_claims_tiny(pomdp.params.visa_prob)

# ╔═╡ f429f2b4-959b-4ed2-bd49-6ba961ba2382
md"""
# Simulations
"""

# ╔═╡ 38b970ed-0fc2-44d4-bf0d-4ad5fa70d1ff
traj_seed = 3

# ╔═╡ 466f9de5-2df5-48a3-b481-7a45858410ae
begin
	Random.seed!(traj_seed)
	mdp_trajectory = simulate_trajectory(mdp, mdp_policy; seed=traj_seed)
end

# ╔═╡ 7ff64ea4-7086-40ad-bd1b-faed62a4d7f2
begin
	Random.seed!(traj_seed)
	pomdplite_trajectory = simulate_trajectory(pomdplite, pomcp_policy_lite; 
											   seed=traj_seed)
end

# ╔═╡ ff841f9f-0603-4d05-a820-736a4cc62e3d
begin
	Random.seed!(traj_seed)
	pomdp_trajectory = simulate_trajectory(pomdp, pomcp_policy; seed=traj_seed)
end

# ╔═╡ 3b3110a8-8322-404f-8ad9-f4361640666e
[plot_claims_tiny(τ[end-1].b) for τ in pomdp_trajectory]

# ╔═╡ 9c10c8d3-71a7-422b-b24e-8e2cb2997cca
map(t->t[end-1].b, pomdp_trajectory[end-2:end])

# ╔═╡ cd3e1287-979d-46f3-84a2-51f97739f409
pomdp_trajectory[end-2:end]

# ╔═╡ 5e000528-bdae-43a7-bd58-a7e2fb3d80be
md"""
# Trajectory plotting
"""

# ╔═╡ 2bb01d5f-e8f4-4986-ad6f-5a44daa464b9
N = 20

# ╔═╡ 52e291aa-15c4-44de-8ba8-a7ca585b2a0a
(length(mdp_trajectory) - 11) + (12 - 11)

# ╔═╡ abc10285-ae4d-4434-9c2a-8f5eb6c2676a
length(mdp_trajectory)

# ╔═╡ 9a44ca83-3b85-4f79-adea-4ffdfb0bbffa
mdp_trajectory[end]

# ╔═╡ 0a08a0ca-0b87-454b-9b30-6afcfc375009
plot_trajectory(mdp, mdp_trajectory, "traj_mdp"; N=N)

# ╔═╡ 281a7c2d-b1ce-4b33-b227-f3b2a6237725
plot_trajectory(pomdplite, pomdplite_trajectory, "traj_pomdplite"; N=N)

# ╔═╡ 69e64da0-121d-4432-87e2-d9d3441725dd
plot_trajectory(pomdp, pomdp_trajectory, "traj_pomdp"; N=N)

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
# ╠═af739bac-2f7d-4c5e-93d4-2b1377509239
# ╠═d0407ec6-3566-409c-a53a-7b9e0501c4ad
# ╠═b2f56210-7de8-4cf2-9f54-1982dde001d8
# ╠═c0ef8c17-bf5d-4a77-bcdb-30d43721c18d
# ╠═e24fb800-57fb-4a7e-8c48-b349b661ae93
# ╠═7b608351-17ef-4dd9-aa17-1c69bacd47d9
# ╠═a17792e3-d4fb-498c-bf8b-d25f9e9efc5c
# ╠═dd15d7f4-a163-44d4-8f8f-7915a78ed4f6
# ╠═8d2b9590-8d76-4675-9b8a-e6a47cbccb8c
# ╟─2b2ed9bd-e033-47ea-b24b-4789f28ab08c
# ╠═0fe49a8e-4ee5-41b8-8802-8e61d997b252
# ╟─d1646d70-426d-4545-ba31-25a7712fe852
# ╠═67139013-44dc-4f83-afcc-eb3aaacb0eab
# ╠═9babaea3-5ed9-4349-ba5b-95f9549213eb
# ╠═29798743-18cc-4c7b-af0d-ffb962a3eec9
# ╟─470e32ce-57da-48b6-a27e-0899083a47a3
# ╠═40430b14-07b2-40c6-95fe-9f60a5c6e75f
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
# ╟─c08f9f26-46d0-405a-80d9-6b766af0bf44
# ╠═4cb1fb17-523e-4eab-a9e7-7e1bf0b9edde
# ╠═6fe49d6b-343a-4250-8744-a6ba734e51d0
# ╟─5629a62b-0532-4736-aa8b-e814192ed9c0
# ╠═8aeff62d-6c27-4f6b-9b0d-8d18df1c2902
# ╠═0c3d6891-74fc-4d39-a0d4-080d929677f8
# ╟─6349762b-1c5e-4b9b-b2eb-90573f19313e
# ╠═05870354-856b-4342-8dce-00219b602342
# ╠═80aef6fb-6c23-41a9-a0e1-76d7a04c503a
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
# ╠═9f2b615d-4187-424f-adb5-f092c78faec5
# ╠═3b3110a8-8322-404f-8ad9-f4361640666e
# ╠═9c10c8d3-71a7-422b-b24e-8e2cb2997cca
# ╠═cd3e1287-979d-46f3-84a2-51f97739f409
# ╟─f429f2b4-959b-4ed2-bd49-6ba961ba2382
# ╠═38b970ed-0fc2-44d4-bf0d-4ad5fa70d1ff
# ╠═466f9de5-2df5-48a3-b481-7a45858410ae
# ╠═7ff64ea4-7086-40ad-bd1b-faed62a4d7f2
# ╠═ff841f9f-0603-4d05-a820-736a4cc62e3d
# ╟─5e000528-bdae-43a7-bd58-a7e2fb3d80be
# ╠═00e07de2-e92b-499d-b446-595a968b1ecc
# ╠═2bb01d5f-e8f4-4986-ad6f-5a44daa464b9
# ╠═52e291aa-15c4-44de-8ba8-a7ca585b2a0a
# ╠═abc10285-ae4d-4434-9c2a-8f5eb6c2676a
# ╠═9a44ca83-3b85-4f79-adea-4ffdfb0bbffa
# ╠═0a08a0ca-0b87-454b-9b30-6afcfc375009
# ╠═281a7c2d-b1ce-4b33-b227-f3b2a6237725
# ╠═69e64da0-121d-4432-87e2-d9d3441725dd
# ╟─9a142f50-9c0b-4d5a-807d-07d4992b5155
# ╠═d3ffbfb2-38ff-4003-a93b-0c804d90d8fc
# ╠═f1e42b93-d895-4706-8044-9863c65908e7
# ╠═9757ae45-a980-42a5-9373-44aad5a34d41
