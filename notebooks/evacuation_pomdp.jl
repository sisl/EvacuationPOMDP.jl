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

# ╔═╡ 9e6cfa03-a98c-4388-a952-7f498c826a75
using KernelDensity

# ╔═╡ 05870354-856b-4342-8dce-00219b602342
using BasicPOMCP

# ╔═╡ d4d4de96-b8aa-484d-b594-afb48dd472bc
using D3Trees

# ╔═╡ 1dae6254-f47b-499a-af1a-ac6524836acc
using ARDESPOT

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

# ╔═╡ 7aea018d-d9bb-44b4-9f16-e6def5ffc249
states(mdp) |> length

# ╔═╡ 06bdc30d-7f1a-4b4e-86a1-76c6030f75e9
(501+12)*1201*13*5 + 1

# ╔═╡ 4bfee8de-cc8d-436d-98ad-5f7002ece4cf
solver = ValueIterationSolver(max_iterations=30, belres=1e-6, verbose=true);

# ╔═╡ c9d38947-173d-4baf-9b3f-81bacf4d16cc
@bind reload_mdp_policy CheckBox(false)

# ╔═╡ fbd975c5-fc0b-43bd-86e0-7ed9c4efc0a5
md"""
> - **Running**: takes about 10753 seconds
> - **Loading**: takes about 1.8 seconds, 73 MB (old)
"""

# ╔═╡ 1e180b5a-34e4-4509-8eba-76445fbbc9ee
# if reload_mdp_policy
# 	mdp_policy = BSON.load("mdp_policy.bson", @__MODULE__)[:mdp_policy]
# else
# 	mdp_policy = solve(solver, mdp)
# end

# ╔═╡ 6816dd65-f198-40f9-bca9-eef180781fb1
# BSON.@save "mdp_policy.bson" mdp_policy

# ╔═╡ 2d58015e-1e3d-4000-ae68-5d5222fa8101
# level1data = experiments(1000, mdp, mdp_policy)

# ╔═╡ 75b0e162-3aee-45ef-91c5-5a8fbfcd1d18
level1data = BSON.load("saved_data/level1data.bson", @__MODULE__)[:level1data]
# BSON.@save "level1data.bson" level1data

# ╔═╡ 5bb11a5a-f9c8-4bec-9f08-c6b00a9a322d
U = kde(convert(Vector{Real}, level1data["AMCITs"]["list_total_accepted_people"]))

# ╔═╡ a0170fcd-7438-4d08-8f75-dfa43249fb6c
plot(U.x, U.density)

# ╔═╡ 2a57deca-5a25-4d3d-a759-74f1f5e136ff
var(Normal(0, 2))

# ╔═╡ 913ae6dc-8012-48a6-8bbc-5195b247a0a9
function plot_accepted(data, label, c; hold=true)
	pf = hold ? plot! : plot
	hα = 0.6
	nz = true
	U = kde(convert(Vector{Real}, data))
	# histogram(data, label=label, alpha=hα, c=colors[5], normalize=nz)
	pf(U.x, U.density, label=label, c=c, lw=2, fill=true, fillalpha=0.2)
end	

# ╔═╡ 3759dd49-0635-454c-8f4d-6f90890fabf8
begin
	histogram(level1data["AMCITs"]["list_total_reward"], label="AMCITs", alpha = 0.7)
	histogram!(level1data["SIV-AMCITs"]["list_total_reward"], label="SIV-AMCITs", alpha = 0.7)
	histogram!(level1data["Level I"]["list_total_reward"], label="Level I", alpha = 0.7)
	# histogram!(level2bdata_roa["Level I"]["list_total_reward"], label="Level IIb", alpha = 0.5)
end

# ╔═╡ 694d7f4d-dc06-4501-9ce1-c89c525d32ba
level1data["Level I"]["list_total_accepted_people"] |> minimum

# ╔═╡ 2d182258-1fa6-48b1-ba45-ee346f9e873e
level1data["Level I"]

# ╔═╡ 50093ef5-1f7a-4a4b-8b7e-49ef14a44261
populations = EvacuationPOMDP.read_population_json()

# ╔═╡ c4738481-fc09-41af-9862-73799436226a
history = manual_simulate(mdp, mdp_policy, deepcopy(populations[8]))

# ╔═╡ 6c8c7500-3ac1-4d1b-b148-90b31b455012
metrics = get_metrics(mdp.params, history)

# ╔═╡ 277ac05c-bc57-4dd0-863a-7f5fb284e825
md"""
# $R(o,a)$ test
"""

# ╔═╡ bb3c370c-54d3-4abd-80fd-2144faf70672
mdp_roa = EvacuationMDP() # EvacuationParameters(roa_reward=true))

# ╔═╡ d2afed13-a379-4f87-8f4f-92bd1407c470
# if false
# 	mdp_roa_policy = BSON.load("mdp_roa_policy.bson", @__MODULE__)[:mdp_roa_policy]
# else
# 	mdp_roa_policy = solve(solver, mdp_roa)
# end

# ╔═╡ c5944f50-8d0d-48a5-b0af-65eb69401e1d
# BSON.@save "mdp_roa_policy.bson" mdp_roa_policy

# ╔═╡ 6029a8c1-48ab-433e-b15a-8cf7f693e5d0
# level2bdata_roa = experiments(1000, mdp_roa, mdp_roa_policy)

# ╔═╡ 88bdff09-8091-4ed4-a6da-da3b9311c3af
level2bdata_roa = BSON.load("saved_data/level2bdata_roa.bson", @__MODULE__)[:level2bdata_roa]
# BSON.@save "level2bdata_roa.bson" level2bdata_roa

# ╔═╡ ef19a1d3-0fe2-45ef-bfc2-6768e6dfb2a3
level2bdata_roa["Level IIb VI"] = level2bdata_roa["Level I"]

# ╔═╡ aa6d9555-a398-4b8a-94db-32a4a0f68128
pomdp_roa = EvacuationPOMDPType(individual_uncertainty=true, population_uncertainty=false)

# ╔═╡ 0c7b6ab7-aafe-44c4-a992-5d003029ac51
function state2obs(s::MDPState)
	
end

# ╔═╡ 0bf669b4-7e12-4ef1-986c-4bcde78169ba
_v = ISIS

# ╔═╡ 3e9e9958-d6a4-461e-90ab-346c446eaffe
_s = EvacuationPOMDP.newstate(MDPState, 1, 100, 2, _v)

# ╔═╡ 0ac99c53-b60c-42f5-bc96-e2ae9d260b83
_a = ACCEPT

# ╔═╡ 99cecdb4-1598-4156-b77d-1e90d86be052
_o = Observation(_s.c, _s.t, _s.f, VisaDocument(Int(_s.v)))

# ╔═╡ 92ea23d4-5200-4d02-9d38-9c968f3b0e20
begin
	_up = DiscreteSubspaceUpdater(pomdp_roa)
	_b₀ = initialize_belief(_up, pomdp_roa.visa_count, _s)
	(_b′ = update(_up, _b₀, _a, _o).b) |> plot_claims_tiny
end

# ╔═╡ 6d99cfd2-db78-4645-9f4b-b1d2922787fe
_b′

# ╔═╡ 67c0c993-8e53-4a56-a17e-e1be15fcb869
println(_v=>_b′)

# ╔═╡ 9ff26f05-446b-4129-8384-f40995344f90
obs_model = Dict(
	ISIS => [1.0, 0.0, 0.0, 0.0, 0.0],
	VulAfghan => [2.3196697811649253e-6, 0.999997680330219, 0.0, 0.0, 0.0],
	P1P2Afghan => [1.7752779331835457e-7, 0.07327466445117516, 0.9267251580210316, 0.0, 0.0],
	SIV => [6.700266384434589e-7, 0.04977967513574174, 0.14726958185760544, 0.8029500729800143, 0.0],
	AMCIT=>[0.0, 0.03703520238186705, 0.24652344950650082, 0.05430733869570854, 0.6621340094159236]
)

# ╔═╡ 4c72e6e4-095a-40f5-aa82-a806fa46bdc7
𝐫 = [reward(mdp_roa, MDPState(_s.c, _s.t, _s.f, v), _a) for v in mdp_roa.params.visa_status]

# ╔═╡ 5d0a4171-818e-406b-a75b-2f7628565072
dot(obs_model[_s.v], 𝐫)

# ╔═╡ a6c04cff-af51-4049-ab4b-c2191ef3602e
reward(mdp_roa, _s, _a; as_obs=true)

# ╔═╡ 91f9a32a-ab53-4b9a-909e-72932dce8929
reward(mdp_roa, _s, _a; as_obs=false)

# ╔═╡ ec5fda31-d346-4129-a117-7b3e10488a69
output_latex = true

# ╔═╡ cb835620-b15c-49e5-9e7e-d20f5c1c0522
colors = ["#FF48CF", "#8770FE", RGB([0,114,178]/255...), RGB([86,180,233]/255...), RGB([86,180,233]/255...), RGB([0,158,115]/255...), RGB([230,159,0]/255...), "#F5615C", "#300A24"]

# ╔═╡ 3196cc4a-79dd-4bd4-8860-86b992578174
begin
	use_latex_accepted = true
	use_latex_accepted ? pgfplotsx() : gr()
    plot_accepted(level1data["AMCITs"]["list_total_accepted_people"], "AMCITs", colors[5]; hold=false)
    plot_accepted(level1data["SIV-AMCITs"]["list_total_accepted_people"], "SIV-AMCITs", colors[6])
    plot_accepted(level1data["SIV-AMCITs-P1P2"]["list_total_accepted_people"], "SIV-AMCITs-P1P2", colors[7])
    plot_accepted(level1data["Level I"]["list_total_accepted_people"], "Level I", colors[1])
    plot!(xlabel="total accepted into airport", ylabel="density", legend=:top)
	p_accepted = ylims!(0, ylims()[2])
	gr()
	p_accepted
end

# ╔═╡ 075db7e6-48cc-48cc-881e-8786f8f71080
use_latex_accepted && savefig(p_accepted, "accepted_distr.pdf")

# ╔═╡ ffc4a077-f50a-4bb9-ab0f-eca86e989f1f
begin
	hα = 0.5
	nz = true
	histogram(level1data["AMCITs"]["list_total_accepted_people"], label="AMCITs", alpha=hα, c=colors[5], normalize=nz)
	histogram!(level1data["SIV-AMCITs"]["list_total_accepted_people"], label="SIV-AMCITs", alpha=hα, c=colors[6], normalize=nz)
	histogram!(level1data["SIV-AMCITs-P1P2"]["list_total_accepted_people"], label="SIV-AMCITs-P1P2", alpha=hα, c=colors[7], normalize=nz)
	histogram!(level1data["Level I"]["list_total_accepted_people"], label="Level I", alpha=hα, c=colors[1], normalize=nz)
	# histogram!(level2bdata_roa["Level I"]["list_total_accepted_people"], label="Level IIb", alpha = 0.5, normalize=nz)
	plot!(xlabel="total accepted into airport", ylabel="density", legend=:top)
	ylims!(0, ylims()[2])
end

# ╔═╡ 8d98f5fc-9f8f-4cb2-a791-3e1a987ed56e
function annotate_policy!(k, x, y)
	pos = :left
	if k == "Level I"
		x, y = (x, y)
	elseif k == "Level IIa"
		y = y - 50
		k = "Level IIa (approx.)"
	elseif k == "Level IIb VI"
		x, y = (x, y+300)
		k = "Level IIb"
	elseif k == "Level III"
		x, y = (x, y)
		k = "Level III (approx.)"
	elseif k == "AfterThresholdAMCITs"
		x, y = (x, y+300)
	elseif k == "Random"
		x, y = (x, y)
	elseif k == "BeforeThresholdAMCITs"
		x, y = (x, y)
	elseif k == "SIV-AMCITs-P1P2"
		x, y = (x, y)
	elseif k == "SIV-AMCITs"
		x, y = (x, y-50)
	else
		x, y = x < 120 ? (x-1, y+20) : (x, y)
	end
	annotate!([(x+15, y, (k, 7, pos))])
	return k
end

# ╔═╡ b2f56210-7de8-4cf2-9f54-1982dde001d8
# level2adata = experiments(1000, mdp_level2a, pomcp_policy_2a, "Level IIa")

# ╔═╡ b7c4ae6e-148a-4bcc-8bf1-8c0b01ef54d2
# level2bdata = experiments(1000, pomdp_level2b, pomcp_policy_2b, "Level IIb")

# ╔═╡ d0407ec6-3566-409c-a53a-7b9e0501c4ad
# level3data = experiments(1000, pomdp_level3, pomcp_policy_3, "Level III")

# ╔═╡ 8e64b232-c2d6-49c7-aa7b-1f1d83d1c8b7
# level3data_despot = experiments(10, pomdp_level3, despot_policy_3, "Level III (DESPOT)")

# ╔═╡ c0ef8c17-bf5d-4a77-bcdb-30d43721c18d
level3data = BSON.load("saved_data/level3data.bson", @__MODULE__)[:level3data]
# BSON.@save "level3data.bson" level3data

# ╔═╡ e24fb800-57fb-4a7e-8c48-b349b661ae93
level2adata = BSON.load("saved_data/level2adata.bson", @__MODULE__)[:level2adata]
# BSON.@save "level2adata.bson" level2adata

# ╔═╡ 595dc813-048c-415b-a98d-d943f8db9328
# BSON.@save "level2bdata.bson" level2bdata

# ╔═╡ 118ab216-0548-4ec4-b933-cedf78f08c55
# BSON.@save "level1data.bson" level1data

# ╔═╡ a17792e3-d4fb-498c-bf8b-d25f9e9efc5c
mean_std(map(sum, level1data["Level I"]["list_reward_over_time"]), false)

# ╔═╡ 10ed9198-8dd3-43e1-b311-d200fc934649
mean_std(map(sum, level2adata["Level IIa"]["list_reward_over_time"]), false)

# ╔═╡ 9f7b2cf0-9765-4846-b510-4cd084a4b760
mean_std(map(sum, level2bdata["Level IIb"]["list_reward_over_time"]), false)

# ╔═╡ bfa95d79-673d-4696-a786-b07ac811919e
mean_std(map(sum, level3data["Level III"]["list_reward_over_time"]), false)

# ╔═╡ c1afbd51-680c-4ebe-9f1e-1a1782feae8b
function filltomatch(A)
	A = deepcopy(A)
    max_length = maximum(map(length, A))    
    for i in 1:length(A)
        len = length(A[i])
        if len < max_length
            # fill last element to `max_length`
			filled_value = 0 # A[i][end]
            A[i] = vcat(A[i], fill(filled_value, max_length-len))
        end
    end
    return A
end

# ╔═╡ 1acce726-dd5b-4bb2-8e82-3603bc054244
filltomatch([[1,2,3], [1,1,1,2]])

# ╔═╡ dd15d7f4-a163-44d4-8f8f-7915a78ed4f6
function mismatch_mean(A)
    max_length = maximum(map(length, A))
    Z = [map(a->i <= length(a) ? a[i] : nothing, A) for i in 1:max_length]
    return map(mean, map(z->filter(!isnothing, z), Z))
end

# ╔═╡ af739bac-2f7d-4c5e-93d4-2b1377509239
begin
	# plot = EvacuationPOMDP.plot
	# plot! = EvacuationPOMDP.plot!
	if output_latex
		pgfplotsx()
	else
		gr()
	end
	plot()
	hline!([0], c=:gray, lw=1, label=false)
	vline!([1200-200+1], c=:gray, lw=0.5, label=false, style=:dash)
	annotate!([(1200-220, 800, ("Threshold (AMCITs)", 7, :right, :gray))])
	policy_keys = ["Level I", "Level IIa", "Level IIb VI", "Level III", "AMCITs", "SIV-AMCITs",  "SIV-AMCITs-P1P2", "AcceptAll", "AfterThresholdAMCITs", "BeforeThresholdAMCITs", "Random"]
	# policy_keys = ["Level I", "Level IIa", "Level IIb", "Level IIb VI", "Level III", "AMCITs", "SIV-AMCITs",  "SIV-AMCITs-P1P2", "AcceptAll", "AfterThresholdAMCITs", "BeforeThresholdAMCITs", "Random"]
	# policy_keys = ["Level I", "Level IIb VI", "AMCITs", "SIV-AMCITs",  "SIV-AMCITs-P1P2", "AcceptAll", "AfterThresholdAMCITs", "BeforeThresholdAMCITs", "Random"]
	# policy_keys = ["Level I", "Level IIa", "Level IIb", "Level III"]

	# colors = [:magenta, :crimson, :red, :darkorange, :gold, :lightgreen, :green, :cyan, :blue, :purple, :black]
	for (i,k) in enumerate(policy_keys)
		style = occursin("Threshold", k) ? :dash : :solid
		style = k == "Random" ? :dot : style
		if k in keys(level1data)
			data = level1data[k]
		elseif k in keys(level2adata)
			data = level2adata[k]
		# elseif k in keys(level2bdata)
			# data = level2bdata[k]
		elseif k in keys(level3data)
			data = level3data[k]
		elseif k in keys(level2bdata_roa)
			data = level2bdata_roa[k]
		# elseif k in keys(level3data_despot)
		# 	data = level3data_despot[k]
		end
		μ_cumulative_reward = cumsum(mismatch_mean(filltomatch(data["list_reward_over_time"])))
		# σ_cumulative_reward = sqrt.(cumsum(mismatch_std(data["list_reward_over_time"])))
		color = i > length(colors) ? colors[i%length(colors)] : colors[i]
		k = annotate_policy!(k, length(μ_cumulative_reward), last(μ_cumulative_reward))
		plot!(μ_cumulative_reward,
			  # ribbon=σ_cumulative_reward,
			  fillalpha=0.5, lw=1, label=k, style=style, c=color)
		scatter!([length(μ_cumulative_reward)], [last(μ_cumulative_reward)], label=false, c=color, ms=3)
		# @show k, last(μ_cumulative_reward)
	end

	plot!(xlabel="simulation time", ylabel="mean cumulative reward", legend=:topleft, title="cumulative reward for each policy")
	ylims!(ylims()[1], ylims()[2]+500)
	agg_plot = xlims!(1, xlims()[2]+400)
	gr()
	agg_plot
end

# ╔═╡ 6d708221-9d27-4e14-a621-420128f2faa3
output_latex && savefig(agg_plot, "agg_plot.tex")

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

# ╔═╡ dca9db35-9c15-4289-bde9-227f7acefec4
begin
	# pgfplotsx()
	gr()
	policy_plot_roa = vis_all(mdp.params, mdp_roa_policy)
	# gr()
	savefig(policy_plot_roa, "policy_plot_roa.pdf")
	policy_plot_roa
end

# ╔═╡ 58c11d04-911c-424c-9b4d-624929eb7c92
value(mdp_policy, MDPState(500, 600, 13, P1P2Afghan), REJECT)

# ╔═╡ 10887c56-a0dc-4157-bc90-0a6bdb4322c2
value(mdp_policy, MDPState(500, 600, 13, P1P2Afghan), ACCEPT)

# ╔═╡ d6b08f6b-a8d5-4853-95d2-587c2f5b0ec5
action(mdp_policy, EvacuationPOMDP.newstate(MDPState, 1, 120, 1, AMCIT))

# ╔═╡ 34b70a62-622e-41e3-a865-7cbd8b6a8fb5
savefig(policy_plot, "policy_plot.pdf")

# ╔═╡ 57c7d1f6-9920-464d-8b14-563fccd6878f
# vis_all(mdp.params, pomcp_policy) # requires action(policy, belief) NOTE `belief`.

# ╔═╡ 7691c8ad-41b8-4fc2-ad77-652794b61092
-sqrt(120-120)

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

# ╔═╡ 27bd873f-c7b1-4323-99e3-6f6be02eb8b5
pomdp_level3 = EvacuationPOMDPType()

# ╔═╡ 3b468bb0-9e90-40b0-9a24-60fba6ca4fc5
fam_distr = plot_family_size_distribution(pomdp_level3.params.family_prob)

# ╔═╡ f9b11f1e-04b9-488c-9884-ab0bb77a945c
savefig(fam_distr, "fam_distr.pdf")

# ╔═╡ d112f66b-bd60-480c-9fc3-275b90206e6c
transitionhidden(pomdp_level3, HiddenState(AMCIT), ACCEPT)

# ╔═╡ 7a301ad6-e31f-4d20-b527-a26573473c0e
begin
	Random.seed!(1)
	simulation(pomdp_level3, pomcp_policy_3)
end

# ╔═╡ 40c84bf1-88c8-47d3-9a5b-6f9308d883e8
observation(pomdp_level3, sh_test, ACCEPT, shp_test)

# ╔═╡ 6fe49d6b-343a-4250-8744-a6ba734e51d0
# pomcp_policy_2a = solve_pomdp(POMCPSolver, mdp_level2a, mdp_policy);

# ╔═╡ fe0b9861-f14a-4ecd-bc3a-8cbf70badb97
# pomcp_policy_2b = solve_pomdp(POMCPSolver, pomdp_level2b);

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
# POMCP Solver
"""

# ╔═╡ 80aef6fb-6c23-41a9-a0e1-76d7a04c503a
function solve_pomdp(::Type{POMCPSolver}, pomdp::EvacuationPOMDPType, policy)
	# rollout_estimator = FORollout(SIVAMCITsP1P2Policy())
	rollout_estimator = BasicPOMCP.SolvedFOValue(MDPValueEstimator(policy))

	pomcp_solver = POMCPSolver(
		max_depth=120,
		tree_queries=500,
		estimate_value=rollout_estimator)
	
	pomcp_policy = solve(pomcp_solver, pomdp)
	return pomcp_policy
end

# ╔═╡ c9e3d5ee-f0d1-4bd3-9d9c-4ddeab2e87d3
# pomcp_policy_3 = solve_pomdp(POMCPSolver, pomdp_level3, mdp_roa_policy)

# ╔═╡ 5684d8f8-49e4-406b-9a98-639450302155
md"""
# DESPOT
"""

# ╔═╡ 91c6299b-0b15-46b1-a890-4e028802dd28
function solve_pomdp_depot(::Type{DESPOTSolver}, pomdp::EvacuationPOMDPType)
	solver = DESPOTSolver(bounds=(0.0, 1300.0))
	policy = solve(solver, pomdp)
	return policy
end

# ╔═╡ 50e5825b-e29c-4b9f-a034-c6bb751097b0
despot_policy_3 = solve_pomdp_depot(DESPOTSolver, pomdp_level3)

# ╔═╡ fdb3db7f-7784-477f-ace2-b65df9031b41
md"""
## Individual beliefs
"""

# ╔═╡ 33beafb5-1fd2-4e0e-892b-1b5b9d2e0a77
up = DiscreteSubspaceUpdater(pomdp_level3)

# ╔═╡ c1295efc-cc37-4547-b948-005ce982b264
up

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

# ╔═╡ 582aa868-aecb-419f-9440-92e01a155081
begin
	Random.seed!(0)
	online_action_despot, info_despot = action_info(despot_policy_3, b₀)
	online_action_despot
end

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
plot_claims(pomdp_level3.params.visa_true_prob; text="Visa probability distribution")

# ╔═╡ 56d742f1-4fb7-4afe-8674-f343d6672364
pomdp_level3.params.visa_true_prob

# ╔═╡ 9f2b615d-4187-424f-adb5-f092c78faec5
plot_claims_tiny(pomdp_level3.params.visa_true_prob)

# ╔═╡ f429f2b4-959b-4ed2-bd49-6ba961ba2382
md"""
# Simulations
"""

# ╔═╡ 38b970ed-0fc2-44d4-bf0d-4ad5fa70d1ff
traj_seed = 2 # 3

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
	level2b_trajectory = simulate_trajectory(mdp_roa, mdp_roa_policy; 
											   seed=traj_seed)
end

# ╔═╡ ff841f9f-0603-4d05-a820-736a4cc62e3d
begin
	Random.seed!(traj_seed)
	level3_trajectory = simulate_trajectory(pomdp_level3, pomcp_policy_3; 
											   seed=traj_seed)
end

# ╔═╡ 3b3110a8-8322-404f-8ad9-f4361640666e
[plot_claims_tiny(τ[end-2].b) for τ in level3_trajectory]

# ╔═╡ 9c10c8d3-71a7-422b-b24e-8e2cb2997cca
map(t->t[end-2].b, level3_trajectory[end-2:end])

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
plot_trajectory(mdp_roa, level2b_trajectory, "traj_level2b"; N=N, show_population=false, show_belief=true)

# ╔═╡ edabc978-aaec-4a09-a234-4fa9caf2b58e
plot_trajectory(pomdp_level3, level3_trajectory, "traj_level3"; N=N, show_population=true, show_belief=true)

# ╔═╡ db20385a-259d-4ed7-b9c6-d29b7156b4cf
md"""
# Prior
"""

# ╔═╡ 9702e989-412b-4f43-bedd-032b587fe6f4
prior = Categorical(EvacuationParameters().visa_prior)

# ╔═╡ a87b3fe7-cbd3-4441-94db-2fae7922965b
EvacuationParameters().visa_true_count / 100_000

# ╔═╡ 74571c80-ee73-485a-8373-63ae21d77e1f
begin
	gr()
	plot_claims(prior.p)
	plot!(size=(400,200), xtickfont=(6), bottom_margin=2Plots.mm)
	ylims!(0 ,0.7)
	title!("(prior) population distribution")
	xlabel!("priority status")
	prior_pop_distr_plt = ylabel!("probability")
end

# ╔═╡ f781c360-a79b-4ff8-8b59-3c0666bb495a
md"""
# Truth (population)
"""

# ╔═╡ 1aa59e06-3213-4d77-b34f-3d1f49fade88
normalize(pomdp_level3.params.visa_true_count, 1)

# ╔═╡ 5b6381d7-4030-47c9-b170-3706681b77a2
begin
	gr()
	plot_claims(normalize(pomdp_level3.params.visa_true_count, 1))
	plot!(size=(400,200), xtickfont=(6), bottom_margin=2Plots.mm)
	ylims!(0 ,0.7)
	title!("population distribution")
	xlabel!("priority status")
	pop_distr_plt = ylabel!("probability")
end

# ╔═╡ 4228f140-2451-4024-9311-97ee3ff5cb92
# savefig(pop_distr_plt, "pop_distr.pdf")

# ╔═╡ 78e98d84-a4bd-4ba7-8348-c05c9f2905c7
dir = Dirichlet(EvacuationParameters().visa_prior_count)

# ╔═╡ 8956ea71-04a2-4d04-b097-fe1745930213
rand(dir) |> plot_claims

# ╔═╡ 89198a61-00c5-415b-bb89-8316291d73f2

begin
	gr()
	plot_claims(mean(dir); yerr=reshape(sqrt.(var(dir)), (1,5)))
	plot!(size=(400,200), xtickfont=(6), bottom_margin=2Plots.mm)
	ylims!(-0.0010, 0.75)
	xlabel!("priority status")
	est_pop_distr_plt = ylabel!("probability")
end

# ╔═╡ 874ee0c0-1152-4e4a-8bca-1c48ca95c0a2
savefig(est_pop_distr_plt, "pop_distr.pdf")

# ╔═╡ 391328c0-7374-4737-a31c-27c5a852a3e8
md"""
# Generate JSON
"""

# ╔═╡ e718c3a3-1469-4fee-b5f9-b15127a9fc5d
EvacuationPOMDP.generate_population_trajectories()

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
# ╠═7aea018d-d9bb-44b4-9f16-e6def5ffc249
# ╠═06bdc30d-7f1a-4b4e-86a1-76c6030f75e9
# ╠═4bfee8de-cc8d-436d-98ad-5f7002ece4cf
# ╠═70dffecd-3db5-45a9-8104-d82f30fdead2
# ╠═c9d38947-173d-4baf-9b3f-81bacf4d16cc
# ╟─fbd975c5-fc0b-43bd-86e0-7ed9c4efc0a5
# ╠═1e180b5a-34e4-4509-8eba-76445fbbc9ee
# ╠═6816dd65-f198-40f9-bca9-eef180781fb1
# ╠═2d58015e-1e3d-4000-ae68-5d5222fa8101
# ╠═75b0e162-3aee-45ef-91c5-5a8fbfcd1d18
# ╠═9e6cfa03-a98c-4388-a952-7f498c826a75
# ╠═5bb11a5a-f9c8-4bec-9f08-c6b00a9a322d
# ╠═a0170fcd-7438-4d08-8f75-dfa43249fb6c
# ╠═2a57deca-5a25-4d3d-a759-74f1f5e136ff
# ╠═913ae6dc-8012-48a6-8bbc-5195b247a0a9
# ╠═3196cc4a-79dd-4bd4-8860-86b992578174
# ╠═075db7e6-48cc-48cc-881e-8786f8f71080
# ╠═ffc4a077-f50a-4bb9-ab0f-eca86e989f1f
# ╠═3759dd49-0635-454c-8f4d-6f90890fabf8
# ╠═694d7f4d-dc06-4501-9ce1-c89c525d32ba
# ╠═2d182258-1fa6-48b1-ba45-ee346f9e873e
# ╠═50093ef5-1f7a-4a4b-8b7e-49ef14a44261
# ╠═c4738481-fc09-41af-9862-73799436226a
# ╠═6c8c7500-3ac1-4d1b-b148-90b31b455012
# ╟─277ac05c-bc57-4dd0-863a-7f5fb284e825
# ╠═bb3c370c-54d3-4abd-80fd-2144faf70672
# ╠═d2afed13-a379-4f87-8f4f-92bd1407c470
# ╠═c5944f50-8d0d-48a5-b0af-65eb69401e1d
# ╠═6029a8c1-48ab-433e-b15a-8cf7f693e5d0
# ╠═88bdff09-8091-4ed4-a6da-da3b9311c3af
# ╠═ef19a1d3-0fe2-45ef-bfc2-6768e6dfb2a3
# ╠═aa6d9555-a398-4b8a-94db-32a4a0f68128
# ╠═0c7b6ab7-aafe-44c4-a992-5d003029ac51
# ╠═0bf669b4-7e12-4ef1-986c-4bcde78169ba
# ╠═3e9e9958-d6a4-461e-90ab-346c446eaffe
# ╠═0ac99c53-b60c-42f5-bc96-e2ae9d260b83
# ╠═99cecdb4-1598-4156-b77d-1e90d86be052
# ╠═92ea23d4-5200-4d02-9d38-9c968f3b0e20
# ╠═6d99cfd2-db78-4645-9f4b-b1d2922787fe
# ╠═67c0c993-8e53-4a56-a17e-e1be15fcb869
# ╠═9ff26f05-446b-4129-8384-f40995344f90
# ╠═4c72e6e4-095a-40f5-aa82-a806fa46bdc7
# ╠═5d0a4171-818e-406b-a75b-2f7628565072
# ╠═a6c04cff-af51-4049-ab4b-c2191ef3602e
# ╠═91f9a32a-ab53-4b9a-909e-72932dce8929
# ╠═c1295efc-cc37-4547-b948-005ce982b264
# ╠═ec5fda31-d346-4129-a117-7b3e10488a69
# ╠═cb835620-b15c-49e5-9e7e-d20f5c1c0522
# ╠═af739bac-2f7d-4c5e-93d4-2b1377509239
# ╠═6d708221-9d27-4e14-a621-420128f2faa3
# ╠═1acce726-dd5b-4bb2-8e82-3603bc054244
# ╠═8d98f5fc-9f8f-4cb2-a791-3e1a987ed56e
# ╠═b2f56210-7de8-4cf2-9f54-1982dde001d8
# ╠═b7c4ae6e-148a-4bcc-8bf1-8c0b01ef54d2
# ╠═d0407ec6-3566-409c-a53a-7b9e0501c4ad
# ╠═8e64b232-c2d6-49c7-aa7b-1f1d83d1c8b7
# ╠═c0ef8c17-bf5d-4a77-bcdb-30d43721c18d
# ╠═e24fb800-57fb-4a7e-8c48-b349b661ae93
# ╠═595dc813-048c-415b-a98d-d943f8db9328
# ╠═118ab216-0548-4ec4-b933-cedf78f08c55
# ╠═a17792e3-d4fb-498c-bf8b-d25f9e9efc5c
# ╠═10ed9198-8dd3-43e1-b311-d200fc934649
# ╠═9f7b2cf0-9765-4846-b510-4cd084a4b760
# ╠═bfa95d79-673d-4696-a786-b07ac811919e
# ╠═c1afbd51-680c-4ebe-9f1e-1a1782feae8b
# ╠═dd15d7f4-a163-44d4-8f8f-7915a78ed4f6
# ╠═8d2b9590-8d76-4675-9b8a-e6a47cbccb8c
# ╟─2b2ed9bd-e033-47ea-b24b-4789f28ab08c
# ╠═0fe49a8e-4ee5-41b8-8802-8e61d997b252
# ╟─d1646d70-426d-4545-ba31-25a7712fe852
# ╠═9babaea3-5ed9-4349-ba5b-95f9549213eb
# ╠═29798743-18cc-4c7b-af0d-ffb962a3eec9
# ╠═67d3b29c-4aeb-45df-a449-cd669590fb58
# ╠═59c54092-e44e-422c-a29d-3fc457718190
# ╟─470e32ce-57da-48b6-a27e-0899083a47a3
# ╠═40430b14-07b2-40c6-95fe-9f60a5c6e75f
# ╠═dca9db35-9c15-4289-bde9-227f7acefec4
# ╠═58c11d04-911c-424c-9b4d-624929eb7c92
# ╠═10887c56-a0dc-4157-bc90-0a6bdb4322c2
# ╠═d6b08f6b-a8d5-4853-95d2-587c2f5b0ec5
# ╠═34b70a62-622e-41e3-a865-7cbd8b6a8fb5
# ╠═57c7d1f6-9920-464d-8b14-563fccd6878f
# ╠═7691c8ad-41b8-4fc2-ad77-652794b61092
# ╟─958c44ef-9c64-484c-ae80-0a62f2142225
# ╠═d112f66b-bd60-480c-9fc3-275b90206e6c
# ╟─4a747878-429e-455f-8894-a92c22be4dcf
# ╠═7a301ad6-e31f-4d20-b527-a26573473c0e
# ╠═56c51f5d-f632-4afe-8e10-32ef28160f48
# ╠═40c84bf1-88c8-47d3-9a5b-6f9308d883e8
# ╠═fded8908-5b32-4e81-9ef1-0fc4349303c9
# ╠═4efe4359-7041-45c9-bedf-939a41954831
# ╟─f66f2579-d2d4-4a5b-9805-4922dbb99e9b
# ╠═4cb1fb17-523e-4eab-a9e7-7e1bf0b9edde
# ╠═06167243-febb-4c48-b088-efabacff64f5
# ╠═27bd873f-c7b1-4323-99e3-6f6be02eb8b5
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
# ╟─5684d8f8-49e4-406b-9a98-639450302155
# ╠═1dae6254-f47b-499a-af1a-ac6524836acc
# ╠═91c6299b-0b15-46b1-a890-4e028802dd28
# ╠═50e5825b-e29c-4b9f-a034-c6bb751097b0
# ╠═582aa868-aecb-419f-9440-92e01a155081
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
# ╟─db20385a-259d-4ed7-b9c6-d29b7156b4cf
# ╠═9702e989-412b-4f43-bedd-032b587fe6f4
# ╠═a87b3fe7-cbd3-4441-94db-2fae7922965b
# ╠═74571c80-ee73-485a-8373-63ae21d77e1f
# ╟─f781c360-a79b-4ff8-8b59-3c0666bb495a
# ╠═1aa59e06-3213-4d77-b34f-3d1f49fade88
# ╠═5b6381d7-4030-47c9-b170-3706681b77a2
# ╠═4228f140-2451-4024-9311-97ee3ff5cb92
# ╠═78e98d84-a4bd-4ba7-8348-c05c9f2905c7
# ╠═8956ea71-04a2-4d04-b097-fe1745930213
# ╠═89198a61-00c5-415b-bb89-8316291d73f2
# ╠═874ee0c0-1152-4e4a-8bca-1c48ca95c0a2
# ╟─391328c0-7374-4737-a31c-27c5a852a3e8
# ╠═e718c3a3-1469-4fee-b5f9-b15127a9fc5d
