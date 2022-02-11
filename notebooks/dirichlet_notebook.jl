### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ 4624989c-ac25-469f-89b1-a151c3a70148
begin
	using Pkg
	Pkg.develop(path="..//MOMDPs.jl//")
	using PlutoUI
	using POMDPModels
	using POMDPs
	using POMDPModelTools
	using Distributions
	using StatsBase

	using Revise
	using BasicPOMCP
	using POMDPPolicies
	using POMDPSimulators
	using MOMDPs
end

# ╔═╡ f23f5c21-d5e4-4878-b85e-356a7619cf0e
using DirichletBeliefs

# ╔═╡ 5745eb4a-8688-4cc5-a262-d86e60d2ebcf
using D3Trees

# ╔═╡ ae451a8c-e790-4b1d-b853-4c592179d9d0
using LinearAlgebra

# ╔═╡ 0c6fda44-14b0-4cb6-b0b9-026752fd2bb7
using Random

# ╔═╡ 2104ce98-ebb0-4258-aeb0-bd48a20dd5fc
using Statistics

# ╔═╡ d1c230d9-674d-4a4b-8c50-13a9bae3c499
using Plots; default(fontfamily="Computer Modern", framestyle=:box)

# ╔═╡ 683dff90-85d3-11ec-2531-05d793550053
md"""
# Dirichlet Belief
"""

# ╔═╡ 7e599ef3-66d4-423e-b213-70acdd9eb7ed
TableOfContents()

# ╔═╡ c392e215-afdd-46ed-b175-1407ff4c9669
D = Dirichlet([1,1])

# ╔═╡ 9f705704-b91e-4b12-8bc8-0a95a093f9ec
md"""
# Abstract typing
"""

# ╔═╡ 3c3a77dc-eb77-4ae9-9c64-88b20d8cbb8a
MOMDP <: POMDP

# ╔═╡ d9bedb8a-927c-454d-bd97-00899e6b5983
md"""
> **TODO**: Merge `MOMDPs` into `POMDPs` so we can call things like `POMDPs.stateshidden` instead of `MOMDPs.stateshidden` and `POMDPs.observations` will not be confused with `MOMDPs.observations`.
"""

# ╔═╡ e23add73-3c18-4d81-ba3c-4f5472fdde42
md"""
# Load `Belief` and `MOMDP`
"""

# ╔═╡ 75c999a2-da6b-40de-a078-901d8d537a47
md"""
# Test POMDP
"""

# ╔═╡ ef423c3a-eae8-4211-b4a4-b3cf4b27d7a8
abstract type TestPOMDP end

# ╔═╡ 8b5d6571-556c-44fb-b0dd-07d8484c2c88
pomdp = RandomPOMDP(5, 2, 5, 0.95);

# ╔═╡ e5d17ad5-10ce-425b-ab8d-962eac5d7f2b
function POMDPs.transition(pomdp::TabularPOMDP, s::Int, a::Int)
	states = ordered_states(pomdp)
	p = zeros(length(states))
	p[s] = 1
	return SparseCat(states, p)
end

# ╔═╡ 05b01eb8-ecee-4f6a-a1aa-305919ef3b44
function POMDPs.observation(pomdp::TabularPOMDP, a::Int, sp::Int)
	obs = ordered_observations(pomdp)
	p = zeros(length(obs))
	p[sp] = 1
	return SparseCat(obs, p)
end

# ╔═╡ 4b3b8e56-c5ad-4f2d-a95b-2930f0d7cb72
# 𝒮 = ordered_states(pomdp)

# ╔═╡ 28293463-3016-4495-a446-bcadb2691e07
# 𝒜 = ordered_actions(pomdp)

# ╔═╡ 60130742-e884-4e59-b36d-9a95c22f336e
# 𝒪 = ordered_observations(pomdp)

# ╔═╡ 431f3131-c1b7-4dbd-91d4-9980366d23cb
md"""
## True distribution
"""

# ╔═╡ 6bf232ba-fa87-4522-87bb-1682975e33b1
D_true = Categorical([0.01, 0.50, 0.31, 0.10, 0.08]);

# ╔═╡ f3f4f965-006b-40f5-a6b6-b43581a4044a
md"""
## Belief updater
"""

# ╔═╡ c3febe96-0247-46a8-a1e1-ead068d799a2
b = DirichletBelief(pomdp);

# ╔═╡ 762a255f-957b-43d9-8bd8-40feb9f779a6
up = DirichletUpdater(pomdp);

# ╔═╡ 22564b5c-31d8-4da2-a9c8-3151ebc209a0
md"""
## Belief updating (simulated)
"""

# ╔═╡ 5ddb7ebf-b202-4fe8-92e7-f6528018ef1e
p_prior = [0.01, 0.50, 0.29, 0.09999, 0.10001];

# ╔═╡ 03a30c5a-9d8d-46a8-be7a-ce1d7e5bb403
D_prior = Categorical(p_prior);

# ╔═╡ e02fa4e0-7caf-4363-8006-6538174dbfba
begin
	act = 1
	b′ = D_prior # deepcopy(b)
	for t in 1:10
		o = rand(D_true)
		b′ = update(up, b′, act, o)
	end
	[mean(b′) b′.b.alpha]
end

# ╔═╡ 405a7108-2434-414b-bbc3-89056b9f2569
md"""
## Example belief updating
"""

# ╔═╡ b01e8019-7b16-489a-b97e-574df652ff24
[mean(b) b.b.alpha]

# ╔═╡ 3257e67d-9975-4f43-8ea4-128b1facd67c
b1 = update(up, b, 1, 1);

# ╔═╡ 0847d794-ade9-40ff-8ee5-0326deb76c43
[mean(b1) b1.b.alpha]

# ╔═╡ 3cece97a-72d3-4bdd-8ca9-290752c74291
b2 = update(up, b1, 1, 1);

# ╔═╡ 9d3b3a03-0620-407d-9e01-64fcd122819b
[mean(b2) b2.b.alpha]

# ╔═╡ 7e58da87-d712-4d64-b58e-5238baa6e557
b3 = update(up, b2, 1, 1);

# ╔═╡ 85802425-17ab-4d1e-bf83-1b28a250d235
[mean(b3) b3.b.alpha]

# ╔═╡ 8c354e0e-18d1-40f2-9da8-3b14c862e039
md"""
## Belief initialization (debug)
"""

# ╔═╡ 697e6d70-ea9e-414e-b6b7-0385e40c80bb
round.(Int, p_prior * 100)

# ╔═╡ a2ce04e4-57d9-44c5-a3c0-b49d4ed7c503
pdf(D_prior, 4)

# ╔═╡ 98ba26bc-f671-4e6c-9ce9-376c6445ef47
belief = DirichletBelief(pomdp);

# ╔═╡ 598a23bf-37b1-4c17-a428-776e3123e681
belief.b

# ╔═╡ ac5cbb0d-6cd9-47fd-a8bc-fc2498eb52e9
initialize_belief(up, D_prior).b

# ╔═╡ e5dc63d1-6ee0-46b8-9506-3f739eeae4a5
md"""
# MOMDP: Mixed Observability MDP
- [S.C.W. Ong, S.W. Png, D. Hsu, and W.S. Lee, "POMDPs for Robotic Tasks with Mixed Observability", _Robotics: Science and Systems (RSS)_, 2009.](http://www.roboticsproceedings.org/rss05/p26.pdf)
"""

# ╔═╡ 95c3541f-5b3e-4adf-8b78-c434d27ad963
md"""
## States
"""

# ╔═╡ 4bdca411-6781-49a4-a33a-1462967694a0
struct VisibleState
	c::Int
	t::Int
	f::Int
end

# ╔═╡ 10057440-2e41-47fa-836e-ad2627d2e4a4
@enum VisaStatus AMCIT SIV OTHER

# ╔═╡ 4f2b1350-100e-43fa-bb8e-ddad9d9bff93
struct HiddenState
	v::VisaStatus
end

# ╔═╡ 76859546-8c03-4b95-90d2-2fe4a30ebc02
const State = Tuple{VisibleState, HiddenState}

# ╔═╡ 9eb261f6-6385-4098-af11-6b6ee9a4d155
𝒮ᵥ = [VisibleState(c,t,f)
		for c in 0:20
			for t in 0:20
				for f in 1:8];

# ╔═╡ b1fe8c15-8b78-4d95-ab65-cd04a15a2887
𝒮ₕ = [HiddenState(v) for v in (AMCIT, SIV, OTHER)];

# ╔═╡ 2ff812d0-75ca-470d-af54-208dd086a1b4
𝒮::Vector{State} = [(v,h) for h in 𝒮ₕ for v in 𝒮ᵥ]

# ╔═╡ e84ee454-fb02-4b52-b336-d0ed16f62b2c
s = (VisibleState(14, 8, 2), HiddenState(AMCIT)) # rand(𝒮)

# ╔═╡ a4989251-7986-41aa-9cce-64468fdff52d
sp = (VisibleState(13, 7, 2), HiddenState(SIV))

# ╔═╡ eb4f0ac0-a3f1-410c-9d50-e3e3bbbf107c
md"""
## Actions
"""

# ╔═╡ 1cd45201-d0dd-49d9-9e6c-eff1b8d1521c
@enum Action ACCEPT REJECT

# ╔═╡ 972075cb-5aa8-42c7-bbc2-ddb531bb547c
md"""
## Observations
"""

# ╔═╡ c620ba50-35ae-40d4-b69d-21984a883734
@enum VisaDocument AMCIT_DOC SIV_DOC OTHER_DOC

# ╔═╡ b29fdaee-006c-4bf4-a5ff-c490d3cef4e1
documentation = (AMCIT_DOC, SIV_DOC, OTHER_DOC)

# ╔═╡ 69c5e487-117f-4120-a6c0-915f72278702
struct Observation
	c::Int
	t::Int
	f::Int
	v_doc::VisaDocument
end

# ╔═╡ 181fc70e-e8f0-4bd8-885d-491890c8259b
begin
	𝒪 = [Observation(c,t,f,v_doc)
			for c in 0:20
				for t in 0:20
					for f in 1:8
						for v_doc in documentation]
	length(𝒪)
end

# ╔═╡ d25623b9-1c35-4bfe-8266-5af07ea6101a
o = Observation(visible(s).c, visible(s).t, visible(s).f, AMCIT_DOC) # rand(𝒪)

# ╔═╡ e0366964-e4b4-44d7-96d5-cba974d43470
# 		oᵢₜₑᵣ = rand(𝒪)
# 		v_doc = documentation[rand(Dₘ_true)]
# 		o = Observation(oᵢₜₑᵣ.c, oᵢₜₑᵣ.t, oᵢₜₑᵣ.f, v_doc)

# ╔═╡ ae35a3ad-210e-421f-8506-591969b6df44
o.v_doc

# ╔═╡ 46835032-5378-4fe7-9966-e9354e0a64d1
# function POMDPs.obs_weight(momdp::TestMOMDP, s, a, sp, o::Observation)
# 	hidden_obs = o.v_doc
# 	return pdf(observation(momdp, s, a, sp), hidden_obs)
# end

# ╔═╡ 66a7d93c-3dcd-4ff6-aeef-d478d49fadd0
md"""
## Transition(s)
"""

# ╔═╡ a2b3c5a8-506b-485b-9799-14fe56475689
(s,o)

# ╔═╡ d4616594-ea3c-4483-8db7-499dd2f29174
md"""
## Generative model
"""

# ╔═╡ 6382deb8-0bb6-438a-a2e8-98939fcf3634
md"""
## Reward
"""

# ╔═╡ f0819505-23cb-44bc-b3c4-7db8dc9ac994
function POMDPs.reward(m::MOMDP, s::State, a::Action)
	rh = reward(m, hidden(s), a)
	rv = reward(m, visible(s), a)
	return rh*rv
end

# ╔═╡ f97a3c29-3598-401c-93fb-f1e19ff151ab
function POMDPs.reward(m::MOMDP, sh::HiddenState, a::Action)
	if sh.v == AMCIT
		return 500
	elseif sh.v == SIV
		return 2
	elseif sh.v == OTHER
		return -10
	end
end

# ╔═╡ 7598f6ab-73e0-45b9-b101-a3ddf6deb4e4
function POMDPs.reward(m::MOMDP, sv::VisibleState, a::Action)
	return sv.f
end

# ╔═╡ 12611914-ab90-42cb-a83c-4a2e90145203
md"""
## MOMDP formulation
"""

# ╔═╡ 3b336040-778d-498c-9ed8-89f0f69b67f9
struct TestMOMDP <: MOMDP{VisibleState, HiddenState, Action, Observation} end

# ╔═╡ 83d100df-4ecf-49f3-b6f4-ea14b6dc508b
MOMDPs.visiblestates(momdp::TestMOMDP) = 𝒮ᵥ

# ╔═╡ 36dbfce8-1c94-41b1-8748-8f5df0142178
MOMDPs.hiddenstates(momdp::TestMOMDP) = 𝒮ₕ

# ╔═╡ b903212f-d438-4a3e-b844-bd4a28015783
POMDPs.states(momdp::TestMOMDP) = 𝒮

# ╔═╡ b2f83903-c6e1-41cb-910f-6f38db222b66
POMDPs.actions(momdp::TestMOMDP) = [ACCEPT, REJECT]

# ╔═╡ aef35f4a-e00d-4d8b-b667-93c10761e2f6
POMDPs.observations(momdp::TestMOMDP) = 𝒪

# ╔═╡ fd4ca58a-33c8-471c-ae91-940906e6202c
function POMDPs.observation(momdp::TestMOMDP, s, a, sp)
	global documentation
	p = 0.05 * ones(length(documentation))
	sₕ_idx = hiddenstateindex(momdp, sp)
	p[sₕ_idx] = 1
	normalize!(p, 1)
	obs = [Observation(visible(s).c, visible(s).t, visible(s).f, v_doc)
		for v_doc in documentation]
	return SparseCat(obs, p)
end

# ╔═╡ 4b06ff50-1ab5-4352-8f13-bf58ed2b5ec2
function POMDPs.observation(momdp::TestMOMDP, sh::HiddenState)
	return observation(momdp, missing, missing, sh)	
end

# ╔═╡ a8f7c03e-02ea-48b1-a50d-fa24e62afd35
O = observation(pomdp, 1, 1)

# ╔═╡ 7934c669-23f0-45b1-aa50-2298bf9cbf16
function MOMDPs.transitionvisible(m::TestMOMDP, sv::VisibleState, a::Action, o=missing)
	visiblestates = ordered_visible_states(m)
	p = ones(length(visiblestates))
	sᵢ = visiblestateindex(m, sv)
	p[sᵢ] = 1
	normalize!(p, 1)
	return SparseCat(visiblestates, p)
	# return Distribution T(v′ | s, a)
end

# ╔═╡ 90c20c89-b7ad-4423-9b6e-440054c22959
function MOMDPs.transitionhidden(m::TestMOMDP, sh::HiddenState, a::Action, o=missing)
	hiddenstates = ordered_hidden_states(m)
	p = 0.05*ones(length(hiddenstates))
	sᵢ = hiddenstateindex(m, sh)
	p[sᵢ] = 1
	normalize!(p, 1)
	return SparseCat(hiddenstates, p)
	# return Distribution T(h′ | s, a, v′)
end

# ╔═╡ 8fe5b28b-5c91-4264-a4eb-4562d7d17753
function POMDPs.transition(m::TestMOMDP, sh::HiddenState, a::Action)
	return transitionhidden(m, sh, a) # TODO.
end

# ╔═╡ eea7efec-34e2-429f-a009-b7ff582304d1
function POMDPs.transition(m::TestMOMDP, sv::VisibleState, a::Action)
	return transitionvisible(m, sv, a) # TODO.
end

# ╔═╡ c58c2fae-d4fd-4349-85c0-e32839f40f08
function POMDPs.transition(m::TestMOMDP, s::State, a::Action)
	sv = visible(s)
	sh = hidden(s)
	t_hidden = transitionhidden(m, sh, a)
	next_states = []
	probs = []
	next_fam = rand(1:8)

	for (spₕ, tpₕ) in weighted_iterator(t_hidden)
		if a == ACCEPT
			next_state = (VisibleState(sv.c-1, sv.t-1, next_fam), spₕ)
		else
			next_state = (VisibleState(sv.c, sv.t-1, next_fam), spₕ)
		end
		push!(next_states, next_state)
		push!(probs, tpₕ)
	end

	return SparseCat(next_states, probs)	
end

# ╔═╡ b0b58647-0098-4557-b0a6-804d952ee5ae
T = transition(pomdp, 1, 1)

# ╔═╡ 1a4def74-873c-481c-8ffd-29f004bc0117
MOMDP

# ╔═╡ 21b8409a-a4f3-41a3-940a-bb067824aaef
momdp = TestMOMDP()

# ╔═╡ 04684b69-cb9b-4160-9498-a5b5f155c152
stateindex(momdp, s)

# ╔═╡ f0b97376-b8b1-4a11-bc05-bfddf6ffabc7
visiblestateindex(momdp, s)

# ╔═╡ 47666999-a264-4791-855a-6edb7db74563
findfirst(map(sv->sv == hidden(s), visiblestates(momdp)))

# ╔═╡ 6f53f391-041a-47f2-bbaf-a6a7cb988842
hiddenstateindex(momdp, s)

# ╔═╡ 8b80bfe2-a104-4b35-9f4a-2cf2f2c94d13
actionindex(momdp, ACCEPT)

# ╔═╡ 0009ca5c-0fe5-4465-a8f4-328916e2fc80
a = rand(actions(momdp))

# ╔═╡ d9f6e366-d63b-4be9-97f7-b8d14a01da92
(s,a,sp,o)

# ╔═╡ 042c4bd5-45c8-49e0-8e5b-ec9d00e3c497
(s,a)

# ╔═╡ 632a8bfd-abb4-4a0c-8c93-08fc77243e75
obsindex(momdp, o)

# ╔═╡ 14225602-6119-4b57-88d5-16c2817b17b2
observation(momdp, s, a, sp)

# ╔═╡ 34f7d327-78a2-4c03-882f-b374ea8c1b4c
obs_weight(momdp, s, a, sp, o)

# ╔═╡ f51dab78-dfca-420e-a06a-341d335c5992
pdf(observation(momdp, s, a, sp), o)

# ╔═╡ b4c03954-be46-41a4-8d08-920d7a30124a
transition(momdp, s, a)

# ╔═╡ 80d236e6-84c1-4b66-a14d-317efc10ea1a
transitionhidden(momdp, hidden(s), a, o)

# ╔═╡ eaf0bc9d-cc9a-4d2c-9a15-6a9dc90ce87c
transitionvisible(momdp, visible(s), ACCEPT, o)

# ╔═╡ 0ad927f8-504b-4932-9715-767e2eb06f5f
@gen(:sp, :o, :r)(momdp, s, a)

# ╔═╡ 8c95dbaf-3156-4ed7-adbe-7754f5fd1d36
reward(momdp, s, a)

# ╔═╡ 3bf57fd8-5ae8-435b-b6a6-c34878259de6
statetype(momdp)

# ╔═╡ bf61582b-4cd4-49f8-bbfc-b33f30d34cac
visiblestatetype(momdp)

# ╔═╡ 4bf1f6e9-dab6-41a7-8b41-8ddd68ba6e36
hiddenstatetype(momdp)

# ╔═╡ 505b9e0f-0119-4d19-941f-24f3dde4d94e
actiontype(momdp)

# ╔═╡ 96e18524-f17b-4693-b2ec-71da8bd659ab
obstype(momdp)

# ╔═╡ 962b44e3-dcc4-4767-b7e9-2a62ee49e42f
states(momdp) |> length

# ╔═╡ 65621624-fb6d-4815-b5fa-c04e37e9125e
visiblestates(momdp) |> length

# ╔═╡ e3bb33f0-613b-457d-a734-7591ad3bcbe1
hiddenstates(momdp) |> length

# ╔═╡ b9deab74-3621-4a25-b5a1-45a9797a4e2e
actions(momdp) |> length

# ╔═╡ 48b4f51f-607c-477e-b54d-6136e3439105
observations(momdp) |> length

# ╔═╡ 3c58cb0c-0216-49b7-af8a-2ee3f344618f
ordered_hidden_states(momdp)

# ╔═╡ b9620f6b-7b9e-474b-8037-0661ef49f493
ordered_visible_states(momdp)

# ╔═╡ b139bd6c-e275-43b6-83be-5fb9095eb0fb
md"""
## Termination
"""

# ╔═╡ 663c3b94-da44-4102-b18b-b40d7b8033c3
POMDPs.isterminal(m::TestMOMDP, s::State) = visible(s).t == 0 || visible(s).c == 0

# ╔═╡ 81a40b69-b994-4a93-960c-1b0a4a4aeced
POMDPs.discount(m::TestMOMDP) = 0.95

# ╔═╡ bc1d8d7f-41e7-42ba-94e2-2c2526060d6b
md"""
## Solver/Policy
"""

# ╔═╡ 3bc39a8f-e9fc-49be-a30f-cf48bed2bdfe
updater(momdp::TestMOMDP) = DirichletSubspaceUpdater(momdp)

# ╔═╡ dd3c1106-7790-4e06-acf0-af3e59c34b32
solver = POMCPSolver(c=100)

# ╔═╡ 9c05ba46-bd4f-4bf8-afb5-38ea470440f6
policy = solve(solver, momdp)

# ╔═╡ f2ab4fff-ac2e-447e-a7d3-916054317994
sv = visible(s)

# ╔═╡ 0cf4ae77-936b-4688-b425-59b869485b68
actiontype(policy.problem)

# ╔═╡ 292f2aa1-b100-4f79-bb77-c3a08c0e5312
md"""
## Dirichlet subspace belief/updater (MOMDP)
"""

# ╔═╡ 84292e01-8b4f-4bba-918e-6df49eea410f
bₘ = DirichletSubspaceBelief(momdp)

# ╔═╡ 358ef5de-d819-4d5a-b0b2-8db1dfc94aff
__a, info = action_info(policy, bₘ(sv), tree_in_info=true); __a

# ╔═╡ a21dcd71-138b-471c-afbf-050f06bc3999
D3Tree(info[:tree], init_expand=1)

# ╔═╡ 311e3a11-59c1-4e08-8e3d-fbc46bb88659
rand(bₘ(sv))

# ╔═╡ fa4cf7f0-e739-477b-98bd-a6d9133870d3
upₘ = DirichletSubspaceUpdater(momdp)

# ╔═╡ 1020efb9-edfb-4d65-a1c0-960cf6d08034
Dₘ_prior = Categorical(normalize(ones(3),1));

# ╔═╡ ac1d4216-daba-4445-a666-4100f579b5da
Dₘ_true = Categorical([0.50, 0.4, 0.1]);

# ╔═╡ 09568f3e-e6f3-47c7-9d4e-baba979d68a9
begin
	bₘ′ = initialize_belief(upₘ, Dₘ_prior) # deepcopy(b)
	bₘ′(sv)
	for t in 1:1000
		# oᵢₜₑᵣ = rand(𝒪)
		v_doc = documentation[rand(Dₘ_true)]
		o = Observation(sv.c, sv.t, sv.f, v_doc)
		bₘ′ = update(upₘ, bₘ′, a, o)
	end
	[mean(bₘ′) bₘ′.b.alpha]
end

# ╔═╡ 1bed633c-f32a-4b47-b142-653441263be6
(s, a, sp, o)

# ╔═╡ d2d82fc4-220a-4db6-88a9-6b9b277f1bfc
# for sh in bₘ′.hidden_state_list
# 	s = (sv, sh)
# 	T = transitionhidden(momdp, sh, a, o)
# 	for (sp, tp) in weighted_iterator(T)
# 		spi = hiddenstateindex(momdp, sp)
# 		op = obs_weight(momdp, s, a, sp, o)
# 		# @info (s, a, sp, o)
# 		# @info op
# 	end
# end

# ╔═╡ e5cef722-1c7f-4b06-959e-71d4fa052d36
pdf(bₘ, s)

# ╔═╡ 81164af3-c209-47ed-8cf9-4476a8ef6909
s

# ╔═╡ 56edc92d-6e0b-4b07-8412-ced25b3cfade
md"""
# Debug
"""

# ╔═╡ 993179df-f791-4df5-b42d-6376bbb609df
function Plots.plot(𝒟::Dirichlet, categories::Vector; kwargs...)
	transposed = reshape(categories, (1, length(categories)))
	bar(
	    transposed,
	    mean(𝒟)',
	    labels = transposed,
	    bar_width = 1;
		kwargs...
	)
end

# ╔═╡ cba884eb-a544-4597-a085-d100e91d2ebe
plot(b′.b, collect(1:5))

# ╔═╡ 8876bf9a-5de9-4443-9705-c076e11ca160
plot(bₘ′.b, map(sₕ->replace(string(sₕ), r"Main.workspace#\d+\."=>""), hiddenstates(momdp)); c=[:forestgreen :gray :crimson])

# ╔═╡ c8595056-ac88-4173-a082-7f5666bb6ae5
begin
	b_test1 = DirichletBelief(pomdp)
	b_test2 = DirichletBelief(pomdp)
end;

# ╔═╡ 71f89039-5a50-47ad-af79-e1b52339422f
begin
	dir1 = Dirichlet([1,1,1])
	dir2 = Dirichlet([1,1,1])
	@assert dir1 == dir2
end

# ╔═╡ 4f4402be-51d2-4361-b593-0978f3a1b9f3
@assert hash(dir1) == hash(dir2)

# ╔═╡ d9d4efad-1d13-4bed-9414-1b3e428714e6
begin
	b_test_fill = DirichletBelief(pomdp)
	@assert fill!(b_test_fill, 20).b.alpha == 20*ones(length(ordered_states(pomdp)))
end

# ╔═╡ 495ff295-56fa-4ebb-81d2-0a05d1ffb309
@assert b_test1 == b_test2

# ╔═╡ 67dd929d-e051-472b-abd5-e12f0ec7040f
@assert hash(b_test1) == hash(b_test2)

# ╔═╡ 95e3097f-bdc4-47b0-829b-d81460e63a74
pdf(b3, 1)

# ╔═╡ 0a6da4e7-7501-4cb7-b992-e4b1734986c0
pdf(b.b, normalize(b.b.alpha, 1))

# ╔═╡ 58bb8d0f-d3ee-4828-afbc-56e101f63eb4
histogram(rand(b3, 1000))

# ╔═╡ Cell order:
# ╟─683dff90-85d3-11ec-2531-05d793550053
# ╠═4624989c-ac25-469f-89b1-a151c3a70148
# ╠═7e599ef3-66d4-423e-b213-70acdd9eb7ed
# ╠═c392e215-afdd-46ed-b175-1407ff4c9669
# ╟─9f705704-b91e-4b12-8bc8-0a95a093f9ec
# ╠═3c3a77dc-eb77-4ae9-9c64-88b20d8cbb8a
# ╟─d9bedb8a-927c-454d-bd97-00899e6b5983
# ╟─e23add73-3c18-4d81-ba3c-4f5472fdde42
# ╠═f23f5c21-d5e4-4878-b85e-356a7619cf0e
# ╟─75c999a2-da6b-40de-a078-901d8d537a47
# ╠═ef423c3a-eae8-4211-b4a4-b3cf4b27d7a8
# ╠═8b5d6571-556c-44fb-b0dd-07d8484c2c88
# ╠═e5d17ad5-10ce-425b-ab8d-962eac5d7f2b
# ╠═05b01eb8-ecee-4f6a-a1aa-305919ef3b44
# ╠═4b3b8e56-c5ad-4f2d-a95b-2930f0d7cb72
# ╠═28293463-3016-4495-a446-bcadb2691e07
# ╠═60130742-e884-4e59-b36d-9a95c22f336e
# ╟─431f3131-c1b7-4dbd-91d4-9980366d23cb
# ╠═6bf232ba-fa87-4522-87bb-1682975e33b1
# ╟─f3f4f965-006b-40f5-a6b6-b43581a4044a
# ╠═c3febe96-0247-46a8-a1e1-ead068d799a2
# ╠═762a255f-957b-43d9-8bd8-40feb9f779a6
# ╠═b0b58647-0098-4557-b0a6-804d952ee5ae
# ╠═a8f7c03e-02ea-48b1-a50d-fa24e62afd35
# ╟─22564b5c-31d8-4da2-a9c8-3151ebc209a0
# ╠═5ddb7ebf-b202-4fe8-92e7-f6528018ef1e
# ╠═03a30c5a-9d8d-46a8-be7a-ce1d7e5bb403
# ╠═e02fa4e0-7caf-4363-8006-6538174dbfba
# ╠═cba884eb-a544-4597-a085-d100e91d2ebe
# ╟─405a7108-2434-414b-bbc3-89056b9f2569
# ╠═b01e8019-7b16-489a-b97e-574df652ff24
# ╠═3257e67d-9975-4f43-8ea4-128b1facd67c
# ╠═0847d794-ade9-40ff-8ee5-0326deb76c43
# ╠═3cece97a-72d3-4bdd-8ca9-290752c74291
# ╠═9d3b3a03-0620-407d-9e01-64fcd122819b
# ╠═7e58da87-d712-4d64-b58e-5238baa6e557
# ╠═85802425-17ab-4d1e-bf83-1b28a250d235
# ╟─8c354e0e-18d1-40f2-9da8-3b14c862e039
# ╠═697e6d70-ea9e-414e-b6b7-0385e40c80bb
# ╠═a2ce04e4-57d9-44c5-a3c0-b49d4ed7c503
# ╠═98ba26bc-f671-4e6c-9ce9-376c6445ef47
# ╠═598a23bf-37b1-4c17-a428-776e3123e681
# ╠═ac5cbb0d-6cd9-47fd-a8bc-fc2498eb52e9
# ╟─e5dc63d1-6ee0-46b8-9506-3f739eeae4a5
# ╟─95c3541f-5b3e-4adf-8b78-c434d27ad963
# ╠═76859546-8c03-4b95-90d2-2fe4a30ebc02
# ╠═4bdca411-6781-49a4-a33a-1462967694a0
# ╠═10057440-2e41-47fa-836e-ad2627d2e4a4
# ╠═4f2b1350-100e-43fa-bb8e-ddad9d9bff93
# ╠═9eb261f6-6385-4098-af11-6b6ee9a4d155
# ╠═b1fe8c15-8b78-4d95-ab65-cd04a15a2887
# ╠═2ff812d0-75ca-470d-af54-208dd086a1b4
# ╠═83d100df-4ecf-49f3-b6f4-ea14b6dc508b
# ╠═36dbfce8-1c94-41b1-8748-8f5df0142178
# ╠═b903212f-d438-4a3e-b844-bd4a28015783
# ╠═e84ee454-fb02-4b52-b336-d0ed16f62b2c
# ╠═a4989251-7986-41aa-9cce-64468fdff52d
# ╠═04684b69-cb9b-4160-9498-a5b5f155c152
# ╠═f0b97376-b8b1-4a11-bc05-bfddf6ffabc7
# ╠═47666999-a264-4791-855a-6edb7db74563
# ╠═6f53f391-041a-47f2-bbaf-a6a7cb988842
# ╟─eb4f0ac0-a3f1-410c-9d50-e3e3bbbf107c
# ╠═1cd45201-d0dd-49d9-9e6c-eff1b8d1521c
# ╠═b2f83903-c6e1-41cb-910f-6f38db222b66
# ╠═8b80bfe2-a104-4b35-9f4a-2cf2f2c94d13
# ╠═0009ca5c-0fe5-4465-a8f4-328916e2fc80
# ╟─972075cb-5aa8-42c7-bbc2-ddb531bb547c
# ╠═c620ba50-35ae-40d4-b69d-21984a883734
# ╠═b29fdaee-006c-4bf4-a5ff-c490d3cef4e1
# ╠═69c5e487-117f-4120-a6c0-915f72278702
# ╠═181fc70e-e8f0-4bd8-885d-491890c8259b
# ╠═aef35f4a-e00d-4d8b-b667-93c10761e2f6
# ╠═d25623b9-1c35-4bfe-8266-5af07ea6101a
# ╠═632a8bfd-abb4-4a0c-8c93-08fc77243e75
# ╠═fd4ca58a-33c8-471c-ae91-940906e6202c
# ╠═4b06ff50-1ab5-4352-8f13-bf58ed2b5ec2
# ╠═e0366964-e4b4-44d7-96d5-cba974d43470
# ╠═d9f6e366-d63b-4be9-97f7-b8d14a01da92
# ╠═14225602-6119-4b57-88d5-16c2817b17b2
# ╠═ae35a3ad-210e-421f-8506-591969b6df44
# ╠═46835032-5378-4fe7-9966-e9354e0a64d1
# ╠═34f7d327-78a2-4c03-882f-b374ea8c1b4c
# ╠═f51dab78-dfca-420e-a06a-341d335c5992
# ╟─66a7d93c-3dcd-4ff6-aeef-d478d49fadd0
# ╠═7934c669-23f0-45b1-aa50-2298bf9cbf16
# ╠═90c20c89-b7ad-4423-9b6e-440054c22959
# ╠═8fe5b28b-5c91-4264-a4eb-4562d7d17753
# ╠═eea7efec-34e2-429f-a009-b7ff582304d1
# ╠═c58c2fae-d4fd-4349-85c0-e32839f40f08
# ╠═a2b3c5a8-506b-485b-9799-14fe56475689
# ╠═b4c03954-be46-41a4-8d08-920d7a30124a
# ╠═80d236e6-84c1-4b66-a14d-317efc10ea1a
# ╠═eaf0bc9d-cc9a-4d2c-9a15-6a9dc90ce87c
# ╟─d4616594-ea3c-4483-8db7-499dd2f29174
# ╠═0ad927f8-504b-4932-9715-767e2eb06f5f
# ╟─6382deb8-0bb6-438a-a2e8-98939fcf3634
# ╠═f0819505-23cb-44bc-b3c4-7db8dc9ac994
# ╠═f97a3c29-3598-401c-93fb-f1e19ff151ab
# ╠═7598f6ab-73e0-45b9-b101-a3ddf6deb4e4
# ╠═8c95dbaf-3156-4ed7-adbe-7754f5fd1d36
# ╠═042c4bd5-45c8-49e0-8e5b-ec9d00e3c497
# ╟─12611914-ab90-42cb-a83c-4a2e90145203
# ╠═3b336040-778d-498c-9ed8-89f0f69b67f9
# ╠═1a4def74-873c-481c-8ffd-29f004bc0117
# ╠═21b8409a-a4f3-41a3-940a-bb067824aaef
# ╠═3bf57fd8-5ae8-435b-b6a6-c34878259de6
# ╠═bf61582b-4cd4-49f8-bbfc-b33f30d34cac
# ╠═4bf1f6e9-dab6-41a7-8b41-8ddd68ba6e36
# ╠═505b9e0f-0119-4d19-941f-24f3dde4d94e
# ╠═96e18524-f17b-4693-b2ec-71da8bd659ab
# ╠═962b44e3-dcc4-4767-b7e9-2a62ee49e42f
# ╠═65621624-fb6d-4815-b5fa-c04e37e9125e
# ╠═e3bb33f0-613b-457d-a734-7591ad3bcbe1
# ╠═b9deab74-3621-4a25-b5a1-45a9797a4e2e
# ╠═48b4f51f-607c-477e-b54d-6136e3439105
# ╠═3c58cb0c-0216-49b7-af8a-2ee3f344618f
# ╠═b9620f6b-7b9e-474b-8037-0661ef49f493
# ╟─b139bd6c-e275-43b6-83be-5fb9095eb0fb
# ╠═663c3b94-da44-4102-b18b-b40d7b8033c3
# ╠═81a40b69-b994-4a93-960c-1b0a4a4aeced
# ╟─bc1d8d7f-41e7-42ba-94e2-2c2526060d6b
# ╠═3bc39a8f-e9fc-49be-a30f-cf48bed2bdfe
# ╠═dd3c1106-7790-4e06-acf0-af3e59c34b32
# ╠═9c05ba46-bd4f-4bf8-afb5-38ea470440f6
# ╠═358ef5de-d819-4d5a-b0b2-8db1dfc94aff
# ╠═5745eb4a-8688-4cc5-a262-d86e60d2ebcf
# ╠═a21dcd71-138b-471c-afbf-050f06bc3999
# ╠═f2ab4fff-ac2e-447e-a7d3-916054317994
# ╠═311e3a11-59c1-4e08-8e3d-fbc46bb88659
# ╠═0cf4ae77-936b-4688-b425-59b869485b68
# ╟─292f2aa1-b100-4f79-bb77-c3a08c0e5312
# ╠═84292e01-8b4f-4bba-918e-6df49eea410f
# ╠═fa4cf7f0-e739-477b-98bd-a6d9133870d3
# ╠═1020efb9-edfb-4d65-a1c0-960cf6d08034
# ╠═ac1d4216-daba-4445-a666-4100f579b5da
# ╠═09568f3e-e6f3-47c7-9d4e-baba979d68a9
# ╠═1bed633c-f32a-4b47-b142-653441263be6
# ╠═8876bf9a-5de9-4443-9705-c076e11ca160
# ╠═d2d82fc4-220a-4db6-88a9-6b9b277f1bfc
# ╠═e5cef722-1c7f-4b06-959e-71d4fa052d36
# ╠═81164af3-c209-47ed-8cf9-4476a8ef6909
# ╟─56edc92d-6e0b-4b07-8412-ced25b3cfade
# ╠═993179df-f791-4df5-b42d-6376bbb609df
# ╠═c8595056-ac88-4173-a082-7f5666bb6ae5
# ╠═71f89039-5a50-47ad-af79-e1b52339422f
# ╠═4f4402be-51d2-4361-b593-0978f3a1b9f3
# ╠═d9d4efad-1d13-4bed-9414-1b3e428714e6
# ╠═495ff295-56fa-4ebb-81d2-0a05d1ffb309
# ╠═67dd929d-e051-472b-abd5-e12f0ec7040f
# ╠═95e3097f-bdc4-47b0-829b-d81460e63a74
# ╠═0a6da4e7-7501-4cb7-b992-e4b1734986c0
# ╠═ae451a8c-e790-4b1d-b853-4c592179d9d0
# ╠═0c6fda44-14b0-4cb6-b0b9-026752fd2bb7
# ╠═2104ce98-ebb0-4258-aeb0-bd48a20dd5fc
# ╠═d1c230d9-674d-4a4b-8c50-13a9bae3c499
# ╠═58bb8d0f-d3ee-4828-afbc-56e101f63eb4
