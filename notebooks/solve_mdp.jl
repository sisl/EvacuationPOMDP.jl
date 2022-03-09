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

# ╔═╡ 2d74e353-7f4a-46ba-a3bc-972cddfe8d54
begin
	using Revise
	using Pkg
	Pkg.develop(path="..")
	using EvacuationPOMDP
	using POMDPs
	using DiscreteValueIteration
	using BSON
	using PlutoUI
end

# ╔═╡ 0a7b2d64-4481-4e1d-8545-900e37f5bbdd
md"""
# Solve Evacuation MDP
"""

# ╔═╡ 1308bb4f-bc92-4151-b700-cdae422e4770
mdp = EvacuationMDP()

# ╔═╡ 4aca44da-0844-4c51-a3cf-321abe4bfb00
solver = ValueIterationSolver(max_iterations=30, belres=1e-6, verbose=true);

# ╔═╡ bc756c7a-2f9c-4d3c-975e-5a55a35ed77b
@bind reload_mdp_policy CheckBox(true)

# ╔═╡ c3ab8785-72d7-4da2-9e76-0732f7475b94
md"""
> - **Running**: takes about 191 seconds
> - **Loading**: takes about 1.8 seconds, 73 MB
"""

# ╔═╡ 75f8fe08-38ad-473f-b546-d417dbf51df1
if reload_mdp_policy
	mdp_policy = BSON.load("mdp_policy.bson", @__MODULE__)[:mdp_policy]
else
	mdp_policy = solve(solver, mdp)
	BSON.@save "mdp_policy.bson" mdp_policy
end

# ╔═╡ 8b9e91df-aa94-426d-9ff9-f29383a5d176
vis_all(mdp.params, mdp_policy)

# ╔═╡ Cell order:
# ╟─0a7b2d64-4481-4e1d-8545-900e37f5bbdd
# ╠═2d74e353-7f4a-46ba-a3bc-972cddfe8d54
# ╠═1308bb4f-bc92-4151-b700-cdae422e4770
# ╠═4aca44da-0844-4c51-a3cf-321abe4bfb00
# ╠═bc756c7a-2f9c-4d3c-975e-5a55a35ed77b
# ╟─c3ab8785-72d7-4da2-9e76-0732f7475b94
# ╠═75f8fe08-38ad-473f-b546-d417dbf51df1
# ╠═8b9e91df-aa94-426d-9ff9-f29383a5d176
