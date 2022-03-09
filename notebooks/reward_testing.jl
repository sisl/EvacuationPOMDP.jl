# ╔═╡ c0292f00-7f92-11ec-2f15-b7975a5e40a1
md"""
# Evacuation POMDP
"""

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

# ╔═╡ fe9dfc73-f502-4d13-a019-8160a6af4617
Random.seed!(0xC0FFEE)

# ╔═╡ ffbb36c6-8e69-46d1-b132-845564de0ae2
md"""
## Environment Parameters
"""

# ╔═╡ 6e7cf801-99d8-4bec-a036-3a0b2475a2eb
params = EvacuationParameters();

# ╔═╡ 5022c6f3-09f8-44bc-b41e-a86cbf8f787c
md"""
## Reward function
"""

# ╔═╡ b5164cb4-7e74-4536-9125-a0732a860690
R(params, 10, 50, 1, AMCIT, REJECT)

# ╔═╡ 9abfcc56-ddbd-4670-8e24-b2472cf35676
@bind current_time Slider(0:120, default=120, show_value=true)

# ╔═╡ 91217d58-7d5b-4559-ba0d-6f07e204ade7
time_penalty(current_cap, current_time)

# ╔═╡ d6ddd6e0-efde-4af3-8885-ddf4f32bf163
time_penalty(c,t) = -c/(t+1) # sqrt?

# ╔═╡ 359069f8-4131-4346-bb75-9d941350b23c
@bind current_cap Slider(0:120, default=120, show_value=true)

# ╔═╡ 3a079763-a17c-4111-b59c-58f8d4391368
Plots.plot(map(t->time_penalty(current_cap,t), 120:-1:0),
           xlabel="time left", ylims=(-120,10), label=false,
           c=get(timecolor, current_cap/params.time))

# ╔═╡ cd445706-0002-45d3-b405-20b2206cde64
timecolor = cgrad(:blues, 0:120)

# ╔═╡ 5e90197e-0eae-47f4-86bd-ba618b3b1c93
get(timecolor, current_time/params.time)

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
