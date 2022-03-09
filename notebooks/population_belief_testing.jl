# ╔═╡ 96176ef1-2919-41a0-a206-bfe228195ad8
md"""
## Population beliefs
"""

# ╔═╡ d47c2f84-ca16-4337-80ef-3abd94a77f6a
prior_belief = Categorical([0.01, 0.50, 0.14, 0.20, 0.15]);

# ╔═╡ 453da136-bd07-4e7c-a47a-0bad8765eb7e
up = DirichletSubspaceUpdater(pomdp)

# ╔═╡ da134037-11fb-4f76-8381-e128a37d43eb
_b′ = initialize_belief(up, prior_belief)

# ╔═╡ c01e8357-c094-4373-ba49-faa149dc7191
begin
    𝒟_true = Categorical([0.01, 0.50, 0.31, 0.10, 0.08])
    b′ = initialize_belief(up, prior_belief)
    sv = VisibleState(params.capacity, params.time, 1)
    # b′(sv)
    for num_updates in 1:10
        vdoc = pomdp.documentation[rand(𝒟_true)]
        o = Observation(sv.c, sv.t, sv.f, vdoc)
        b′ = update(up, b′, ACCEPT, o)
    end
    [mean(b′) b′.b.alpha]
end

# ╔═╡ b1f02e06-7131-4de3-9b40-b9d7e87ce99e
Plots.plot(b′.b, bar_labels)

# ╔═╡ 230fd9b3-837c-491e-85e6-d27be29618e3
bar_labels = map(sₕ->replace(replace(string(sₕ), r"EvacuationPOMDP\."=>""), "HiddenState"=>""), hiddenstates(pomdp))
