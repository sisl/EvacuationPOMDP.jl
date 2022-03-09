# turn probabilities into counts 
𝒟₀ = Dirichlet(round.(normalize(params.visa_prob, 1) .* 100) .+ 1)

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

Plots.plot(𝒟₀, visa_statuses, cmap_bar, title="Dirichlet expert prior")

## Updating Dirichlet belief
𝒟_true = Categorical([0.01, 0.50, 0.31, 0.10, 0.08]);
is_uniform_initial = true

if is_uniform_initial
    𝒟_belief = Dirichlet(ones(length(params.visa_status)))
else
    𝒟_belief = deepcopy(𝒟₀) # initial belief
end

# Updating pseudocounts 
# Could seed out belief with the expert prior or use a uniform prior
begin
    b_p = Plots.plot(𝒟_belief, visa_statuses, cmap_bar, title="updated Dirichlet posterior")
    v′ = rand(𝒟_true) # sample a new visa status from the latent _true_ distribution
    𝒟_belief.alpha[v′] += 1 # update pseudocount
    b_p
end

new_visa_probs = normalize(𝒟_belief.alpha ./ 𝒟_belief.alpha0, 1)



# macro seeprints(expr)
#   quote
#       stdout_bk = stdout
#       rd, wr = redirect_stdout()
#       $expr
#       redirect_stdout(stdout_bk)
#       close(wr)
#       read(rd, String) |> Text
#   end
# end


function MOMDPs.transitionvisible(pomdp::EvacuationPOMDPType, sv::VisibleState, a::Action, o=missing)
    visiblestates = ordered_visible_states(pomdp)
    p = ones(length(visiblestates))
    sᵢ = visiblestateindex(pomdp, sv)
    p[sᵢ] = 1
    normalize!(p, 1)
    return SparseCat(visiblestates, p)
    # return Distribution T(v′ | s, a)
end



using D3Trees
D3Tree(info[:tree], init_expand=1)