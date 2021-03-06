# turn probabilities into counts 
๐โ = Dirichlet(round.(normalize(params.visa_prob, 1) .* 100) .+ 1)

# Plot Dirichlet distribution counts over each provided category.
function Plots.plot(๐::Dirichlet, categories::Vector, cmap; kwargs...)
    transposed = reshape(categories, (1, length(categories)))
    bar(
        transposed,
        ๐.alpha',
        labels = transposed,
        bar_width = 1,
        c = [get(cmap, i/length(categories)) for i in 1:length(categories)]';
        kwargs...
    )
end

Plots.plot(๐โ, visa_statuses, cmap_bar, title="Dirichlet expert prior")

## Updating Dirichlet belief
๐_true = Categorical([0.01, 0.50, 0.31, 0.10, 0.08]);
is_uniform_initial = true

if is_uniform_initial
    ๐_belief = Dirichlet(ones(length(params.visa_status)))
else
    ๐_belief = deepcopy(๐โ) # initial belief
end

# Updating pseudocounts 
# Could seed out belief with the expert prior or use a uniform prior
begin
    b_p = Plots.plot(๐_belief, visa_statuses, cmap_bar, title="updated Dirichlet posterior")
    vโฒ = rand(๐_true) # sample a new visa status from the latent _true_ distribution
    ๐_belief.alpha[vโฒ] += 1 # update pseudocount
    b_p
end

new_visa_probs = normalize(๐_belief.alpha ./ ๐_belief.alpha0, 1)



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
    sแตข = visiblestateindex(pomdp, sv)
    p[sแตข] = 1
    normalize!(p, 1)
    return SparseCat(visiblestates, p)
    # return Distribution T(vโฒ | s, a)
end



using D3Trees
D3Tree(info[:tree], init_expand=1)