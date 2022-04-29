function create_family_size_distribution(N=10000)
    fam_distr = MixtureModel([TruncatedNormal(1, 0.6, 1, 13), TruncatedNormal(8, 2, 1, 13)])
    fam_samples = round.(Int, rand(fam_distr, N))
    fam_fit = fit(Histogram, fam_samples)
    fam_probs = normalize(fam_fit.weights, 1)
    fam_plot = Plots.plot(0:0.1:15, x->pdf(fam_distr, x), lab=false)
    return fam_probs, fam_plot
end


function plot_family_size_distribution(fam_probs)
    return bar(fam_probs,
               bar_width=1,
               xticks=(1:13),
               label=false,
               c=:gray,
               ylims=(0, 0.31),
               xlabel="family size at gate",
               ylabel="likelihood",
               size=(500,300))
end