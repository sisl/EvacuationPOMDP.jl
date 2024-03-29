# https://en.wikipedia.org/wiki/Flag_of_Afghanistan

default(fontfamily="Computer Modern", framestyle=:box)

afghan_red = colorant"#be0000";
afghan_green = colorant"#007a36";

cmap = ColorScheme([afghan_red, afghan_green])
cmap_bar = ColorScheme([afghan_red, colorant"lightgray", afghan_green])

visa_statuses = ["ISIS-K", "Vul. Afghan", "P1/P2 Afghan", "SIV", "AMCIT"] # NOTE: ordering
visa_statuses_short = ["ISIS", "Afg.", "P1/P2", "SIV", "AMCIT"] # NOTE: ordering


function plot_claims(p; small=false, text="", xticks=false, legend=false, kwargs...)
    categories = visa_statuses
    transposed = reshape(categories, (1, length(categories)))
    p = reshape(p, (1, length(p)))
    plegend = false # legend ? :left : false
    bar(
        transposed,
        p,
        labels=transposed,
        bar_width=1,
        legend=plegend,
        topmargin=small ? -2Plots.mm : 0Plots.mm,
        bottommargin=small ? -3Plots.mm : 0Plots.mm,
        leftmargin=3Plots.mm,
        xticks=xticks || !small ? true : (1:length(p), fill("", length(p))),
        yticks=([0, 0.5, 1], [0, 0.5,  1]),
        ylims=(0, 1.25),
        size=small ? (600,150) : (500,150),
        legendfontsize=2,
        c=[get(cmap_bar, i/length(categories)) for i in 1:length(categories)]';
        kwargs...
    )
    annotate!([(0, 1.125, (text, 10, :black, :left))])
end


function plot_claims_tiny(p; kwargs...)
    categories = visa_statuses
    transposed = reshape(categories, (1, length(categories)))
    p = reshape(p, (1, length(p)))
    plegend = false # legend ? :left : false
    bar(
        transposed,
        p,
        labels=transposed,
        bar_width=1.5,
        legend=plegend,
        xticks=false,
        yticks=false,
        ylims=(0, 1),
        size=(100,100),
        framestyle=:none,
        c=[get(cmap_bar, i/length(categories)) for i in 1:length(categories)]';
        kwargs...
    )
end


function plot_all_claims(pomdp::EvacuationPOMDPType)
    claims = pomdp.claims
    p1 = plot_claims(claims.p_amcit; small=true, legend=true, text=L"P(v_\mathrm{obs} \mid v=\texttt{AMCIT})")
    p2 = plot_claims(claims.p_siv; small=true, text=L"P(v_\mathrm{obs} \mid v=\texttt{SIV})")
    p3 = plot_claims(claims.p_p1p2; small=true, text=L"P(v_\mathrm{obs} \mid v=\texttt{P1/P2})")
    p4 = plot_claims(claims.p_afghan; small=true, text=L"P(v_\mathrm{obs} \mid v=\texttt{Vul. Afghan})")
    p5 = plot_claims(claims.p_isis; small=true, xticks=true, text=L"P(v_\mathrm{obs} \mid v=\texttt{ISIS})")
    Plots.plot(p1, p2, p3, p4, p5, layout=@layout([a;b;c;d;e]), size=(450,500))
end


"""
Pass in policy and chairs and time remaing. Spit out graph of family size versus visa status.
"""
function vis_time_step(params, policy, c, t)
    v_size = length(params.visa_status)
    f_size = length(params.family_sizes)
    policyGraph = zeros(v_size, f_size) 
    
    visa_statuses = params.visa_status
    x = Int.(visa_statuses)
    family_sizes = params.family_sizes
    y = family_sizes
        
    for f in 1:f_size
        for v in 1:v_size
            act = action(policy, MDPState(c, t, family_sizes[f], visa_statuses[v])) 
            if act == ACCEPT
                policyGraph[v,f] = 500
            else
                policyGraph[v,f] = 0
            end
        end
    end
    
    z = policyGraph'

    timeVal = string(t)
    capVal = string(c)
    titleX = string("\$t_{$timeVal}, c_{$capVal}\$")

    return heatmap(x, y, z,
         # aspect_ratio = 1,
         legend = :none, 
         xlims = (x[1]-0.5, x[end]+0.5-0.1),
         ylims = (params.family_sizes[1], params.family_sizes[end]),
         yticks = (params.family_sizes[1]:params.family_sizes[end], [1, "", "", 4, "", "", 7, "", "", 10, "", "", 13]),
         xticks = (x[1]:x[end], visa_statuses_short),
         xlabel = "visa status",
         ylabel = "family size",
         title = titleX,
         xtickfont = font(5, "Courier"), 
         ytickfont = font(5, "Courier"),
         thickness_scaling = .9,
         rightmargin = 5Plots.mm,
         color=cmap.colors,
    )
end


function vis_all(params, policy)
    total_time = params.time 
    total_capacity = params.capacity
    graph_per_n_time = 600 # 60
    graph_per_n_cap = 250 # 60
    heat_maps = []
    time_points = (total_time/graph_per_n_time) + 1 # to include 0 
    capacity_points = (total_capacity/graph_per_n_cap) + 1 
    num_graphs = trunc(Int, time_points*capacity_points)
    
    for t in reverse(0:total_time)
        if t % graph_per_n_time == 0 
            for c in reverse(0:total_capacity)
                if c % graph_per_n_cap == 0
                    if c == 0
                        c = 1
                    end
                    if t == 0
                        t = 1
                    end
                    push!(heat_maps, vis_time_step(params, policy, c, t))
                end
            end
        end
    end 

    Plots.plot(heat_maps..., layout=num_graphs, margin=2mm)
end


function Plots.plot(𝒟::Dirichlet, categories::Vector; kwargs...)
    transposed = reshape(categories, (1, length(categories)))
    bar(
        transposed,
        𝒟.alpha',
        labels = transposed,
        bar_width = 1,
        c = [get(cmap_bar, i/length(categories)) for i in 1:length(categories)]';
        kwargs...
    )
end