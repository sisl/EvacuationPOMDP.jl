function simulate_trajectory(pomdp::POMDP, pomdp_policy)
    up = updater(pomdp)
    s₀ = rand(initialstate(pomdp))
    prior_belief = initialize_belief(up, pomdp.params.visa_count, visible(s₀))
    o₀ = rand(observation(pomdp, s₀, REJECT, s₀))
    initial_belief = update(up, prior_belief, REJECT, o₀)

    # global variable hack (due to @seeprints)
    # initial_obs[1] = o₀
    # empty!(trajectory)
    # trajectory = [(nothing, nothing, o₀, nothing, nothing)]
    trajectory = []
    
    for (t, (s, a, o, b, sp, r)) in enumerate(stepthrough(pomdp, pomdp_policy,
                                                          up,
                                                          initial_belief,
                                                          s₀,
                                                          "s,a,o,b,sp,r",
                                                          max_steps=130)) # 121
        @show t
        println("Capacity=$(visible(s).c), time remaining=$(visible(s).t)")
        println(hidden(s).v," of size ", visible(s).f)
        if a == ACCEPT
            println("\t——————ACCEPT—————— ✅")
            @info ACCEPT, t
        else
            println("\t(reject) ❌")
        end
        @show o
        @show round.(b.b, digits=3)
        @show VisaStatus(argmax(b.b) - 1) # 0-based enums
        if hidden(s).v != VisaStatus(argmax(b.b) - 1)
            @show "~~~~~ M I S M A T C H ~~~~~"
            @warn "mismatch ($t)" # Only happens in the noisy case due to sampling
        end
        println(hidden(sp).v," of size ", visible(sp).f)
        @show r
        println("—"^20)
        push!(trajectory, (s,a,o,b,r))
    end

    trajectory′ = Tuple{Any,Any,Any,Any,Any}[(trajectory[1][1], trajectory[1][2], o₀, trajectory[1][4], trajectory[1][5])]
    for i in 2:length(trajectory)
        push!(trajectory′, (trajectory[i][1], trajectory[i][2], trajectory[i-1][3], trajectory[i][4], trajectory[i][5]))
    end
    return trajectory′
end


function simulate_trajectory(mdp::MDP, mdp_policy)
    trajectory = []
    
    for (t, (s, a, sp, r)) in enumerate(stepthrough(mdp, mdp_policy,
                                                    "s,a,sp,r",
                                                    max_steps=130))
        @show t
        println("Capacity=$(s.c), time remaining=$(s.t)")
        println(s.v," of size ", s.f)
        if a == ACCEPT
            println("\t——————ACCEPT—————— ✅")
            @info ACCEPT, t
        else
            println("\t(reject) ❌")
        end
        println(sp.v," of size ", sp.f)
        @show r
        println("—"^20)
        push!(trajectory, (s,a,r))
    end

    return trajectory
end


_color_accept = "green!70!black"
_color_reject = "red!70!black"
_visa_status_labels = ["ISIS", "VulAfghan", "P1/P2", "SIV", "AMCIT", ""]

function plot_trajectory(mdp::MDP, trajectory, filename; N=length(trajectory))
    g = DiGraph(N)
    node_styles = Dict()
    node_tags = fill("", nv(g))
    for i in 1:nv(g)
        (s,a,r) = trajectory[i]

        add_edge!(g, i, i+1)
        color = a == ACCEPT ? _color_accept : _color_reject
        rcolor = r <= 0 ? _color_reject : _color_accept

        node_styles[i] =
        "circle, draw=black, fill=$color, minimum size=$(s.f)mm,
         label={[align=center]below:\$t_{$(mdp.params.time-i+1)}\$\\\\
                {\\scriptsize\\color{$rcolor}\$($(round(r, digits=2)))\$}},
         label={[align=center]above:$(_visa_status_labels[Int(s.v)+1])}"
        node_tags[i] = ""
    end
    node_tags[nv(g)] = raw"\ldots"
    node_styles[nv(g)] = ""
    tp = TikzGraphs.plot(g, node_tags, node_styles=node_styles,
                         options="grow'=right, level distance=22mm, semithick, >=stealth'")
    TikzGraphs.save(TikzGraphs.PDF(filename), tp)
    return tp
end


function plot_trajectory(pomdp::POMDP, trajectory, filename; N=length(trajectory))
    g = DiGraph(N)
    node_styles = Dict()
    node_tags = fill("", nv(g))
    for i in 1:nv(g)
        (s,a,o,b,r) = trajectory[i]
        sv = visible(s)
        sh = hidden(s)

        add_edge!(g, i, i+1)
        color = a == ACCEPT ? _color_accept : _color_reject
        rcolor = r <= 0 ? _color_reject : _color_accept

        node_styles[i] =
        "circle, draw=black, fill=$color, minimum size=$(sv.f)mm,
         label={[align=center]below:\$t_{$(pomdp.params.time-i+1)}\$\\\\
                {\\scriptsize\\color{$rcolor}\$($(round(r, digits=2)))\$}},
         label={[align=center]above:$(_visa_status_labels[Int(sh.v)+1])\\\\
                {\\color{gray}($(_visa_status_labels[Int(o.vdoc)+1]))}}"
        node_tags[i] = Int(o.vdoc) != Int(sh.v) ? "{\\color{white}x}" : ""
    end
    node_tags[nv(g)] = raw"\ldots"
    node_styles[nv(g)] = ""
    tp = TikzGraphs.plot(g, node_tags, node_styles=node_styles,
                         options="grow'=right, level distance=22mm, semithick, >=stealth'")
    TikzGraphs.save(TikzGraphs.PDF(filename), tp)
    return tp
end