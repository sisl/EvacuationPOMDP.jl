function simulate_trajectory(pomdp::POMDP, pomdp_policy; seed=nothing)
    up = updater(pomdp)
    trajectory = []

    if isnothing(seed)
        s₀ = rand(initialstate(pomdp))
        prior_belief = initialize_belief(up, pomdp.params.visa_count, visible(s₀))
        o₀ = rand(observation(pomdp, s₀, REJECT, s₀))
        initial_belief = update(up, prior_belief, REJECT, o₀)
        iterator = enumerate(stepthrough(pomdp, pomdp_policy, up, initial_belief, s₀, "s,a,r,sp,b,o", max_steps=130))
    else
        populations = read_population_json()
        population = populations[seed]
        iterator = enumerate(manual_simulate(pomdp, pomdp_policy, up, population; include_all=true))
        o₀ = last(collect(iterator))[end].o0
    end
    
    for (t, (s, a, r, sp, b, o)) in iterator
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


function simulate_trajectory(mdp::MDP, mdp_policy; seed=nothing)
    trajectory = []
    
    if isnothing(seed)
        iterator = enumerate(stepthrough(mdp, mdp_policy, "s,a,r,sp", max_steps=130))
    else
        populations = read_population_json()
        population = populations[seed]
        iterator = enumerate(manual_simulate(mdp, mdp_policy, population))
    end

    for (t, (s, a, r, sp)) in iterator
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

function plot_trajectory(m::Union{MDP,POMDP}, trajectory, filename; N=length(trajectory))
    half = N÷2 + 1
    N += 1
    g = DiGraph(N+1)
    node_styles = Dict()
    node_tags = fill("", nv(g))
    for i in 1:nv(g)
        nodei = i
        if i != nv(g)
            add_edge!(g, nodei, nodei+1)
        end

        if i == half
            node_tags[nodei] = raw"\ldots"
            node_styles[nodei] = "rectangle, draw=gray, minimum size=5mm"
            continue
        elseif i >= half
            i = (length(trajectory) - half) + (i - half)
        end
        if m isa MDP
            (s,a,r) = trajectory[i]
            obs = ""
            node_tags[nodei] = ""
        elseif m isa POMDP
            (s,a,o,b,r) = trajectory[i]
            obs = "\\\\{\\color{gray}($(_visa_status_labels[Int(o.vdoc)+1]))}"
            node_tags[nodei] = Int(o.vdoc) != Int(getstatus(s)) ? "{\\color{white}x}" : ""
        end
        t = gettime(s)
        c = getcapacity(s)
        f = getfamilysize(s)
        v = getstatus(s)

        color = a == ACCEPT ? _color_accept : _color_reject
        color = nodei == nv(g) ? "gray" : color
        rcolor = r <= 0 ? _color_reject : _color_accept

        node_styles[nodei] =
        "circle, draw=black, fill=$color, minimum size=$(f)mm,
         label={[align=center]below:\$t_{$t}\$\\\\
                \$(c_{$c})\$\\\\
                {\\scriptsize\\color{$rcolor}\$($(round(r, digits=2)))\$}},
         label={[align=center]above:$(_visa_status_labels[Int(v)+1]) $obs}"
    end
    tp = TikzGraphs.plot(g, node_tags, node_styles=node_styles,
                         options="grow'=right, level distance=22mm, semithick, >=stealth'")
    TikzGraphs.save(TikzGraphs.PDF(filename), tp)
    return tp
end
