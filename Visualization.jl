using POMDPs
using POMDPModelTools 
using POMDPPolicies 
using QuickPOMDPs
using Parameters, Random 

using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

```
PSEUDOCODE 

1. Figure out how to generate plot with j
```

struct State
	x::Int
	y::Int
end

@with_kw struct GridWorldParameters
	size::Tuple{Int,Int} = (10, 10)   # size of the grid
	null_state::State = State(-1, -1) # terminal state outside of the grid
	p_transition::Real = 0.7 # probability of transitioning to the correct next state
end	

params = GridWorldParameters();

𝒮 = [[State(x,y) for x=1:params.size[1], y=1:params.size[2]]..., params.null_state]

# the possible actions \scrs<TAB> are wither accepting or rejecting a family
@enum Action REJECT ACCEPT

#A = [REJECT::Action=0, ACCEPT::Action=1]
A = Action

𝒜 = [REJECT, ACCEPT]

inbounds(s::State) = 1 ≤ s.x ≤ params.size[1] && 1 ≤ s.y ≤ params.size[2]

function R(s, a)
    if s == State()
        return 1
    elseif s == State()
        return 5
    elseif s == State()
        return 10
    elseif s == State()
        return -10
    else 
        return 0
    end 
end

function T(s::State, a::Action)
	if R(s) != 0
		return Deterministic(params.null_state)
	end

	Nₐ = length(𝒜)
	next_states = Vector{State}(undef, Nₐ + 1)
	probabilities = zeros(Nₐ + 1)
	p_transition = params.p_transition

	for (i, a′) in enumerate(𝒜)
		prob = (a′ == a) ? p_transition : (1 - p_transition) / (Nₐ - 1)
		destination = s + MOVEMENTS[a′]
		next_states[i+1] = destination

		if inbounds(destination)
			probabilities[i+1] += prob
		end
	end
	
	# handle out-of-bounds transitions
	next_states[1] = s
	probabilities[1] = 1 - sum(probabilities)

	return SparseCat(next_states, probabilities)
end

abstract type GridWorld <: MDP{State, Action} end

mdp = QuickMDP(GridWorld,
    states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    #discount     = γ,
    initialstate = 𝒮,
    #isterminal   = termination,
    render       = render);

render(mdp, mdp)

render = plot_grid_world

render(mdp)


