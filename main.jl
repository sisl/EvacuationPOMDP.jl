https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/1-MDPs.jl.html

struct S
    c # chairs remaining 
    t # time remaining 
    f # family size 
    v # visa status 
end 

struct evacProblem 
    family_distribution
    visa_distribution
  end

  function update(problem, s, U, a)
    if a
        for v in s.v 
            for f in s.f'

              # Apply Bellman Equation
              lookahead



              Transition(s, a) # apply Bellman equation  Q(s,a) = R(s,a)
              # take weighted average of utilities at next state

            # Iterate over all possible next states, weight them by their likelihoods (get from T)


            # I feel like the reward would need to be applied here, how do we do that? 
            # How does the transition function relate to the update function? 

end

function update(s, U)
    
    max(update(s, U, true), update(s, U, false))

end

function Transition(s, a, s')
    if a
        s'.c = s - s.c 
    s'.t = s.t - 1 # how to update  time
end

# dictionary U that maps states to utilities 
function up

    transition function: 
    - give you a probability of landing in a different state (can skip for the project)
    - (ex for state: on or off, actions you can take)
    - transition to the next state 
    
    function update(s,U)
      update()
    end 
    
    function update(s, U, a)
      if a
        for v in visa status 
          for f in family size 


struct MDP
    gamma 
    S
    A
    T
    R
    TR
end 

function lookahead(P::MDP, U, s, a)
    S, T, R, gamma = P.S, P.T, R, P, gamma 
    return R(s,a) + gamma * sum(T(s,a,s_p)*U(s_p) for s_p in S)
end 


function lookahead(P:MDP, U::Vector, s, a)
    S, T, R, gamma = P.S, P.T, P.R, P.gamma 
    return R(s,a) + gamma*sum(T(s,a,s_p) * U[i] for (i,s_p) in enumerate(S))
end 

function backup(P::MDP, U, s)
    maximum(lookahead(P, U, s, a) for a in P.A)
end 

struct ValueIteration 
    k_max
end 

function solve(M::ValueIteration, P::MDP)
    U = [0.0 for s in P.S]
    for k = 1:M.k_max
        U = [backup(P,U,s) for s in P.S]
    end 
    return ValueFunctionPolicy(P,U)
end 