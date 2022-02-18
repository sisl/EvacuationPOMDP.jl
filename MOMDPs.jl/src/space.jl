"""visiblestates(problem::MOMDP)"""
function visiblestates end

"""hiddenstates(problem::MOMDP)"""
function hiddenstates end

"""visible(problem::MOMDP, s)"""
visible(s::Tuple) = s[1]
visible(m::MOMDP{Sv,Sh,A,O}, s::Tuple{Sv,Sh}) where {Sv,Sh,A,O} = visible(s)

"""
    visible(problem::MOMDP{Tuple{Sv,Sh},A,O}, o::O)

Extract the visible state portion of the observation.
"""
function visible(m::MOMDP{Sv,Sh,A,O}, o::O) where {Sv,Sh,A,O} end

"""hidden(problem::MOMDP, s)"""
hidden(s::Tuple) = s[2]
hidden(m::MOMDP{Sv,Sh,A,O}, s::Tuple{Sv,Sh}) where {Sv,Sh,A,O} = hidden(s)

"""
    hidden(problem::MOMDP{Tuple{Sv,Sh},A,O}, o::O)

Extract the hidden state portion of the observation.
"""
function hidden(m::MOMDP{Sv,Sh,A,O}, o::O) where {Sv,Sh,A,O} end