"""visiblestates(problem::MOMDP)"""
function visiblestates end

"""hiddenstates(problem::MOMDP)"""
function hiddenstates end

"""visible(problem::MOMDP, s)"""
visible(s) = s[1]
visible(m::MOMDP, s) = visible(s)

"""hidden(problem::MOMDP, s)"""
hidden(s) = s[2]
hidden(m::MOMDP, s) = hidden(s)
