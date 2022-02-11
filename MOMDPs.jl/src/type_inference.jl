"""
```
type A <: MOMDP{Int, Float64, Bool, Bool} end
visiblestatetype(A) # returns Int
```
"""
visiblestatetype(t::Type) = visiblestatetype(supertype(t))
visiblestatetype(t::Type{MOMDP{Sv,Sh,A,O}}) where {Sv,Sh,A,O} = Sv
visiblestatetype(p::MOMDP) = visiblestatetype(typeof(p))

"""
```
type A <: MOMDP{Int, Float64, Bool, Bool} end
hiddenstatetype(A) # returns Float64
```
"""
hiddenstatetype(t::Type) = hiddenstatetype(supertype(t))
hiddenstatetype(t::Type{MOMDP{Sv,Sh,A,O}}) where {Sv,Sh,A,O} = Sh
hiddenstatetype(p::MOMDP) = hiddenstatetype(typeof(p))
