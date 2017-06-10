module Algorithms

export replace!, replace

import Base: replace


"""
    replace!(pred, [f::Function], A, [new], [count])

Replace all occurrences `x` in collection `A` for which `pred(x)` is true
by `new` or `f(x)` (exactly one among `f` and `new` must be specified).
If `count` is specified, then replace at most `count` occurrences.
This is the in-place version of [`replace`](@ref).

# Examples
```jldoctest
julia> a = [1, 2, 3, 1];

julia> replace!(isodd, a, 0, 2); a
4-element Array{Int64,1}:
 0
 2
 0
 1

julia> replace!(x->x.first=>3, Dict(1=>2, 3=>4), 1) do (k, v)
           v < 3
       end

Dict{Int64,Int64} with 2 entries:
  3 => 4
  1 => 3
```

!!! note
When `A` is an `Associative` or `AbstractSet` collection, if
there are collisions among old and newly created keys, the result
can be unexpected:

```jldoctest
julia> replace!(x->true, x->2x, Set([3, 6]))
Set([12])
```
"""
function replace!(pred, new::Function, A::AbstractArray, n::Integer=-1)
    n == 0 && return A
    count = 0
    @inbounds for i in eachindex(A)
        if pred(A[i])
            A[i] = new(A[i])
            count += 1
            count == n && break
        end
    end
    A
end

askey(k, ::Associative) = k.first
askey(k, ::AbstractSet) = k

function replace!(pred, new::Function, A::Union{Associative,AbstractSet}, n::Integer=-1)
    n == 0 && return A
    del = eltype(A)[]
    count = 0
    for x in A
        if pred(x)
            push!(del, x)
            count += 1
            count == n && break
        end
    end
    for k in del
        pop!(A, askey(k, A))
        push!(A, new(k))
    end
    A
end

const ReplaceCollection = Union{AbstractArray,Associative,AbstractSet}

replace!(pred, A::ReplaceCollection, new, n::Integer=-1) = replace!(pred, y->new, A, n)

"""
    replace(pred, [f::Function], A, [new], [count])

Return a copy of collection `A` where all occurrences `x` for which
`pred(x)` is true are replaced by `new` or `f(x)` (exactly one among
`f` and `new` must be specified).
If `count` is given, then replace at most `count` occurrences.
See the in-place version [`replace!`](@ref) for examples.
"""
replace(pred, new::Function, A::ReplaceCollection, n::Integer=-1) = replace!(pred, new, copy(A), n)
replace(pred, A::ReplaceCollection, new, n::Integer=-1) = replace!(pred, copy(A), new, n)

"""
    replace!(A, old, new, [count])

Replace all occurrences of `old` in collection `A` by `new`.
If `count` is given, then replace at most `count` occurrences.

# Examples
```jldoctest
julia> replace!(Set([1, 2, 3]), 1, 0)
Set([0, 2, 3])
```
"""
replace!(A::ReplaceCollection, old, new, n::Integer=-1) = replace!(x->x==old, A, new, n)

"""
    replace(A, old, new, [count])

Return a copy of collection `A` where all occurrences of `old` are
replaced by `new`.
If `count` is given, then replace at most `count` occurrences.
"""
replace(A::ReplaceCollection, old, new, n::Integer=-1) = replace!(copy(A), old, new, n)

end
