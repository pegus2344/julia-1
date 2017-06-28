# This file is a part of Julia. License is MIT: https://julialang.org/license

## Create an extractor that extracts the modified original matrix, e.g.
## LD for BunchKaufman, UL for CholeskyDense, LU for LUDense and
## define size methods for Factorization types using it.

struct BunchKaufman{T,S<:AbstractMatrix} <: Factorization{T}
    LD::S
    ipiv::Vector{BlasInt}
    uplo::Char
    symmetric::Bool
    rook::Bool
    info::BlasInt
end
BunchKaufman(A::AbstractMatrix{T}, ipiv::Vector{BlasInt}, uplo::Char, symmetric::Bool,
             rook::Bool, info::BlasInt) where {T} =
        BunchKaufman{T,typeof(A)}(A, ipiv, uplo, symmetric, rook, info)

"""
    bkfact!(A, uplo::Symbol=:U, symmetric::Bool=issymmetric(A), rook::Bool=false) -> BunchKaufman

`bkfact!` is the same as [`bkfact`](@ref), but saves space by overwriting the
input `A`, instead of creating a copy.
"""
function bkfact!(A::StridedMatrix{<:BlasReal}, uplo::Symbol = :U,
    symmetric::Bool = issymmetric(A), rook::Bool = false)

    if !symmetric
        throw(ArgumentError("Bunch-Kaufman decomposition is only valid for symmetric matrices"))
    end
    if rook
        LD, ipiv, info = LAPACK.sytrf_rook!(char_uplo(uplo), A)
    else
        LD, ipiv, info = LAPACK.sytrf!(char_uplo(uplo), A)
    end
    BunchKaufman(LD, ipiv, char_uplo(uplo), symmetric, rook, info)
end
function bkfact!(A::StridedMatrix{<:BlasComplex}, uplo::Symbol=:U,
    symmetric::Bool=issymmetric(A), rook::Bool=false)

    if rook
        if symmetric
            LD, ipiv, info = LAPACK.sytrf_rook!(char_uplo(uplo), A)
        else
            LD, ipiv, info = LAPACK.hetrf_rook!(char_uplo(uplo), A)
        end
    else
        if symmetric
            LD, ipiv, info = LAPACK.sytrf!(char_uplo(uplo),  A)
        else
            LD, ipiv, info = LAPACK.hetrf!(char_uplo(uplo), A)
        end
    end
    BunchKaufman(LD, ipiv, char_uplo(uplo), symmetric, rook, info)
end

"""
    bkfact(A, uplo::Symbol=:U, symmetric::Bool=issymmetric(A), rook::Bool=false) -> BunchKaufman

Compute the Bunch-Kaufman [^Bunch1977] factorization of a symmetric or Hermitian
matrix `A` and return a `BunchKaufman` object.
`uplo` indicates which triangle of matrix `A` to reference.
If `symmetric` is `true`, `A` is assumed to be symmetric. If `symmetric` is `false`,
`A` is assumed to be Hermitian. If `rook` is `true`, rook pivoting is used. If
`rook` is false, rook pivoting is not used.
The following functions are available for
`BunchKaufman` objects: [`size`](@ref), `\\`, [`inv`](@ref), [`issymmetric`](@ref), [`ishermitian`](@ref).

[^Bunch1977]: J R Bunch and L Kaufman, Some stable methods for calculating inertia and solving symmetric linear systems, Mathematics of Computation 31:137 (1977), 163-179. [url](http://www.ams.org/journals/mcom/1977-31-137/S0025-5718-1977-0428694-0/).

"""
bkfact(A::StridedMatrix{<:BlasFloat}, uplo::Symbol=:U, symmetric::Bool=issymmetric(A),
    rook::Bool=false) =
        bkfact!(copy(A), uplo, symmetric, rook)
bkfact(A::StridedMatrix{T}, uplo::Symbol=:U, symmetric::Bool=issymmetric(A),
    rook::Bool=false) where {T} =
        bkfact!(convert(Matrix{promote_type(Float32, typeof(sqrt(one(T))))}, A),
                uplo, symmetric, rook)

convert(::Type{BunchKaufman{T}}, B::BunchKaufman{T}) where {T} = B
convert(::Type{BunchKaufman{T}}, B::BunchKaufman) where {T} =
    BunchKaufman(convert(Matrix{T}, B.LD), B.ipiv, B.uplo, B.symmetric, B.rook, B.info)
convert(::Type{Factorization{T}}, B::BunchKaufman{T}) where {T} = B
convert(::Type{Factorization{T}}, B::BunchKaufman) where {T} = convert(BunchKaufman{T}, B)

size(B::BunchKaufman) = size(B.LD)
size(B::BunchKaufman, d::Integer) = size(B.LD, d)
issymmetric(B::BunchKaufman) = B.symmetric
ishermitian(B::BunchKaufman) = !B.symmetric

function _ipiv2perm_bk(v::AbstractVector{T}, maxi::Integer, uplo::Char) where T
    p = T[1:maxi;]
    uploL = uplo == 'L'
    i = uploL ? 1 : maxi
    # if uplo == 'U' we construct the permution backwards
    @inbounds while 1 <= i <= length(v)
        vi = v[i]
        if vi > 0 # the 1x1 blocks
            p[i], p[vi] = p[vi], p[i]
            i += uploL ? 1 : -1
        else # the 2x2 blocks
            if uploL
                p[i + 1], p[-vi] = p[-vi], p[i + 1]
                i += 2
            else # 'U'
                p[i - 1], p[-vi] = p[-vi], p[i - 1]
                i -= 2
            end
        end
    end
    return p
end

"""
    getindex(B::BunchKaufman, d::Symbol)

Extract the factors of the Bunch-Kaufman factorization `B`. The factorization can take the
two forms `L*D*L.'` or `U*D*U.'` where `L` is a `UnitLowerTriangular` matrix, `U` is a
`UnitUpperTriangular`, and `D` is a block diagonal matrix with 1x1 or 2x2 blocks. The argument
`d` can be
- `:D`: the block diagonal matrix
- `:L`: the lower triangular factor (if factorization is `L*D*L.'`)
- `:U`: the lower triangular factor (if factorization is `U*D*U.'`)
- `:p`: permutation vector
- `:P`: permutation matrix

```jldoctest
julia> A = [1 2 3; 2 1 2; 3 2 1]
3×3 Array{Int64,2}:
 1  2  3
 2  1  2
 3  2  1

julia> F = bkfact(Symmetric(A, :L));

julia> F[:L]*F[:D]*F[:L].' - A[F[:p], F[:p]]
3×3 Array{Float64,2}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> F = bkfact(Symmetric(A));

julia> F[:U]*F[:D]*F[:U].' - F[:P]*A*F[:P]'
3×3 Array{Float64,2}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
```
"""
function getindex(B::BunchKaufman{T}, d::Symbol) where {T<:BlasFloat}
    n = size(B, 1)
    if d == :p
        return _ipiv2perm_bk(B.ipiv, n, B.uplo)
    elseif d == :P
        return eye(T, n)[:,invperm(B[:p])]
    elseif d == :L || d == :U || d == :D
        if B.rook
            throw(ArgumentError("reconstruction rook pivoted Bunch-Kaufman factorization not implemented yet"))
        else
            LUD, od = LAPACK.syconv!(B.uplo, copy(B.LD), B.ipiv)
        end
        if d == :D
            D = diagm(diag(LUD))
            for i in 1:n
                if !iszero(od[i])
                    odi = od[i]
                    if B.uplo == 'L'
                        D[i, i + 1] = B.symmetric ? odi : odi'
                        D[i + 1, i] = odi
                    else # 'U'
                        D[i, i - 1] = B.symmetric ? odi : odi'
                        D[i - 1, i] = odi
                    end
                end
            end
            return D
        elseif d == :L
            if B.uplo == 'L'
                return UnitLowerTriangular(LUD)
            else
                throw(ArgumentError("factorization is U*D*U.' but you requested L"))
            end
        else # :U
            if B.uplo == 'U'
                return UnitUpperTriangular(LUD)
            else
                throw(ArgumentError("factorization is L*D*L.' but you requested U"))
            end
        end
    else
        throw(KeyError(d))
    end
end

issuccess(B::BunchKaufman) = B.info == 0

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, B::BunchKaufman)
    println(io, summary(B))
    println(io, "D factor:")
    show(io, mime, B[:D])
    println(io, "\n$(B.uplo) factor:")
    show(io, mime, B[Symbol(B.uplo)])
    println(io, "\npermutation:")
    show(io, mime, B[:p])
    print(io, "\nsuccessful: $(issuccess(B))")
end

function inv(B::BunchKaufman{<:BlasReal})
    if !issuccess(B)
        throw(SingularException(B.info))
    end

    if B.rook
        copytri!(LAPACK.sytri_rook!(B.uplo, copy(B.LD), B.ipiv), B.uplo, true)
    else
        copytri!(LAPACK.sytri!(B.uplo, copy(B.LD), B.ipiv), B.uplo, true)
    end
end

function inv(B::BunchKaufman{<:BlasComplex})
    if !issuccess(B)
        throw(SingularException(B.info))
    end

    if issymmetric(B)
        if B.rook
            copytri!(LAPACK.sytri_rook!(B.uplo, copy(B.LD), B.ipiv), B.uplo)
        else
            copytri!(LAPACK.sytri!(B.uplo, copy(B.LD), B.ipiv), B.uplo)
        end
    else
        if B.rook
            copytri!(LAPACK.hetri_rook!(B.uplo, copy(B.LD), B.ipiv), B.uplo, true)
        else
            copytri!(LAPACK.hetri!(B.uplo, copy(B.LD), B.ipiv), B.uplo, true)
        end
    end
end

function A_ldiv_B!(B::BunchKaufman{T}, R::StridedVecOrMat{T}) where T<:BlasReal
    if !issuccess(B)
        throw(SingularException(B.info))
    end

    if B.rook
        LAPACK.sytrs_rook!(B.uplo, B.LD, B.ipiv, R)
    else
        LAPACK.sytrs!(B.uplo, B.LD, B.ipiv, R)
    end
end
function A_ldiv_B!(B::BunchKaufman{T}, R::StridedVecOrMat{T}) where T<:BlasComplex
    if !issuccess(B)
        throw(SingularException(B.info))
    end

    if B.rook
        if issymmetric(B)
            LAPACK.sytrs_rook!(B.uplo, B.LD, B.ipiv, R)
        else
            LAPACK.hetrs_rook!(B.uplo, B.LD, B.ipiv, R)
        end
    else
        if issymmetric(B)
            LAPACK.sytrs!(B.uplo, B.LD, B.ipiv, R)
        else
            LAPACK.hetrs!(B.uplo, B.LD, B.ipiv, R)
        end
    end
end
# There is no fallback solver for Bunch-Kaufman so we'll have to promote to same element type
function A_ldiv_B!(B::BunchKaufman{T}, R::StridedVecOrMat{S}) where {T,S}
    TS = promote_type(T,S)
    return A_ldiv_B!(convert(BunchKaufman{TS}, B), convert(AbstractArray{TS}, R))
end

function logabsdet(F::BunchKaufman)
    M = F.LD
    p = F.ipiv
    n = size(F.LD, 1)

    if !issuccess(F)
        return eltype(F)(-Inf), zero(eltype(F))
    end
    s = one(real(eltype(F)))
    i = 1
    abs_det = zero(real(eltype(F)))
    while i <= n
        if p[i] > 0
            elm = M[i,i]
            s *= sign(elm)
            abs_det += log(abs(elm))
            i += 1
        else
            # 2x2 pivot case. Make sure not to square before the subtraction by scaling
            # with the off-diagonal element. This is safe because the off diagonal is
            # always large for 2x2 pivots.
            if F.uplo == 'U'
                elm = M[i, i + 1]*(M[i,i]/M[i, i + 1]*M[i + 1, i + 1] -
                    (issymmetric(F) ? M[i, i + 1] : conj(M[i, i + 1])))
                s *= sign(elm)
                abs_det += log(abs(elm))
            else
                elm = M[i + 1,i]*(M[i, i]/M[i + 1, i]*M[i + 1, i + 1] -
                    (issymmetric(F) ? M[i + 1, i] : conj(M[i + 1, i])))
                s *= sign(elm)
                abs_det += log(abs(elm))
            end
            i += 2
        end
    end
    return abs_det, s
end

## reconstruct the original matrix
## TODO: understand the procedure described at
## http://www.nag.com/numeric/FL/nagdoc_fl22/pdf/F07/f07mdf.pdf
