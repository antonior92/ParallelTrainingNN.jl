#*************************************************************************#
#
# Linear Type
#
#*************************************************************************#
"""
    LinearSparse(ninputs, noutputs, i, j)

Initialize `ParametricModel` representing sparse linear operator `z = A*x`.
Where `A` is a sparse matrix with dimension `(noutputs, ninputs) `
cointaining the the parameter vector `Θ` in the position specified
in the elements of `i` and `j`.
"""
struct LinearSparse <: ParametricModel
    ninputs::Int
    noutputs::Int
    nparams::Int
    i::Vector{Int}
    j::Vector{Int}
end

# Constructor
function LinearSparse(ninputs::Int, noutputs::Int, i::Vector{Int}, j::Vector{Int})
    if (length(i) != length(j))
        throw(ArgumentError("Index vector `i` and `j` should have the same length."))
    end
    nparams = length(i)
    LinearSparse(ninputs, noutputs, nparams, i, j)
end

#*************************************************************************#
#
# Generate Buffers
#
#*************************************************************************#
function dx_buffer(lin::LinearSparse)
    return sparse(lin.i, lin.j, ones(lin.nparams), lin.noutputs, lin.ninputs)
end

function dΘ_buffer(lin::LinearSparse)
    row = lin.i
    column = 1:lin.nparams
    return sparse(row, column, ones(lin.nparams), lin.noutputs, lin.nparams)
end

#*************************************************************************#
#
# Compute Output and its Derivatives
#
#*************************************************************************#
function _evaluate!(lin::LinearSparse,
                    x::AbstractVector{<:Real},
                    Θ::AbstractVector{<:Real},
                    z::AbstractVector{Float64},
                    dx::AbstractMatrix{Float64},
                    dΘ::AbstractMatrix{Float64})

    # TODO: Improve memory allocation

    # Get Matrix
    matrix = sparse(lin.i, lin.j, Θ, lin.noutputs, lin.ninputs)

    # Compute z
    z .= matrix*x

    # Compute dz
    if !isempty(dx)
        dx .= matrix
    end

    # Compute dΘ
    if !isempty(dΘ)
        row = lin.i
        column = 1:lin.nparams
        value = [x[j] for j in lin.j]
        dΘ .= sparse(row, column, value, lin.noutputs, lin.nparams)
    end
end
