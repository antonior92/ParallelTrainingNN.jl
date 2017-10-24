#*************************************************************************#
#
# AbstractType
#
#*************************************************************************#
abstract type ElementwiseOperator <: AparametricModel end


#*************************************************************************#
#
# Generate Buffers
#
#*************************************************************************#
function dx_buffer(f::ElementwiseOperator)
    return Diagonal{Float64}(zeros(f.noutputs))
end


#*************************************************************************#
#
# LogisticFunction
#
#*************************************************************************#
"""
    LogisticFunction(n)

Initialize `ParametricModel` representing logistic function. It applies
`f(x) = 1 / (1+exp(-x))` elementwise to a given input vector.
`n` is the input vector dimension.
"""
struct LogisticFunction <: ElementwiseOperator
    ninputs::Int
    noutputs::Int
end

function LogisticFunction(n::Int)
    LogisticFunction(n, n)
end

function _evaluate!(f::LogisticFunction,
                    x::AbstractVector{<:Real},
                    z::AbstractVector{Float64},
                    dx::AbstractMatrix{Float64})

    z .= 1 ./ (1+exp.(-x))

    if !isempty(dx)
        dx.diag .=  z .* (1-z)
    end
end


#*************************************************************************#
#
# Hyperbolic Tangent
#
#*************************************************************************#
"""
    HyperbolicTangent(n)

Initialize `ParametricModel` representing hyperbolic tangent. It
applies `f(x) = tanh(x)` elementwise to a given input vector.
 `n` is the input vector dimension.
"""
struct HyperbolicTangent <: ElementwiseOperator
    ninputs::Int
    noutputs::Int
end

function HyperbolicTangent(n::Int)
    HyperbolicTangent(n, n)
end

function _evaluate!(f::HyperbolicTangent,
                    x::AbstractVector{<:Real},
                    z::AbstractVector{Float64},
                    dx::AbstractMatrix{Float64})
    z .= tanh.(x)

    if !isempty(dx)
        dx.diag .=  1 - z.^2
    end
end


#*************************************************************************#
#
# AffineMap
#
#*************************************************************************#
"""
    AffineMap(a[, b])

Initialize `ParametricModel` representing affine function. It applies
`f(x) = a*x + b` elementwise to a given input vector. `a` and
`b` are vectors.
"""
struct AffineMap <: ElementwiseOperator
    ninputs::Int
    noutputs::Int
    a::Vector{Float64}
    b::Vector{Float64}
end

function AffineMap{R<:Real, S<:Real}(a::Vector{R}, b::Vector{S}=zeros(length(a)))
    if length(a) != length(b)
        throw(ArgumentError("Vector `a` and `b` should have the same size."))
    end

    n = length(a)
    AffineMap(n, n, a, b)
end

function _evaluate!(f::AffineMap,
                    x::AbstractVector{<:Real},
                    z::AbstractVector{Float64},
                    dx::AbstractMatrix{Float64})
    z .= f.a.*x + f.b

    if !isempty(dx)
        dx.diag .=  f.a
    end
end


#*************************************************************************#
#
# Identity
#
#*************************************************************************#
"""
    Identity(n)

Initialize `ParametricModel` representing identity function.  It applies
`f(x) = x` elementwise to a given input vector. `n` is the input vector
dimension.
"""
struct Identity <: ElementwiseOperator
    ninputs::Int
    noutputs::Int
end

function Identity(n::Int)
    Identity(n, n)
end

function _evaluate!(f::Identity,
                    x::AbstractVector{<:Real},
                    z::AbstractVector{Float64},
                    dx::AbstractMatrix{Float64})
    z .= x

    if !isempty(dx)
        dx.diag .=  1
    end
end
