#*************************************************************************#
#
# Bias Type
#
#*************************************************************************#
"""
    Bias(n)

Initialize `ParametricModel` representing bias operation `z = x + Θ`.
`n` is the input vector dimension.
"""
struct Bias <: ParametricModel
    ninputs::Int
    noutputs::Int
    nparams::Int
end

function Bias(n::Int)
    Bias(n, n, n)
end


#*************************************************************************#
#
# Generate Buffers
#
#*************************************************************************#
function dx_buffer(bias::Bias)
    return Diagonal{Float64}(zeros(bias.noutputs))
end

function dΘ_buffer(bias::Bias)
    return Diagonal{Float64}(zeros(bias.nparams))
end


#*************************************************************************#
#
# Compute Output and its Derivatives
#
#*************************************************************************#
function _evaluate!(bias::Bias,
                    x::AbstractVector{<:Real},
                    Θ::AbstractVector{<:Real},
                    z::AbstractVector{Float64},
                    dx::AbstractMatrix{Float64},
                    dΘ::AbstractMatrix{Float64})

    # Compute z
    z .= x + Θ

    # Compute dx
    if !isempty(dx)
        dx.diag .= 1
    end

    # Compute dΘ
    if !isempty(dΘ)
        dΘ.diag .= 1
    end
end
