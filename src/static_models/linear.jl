#*************************************************************************#
#
# Linear Type
#
#*************************************************************************#
"""
    Linear(ninputs, noutputs)

Initialize `ParametricModel` representing linear operator `z = A*x`.
Where `A` is a dense matrix with dimension `(noutputs, ninputs) `
cointaining the reshaped parameter vector `Θ`.
"""
struct Linear <: ParametricModel
    ninputs::Int
    noutputs::Int
    nparams::Int
end

# Constructor
function Linear(ninputs::Int, noutputs::Int)
    nparams = ninputs*noutputs
    Linear(ninputs, noutputs, nparams)
end


#*************************************************************************#
#
# Generate Buffers
#
#*************************************************************************#
function dΘ_buffer(lin::Linear)
    if lin.noutputs == 1
        return Matrix{Float64}(lin.noutputs, lin.nparams)
    else
        return hcat((speye(lin.noutputs) for i=1:lin.ninputs)...)
    end
end


#*************************************************************************#
#
# Compute Output and its Derivatives
#
#*************************************************************************#
function _evaluate!(lin::Linear,
                    x::AbstractVector{<:Real},
                    Θ::AbstractVector{<:Real},
                    z::AbstractVector{Float64},
                    dx::AbstractMatrix{Float64},
                    dΘ::AbstractMatrix{Float64})
    # Get Matrix
    matrix = reshape(Θ, lin.noutputs, lin.ninputs)

    # Compute z
    z .= matrix*x

    # Compute dz
    if !isempty(dx)
        dx .= matrix
    end

    # Compute dΘ
    if !isempty(dΘ)
        for i = 1:lin.ninputs
            for j = 1:lin.noutputs
                dΘ[j, (i-1)*lin.noutputs+j] = x[i]
            end
        end
    end
end


#*************************************************************************#
#
# Initializer
#
#*************************************************************************#
"""
    ScaledGaussianInitializer()

The `ScaledGaussianInitializer` will initialize the parameter vector
with of random values picked from normal distribution with standard
deviation of `1/sqrt(ninputs)`.

Valid only for Linear static model.
"""
immutable ScaledGaussianInitializer <: Initializer
end

function initial_guess(lin::Linear, init::ScaledGaussianInitializer)
    d = Distributions.Normal(0, 1/sqrt(lin.ninputs))
    
    return rand(d, lin.nparams)
end

function initial_guess(lin::Linear, init::DefaultInitializer)
    initial_guess(lin, ScaledGaussianInitializer())
end
