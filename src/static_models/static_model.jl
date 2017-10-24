#*************************************************************************#
#
# Abstract Types
#
#*************************************************************************#
"""
    ParametricModel

Define a static model relating input and output.
"""
abstract type StaticModel end

"""
    ParametricModel

Define a static parametric model `z = f(x, Θ)`.

Basic Functions
---------------

| Function                          | Description                                                 |
|:--------------------------------- |:----------------------------------------------------------- |
| `evaluate!(f, x, Θ, z[, dx, dΘ])` | Compute the output `z` and its first derivatives            |
| `z_buffer(f)`                     | Generate `z` buffer to be used in `evaluate!`               |
| `dx_buffer(f)`                    | Generate `dx` buffer to be used in `evaluate!`              |
| `dΘ_buffer(f)`                    | Generate `dΘ` buffer to be used in `evaluate!`              |
| `isparametric(f)`                 | Return `true` if the given StaticModel is a ParametricModel |
| `length_input(f)`                 | Return number of model inputs                               |
| `length_output(f)`                | Return number of model outputs                              |
| `length_params(f)`                | Return number of model parameters                           |
"""
abstract type ParametricModel <: StaticModel end


"""
    AparametricModel

Define a static aparametric model `z = f(x)`.

Basic Functions
---------------

| Function                   | Description                                                   |
|:-------------------------- |:-----------------------------------------------------------   |
| `evaluate!(f, x, z[, dx])` | Compute the output `z` and its first derivatives              |
| `z_buffer(f)`              | Generate `z` buffer to be used in `evaluate!`                 |
| `dx_buffer(f)`             | Generate `dx` buffer to be used in `evaluate!`                |
| `isparametric(f)`          | Return `false` if the given StaticModel is a AparametricModel |
| `length_input(f)`          | Return number of model inputs                                 |
| `length_output(f)`         | Return number of model outputs                                |
"""
abstract type AparametricModel <: StaticModel end


#*************************************************************************#
#
# Generate Buffers
#
#*************************************************************************#
"""
    z_buffer(f::StaticModel) -> z

Return the array to be used as buffer for the output vector `z`
"""
function z_buffer(f::StaticModel)
    return Vector{Float64}(f.noutputs)
end


"""
    dx_buffer(f::StaticModel) -> dx

Return the array to be used as buffer for the Jacobian matrix `dx`.
"""
function dx_buffer(f::StaticModel)
    return Matrix{Float64}(f.noutputs, f.ninputs)
end

"""
    dΘ_buffer(f::ParametricModel) -> dΘ

Return the array to be used as buffer for the Jacobian matrix `dΘ`.
"""
function dΘ_buffer(f::ParametricModel)
    return Matrix{Float64}(f.noutputs, f.nparams)
end


#*************************************************************************#
#
# Compute Output and its Derivatives
#
#*************************************************************************#
"""
    evaluate!(f::ParametricModel, x, Θ, z[, dx, dΘ])

For a static function `z = f(x, Θ)` compute the
output `z` and its first derivatives `dx` and `dΘ.

The real vectors `x` and `Θ` contains respectively the
input and parameters to the static model `f`.  The computed
values are written in `z`, `dx` and `dΘ`. Which are the static
function output and the jacobian matrices containing derivatives in
relation to `x` and `Θ`.
"""
function evaluate!(f::ParametricModel,
                   x::AbstractVector{<:Real},
                   Θ::AbstractVector{<:Real},
                   z::AbstractVector{Float64};
                   dx::AbstractMatrix{Float64}=Matrix{Float64}(0, 0),
                   dΘ::AbstractMatrix{Float64}=Matrix{Float64}(0, 0))

    # ---
    # Check Arguments
    # ---
    nx = length(x)
    nparams = length(Θ)
    nz = length(z)
    size_dx = size(dx)
    size_dΘ = size(dΘ)

    if nx != f.ninputs
        throw(ArgumentError(string("Vector `x` have $nx elements when it should ",
                                   "have $(f.ninputs).")))
    end
    if nparams != f.nparams
        throw(ArgumentError(string("Vector `Θ` have $nparams elements when it ",
                                   "should have $(f.nparams).")))
    end
    if nz != f.noutputs
        throw(ArgumentError(string("Vector `z` have $nz elements when it should ",
                                   "have $(f.noutputs).")))
    end
    if !isempty(dx) && size_dx != (f.noutputs, f.ninputs)
        throw(ArgumentError(string("The matrix `dx` have dimension $(size_dx) ",
                                   "when it should have dimension ",
                                   "$((f.noutputs, f.ninputs)).")))
    end
    if !isempty(dΘ) && size_dΘ != (f.noutputs, f.nparams)
        throw(ArgumentError(string("The matrix `dΘ` have ",
                                   "dimension $(size_dΘ) ",
                                   "when it should have dimension ",
                                   "$((f.noutputs, f.nparams)).")))
    end
    # ---

    # Evaluate
    _evaluate!(f, x, Θ, z, dx, dΘ)
end


"""
    evaluate!(f::ParametricModel, x, z[, dx])

For a static function `z = f(x)` compute the
output `z` and its first derivatives `dx`.

The real vector `x` contain respectively the
input and parameters to the static model `f`.  The computed
values are written in `z`, `dx`. Which are the static
function output and the  jacobian matrix.
"""
function evaluate!(f::AparametricModel,
                   x::AbstractVector{<:Real},
                   z::AbstractVector{Float64};
                   dx::AbstractMatrix{Float64}=Matrix{Float64}(0, 0))

    # ---
    # Check Arguments
    # ---
    nx = length(x)
    nz = length(z)
    size_dx = size(dx)

    if nx != f.ninputs
        throw(ArgumentError(string("Vector `x` have $nx elements when it should ",
                                   "have $(f.ninputs).")))
    end
    if nz != f.noutputs
        throw(ArgumentError(string("Vector `z` have $nz elements when it should ",
                                   "have $(f.noutputs).")))
    end
    if !isempty(dx) && size_dx != (f.noutputs, f.ninputs)
        throw(ArgumentError(string("The matrix `dx` have dimension $(size_dx) ",
                                   "when it should have dimension ",
                                   "$((f.noutputs, f.ninputs)).")))
    end
    # ---

    # Evaluate
    _evaluate!(f, x, z, dx)
end


#*************************************************************************#
#
# Auxiliar Functions
#
#*************************************************************************#
function isparametric(f::StaticModel)
    typeof(f) <: ParametricModel
end

function length_input(f::StaticModel)
    return f.ninputs
end

function length_output(f::StaticModel)
    return f.noutputs
end

function length_params(f::ParametricModel)
    return f.nparams
end
