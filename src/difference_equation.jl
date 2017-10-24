#*************************************************************************#
#
# DifferenceEquation Type
#
#*************************************************************************#
"""
    DifferenceEquation(mdl::ParametricModel, yterms, uterms)

Type representing a difference equation. `mdl` is a static function.
It relates the output at a given instant with its previous values
and with the input. The `yterms` and `uterms` are vector of vectors
containing integers, each one specifying a dependence with previous
terms of `y` (or with input terms):

    y[k] = mdl(y_1[k-yterms[1][1]], y_1[k-yterms[1][2]],..., y_1[k-yterms[1][end]],
               y_2[k-yterms[2][1]], y_2[k-yterms[2][2]],..., y_2[k-yterms[2][end]],
               ...
               y_end[k-yterms[end][1]], y_end[k-yterms[end][2]],..., y_1[k-yterms[end][end]],
               u_1[k-uterms[1][1]], u_1[k-uterms[1][2]],..., u_1[k-uterms[1][end]],
               u_2[k-uterms[2][1]], u_2[k-uterms[2][2]],..., u_2[k-uterms[2][end]],
               ...
               u_end[k-uterms[end][1]], u_end[k-uterms[end][2]],..., u_1[k-yterms[end][end]], Θ)

The following values are internally stored:

| Atributes | Type                  | Brief Description                                     |
|:--------- |:--------------------- |:----------------------------------------------------- |
| mdl       | `StaticModel`         | Static model.                                         |
| yterms    | `Vector{Vector{Int}}` | Contains dependencies on previous terms of `y`.       |
| uterms    | `Vector{Vector{Int}}` | Contains dependencies on previous terms of `u`.       |
| ny        | `Int`                 | Number of inputs `ny = length(yterms)`.               |
| nu        | `Int`                 | Number of outputs `nu = length(uterms)`.              |
| n         | `Int`                 | System order. Maximum value in `yterms`.              |
| iodelay   | `Int`                 | Input-output delay. Minimum value in `uterms`.        |
| m         | `Int`                 | Distance between maximum and minimum `uterms` values. |

The previous  mentioned difference equation can be compactedly
represented using mathematical form:

   y[k] = F(y[k-1],..., y[k-n], u[k-iodelay],..., u[k-iodelay-m+1], Θ)
"""
struct DifferenceEquation{M<:StaticModel}
    mdl::M
    yterms::Vector{Vector{Int}}
    uterms::Vector{Vector{Int}}
    ny::Int
    nu::Int
    n::Int
    m::Int
    iodelay::Int

    # Internal buffers
    x_buf::AbstractVector{Float64}
    z_buf::AbstractVector{Float64}
    dΘ_buf::AbstractMatrix{Float64}
    dx_buf::AbstractMatrix{Float64}
end


function DifferenceEquation(mdl::ParametricModel;
                            yterms::Vector{Vector{Int}}=Vector{Vector{Int}}(0),
                            uterms::Vector{Vector{Int}}=Vector{Vector{Int}}(0))

    # Get number of inputs, outputs terms
    ny = length(yterms)
    nu = length(uterms)

    # Compute order of terms
    n = isempty(yterms) ? 0 : maximum([maximum(z) for z in yterms])
    iodelay = isempty(uterms) ? 0 : minimum([minimum(z) for z in uterms])
    m = isempty(uterms) ? 0 : maximum([maximum(z)-iodelay+1 for z in uterms])


    # Number of input to the static model
    ninputs = sum([length(z) for z in [yterms; uterms]])

    # Check coherence with static model
    if ny != 0 && ny != length_output(mdl)
        throw(ArgumentError("yterms should have $(length_output(mdl)) vectors, one
                             for each output."))
    end
    ny = length_output(mdl)

    if ninputs != length_input(mdl)
        throw(ArgumentError("The total number of inputs to the static model
                             is $ninputs when it should be $(length_input(mdl))."))
    end

    # Check yterms
    for ynterms in yterms
        for j in ynterms
            if j <= 0
                throw(ArgumentError("yterms should not contain non-positive elements."))
            end
        end
    end

    # Initialize buffers
    x_buf = Vector{Float64}(length_input(mdl))
    z_buf = z_buffer(mdl)
    dΘ_buf = dΘ_buffer(mdl)
    dx_buf = dx_buffer(mdl)

    # Call original constructor
    DifferenceEquation(mdl, yterms, uterms, ny, nu, n, m, iodelay,
                       x_buf, z_buf, dΘ_buf, dx_buf)
end


#*************************************************************************#
#
# System Simulation
#
#*************************************************************************#

"""
    simulate(diffeq::DifferenceEquation, u, Θ, y0) -> y

For a given input vector `u` and initial conditions `y0` simulate the system,
returning the output `y`.

Inputs:
------
* `diffeq` is a structure representing a difference equation.
* `u` is a matrix containing the sequence of inputs. It should have
 dimension `nu`-by-`nd`, where `nd` is the number of observations
* `Θ` is the parameter vector, containing `nparams` elements.
* `y0` is a matrix containing the initial conditions. It should have
 dimension `ny`-by-`n`.

Outputs:
-------
* `y` is the sequence of ouputs. It is a matrix with dimension `ny`-by-`nd-m+1`.

The dimension `nd` is infered from the provided vectors.
"""
function simulate{R<:Real,
                  S<:Real,
                  T<:Real}(diffeq::DifferenceEquation, u::AbstractMatrix{T},
                           Θ::AbstractVector{R}, y0::AbstractMatrix{S})

    # Get sizes
    nu, nd = size(u)
    ny = diffeq.ny
    m = diffeq.m

    # Create vector
    y = Matrix{Float64}(ny, nd-m+1)

    # Simulate
    simulate!(diffeq, u, Θ, y0, y)

    # Return the computed vector
    return y
end


"""
    simulate!(diffeq::DifferenceEquation, u, Θ, y0, y[, dΘ, dy0])

For a given input vector `u` and initial conditions `y0` simulate the system
and store the results in the given vector `y`. When the arrays `dΘ`
and `dy0` are provided they are subscribed with the partial derivatives
in relation to `Θ` and `y0`.

Inputs:
------
* `diffeq` is a structure representing a difference equation.
* `u` is a matrix containing the sequence of inputs. It should  have
 dimension `nu`-by-`nd+m-1`, where `nd` is the number of observations
* `Θ` is the parameter vector, containing `nparams` elements.
* `y0` is a matrix containing the initial conditions. It should have
 dimension `ny`-by-`n`.

Outputs:
-------
* `y` is buffer where the computed output will be written. It
should be a matrix with dimension `ny`-by-`nd`, where `nd` is
 the number of observations
* `dΘ`  is buffer where the computed derivatives `dy/dΘ`
 will be written.  It should have dimensions `ny`-by-`nd`-by-`nparams`.
* `dy0` is buffer where the computed derivatives `dy/dy0`
 will be written. It should have dimensions `ny`-by-`nd`-by-`ny`-by-`n`.

If empty arrays are provided for `y`, `dΘ` or `dy0` the respective
computations are bypassed.

The dimension `nd` is infered from the provided vectors.
"""
function simulate!(diffeq::DifferenceEquation,
                   u::AbstractMatrix{<:Real},
                   Θ::AbstractVector{<:Real},
                   y0::AbstractMatrix{<:Real},
                   y::AbstractMatrix{<:Real};
                   dΘ::AbstractArray{<:Real, 3}=Array{Float64}(0, 0, 0),
                   dy0::AbstractArray{<:Real, 4}=Array{Float64}(0, 0, 0, 0))
    # Check Arguments
    argcheck_simulate(diffeq, u, Θ, y0, y, dΘ, dy0)

    # Get Model
    mdl = diffeq.mdl

    # Get model values
    n = diffeq.n
    m = diffeq.m
    iodelay = diffeq.iodelay
    ny = diffeq.ny
    nu = diffeq.nu

    # Get data set size
    nd = size(y, 2)

    # Initialize buffers
    x = diffeq.x_buf
    z = diffeq.z_buf
    if isempty(dΘ)
        dtheta = Matrix{Float64}(0, 0)
    else
        dtheta = diffeq.dΘ_buf
    end

    if isempty(dΘ) && isempty(dy0)
        dx = Matrix{Float64}(0, 0)
    else
        dx = diffeq.dx_buf
    end

    # TODO: vectorize this computation to improve legibility and versatility of the code
    # Compute output sequence
    for k = 1:nd
        # Assemble input vector
        ind = 1
        if n != 0
            for i = 1:ny
                for yterm in diffeq.yterms[i]
                    x[ind] = k-yterm>0 ? y[i, k-yterm]:y0[i, k+n-yterm]
                    ind += 1
                end
            end
        end
        for i = 1:nu
            for uterm in diffeq.uterms[i]
                x[ind] = u[i, k+m-uterm+iodelay-1]
                ind += 1
            end
        end

        # Evaluate input function
        evaluate!(mdl, x, Θ, z, dx=dx, dΘ=dtheta)
        y[:, k] .= z
        
        # Compute dΘ
        if !isempty(dΘ)
            dΘ[:, k, :] = dtheta
            for i = 1:ny
                ind = 1
                for j = 1:length(diffeq.yterms)
                    for yterm in diffeq.yterms[j]
                        if k-yterm>0
                            dΘ[i, k, :] += dx[i, ind]*dΘ[j, k-yterm, :]
                        end
                        ind += 1
                    end
                end
            end
        end # End compute dΘ

        # Compute dy0
        if !isempty(dy0)
            for i = 1:ny
                for j = 1:ny
                    for p = 1:n
                        ind = 1
                        dy0[i, k, j, p] = 0
                        for l = 1:length(diffeq.yterms)
                            for yterm in diffeq.yterms[l]
                                if k-yterm>0
                                    dy0[i, k, j, p] += dx[i, ind]*dy0[l, k-yterm, j, p]
                                elseif p==k-yterm+n && j==l
                                    dy0[i, k, j, p] += dx[i, ind]
                                end
                                ind += 1
                            end
                        end
                    end
                end
            end
        end # End compute dy0

    end
end


function argcheck_simulate(diffeq::DifferenceEquation,
                           u::AbstractMatrix{<:Real},
                           Θ::AbstractVector{<:Real},
                           y0::AbstractMatrix{<:Real},
                           y::AbstractMatrix{<:Real},
                           dΘ::AbstractArray{<:Real, 3},
                           dy0::AbstractArray{<:Real, 4})

    # Get dimensions
    size_u = size(u)
    nparams = length(Θ)
    size_y0 = size(y0)
    ny, nd = size(y)
    size_dΘ = size(dΘ)
    size_dy0 = size(dy0)
    mdl = diffeq.mdl

    # Argument Check
    if ny != diffeq.ny
        throw(ArgumentError(string("The matrix `y` have $ny rows when it should ",
                                   "have $(diffeq.ny).")))
    end
    if !isempty(u) && size_u != (diffeq.nu, nd+diffeq.m-1)
        throw(ArgumentError(string("The matrix `u` have dimension $(size_u) ",
                                   "when it should have dimension ",
                                   "$((diffeq.nu, nd+diffeq.m-1)).")))
    end
    if nparams != length_params(mdl)
        throw(ArgumentError(string("Parameter vector have $nparams elements when",
                                   "it should have $(length_params(mdl)).")))
    end
    if !isempty(y0) && size_y0 != (ny, diffeq.n)
        throw(ArgumentError(string("The matrix `y0` have dimension $(size_y0) ",
                                   "when it should have dimension ",
                                   "$((ny, diffeq.n)).")))
    end
    if !isempty(dΘ) && size_dΘ != (ny, nd, nparams)
        throw(ArgumentError(string("The array `dΘ` have dimension $(size_dΘ) ",
                                   "when it should have dimension ",
                                   "$((ny, nd, nparams)).")))
    end
    if !isempty(dy0) && size_dy0 != (ny, nd, ny, diffeq.n)
        throw(ArgumentError(string("The array `dy0` have dimension $(size_dy0) ",
                                   "when it should have dimension ",
                                   "$((ny, nd, ny, diffeq.n)).")))
    end
end
