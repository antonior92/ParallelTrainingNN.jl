#*************************************************************************#
#
# Predictor Error Models
#
#*************************************************************************#

"""
    narx(mdl, yterms, uterms, iddata, [init, kwargs...]) -> Θ, results, e

Compute a NARX model with the given structure for the data
provided in `iddata`.

Estimate the parameters minimizing the one-step-ahead prediction error,
between the predicted output and the measured one. 

Use the given initializer `init` to compute the initial guess used
by the optimization algorithm. Aditional arguments will be passed
on the to the nonlinear least squares solver (levenberg-marquadt).

Return the estimated parameter vector Θ and a structure `results`
containing information about the optimization procedure.
"""
function narx(mdl::ParametricModel, yterms::Vector{Vector{Int}},
              uterms::Vector{Vector{Int}}, iddata::IdData;
              init::Initializer=DefaultInitializer(), kwargs...)

    pred = one_step_ahead(mdl, yterms, uterms, iddata)

    e = SimpleError(pred, iddata)

    Θ, results = estimate_parameters(e, init; kwargs...)

    return Θ, results, e
end


"""
    noe(mdl, yterms, uterms, iddata, [init, use_extended,  kwargs...]) -> Θ, results, e

Compute a NOE  model with the given structure for the data
provided in `iddata`.

Estimate the parameters minimizing the free-run simulation error,
between the predicted output and the measured one.

If `extended_parameter` is true include initial conditions
on the optimization problem. Otherwise the measured values
will be used as initial conditions. By default it is false.

Use the given Initializer `init` to compute the initial guess
used by the optimization algorithm. Aditional arguments will
be passed on the to the nonlinear least squares solver
(levenberg-marquadt).

Return the estimated parameter vector Θ and a structure `results`
containing information about the optimization procedure.
"""
function noe(mdl::ParametricModel, yterms::Vector{Vector{Int}},
             uterms::Vector{Vector{Int}}, iddata::IdData;
             init::Initializer=DefaultInitializer(),
             use_extended::Bool=false,
             kwargs...)
    pred = free_run_simulation(mdl, yterms, uterms, iddata)

    if use_extended
        e = ExtendedParamsError(pred, iddata)
    else
        e = SimpleError(pred, iddata)
    end

    Θ, results = estimate_parameters(e, init; kwargs...)

    return Θ, results, e
end


#*************************************************************************#
#
# PredictorError
#
#*************************************************************************#
abstract type PredictorError end

function estimate_parameters(pred_error::PredictorError, init::Initializer=DefaultInitializer();
                             kwargs...)
    # Get Initial Guess
    ϕ₀ = initial_guess(pred_error, init)

    # Error Lenght
    len = pred_error.pred.length*pred_error.pred.dynmdl.ny

    # Create Buffer
    e_buffer = Vector{Float64}(len)

    # Optimize
    f!(ϕ, storage) = error_function!(pred_error, storage, ϕ)
    g!(ϕ, storage) = error_function!(pred_error, e_buffer, ϕ; dϕ=storage)
    results = levenberg_marquardt(f!, g!, ϕ₀, len; kwargs...)

    Θ = get_Θ(pred_error, Optim.minimizer(results))
    return Θ, results
end

#*************************************************************************#
#
# SimpleError
#
#*************************************************************************#
struct SimpleError <: PredictorError
    pred::Predictor
    iddata::IdData
end


get_Θ{R<:Real}(e::SimpleError, Θ::Vector{R}) = Θ


function error_function!{R<:Real,
                         S<:Real,
                         T<:Real}(pred_error::SimpleError, e::AbstractVector{R},
                                  ϕ::AbstractVector{S};
                                  dϕ::AbstractMatrix{T}=Matrix{Float64}(0, 0))
    # Get info
    pred = pred_error.pred
    iddata = pred_error.iddata
    nparams = length_params(pred.dynmdl.mdl)
    ny = pred.dynmdl.ny
    n = pred.dynmdl.n
    n0 = pred.start
    nd = pred.length

    Θ = get_Θ(pred_error, ϕ)

    # Reshape buffers
    y = reshape(e, ny, nd)
    if !isempty(dϕ)
        dΘ = reshape(dϕ, ny, nd, nparams)
    else
        dΘ = Array{Float64}(0, 0, 0)
    end

    # Compute output and its jacobian
    predict!(pred, y, Θ; dΘ=dΘ)

    # Measured values
    y_measured = get_slice(pred, iddata.y)
    y .-= y_measured

    return
end


function initial_guess(e::SimpleError, init::Initializer=DefaultInitializer())
    initial_guess(e.pred.dynmdl.mdl, init)
end


#*************************************************************************#
#
# ExtendedParamsError
#
#*************************************************************************#
struct ExtendedParamsError <: PredictorError
    pred::Predictor
    iddata::IdData
end

function get_Θ{R<:Real}(e::ExtendedParamsError, ϕ::Vector{R})
    # Get parameters length
    nparams = length_params(e.pred.dynmdl.mdl)

    # Get parameters from extended parameters
    Θ = ϕ[1:nparams]

    return Θ
end

function get_y0{R<:Real}(e::ExtendedParamsError, ϕ::Vector{R})
    # Get initial condition length
    ny, n = size(e.pred.y0)

     # Get parameters length
    nparams = length_params(e.pred.dynmdl.mdl)

    # Get initial conditions from extended parameters
    y0 = reshape(ϕ[nparams+1:end], ny, n)

    return y0
end

function assemble_ϕ{R<:Real, S<:Real}(e::ExtendedParamsError,
                                      Θ::Vector{R},
                                      y0::Matrix{S})
    y0 = vec(y0)
    ϕ = [Θ; y0]
    return ϕ
end

function error_function!{R<:Real,
                         S<:Real,
                         T<:Real}(pred_error::ExtendedParamsError, e::AbstractVector{R},
                                  ϕ::AbstractVector{S};
                                  dϕ::AbstractMatrix{T}=Matrix{Float64}(0, 0))
    # Get info
    pred = pred_error.pred
    iddata = pred_error.iddata
    nparams = length_params(pred.dynmdl.mdl)
    ny = pred.dynmdl.ny
    n = pred.dynmdl.n
    n0 = pred.start
    nd = pred.length

    # Get initial conditions and parameters from extended parameters
    Θ = get_Θ(pred_error, ϕ)
    y0 = get_y0(pred_error, ϕ)

    # Reshape buffers
    y = reshape(e, ny, nd)
    if !isempty(dϕ)
        dΘ = view(dϕ, :, 1:nparams)
        dΘ = reshape(dΘ, ny, nd, nparams)

        dy0 = view(dϕ, :, nparams+1:nparams+n*ny)
        dy0 = reshape(dy0, ny, nd, ny, n)
    else
        dΘ = Array{Float64}(0, 0, 0)
        dy0 = Array{Float64}(0, 0, 0, 0)
    end

    # Compute output and its jacobian
    predict!(pred, y, Θ, y0; dΘ=dΘ, dy0=dy0)

    # Measured values
    y_measured = get_slice(pred, iddata.y)
    y .-= y_measured

    return
end

function initial_guess(e::ExtendedParamsError, init::Initializer=DefaultInitializer())
    Θ = initial_guess(e.pred.dynmdl.mdl, init)
    
    return assemble_ϕ(e, Θ, e.pred.y0)
end
