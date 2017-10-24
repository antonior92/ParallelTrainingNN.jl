#*************************************************************************#
#
# Predictor type
#
#*************************************************************************#
"""
    Predictor

Type used to make a prediction about a temporal series.
"""
struct Predictor
    # Range of values of interest.
    start::Int
    length::Int
    # Internal Dynamic Model of the predictor.
    dynmdl::DifferenceEquation
    # Input to `dynmdl`.
    u::Matrix{Float64}
    # Initial conditions used by the `dynmdl`.
    y0::Matrix{Float64}
end


#*************************************************************************#
#
# Usefull constructors
#
#*************************************************************************#
"""
    one_step_ahead(mdl, yterms, uterms, y, u) -> pred::Predictor

One-step-ahead predictor for the given data set.
"""
function one_step_ahead{R<:Real,
                        S<:Real}(mdl::ParametricModel,
                                 yterms::Vector{Vector{Int}},
                                 uterms::Vector{Vector{Int}},
                                 y::Matrix{R},
                                 u::Matrix{S})
    # Get dynmdl
    dynmdl = DifferenceEquation(mdl, uterms=[yterms; uterms])

    # Get info about dynmdl
    iodelay = dynmdl.iodelay
    m = dynmdl.m
    n = dynmdl.n
    ny, nd = size(y)

    # Data for Simulation
    u_pred = [y[:, 1:nd-iodelay]; u[:, 1:nd-iodelay]]
    y0 = Matrix{Float64}(0, 0)

    Predictor(iodelay+m, nd-iodelay-m+1, dynmdl, u_pred, y0)
end


"""
    one_step_ahead(mdl, yterms, uterms, iddata) -> pred::Predictor

One-step-ahead predictor for the given data set.
"""
function one_step_ahead(mdl::ParametricModel,
                        yterms::Vector{Vector{Int}},
                        uterms::Vector{Vector{Int}},
                        iddata::IdData)
    one_step_ahead(mdl, yterms, uterms, iddata.y, iddata.u)
end


"""
    free_run_simulation(mdl, yterms, uterms, y, u) -> pred::Predictor

Free-run simulation for the given data set.
"""
function free_run_simulation(mdl::ParametricModel,
                             yterms::Vector{Vector{Int}},
                             uterms::Vector{Vector{Int}},
                             y::Matrix{<:Real},
                             u::Matrix{<:Real})
    # Get optimal predictor
    dynmdl = DifferenceEquation(mdl, yterms=yterms, uterms=uterms)

    # Get info about optimal predictor
    iodelay = dynmdl.iodelay
    m = dynmdl.m
    n = dynmdl.n
    n0 = max(iodelay+m, n+1)
    ny, nd = size(y)

    # Data for computing error
    u_pred = u[:, n0-iodelay-m+1:nd-iodelay]
    y0 = y[:, n0-n:n0-1]

    Predictor(n0, nd-n0+1, dynmdl, u_pred, y0)
end


"""
    free_run_simulation(mdl, yterms, uterms, iddata) -> pred::Predictor

Free-run simulation for the given data set.
"""
function free_run_simulation(mdl::ParametricModel,
                             yterms::Vector{Vector{Int}},
                             uterms::Vector{Vector{Int}},
                             iddata::IdData)
    free_run_simulation(mdl, yterms, uterms, iddata.y, iddata.u)
end


#*************************************************************************#
#
# Generate Buffers
#
#*************************************************************************#
"""
    y_buffer(pred::Predictor) -> y

Return the array to be used as buffer for the predicted output sequence `y`.
"""
function y_buffer(pred::Predictor)
    return Matrix{Float64}(pred.dynmdl.ny, pred.length)
end


"""
    dΘ_buffer(pred::Predictor) -> dΘ

Return the array to be used as buffer for the array of derivatives `dΘ`.
"""
function dΘ_buffer(pred::Predictor)
    return Array{Float64}(pred.dynmdl.ny, pred.length,
                          length_params(pred.dynmdl.mdl))
end

"""
    dy0_buffer(pred::Predictor) -> dy0

Return the array to be used as buffer for the array of derivatives `dy0`.
"""
function dy0_buffer(pred::Predictor)
    return Array{Float64}(pred.dynmdl.ny, pred.length,
                          pred.dynmdl.ny, pred.dynmdl.n)
end

#*************************************************************************#
#
# predict!
#
#*************************************************************************#
function predict!(pred::Predictor,
                  y::AbstractMatrix{<:Real},
                  Θ::AbstractVector{<:Real},
                  y0::AbstractMatrix{<:Real}=pred.y0;
                  dΘ::AbstractArray{<:Real, 3}=Array{Float64}(0, 0, 0),
                  dy0::AbstractArray{<:Real, 4}=Array{Float64}(0, 0, 0, 0))

    simulate!(pred.dynmdl, pred.u, Θ, y0, y,  dΘ=dΘ, dy0=dy0)

end


#*************************************************************************#
#
# predict
#
#*************************************************************************#
function predict(pred::Predictor,
                 Θ::AbstractVector{<:Real},
                 y0::AbstractMatrix{<:Real}=pred.y0)

    y = y_buffer(pred)

    predict!(pred, y, Θ, y0)

    return y
end


#*************************************************************************#
#
# get_slice
#
#*************************************************************************#
function get_slice(pred::Predictor,
                   v::AbstractVector{<:Real})
    return v[pred.start:(pred.start+pred.length-1)]
end


function get_slice(pred::Predictor,
                   v::AbstractMatrix{<:Real})
    return v[:, pred.start:(pred.start+pred.length-1)]
end
