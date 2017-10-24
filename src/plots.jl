function _plot(iddata::IdData, x::Matrix{Float64},
               name::Vector{String}, unit::Vector{String},
               line::Symbol; kwargs...)
    t = get_time_vector(iddata)
    nx = size(x, 1)

    yguide = [merge_name_unit(name[i], unit[i]) for i=1:nx]
    xguide = [merge_name_unit("t", iddata.time_unit) for i=1:nx]
    grid_width = Int(ceil(sqrt(nx)))
    
    p = [(i <= nx - grid_width) ?
         Plots.plot(t, x[i, :],
                    color="black",
                    yguide=yguide[i],
                    label=string("measured ", name[i]),
                    line=line) :
         Plots.plot(t, x[i, :],
                    color="black",
                    xguide=xguide[i],
                    yguide=yguide[i],
                    label=string("measured ", name[i]),
                    line=line)
         for i=1:nx]


    Plots.plot(p..., layout=nx; kwargs...)
end


function _plot_prediction!{R<:Real, S<:Real}(iddata::IdData,
                                             pred::Predictor,
                                             Θ::AbstractVector{R},
                                             y0::AbstractMatrix{S}=pred.y0;
                                             label::String="predicted",
                                             kwargs...)
    

    y = predict(pred, Θ, y0)
    t = iddata.t_start + iddata.ts*(pred.start-1:pred.start+pred.length-2)
    
    ny = size(y, 1)

    p = [Plots.plot!(t, y[i, :],
        label=string(label, " ", iddata.output_name[i])) 
        for i=1:ny]


    Plots.plot!(p..., layout=ny; kwargs...)
end


merge_name_unit(name::String, unit::String) = isempty(unit) ? name : string(name," (", unit, ")")



"""
    plot_input(iddata::IdData)

Plot each one of the system inputs.
"""
function plot_input(iddata::IdData; kwargs...)
    u = get_input(iddata)
    _plot(iddata, u, iddata.input_name, iddata.input_unit, :steppost; kwargs...)
end


function plot_input(e::PredictorError, args...; kwargs...)
    plot_input(e.iddata, args...; kwargs...)
end


"""
    plot_output(iddata::IdData)

Plot each one of the system outputs.
"""
function plot_output(iddata::IdData; kwargs...)
    y = get_output(iddata)
    _plot(iddata, y, iddata.output_name, iddata.output_unit, :line; kwargs...)
end


function plot_output{R<:Real, S<:Real}(iddata::IdData, pred::Predictor,
                                       Θ_list::Vector{Vector{R}},
                                       y0_list::Vector{Matrix{S}}=fill(pred.y0,
                                                                       length(Θ_list));
                                       label::Vector{String}=fill("predicted",
                                                                  length(Θ_list)),
                                       kwargs...)
    n_model_types = length(Θ_list)

    plot_output(iddata; kwargs...)
    p = _plot_prediction!(iddata, pred, Θ_list[1], y0_list[1], label=label[1]; kwargs...)
    for i=2:n_model_types
        p = _plot_prediction!(iddata, pred, Θ_list[i], y0_list[i], label=label[i]; kwargs...)
    end
    p
end


function plot_output{R<:Real, S<:Real}(iddata::IdData, pred::Predictor,
                                       Θ::Vector{R},
                                       y0::Matrix{S}=pred.y0;
                                       label::String="predicted",
                                       kwargs...)
    plot_output(iddata; kwargs...)
    _plot_prediction!(iddata, pred, Θ, y0, label=label; kwargs...)
end


function plot_output(e::PredictorError, args...; kwargs...)
    plot_output(e.iddata, e.pred, args...; kwargs...)
end
