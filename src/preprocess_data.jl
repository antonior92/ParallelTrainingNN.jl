#*************************************************************************#
#
# Preprocessing data
#
#*************************************************************************#
function _compute_stats(mdl::StaticModel,
                        yterms::Vector{Vector{Int}},
                        uterms::Vector{Vector{Int}},
                        iddata::IdData)
    # Get std and mean
    y_std = std(iddata.y, 2)
    u_std = std(iddata.u, 2)
    y_mean = mean(iddata.y, 2)
    u_mean = mean(iddata.u, 2)

    # Compute mean μ and standard deviation σ
    # for each element of the model input x
    ninputs = length_input(mdl)
    k = 1
    σ_x = Vector{Float64}(ninputs)
    μ_x  = Vector{Float64}(ninputs)
    for (i, ynterms) in Enumerate(yterms)
        for yn in ynterms
            σ_x[k] = y_std[i]
            μ_x[k] = y_mean[i]
            k += 1
        end
    end

    for (i, unterms) in Enumerate(uterms)
        for un in unterms
            σ_x[k] = u_std[i]
            μ_x[k] = u_mean[i]
            k += 1
        end
    end

    # Compute mean μ and standard deviation σ
    # for each element of the model input z
    σ_z = vec(y_std)
    μ_z = vec(y_mean)

    return σ_x, μ_x, σ_z, μ_z
end


"""
    learn_offset(mdl, yterms, uterms, iddata) -> new_mdl

Learn offset from data and generate new model with the given offset
embbeded.
"""
function learn_offset(mdl::StaticModel,
                      yterms::Vector{Vector{Int}},
                      uterms::Vector{Vector{Int}},
                      iddata::IdData)

    σ_x, μ_x, σ_z, μ_z = _compute_stats(mdl, yterms, uterms, iddata)
    
    # Build New Series Model
    new_model = SeriesModel([AffineMap(ones(σ_x), -μ_x), mdl, AffineMap(ones(σ_z), μ_z)])

    return new_model
end


"""
    learn_normalization(mdl, yterms, uterms, iddata[, nσ]) -> new_mdl

Learn offset and scale from data and generate new model with the given offset
and scale embbeded. Where `nσ` is the number of standard deviations the data
will be internally divided by (default is 3).
"""
function learn_normalization{R<:Real}(mdl::StaticModel,
                                      yterms::Vector{Vector{Int}},
                                      uterms::Vector{Vector{Int}},
                                      iddata::IdData;
                                      nσ::R=3.0)

    σ_x, μ_x, σ_z, μ_z = _compute_stats(mdl, yterms, uterms, iddata)

    # Replace σ_x, σ_z by ones when needed
    σ_x = [σ <= 1e-10 ? 1.0 : σ for σ in σ_x]
    σ_z = [σ <= 1e-10 ? 1.0 : σ for σ in σ_z]
    
    # Build New Series Model
    new_model = SeriesModel([AffineMap(1./(nσ*σ_x) , -μ_x./(nσ*σ_x)), mdl,
                             AffineMap((nσ*σ_z), μ_z)])

    return new_model
end
