#*************************************************************************#
#
# Series Model Type
#
#*************************************************************************#
struct SeriesModel{M<:StaticModel} <: ParametricModel
    ninputs::Int
    noutputs::Int
    nparams::Int
    nmodels::Int
    mdls::Vector{M}

    # Internal buffers
    vec_z::Vector{AbstractVector{Float64}}
    vec_dx::Vector{AbstractMatrix{Float64}}
    vec_dΘ::Vector{AbstractMatrix{Float64}}
end

"""
    SeriesModel([mdl1, mdl2, mdl3,..., mdln])

Initialize `ParametricModel` representing composition of models
`z = mdln(... mdl3(mdl2(mdl1(x))))`.
"""
function SeriesModel{M<:StaticModel}(mdls::Vector{M})
    
    # Check coherence between models
    nmodels = length(mdls)
    for i=1:length(mdls)-1
        if mdls[i].noutputs != mdls[i+1].ninputs
            throw(DimensionMismatch(string("The number of outputs of a ",
                                           "model should match the number ",
                                           "of inputs of the next one.")))
        end
    end

    # Get input, outputs and number of parameters
    ninputs = mdls[1].ninputs
    noutputs = mdls[end].noutputs

    nparams = 0
    nparametricmodels = 0
    for mdl in mdls
        if isparametric(mdl)
            nparams += mdl.nparams
        end
    end

    # Check if there are nested Series Models
    expand_model = false
    for mdl in mdls
        if typeof(mdl) <: SeriesModel
            nmodels += mdl.nmodels-1
            expand_model = true
        end
    end

    # Expand Series Models if needed
    if expand_model
        new_mdls = Vector{StaticModel}(nmodels)

        k = 1
        for mdl in mdls
            if typeof(mdl) <: SeriesModel
                for internal_mdl in mdl.mdls
                    new_mdls[k] = internal_mdl
                    k += 1
                end
            else
                new_mdls[k] = mdl
                k += 1
            end
        end
        mdls = new_mdls
    end

    # ---
    # Allocate Memory
    # ---
    vec_z = Vector{AbstractVector{Float64}}(nmodels+1)
    vec_dx = Vector{AbstractMatrix{Float64}}(nmodels)
    vec_dΘ = Vector{AbstractMatrix{Float64}}(nmodels)

    vec_z[1] = Vector{Float64}(ninputs)
    for (i, mdl) in Enumerate(mdls)
        vec_z[i+1] = z_buffer(mdl)
        vec_dx[i] = dx_buffer(mdl)
        if isparametric(mdl)
            vec_dΘ[i] = dΘ_buffer(mdl)
        end
    end
    # ---

    SeriesModel(ninputs, noutputs, nparams, nmodels, mdls, vec_z, vec_dx, vec_dΘ)
end


#*************************************************************************#
#
# Compute Output and its Derivatives
#
#*************************************************************************#
function _evaluate!(seriesmdl::SeriesModel,
                    x::AbstractVector{<:Real},
                    Θ::AbstractVector{<:Real},
                    z::AbstractVector{Float64},
                    dx::AbstractMatrix{Float64},
                    dΘ::AbstractMatrix{Float64})

    # Get memory allocated for vec_z
    vec_z = seriesmdl.vec_z
    vec_z[1] .= x

    # Empty auxiliar matrix
    empty = Vector{AbstractMatrix{Float64}}(seriesmdl.nmodels)
    fill!(empty, (Matrix{Float64}(0, 0)))

    # Use memory allocated for vec_dx and vec_dΘ
    if isempty(dx) && isempty(dΘ)
        vec_dx = empty
        vec_dΘ = empty
    elseif !isempty(dΘ)
        vec_dx = seriesmdl.vec_dx
        vec_dΘ = seriesmdl.vec_dΘ
    else
        vec_dx[i] = seriesmdl.vec_dx
        vec_dΘ[i] = empty
    end

    # ---
    # Forward Stage
    # ---
    k = 1
    for (i, mdl) in Enumerate(seriesmdl.mdls)
        if isparametric(mdl)
            _evaluate!(mdl, vec_z[i], Θ[k:k+mdl.nparams-1],
                       vec_z[i+1], vec_dx[i], vec_dΘ[i])
            k += mdl.nparams
        else
            _evaluate!(mdl, vec_z[i], vec_z[i+1], vec_dx[i])
        end
    end

    z .= vec_z[end]
    # ---

    if isempty(dx) && isempty(dΘ)
        return
    end

    # ---
    # Backward Stage
    # ---
    node_derivative = Matrix{Float64} # TODO: reuse the same `node_derivative` buffer for all `i`
    for i = seriesmdl.nmodels:-1:1

        # Get static model
        mdl = seriesmdl.mdls[i]

        # compute dΘ
        if !isempty(dΘ) && isparametric(mdl)
            k -= mdl.nparams
            if i == seriesmdl.nmodels
                dΘ[1:end, k:k+mdl.nparams-1] .= vec_dΘ[i]
            else
                dΘ[1:end, k:k+mdl.nparams-1] .= node_derivative * vec_dΘ[i]
            end
        end

        # Propagate derivatives backwards
        if i == seriesmdl.nmodels
            node_derivative = vec_dx[i]
        elseif  i>1 || !isempty(dx)
            node_derivative = node_derivative * vec_dx[i]
        end
    end

    if  !isempty(dx)
        dx .= node_derivative
    end
    # ---
end


#*************************************************************************#
#
# Initializer
#
#*************************************************************************#
"""
    ComposedInitializer()

The `ComposedInitializer` will initialize the parameter vector
according to its subinitializers.

Valid only for series static model.
"""
struct ComposedInitializer{M<:Initializer} <: Initializer
    init_vec::Vector{M}
end

function initial_guess(seriesmdl::SeriesModel, init::ComposedInitializer)

    Θ = Vector{Float64}(seriesmdl.nparams)
    k = 1
    i = 1
    for mdl in seriesmdl.mdls
        if isparametric(mdl)
            Θ[k:k+mdl.nparams-1] = initial_guess(mdl, init.init_vec[i])
            k += mdl.nparams
            i += 1
        end
    end
    return Θ
end

function initial_guess(seriesmdl::SeriesModel, init::DefaultInitializer)

    # Use default initializer in each one of the submodels
    init_vec = fill(DefaultInitializer(), seriesmdl.nmodels)
    initial_guess(seriesmdl, ComposedInitializer(init_vec))
end

