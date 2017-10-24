#*************************************************************************#
#
# FeedforwardNetwork Type
#
#*************************************************************************#

# Constructor I
"""
    FeedforwardNetwork(ninputs, activfunc)

Initialize `ParametricModel` representing feedforward neural network.
`ninputs` is the network number of inputs and `activfunc`
is a vector of `AparametricModel` containg activation
functions for each one of the layers.
"""
function FeedforwardNetwork{T<:AparametricModel}(ninputs::Int,
                                                 activfunc::Vector{T})
    nlayers = length(activfunc)
    mdls = Vector{StaticModel}(3*nlayers)
    node_noutputs = ninputs
    k = 1
    for i = 1:nlayers
        node_ninputs = node_noutputs
        node_noutputs = activfunc[i].ninputs
        mdls[k] = Linear(node_ninputs, node_noutputs)
        k += 1
        mdls[k] = Bias(node_noutputs)
        k += 1
        mdls[k] = activfunc[i]
        k += 1
    end

    SeriesModel(mdls)
end

# Constructor II
"""
    FeedforwardNetwork(ninputs, noutputs, nhidden)

Initialize `ParametricModel` representing feedforward neural network.
`ninputs` and `outputs` are the network number of inputs and outputs
and `nhidden` is a vector containg the number of nodes in each hidden
layer.
"""
function FeedforwardNetwork(ninputs::Int,
                            noutputs::Int,
                            nhidden::Vector{Int})
    nlayers = length(nhidden)+1
    
    activfunc = Vector{ElementwiseOperator}(nlayers)
    for k = 1:nlayers
        activfunc[k] = (k==nlayers) ? Identity(noutputs) : HyperbolicTangent(nhidden[k])
    end

    FeedforwardNetwork(ninputs, activfunc)
end
