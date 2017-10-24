#*************************************************************************#
#
# Initializer Abstract Type
#
#*************************************************************************#
abstract type Initializer end

"""
    initial_guess(f::StaticModel[, init::Initializer]) -> params

For a given static model `f` provide a suitable initial guess
for the parameters vectors `params`. This could be used
as start point for an optimization procedure. The provided
`Initializer` will determine what strategy will be used.
"""
function initial_guess(f::ParametricModel)
    initial_guess(f, DefaultInitializer())
end


"""
    DefaultInitializer()

Construct `Initializer` for static model using the default settings
for the given class.

Valid for any ParametricModel.
"""
struct DefaultInitializer <: Initializer
end

function initial_guess(f::ParametricModel, init::DefaultInitializer)
    initial_guess(f, ZeroInitializer())
end


"""
    ZeroInitializer()

Construct `Initializer` for static model. The `ZeroInitializer`
will initialize any parameter vector with zeros.

Valid for any ParametricModel.
"""
struct ZeroInitializer <: Initializer
end

function initial_guess(f::ParametricModel, init::ZeroInitializer)
    return zeros(f.nparams)
end

"""
    CustomRandomInitializer([s::Sampleable])

Construct `Initializer` for static model. The `CustomRandomInitializer`
will initialize any parameter vector with of random values picked
from the given distribution.

Valid for any ParametricModel.
"""
struct CustomRandomInitializer{F<:Distributions.VariateForm,
                               S<:Distributions.ValueSupport} <: Initializer
    s::Distributions.Sampleable{F, S}
end

function initial_guess(f::ParametricModel, init::CustomRandomInitializer)
    return rand(init.s, f.nparams)
end





