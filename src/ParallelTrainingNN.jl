module ParallelTrainingNN

import Distributions
import Optim
import Plots

export
    # Static Models
    StaticModel,
    ParametricModel,
    AparametricModel,
    ElementwiseOperator,
    Linear,
    LinearSparse,
    LogisticFunction,
    HyperbolicTangent,
    AffineMap,
    Identity,
    Bias,
    SeriesModel,
    FeedforwardNetwork,

    # Initializers
    Initializer,
    DefaultInitializer,
    ZeroInitializer,
    CustomRandomInitializer,
    ScaledGaussianInitializer,
    ComposedInitializer,

    # Static Models Methods
    evaluate!,
    z_buffer,
    dx_buffer,
    dΘ_buffer,
    initial_guess,
    isparametric,
    length_input,
    length_output,
    length_params,

    # Difference Equation Methods
    DifferenceEquation,
    simulate!,
    simulate,

    # Identification Data
    IdData,
    get_time_vector,
    get_input,
    get_output,

    # Predictor
    Predictor,
    one_step_ahead,
    free_run_simulation,
    predict!,
    predict,
    y_buffer,
    dy0_buffer,
    get_slice,

    # Predictor Error
    PredictorError,
    SimpleError,
    ExtendedParamsError,
    error_function!,
    estimate_parameters,
    narx,
    noe,

    # Preprocess Data
    learn_offset,
    learn_normalization,

    # Plots
    plot_input,
    plot_output


# Static Models
include("static_models/static_model.jl")
include("static_models/initializers.jl")
include("static_models/linear.jl")
include("static_models/linear_sparse.jl")
include("static_models/elementwise_operator.jl")
include("static_models/bias.jl")
include("static_models/series_model.jl")
include("static_models/feedforward_network.jl")

# Difference Equation
include("difference_equation.jl")

# IdData
include("iddata.jl")

# Include Levenberg-Marquadt algorithm
include("optimization/levenberg_marquardt.jl")

# Predictor
include("predictor.jl")
include("predictor_error.jl")

# Preprocessing data
include("preprocess_data.jl")

# Plots
include("plots.jl")

end # module
