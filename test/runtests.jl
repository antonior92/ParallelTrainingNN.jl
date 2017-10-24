using ParallelTrainingNN
using Base.Test
using Calculus
using Distributions
using Optim

@testset "Static Models" begin
    @testset "Test Linear" begin
        include("static_models/test_linear.jl")
    end
    @testset "Test Linear Sparse" begin
        include("static_models/test_linear_sparse.jl")
    end
    @testset "Test Elementwise Operator" begin
        include("static_models/test_elementwise_operator.jl")
    end
    @testset "Test Bias" begin
        include("static_models/test_bias.jl")
    end
    @testset "Test Series Model" begin
        include("static_models/test_series_model.jl")
    end
    @testset "Test Feedforward Network Model" begin
        include("static_models/test_feedforward_network.jl")
    end
end

@testset "Test Difference Equation" begin
    include("test_difference_equation.jl")
end

@testset "Test Identification Data" begin
    include("test_iddata.jl")
end

@testset "Test Levemberg-Marquadt" begin
    include("optimization/test_levenberg_marquardt.jl")
end

@testset "Test Predictor" begin
    include("test_predictor.jl")
end

@testset "Test Predictor Error" begin
    include("test_predictor_error.jl")
end

@testset "Test Preprocess Data" begin
    include("test_preprocess_data.jl")
end

