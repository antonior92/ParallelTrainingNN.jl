@testset "Test Constructor" begin
    mdl1 = Linear(10, 2)
    mdl2 = Linear(2, 5)
    mdl3 = Linear(5, 3)
    mdl4 = Linear(3, 4)

    mdl2wrong = Linear(4, 5)
    mdl3wrong = Linear(5, 2)

    mdl = SeriesModel([mdl1, mdl2, mdl3, mdl4])

    @test mdl.ninputs == 10
    @test mdl.noutputs == 4
    @test mdl.nparams == 10*2+2*5+5*3+3*4
    @test mdl.nmodels == 4

    @test_throws DimensionMismatch SeriesModel([mdl1, mdl2wrong, mdl3, mdl4])
    @test_throws DimensionMismatch SeriesModel([mdl1, mdl2, mdl3wrong, mdl4])
end

@testset "Test Constructor with Multiple Series Model" begin
    mdl1 = Linear(10, 5)
    mdl2 = SeriesModel([Identity(5), Bias(5)])
    mdl3 = Linear(5, 3)
    mdl4 = SeriesModel([Identity(3), Bias(3)])

    mdl = SeriesModel([mdl1, mdl2, mdl3, mdl4])

    @test mdl.ninputs == 10
    @test mdl.noutputs == 3
    @test mdl.nparams == 10*5+5+5*3+3
    @test mdl.nmodels == 6
    @test isequal(mdl.mdls, [Linear(10, 5), Identity(5),  Bias(5), Linear(5, 3), Identity(3), Bias(3)])
end

@testset "Test Evaluation" begin
    mdl1 = Linear(2, 2)
    mdl2 = Identity(2)
    mdl3 = Linear(2, 3)
    mdl4 = Identity(3)

    mdl = SeriesModel([mdl1, mdl2, mdl3, mdl4])

    A1 = [1 2;
          3 4]
    A2 = [5 6;
          7 8;
          9 10]

    Θ = [vec(A1); vec(A2)]

    z = z_buffer(mdl)
    dx = dx_buffer(mdl)
    dΘ = dΘ_buffer(mdl)
    x = [2.3, 4]
    evaluate!(mdl, x, Θ, z; dx=dx, dΘ=dΘ)
    @test z ≈ A2*A1*x
    @test dx ≈ A2*A1
    @test dΘ ≈ [11.5 13.8 20.0 24.0 10.3 0.0 0.0 22.9 0.0 0.0;
                 16.1 18.4 28.0 32.0 0.0 10.3 0.0 0.0 22.9 0.0;
                 20.7 23.0 36.0 40.0 0.0 0.0 10.3 0.0 0.0 22.9]

    z = z_buffer(mdl)
    x = [5, 4]
    evaluate!(mdl, x, Θ, z)
    @test z ≈ A2*A1*x

    z = z_buffer(mdl)
    dx = dx_buffer(mdl)
    x = [10.2, 4]
    evaluate!(mdl, x, Θ, z; dx=dx, dΘ=dΘ)
    @test z ≈ A2*A1*x
    @test dx ≈ A2*A1

    z = z_buffer(mdl)
    dΘ = dΘ_buffer(mdl)
    x = [2.3, 4]
    evaluate!(mdl, x, Θ, z; dΘ=dΘ)
    @test z ≈ A2*A1*x
    @test dΘ ≈ [11.5 13.8 20.0 24.0 10.3 0.0 0.0 22.9 0.0 0.0;
                 16.1 18.4 28.0 32.0 0.0 10.3 0.0 0.0 22.9 0.0;
                 20.7 23.0 36.0 40.0 0.0 0.0 10.3 0.0 0.0 22.9]
end

@testset "Test Initializer" begin
    N = 1000
    srand(0)
    
    mdl1 = Linear(10, 5)
    mdl2 = HyperbolicTangent(5)
    mdl3 = Linear(5, 3)
    mdl4 = Bias(3)

    mdl = SeriesModel([mdl1, mdl2, mdl3, mdl4])

    init1 = CustomRandomInitializer(Distributions.Normal(0, 1))
    init2 = CustomRandomInitializer(Distributions.Normal(0, 2))
    init3 = ZeroInitializer()

    init = ComposedInitializer([init1, init2, init3])

    # First System
    mean_estimated1 = 0
    var_estimated1 = 0
    mean_estimated2 = 0
    var_estimated2 = 0
    mean_estimated3 = 0
    var_estimated3 = 0
    for i=1:N
        params = initial_guess(mdl, init)
        params1 = params[1:50]
        params2 = params[51:65]
        params3 = params[66:end]
        mean_estimated1 += sum(params1)/(N*50)
        var_estimated1 += sum(params1.^2)/(N*50)
        mean_estimated2 += sum(params2)/(N*15)
        var_estimated2 += sum(params2.^2)/(N*15)
        mean_estimated3 += sum(params3)/(N*3)
        var_estimated3 += sum(params3.^2)/(N*3)
    end

    std_estimated1 = sqrt(var_estimated1)
    std_estimated2 = sqrt(var_estimated2)
    std_estimated3 = sqrt(var_estimated3)

    @test std_estimated1 ≈ 1.0 atol=1e-1
    @test mean_estimated1 ≈ 0.0 atol=1e-1

    @test std_estimated2 ≈ 2.0 atol=1e-1
    @test mean_estimated2 ≈ 0.0 atol=1e-1

    @test std_estimated3 ≈ 0.0 atol=1e-16
    @test mean_estimated3 ≈ 0.0 atol=1e-16
end


@testset "Test Default Initializer" begin
    N = 1000
    srand(0)
    
    mdl1 = Linear(10, 5)
    mdl2 = HyperbolicTangent(5)
    mdl3 = Linear(5, 3)
    mdl4 = Bias(3)

    mdl = SeriesModel([mdl1, mdl2, mdl3, mdl4])

    # First System
    mean_estimated1 = 0
    var_estimated1 = 0
    mean_estimated2 = 0
    var_estimated2 = 0
    mean_estimated3 = 0
    var_estimated3 = 0
    for i=1:N
        params = initial_guess(mdl)
        params1 = params[1:50]
        params2 = params[51:65]
        params3 = params[66:end]
        mean_estimated1 += sum(params1)/(N*50)
        var_estimated1 += sum(params1.^2)/(N*50)
        mean_estimated2 += sum(params2)/(N*15)
        var_estimated2 += sum(params2.^2)/(N*15)
        mean_estimated3 += sum(params3)/(N*3)
        var_estimated3 += sum(params3.^2)/(N*3)
    end

    std_estimated1 = sqrt(var_estimated1)
    std_estimated2 = sqrt(var_estimated2)
    std_estimated3 = sqrt(var_estimated3)

    @test std_estimated1 ≈ 1/sqrt(10) atol=1e-1
    @test mean_estimated1 ≈ 0.0 atol=1e-1

    @test std_estimated2 ≈ 1/sqrt(5) atol=1e-1
    @test mean_estimated2 ≈ 0.0 atol=1e-1

    @test std_estimated3 ≈ 0.0 atol=1e-16
    @test mean_estimated3 ≈ 0.0 atol=1e-16
end
