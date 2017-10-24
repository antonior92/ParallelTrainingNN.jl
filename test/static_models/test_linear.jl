@testset "Test Type Constructor" begin
    lin = Linear(2, 3)
    @test lin.ninputs == 2
    @test lin.noutputs == 3
    @test lin.nparams == 6
end

@testset "Test dx Buffer" begin
    lin = Linear(2, 3)
    dx = dx_buffer(lin)
    @test typeof(dx) == Matrix{Float64}
    @test size(dx) == (lin.noutputs, lin.ninputs)
end

@testset "Test dΘ Buffer" begin
    @testset "Test Single Output" begin
        lin = Linear(3, 1)
        dΘ = dΘ_buffer(lin)
        @test typeof(dΘ) == Matrix{Float64}
        @test size(dΘ) == (lin.noutputs, lin.ninputs)
    end
    @testset "Test Multiple Output" begin
        lin = Linear(2, 3)
        dΘ = dΘ_buffer(lin)
        dΘ_expected = hcat((speye(lin.noutputs) for i=1:lin.ninputs)...)
        @test typeof(dΘ) == typeof(dΘ_expected)
        @test dΘ.m  == dΘ_expected.m
        @test dΘ.n  == dΘ_expected.n
        @test dΘ.colptr == dΘ.colptr
        @test dΘ.rowval == dΘ.rowval
    end
end

@testset "Test Linear Evaluation" begin
    lin = Linear(3, 2)
    A = [1 2 3;
         4 5 6]
    z = z_buffer(lin)
    evaluate!(lin, [2, 4, 6], vec(A), z)
    @test z == [28; 64]
    evaluate!(lin, [1, 1, 1], vec(A), z)
    @test z == [6; 15]
    evaluate!(lin, [3, 3, 3], vec(A), z)
    @test z == [18; 45]
end

@testset "Test Jacobian Matrix Computation" begin
    @testset "Test Dense" begin
        lin = Linear(3, 2)
        A = [1 2 3;
             4 5 6]
        z = z_buffer(lin)
        dx = dx_buffer(lin)
        dΘ = dΘ_buffer(lin)
        evaluate!(lin, [2, 3, 4], vec(A), z,
                  dx=dx, dΘ=dΘ)
        @test dx == A
        @test dΘ == [2 0 3 0 4 0;
                     0 2 0 3 0 4]
    end

    @testset "Test Finite Diferences" begin
        lin = Linear(4, 5)
        A = [1.0 2.0 3.1 1.2;
             4.0 5.0 6.1 1.2;
             7.0 8.0 9.1 1.2;
             1.0 2.0 3.1 1.2;
             4.0 5.0 6.1 1.2]
        x = [1.2, 2.3, 4.3, 5]
        Θ = vec(A)

        function cal_x(y)
            z = z_buffer(lin)
            evaluate!(lin, y, Θ, z)
            return z
        end
        function cal_Θ(theta)
            z = z_buffer(lin)
            evaluate!(lin, x, theta, z)
            return z
        end
        dx_numeric = Calculus.finite_difference_jacobian(cal_x, x)
        dΘ_numeric = Calculus.finite_difference_jacobian(cal_Θ, Θ)

        z = z_buffer(lin)
        dx = dx_buffer(lin)
        dΘ = dΘ_buffer(lin)
        evaluate!(lin, x, Θ, z, dx=dx, dΘ=dΘ)
        @test dx ≈ dx_numeric
        @test dΘ ≈ dΘ_numeric
    end

    @testset "Test Single Output" begin
        lin = Linear(6, 1)
        Θ = [1, 2, 3, 4, 5, 6]
        x = [10, 20, 30, 40, 50, 60]

        z = z_buffer(lin)
        dx = dx_buffer(lin)
        dΘ = dΘ_buffer(lin)
        evaluate!(lin, x, Θ, z, dx=dx, dΘ=dΘ)
        @test dx ≈ reshape(Θ, 1, 6)
        @test dΘ ≈ reshape(x, 1, 6)
    end
end


@testset "Test Initial Guess" begin
    N = 1000
    srand(0)
    lin = Linear(5, 2)
    init = ScaledGaussianInitializer()
    
    mean_estimated = 0
    var_estimated = 0
    for i=1:N
        params = initial_guess(lin, init)
        mean_estimated += sum(params)/(N*lin.nparams)
        var_estimated += sum(params.^2)/(N*lin.nparams)
    end

    std_estimated = sqrt(var_estimated)

    @test std_estimated ≈ 1/sqrt(5) atol=1e-1
    @test mean_estimated ≈ 0.0 atol=1e-1
end
