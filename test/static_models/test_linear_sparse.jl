@testset "Test Type Constructor" begin
    I = [1, 4, 3, 5]; J = [4, 7, 18, 9];
    lin = LinearSparse(18, 5, I, J)
    @test lin.ninputs == 18
    @test lin.noutputs == 5
    @test lin.nparams == 4
    @test lin.i == I
    @test lin.j == J
end

@testset "Test dx Buffer" begin
    I = [1, 4, 3, 5]; J = [4, 7, 18, 9];
    lin = LinearSparse(18, 5, I, J)
    dx = dx_buffer(lin)
    dx_expected = sparse([1, 4, 3, 5], [1, 4, 3, 5], zeros(4), 5, 18)
    @test typeof(dx) == typeof(dx_expected)
    @test dx.m  == dx_expected.m
    @test dx.n  == dx_expected.n
    @test dx.colptr == dx.colptr
    @test dx.rowval == dx.rowval
end

@testset "Test dΘ Buffer" begin
    I = [1, 4, 3, 5]; J = [4, 7, 18, 9];
    lin = LinearSparse(18, 5, I, J)
    dΘ = dΘ_buffer(lin)
    dΘ_expected = sparse([1, 4, 3, 5], [1, 2, 3, 4], zeros(4),
                         5, 4)
    @test typeof(dΘ) == typeof(dΘ_expected)
    @test dΘ.m  == dΘ_expected.m
    @test dΘ.n  == dΘ_expected.n
    @test dΘ.colptr == dΘ.colptr
    @test dΘ.rowval == dΘ.rowval
end

@testset "Test LinearSparse Evaluation" begin
    I = [1, 2, 3, 4]; J = [4, 3, 2, 1]; K = [1, 1, 1, 1]
    lin = LinearSparse(4, 4, I, J)
    z = z_buffer(lin)
    evaluate!(lin, [2, 4, 6, 5], K, z)
    @test z == [5, 6, 4, 2]
    evaluate!(lin, [1, 1, 1, 5], K, z)
    @test z == [5, 1, 1, 1]
    evaluate!(lin, [3, 3, 3, 5], K, z)
    @test z == [5, 3, 3, 3]
end

@testset "Test Jacobian Matrix Computation" begin
    @testset "Test Sparse" begin
        I = [1, 2, 3, 4]; J = [4, 3, 2, 1];
        lin = LinearSparse(4, 4, I, J)
        x = [10, 20, 30, 40]
        Θ = [1, 2, 3, 4]

        z = z_buffer(lin)
        dx = dx_buffer(lin)
        dΘ = dΘ_buffer(lin)
        evaluate!(lin, x, Θ, z, dx=dx, dΘ=dΘ)
        @test dx  == [0 0 0 1;
                      0 0 2 0;
                      0 3 0 0;
                      4 0 0 0]
        @test dΘ == [40 0 0 0;
                     0 30 0 0;
                     0 0 20 0;
                     0 0 0 10]
    end

    @testset "Test Finite Diferences" begin
        I = [1, 2, 3, 4, 1, 2, 4]; J = [4, 3, 2, 1, 3, 2, 2];
        lin = LinearSparse(4, 4, I, J)
        x = [10.0, 20.0, 30.0, 40]
        Θ = [1, 2, 3, 4, 4.5, 5.5, 6.1]

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
    @testset "Test Argument Check" begin
        I = [1, 2, 3, 4, 1, 2, 4]; J = [4, 3, 2, 1, 3, 2, 2];
        lin = LinearSparse(4, 4, I, J)
        x = [10.0, 20.0, 30.0, 40]
        Θ = [1, 2, 3, 4, 4.5, 5.5, 6.1]

        z = z_buffer(lin)
        dx = dx_buffer(lin)
        dΘ = dΘ_buffer(lin)

        @test_throws ArgumentError evaluate!(lin, [1, 2, 3],
                                             Θ,
                                             z,
                                             dx=dx,
                                             dΘ=dΘ)
        @test_throws ArgumentError evaluate!(lin, x,
                                             [1, 2, 3, 4, 5],
                                             z,
                                             dx=dx,
                                             dΘ=dΘ)
        @test_throws ArgumentError evaluate!(lin, x, Θ,
                                             Vector{Float64}(10),
                                             dx=dx,
                                             dΘ=dΘ)
        @test_throws ArgumentError evaluate!(lin, x, Θ, z,
                                             dx=Matrix{Float64}(3, 4),
                                             dΘ=dΘ)
        @test_throws ArgumentError evaluate!(lin, x, Θ, z,
                                             dx=Matrix{Float64}(4, 3),
                                             dΘ=dΘ)
        @test_throws ArgumentError evaluate!(lin, x, Θ, z,
                                             dx=dx,
                                             dΘ=Matrix{Float64}(5, 7))
        @test_throws ArgumentError evaluate!(lin, x, Θ, z,
                                             dx=dx,
                                             dΘ=Matrix{Float64}(4, 2))
    end
end
