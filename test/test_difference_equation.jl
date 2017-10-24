@testset "Test Constructor I" begin
    mdl = Linear(7, 2)

    yterms = [[1, 4, 5], [1]]
    uterms = [[1, 2], [3]]

    dynmdl = DifferenceEquation(mdl,
                                yterms=yterms,
                                uterms=uterms)
    @test dynmdl.ny == 2
    @test dynmdl.nu == 2
    @test dynmdl.n == 5
    @test dynmdl.m == 3
    @test dynmdl.iodelay == 1
    @test_throws ArgumentError DifferenceEquation(Linear(8, 3),
                                                  yterms=yterms,
                                                  uterms=uterms)
    @test_throws ArgumentError DifferenceEquation(Linear(7, 4),
                                                  yterms=yterms,
                                                  uterms=uterms)
    @test_throws ArgumentError DifferenceEquation(mdl,
                                                  yterms=[[0, 4, 5], [1]],
                                                  uterms=uterms)
end

@testset "Test Constructor II" begin
    mdl = Linear(7, 1)

    yterms = [[1, 2, 3]]
    uterms = [[0, 4, 5, 6]]

    dynmdl = DifferenceEquation(mdl, yterms=yterms,
                                uterms=uterms)
    @test dynmdl.ny == 1
    @test dynmdl.nu == 1
    @test dynmdl.n == 3
    @test dynmdl.m == 7
    @test dynmdl.iodelay == 0
    @test_throws ArgumentError DifferenceEquation(Linear(11, 1),
                                                  yterms=yterms,
                                                  uterms=uterms)
    @test_throws ArgumentError DifferenceEquation(Linear(10, 2),
                                                  yterms=yterms,
                                                  uterms=uterms)
    @test_throws ArgumentError DifferenceEquation(mdl,
                                                  yterms=[[0, 1, 2]],
                                                  uterms=uterms)
end

@testset "Test Type Construction III" begin
    mdl = Linear(4, 1)

    uterms = [[0, 4, 5, 6]]

    dynmdl = DifferenceEquation(mdl,
                                uterms=uterms)
    @test dynmdl.ny == 1
    @test dynmdl.nu == 1
    @test dynmdl.n == 0
    @test dynmdl.m == 7
    @test dynmdl.iodelay == 0
    @test_throws ArgumentError DifferenceEquation(Linear(6, 1),
                                                  uterms=uterms)
end

@testset "Test Type Construction IV" begin
    mdl = Linear(4, 1)

    yterms = [[1, 4, 5, 6]]

    dynmdl = DifferenceEquation(mdl,
                                yterms=yterms)
    @test dynmdl.ny == 1
    @test dynmdl.nu == 0
    @test dynmdl.n == 6
    @test dynmdl.m == 0
    @test dynmdl.iodelay == 0
    @test_throws ArgumentError DifferenceEquation(Linear(5, 1),
                                                  yterms=yterms)
    @test_throws ArgumentError DifferenceEquation(Linear(4, 2),
                                                  yterms=yterms)
    @test_throws ArgumentError DifferenceEquation(mdl,
                                                  yterms=[[0, 1, 2, 6]])
end

@testset "Test Single Step Prediction" begin
    mdl = LinearSparse(13, 2, [1, 1, 1, 2, 2], [1, 6, 11, 1, 2])

    yterms = [[1, 2, 3], [1, 2]]
    uterms = [[0, 1, 2], [1], [2], [0, 1], [1]]

    dynmdl = DifferenceEquation(mdl,
                                yterms=yterms,
                                uterms=uterms)

    y0 = Float64[1 2 3;
                 4 5 6]
    u = [7  8  9;
         10 11 12;
         13 14 15;
         0 16 17;
         0 18 19]
    y = Array{Float64}(2, 1)

    Θ = Float64[1, 1, 1, 1, 2]


    @testset "Simulation" begin
        simulate!(dynmdl, u, Θ, y0, y)

        @test y == reshape([29; 7], 2, 1)

        y2 = simulate(dynmdl, u, Θ, y0)

        @test y2 == reshape([29; 7], 2, 1)
    end

    @testset "Derivatives" begin
        function cal_Θ(theta)
            yy = Matrix{Float64}(size(y))
            simulate!(dynmdl, u, theta, y0, yy)
            return vec(yy)
        end
        function cal_y0(x0)
            yy = Matrix{Float64}(size(y))
            simulate!(dynmdl, u, Θ, reshape(x0, size(y0)), yy)
            return vec(yy)
        end

        dy0_numeric = Calculus.finite_difference_jacobian(cal_y0, vec(y0))
        dΘ_numeric = Calculus.finite_difference_jacobian(cal_Θ, Θ)

        dΘ = Array{Float64}(2, 1, 5)
        dy0 = Array{Float64}(2, 1, 2, 3)
        simulate!(dynmdl, u, Θ, y0, y, dΘ=dΘ, dy0=dy0)

        @test vec(dΘ) ≈ vec(dΘ_numeric)
        @test vec(dy0) ≈ vec(dy0_numeric)
    end
end

@testset "Test Single Step Prediction empty u" begin
    mdl = LinearSparse(5, 2, [1, 2, 2], [1, 1, 2])

    yterms = [[1, 2, 3], [1, 2]]

    dynmdl = DifferenceEquation(mdl,
                                yterms=yterms)

    y0 = Float64[1 2 3;
                 4 5 6]
    u = Matrix{Float64}(0, 0)
    y = Array{Float64, 2}(2, 1)

    Θ = Float64[1, 1, 1]

    @testset "Simulation" begin
        simulate!(dynmdl, u, Θ, y0, y)

        @test y == reshape([3; 5], 2, 1)

        y2 = simulate(dynmdl, u, Θ, y0)

        @test y2 == reshape([3; 5], 2, 1)
    end

    @testset "Derivatives" begin
        function cal_Θ(theta)
            yy = Matrix{Float64}(size(y))
            simulate!(dynmdl, u, theta, y0, yy)
            return vec(yy)
        end
        function cal_y0(x0)
            yy = Matrix{Float64}(size(y))
            simulate!(dynmdl, u, Θ, reshape(x0, size(y0)), yy)
            return vec(yy)
        end

        dy0_numeric = Calculus.finite_difference_jacobian(cal_y0, vec(y0))
        dΘ_numeric = Calculus.finite_difference_jacobian(cal_Θ, Θ)

        dΘ = Array{Float64}(2, 1, 3)
        dy0 = Array{Float64}(2, 1, 2, 3)
        simulate!(dynmdl, u, Θ, y0, y, dΘ=dΘ, dy0=dy0)

        @test vec(dΘ) ≈ vec(dΘ_numeric)
        @test vec(dy0) ≈ vec(dy0_numeric)
    end
end

@testset "Test Single Step Prediction Empty y" begin
    mdl = LinearSparse(5, 2, [1, 2, 2], [1, 1, 2])

    uterms = [[1, 2, 3], [1, 2]]

    dynmdl = DifferenceEquation(mdl, uterms=uterms)

    u = [1 2 3;
         4 5 6]
    y0 = Matrix{Float64}(0, 0)
    y = Matrix{Float64}(2, 1)

    Θ = Float64[1, 1, 1]

    @testset "Simulation" begin
        simulate!(dynmdl, u, Θ, y0, y)

        @test y == reshape([3; 5], 2, 1)

        y2 = simulate(dynmdl, u, Θ, y0)

        @test y2 == reshape([3; 5], 2, 1)
    end

    @testset "Derivatives" begin
        function cal_Θ(theta)
            yy = Matrix{Float64}(size(y))
            simulate!(dynmdl, u, theta, y0, yy)
            return vec(yy)
        end
        function cal_y0(x0)
            yy = Matrix{Float64}(size(y))
            simulate!(dynmdl, u, Θ, reshape(x0, size(y0)), yy)
            return vec(yy)
        end

        dy0_numeric = Calculus.finite_difference_jacobian(cal_y0, vec(y0))
        dΘ_numeric = Calculus.finite_difference_jacobian(cal_Θ, Θ)

        dΘ = Array{Float64}(2, 1, 3)
        dy0 = Array{Float64}(2, 1, 0, 0)
        simulate!(dynmdl, u, Θ, y0, y, dΘ=dΘ, dy0=dy0)

        @test vec(dΘ) ≈ vec(dΘ_numeric)
        @test vec(dy0) ≈ vec(dy0_numeric)
    end
end

@testset "Test SISO Multi Step Simulation" begin
    mdl = Linear(6, 1)

    yterms = [[1, 2, 3]]
    uterms = [[1, 2, 3]]

    dynmdl = DifferenceEquation(mdl, yterms=yterms,  uterms=uterms)
    y0 = reshape([0.0; 4.0; 13.0], 1, 3)

    yref = reshape([24.8; 30.2; 27.96; 11.76;
                    -16.568; -50.384; -64.7776;
                    -57.784; -26.03872; 22.23296;
                    76.507136; 98.513024; 86.7539072;
                    96.2449152; 154.54439424; 244.89924352;
                    290.512711168], 1, 17)
    u = reshape([1; 1; 1;  1;  1;  -2;  -2;
                 -2;  -2;  -2;  3;  3;  3;
                 3;  3;  10;  10;  10;  10], 1, 19)
    y = Array{Float64, 2}(1, 17)

    Θ = [1, -0.8, 0.2, 4, 5, 6]

    @testset "Simulation" begin
        simulate!(dynmdl, u, Θ, y0, y)

        @test y ≈ yref

        y2 = simulate(dynmdl, u, Θ, y0)

        @test y2 ≈ yref
    end

    @testset "Derivatives" begin
        function cal_Θ(theta)
            yy = Matrix{Float64}(size(y))
            simulate!(dynmdl, u, theta, y0, yy)
            return vec(yy)
        end
        function cal_y0(x0)
            yy = Matrix{Float64}(size(y))
            simulate!(dynmdl, u, Θ, reshape(x0, size(y0)), yy)
            return vec(yy)
        end

        dy0_numeric = Calculus.finite_difference_jacobian(cal_y0, vec(y0))
        dΘ_numeric = Calculus.finite_difference_jacobian(cal_Θ, Θ)

        dΘ = Array{Float64}(1, 17, 6)
        dy0 = Array{Float64}(1, 17, 1, 3)
        simulate!(dynmdl, u, Θ, y0, y, dΘ=dΘ, dy0=dy0)

        @test vec(dΘ) ≈ vec(dΘ_numeric)
        @test vec(dy0) ≈ vec(dy0_numeric)
    end
end

@testset "Test MIMO multistep prediction" begin
    mdl = Linear(12, 2)

    yterms = [[1, 2, 3], [1, 2, 3]]
    uterms = [[1, 2, 3], [1, 2, 3]]

    dynmdl = DifferenceEquation(mdl,
                                yterms=yterms,
                                uterms=uterms)

    y0 = Float64[0 0;
                 4 8;
                 13 26]'
    yref = [24.8 49.6;
            30.2 60.4;
            27.96 63.92;
            11.76 61.52;
            -16.568 55.464;
            -50.384 34.032;
            -64.7776 8.9648;
            -57.784 -10.1680;
            -26.03872 -5.53344;
            22.23296 34.39392;
            76.507136 112.787072;
            98.513024 180.165248;
            86.7539072 216.8143744;
            96.2449152 243.2395904;
            154.54439424 260.82114048;
            244.89924352 271.59234304;
            290.512711168 231.583348736]'
    u = [1 1;
         1 1;
         1 1;
         1 1;
         1 3;
         -2 3;
         -2 3;
         -2 3;
         -2 1;
         -2 1;
         3 1;
         3 1;
         3 5;
         3 5;
         3 5;
         10 5;
         10 -2;
         10 -2;
         10 -2]'
    y = Array{Float64, 2}(2, 17)

    Θ = Float64[1, 0,
                -0.8, 0,
                0.2, 0,
                0, 1,
                0,-0.8,
                0, 0.2,
                4, 4,
                5, 5,
                6, 6,
                0, 4,
                0, 5,
                0, 6]

    @testset "Simulation" begin
        simulate!(dynmdl, u, Θ, y0, y)

        @test y ≈ yref

        y2 = simulate(dynmdl, u, Θ, y0)

        @test y2 ≈ yref
    end

    @testset "Derivatives" begin
        function cal_Θ(theta)
            yy = Matrix{Float64}(size(y))
            simulate!(dynmdl, u, theta, y0, yy)
            return vec(yy)
        end
        function cal_y0(x0)
            yy = Matrix{Float64}(size(y))
            simulate!(dynmdl, u, Θ, reshape(x0, size(y0)), yy)
            return vec(yy)
        end

        dy0_numeric = Calculus.finite_difference_jacobian(cal_y0, vec(y0))
        dΘ_numeric = Calculus.finite_difference_jacobian(cal_Θ, Θ)

        dΘ = Array{Float64}(2, 17, 24)
        dy0 = Array{Float64}(2, 17, 2, 3)
        simulate!(dynmdl, u, Θ, y0, y, dΘ=dΘ, dy0=dy0)

        @test vec(dΘ) ≈ vec(dΘ_numeric)
        @test vec(dy0) ≈ vec(dy0_numeric)
    end
end

@testset "Test Predict Argument Check" begin
    mdl = Linear(12, 2)

    yterms = [[1, 2, 3], [1, 2, 3]]
    uterms = [[1, 2, 3], [1, 2, 3]]

    dynmdl = DifferenceEquation(mdl,
                                yterms=yterms,
                                uterms=uterms)

    y0 = Float64[0 0;
                 4 8;
                 13 26]'
    yref = [24.8 49.6;
            30.2 60.4;
            27.96 63.92;
            11.76 61.52;
            -16.568 55.464;
            -50.384 34.032;
            -64.7776 8.9648;
            -57.784 -10.1680;
            -26.03872 -5.53344;
            22.23296 34.39392;
            76.507136 112.787072;
            98.513024 180.165248;
            86.7539072 216.8143744;
            96.2449152 243.2395904;
            154.54439424 260.82114048;
            244.89924352 271.59234304;
            290.512711168 231.583348736]'
    u = [1 1;
         1 1;
         1 1;
         1 1;
         1 3;
         -2 3;
         -2 3;
         -2 3;
         -2 1;
         -2 1;
         3 1;
         3 1;
         3 5;
         3 5;
         3 5;
         10 5;
         10 -2;
         10 -2;
         10 -2]'
    y = Array{Float64, 2}(2, 17)

    Θ = Float64[1, 0,
                -0.8, 0,
                0.2, 0,
                0, 1,
                0,-0.8,
                0, 0.2,
                4, 4,
                5, 5,
                6, 6,
                0, 4,
                0, 5,
                0, 6]

    dΘ = Array{Float64}(2, 17, 24)
    dy0 = Array{Float64}(2, 17, 2, 3)


    @test_throws ArgumentError simulate!(dynmdl, u, Θ, y0,
                                         y[1:end-1, :], dΘ=dΘ,
                                         dy0=dy0)
    @test_throws ArgumentError simulate!(dynmdl, u[1:end-1, :], Θ, y0,
                                         y, dΘ=dΘ, dy0=dy0)
    @test_throws ArgumentError simulate!(dynmdl, u[:, 1:end-1], Θ, y0,
                                         y, dΘ=dΘ, dy0=dy0)
    @test_throws ArgumentError simulate!(dynmdl, u, Θ[1:end-1], y0,
                                         y, dΘ=dΘ, dy0=dy0)
    @test_throws ArgumentError simulate!(dynmdl, u, Θ, y0[:, 1:end-1],
                                         y, dΘ=dΘ, dy0=dy0)
    @test_throws ArgumentError simulate!(dynmdl, u, Θ, y0[1:end-1, :],
                                         y, dΘ=dΘ, dy0=dy0)
    @test_throws ArgumentError simulate!(dynmdl, u, Θ, y0,
                                         y, dΘ=dΘ[:, :, 1:end-1], dy0=dy0)
    @test_throws ArgumentError simulate!(dynmdl, u, Θ, y0,
                                         y, dΘ=dΘ, dy0=dy0[:, :, :, 1:end-1])
end
