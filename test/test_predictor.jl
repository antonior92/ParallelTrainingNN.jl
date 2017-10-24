@testset "Test for SISO data" begin
    # Generate Input Data
    len = 8
    srand(1)
    u = rand(1, len)-0.5;

    # Define Dynamic Model
    mdl = Linear(5, 1)
    yterms = [[1, 2]]
    uterms = [[0, 1, 2]]
    diffeq = DifferenceEquation(mdl, yterms=yterms, uterms=uterms)

    # Simulate system for a given parameter
    Θ = [-0.1, 0.3, 1, 0.4, 0.3]
    y0 = reshape([0.0, 0.0], 1, 2)
    ϕ = [Θ; vec(y0)]
    yaux = simulate(diffeq, u, Θ, y0)
    y = [y0 yaux]

    @testset "Test one-step-ahead prediction" begin
        pred = one_step_ahead(mdl, yterms, uterms, IdData(y, u))

        @testset "Test Basic Construction" begin
            @test pred.dynmdl.yterms == Vector{Vector{Int}}(0)
            @test pred.dynmdl.uterms == [[1, 2], [0, 1, 2]]
            @test pred.u == [y; u]
            @test pred.y0 == Matrix{Float64}(0, 0)
            @test pred.length == size(yaux, 2)
            @test pred.start == 3
        end

        @testset "Test Prediction" begin
            @test predict(pred, Θ) ≈ yaux
        end

        @testset "Test Params Derivatives" begin

            function cal_Θ(theta)
                vec(predict(pred, theta))
            end

            dΘ_numeric = Calculus.finite_difference_jacobian(cal_Θ, Θ)

            ys = y_buffer(pred)
            dΘ = dΘ_buffer(pred)
            predict!(pred, ys, Θ; dΘ=dΘ)

            @test dΘ ≈ reshape(dΘ_numeric, size(dΘ))
        end

    end

    @testset "Test free-run simulation" begin
        # Test Basic Construction
        pred = free_run_simulation(mdl, yterms, uterms, IdData(y, u))
        
        @testset "Test Basic Construction" begin
            @test pred.dynmdl.yterms == [[1, 2]]
            @test pred.dynmdl.uterms == [[0, 1, 2]]
            @test pred.u == u
            @test pred.y0 == y0
            @test pred.length == size(yaux, 2)
            @test pred.start == 3
        end

        @testset "Test Prediction" begin
            @test predict(pred, Θ) ≈ yaux
        end

        @testset "Test Params Derivatives" begin

            function cal_Θ(theta)
                vec(predict(pred, theta))
            end

            dΘ_numeric = Calculus.finite_difference_jacobian(cal_Θ, Θ)

            ys = y_buffer(pred)
            dΘ = dΘ_buffer(pred)
            predict!(pred, ys, Θ; dΘ=dΘ)

            @test dΘ ≈ reshape(dΘ_numeric, size(dΘ))
        end

        
        @testset "Test Initial Conditions Derivatives" begin

            function cal_y0(x0)
                vec(predict(pred, Θ, reshape(x0, size(y0))))
            end

            dy0_numeric = Calculus.finite_difference_jacobian(cal_y0, vec(y0))

            ys = y_buffer(pred)
            dy0 = dy0_buffer(pred)
            predict!(pred, ys, Θ; dy0=dy0)

            @test dy0 ≈ reshape(dy0_numeric, size(dy0))
        end
    end
end

@testset "Test for MIMO data" begin
    # Generate Input Data
    len = 100
    srand(1)
    u = rand(2, len)-0.5;

    # Define Dynamic Model
    mdl = Linear(12, 2)
    yterms = [[1, 2, 5],[1, 2, 3]]
    uterms = [[2, 3, 4], [5, 6, 7]]
    diffeq = DifferenceEquation(mdl, yterms=yterms, uterms=uterms)

    # Simulate System for a given parameter
    Θ = rand(12*2)
    y0 = zeros(2, 5)
    ϕ = [Θ; vec(y0)]
    yaux = simulate(diffeq, u[:, 1:(end-2)], Θ, y0)
    y = [zeros(2, 2) y0 yaux]

    @testset "Test one-step-ahead prediction" begin
        pred = one_step_ahead(mdl, yterms, uterms, IdData(y, u))

        @testset "Test Basic Construction" begin
            @test pred.dynmdl.yterms == Vector{Vector{Int}}(0)
            @test pred.dynmdl.uterms == [[1, 2, 5],
                                         [1, 2, 3],
                                         [2, 3, 4],
                                         [5, 6, 7]]
            @test pred.u == [y; u][:, 1:(end-1)]
            @test pred.y0 == Matrix{Float64}(0, 0)
            @test pred.length == size(yaux, 2)
            @test pred.start == 8
        end

        @testset "Test Prediction" begin
            @test predict(pred, Θ) ≈ yaux
        end

        @testset "Test Params Derivatives" begin

            function cal_Θ(theta)
                vec(predict(pred, theta))
            end

            dΘ_numeric = Calculus.finite_difference_jacobian(cal_Θ, Θ)

            ys = y_buffer(pred)
            dΘ = dΘ_buffer(pred)
            predict!(pred, ys, Θ; dΘ=dΘ)

            @test dΘ ≈ reshape(dΘ_numeric, size(dΘ))
        end

    end

    @testset "Test free-run simulation" begin
        pred = free_run_simulation(mdl, yterms, uterms, IdData(y, u))

        @testset "Test Basic Construction" begin
            @test pred.dynmdl.yterms == [[1, 2, 5], [1, 2, 3]]
            @test pred.dynmdl.uterms == [[2, 3, 4], [5, 6, 7]]
            @test pred.u == u[:, 1:(end-2)]
            @test pred.y0 == y0
            @test pred.length == size(yaux, 2)
            @test pred.start == 8
        end

        @testset "Test Prediction" begin
            @test predict(pred, Θ) ≈ yaux
        end

        @testset "Test Params Derivatives" begin

            function cal_Θ(theta)
                vec(predict(pred, theta))
            end

            dΘ_numeric = Calculus.finite_difference_jacobian(cal_Θ, Θ)

            ys = y_buffer(pred)
            dΘ = dΘ_buffer(pred)
            predict!(pred, ys, Θ; dΘ=dΘ)

            @test dΘ ≈ reshape(dΘ_numeric, size(dΘ))
        end

        @testset "Test Initial Conditions Derivatives" begin

            function cal_y0(x0)
                vec(predict(pred, Θ, reshape(x0, size(y0))))
            end

            dy0_numeric = Calculus.finite_difference_jacobian(cal_y0, vec(y0))

            ys = y_buffer(pred)
            dy0 = dy0_buffer(pred)
            predict!(pred, ys, Θ; dy0=dy0)

            @test dy0 ≈ reshape(dy0_numeric, size(dy0))
        end
    end
end
