@testset "Test for SISO data" begin
    # Generate Input Data
    len = 50
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
    iddata = IdData(y, u)

    @testset "Test one-step-ahead prediction" begin
        pred = one_step_ahead(mdl, yterms, uterms, iddata)

        e = SimpleError(pred, iddata)

        function er(p)
            e_buffer = Vector{Float64}(len-2)
            error_function!(e, e_buffer, p)
            return e_buffer
        end

        function jac(p)
            e_buffer = Vector{Float64}(len-2)
            dΘ_buffer = Matrix{Float64}(len-2, 5)
            error_function!(e, e_buffer, p, dϕ=dΘ_buffer)
            return dΘ_buffer
        end

        J_numeric = Calculus.finite_difference_jacobian(er, Θ)

        @test er(Θ) ≈ zeros(len-2)
        @test jac(Θ) ≈ J_numeric
    end
    
    @testset "Test free-run simulation" begin
        pred = free_run_simulation(mdl, yterms, uterms, iddata)

        @testset "Test SimpleError" begin
            e = SimpleError(pred, iddata)

            function er(p)
                e_buffer = Vector{Float64}(len-2)
                error_function!(e, e_buffer, p)
                return e_buffer
            end

            function jac(p)
                e_buffer = Vector{Float64}(len-2)
                dΘ_buffer = Matrix{Float64}(len-2, 5)
                error_function!(e, e_buffer, p, dϕ=dΘ_buffer)
                return dΘ_buffer
            end

            J_numeric = Calculus.finite_difference_jacobian(er, Θ)

            @test er(Θ) ≈ zeros(len-2)
            @test jac(Θ) ≈ J_numeric
        end

        @testset "Test ExtendedParamsError" begin
            e = ExtendedParamsError(pred, iddata)

            function er(p)
                e_buffer = Vector{Float64}(len-2)
                error_function!(e, e_buffer, p)
                return e_buffer
            end

            function jac(p)
                e_buffer = Vector{Float64}(len-2)
                dΘ_buffer = Matrix{Float64}(len-2, 5+2)
                error_function!(e, e_buffer, p, dϕ=dΘ_buffer)
                return dΘ_buffer
            end

            J_numeric = Calculus.finite_difference_jacobian(er, ϕ)

            @test er(ϕ) ≈ zeros(len-2)
            @test jac(ϕ) ≈ J_numeric
        end
    end
    
    @testset "Test NARX" begin
        Θs, opt, e = narx(mdl, yterms, uterms, IdData(y, u))

        @test Θs ≈ Θ
    end

    @testset "Test NOE" begin
        Θs, opt, e = noe(mdl, yterms, uterms, IdData(y, u))

        @test Θs ≈ Θ
    end

    @testset "Test NOE (Extended Parameter)" begin
        Θs, opt, e = noe(mdl, yterms, uterms, IdData(y, u); use_extended=true)

        @test Θs ≈ Θ
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
        iddata = IdData(y, u)
        pred = one_step_ahead(mdl, yterms, uterms, iddata)

        e = SimpleError(pred, iddata)

        function er(p)
            e_buffer = Vector{Float64}(93*2)
            error_function!(e, e_buffer, p)
            return e_buffer
        end

        function jac(p)
            e_buffer = Vector{Float64}(93*2)
            dΘ_buffer = Matrix{Float64}(93*2, 12*2)
            error_function!(e, e_buffer, p, dϕ=dΘ_buffer)
            return dΘ_buffer
        end

        J_numeric = Calculus.finite_difference_jacobian(er, Θ)

        @test er(Θ) ≈ zeros(93*2)
        @test jac(Θ) ≈ J_numeric
    end

    @testset "Test free-run simulation" begin
        iddata = IdData(y, u)
        pred = free_run_simulation(mdl, yterms, uterms, iddata)

        @testset "Test SimpleError" begin
            e = SimpleError(pred, iddata)
            
            function er(p)
                e_buffer = Vector{Float64}(93*2)
                error_function!(e, e_buffer, p)
                return e_buffer
            end

            function jac(p)
                e_buffer = Vector{Float64}(93*2)
                dΘ_buffer = Matrix{Float64}(93*2, 12*2)
                error_function!(e, e_buffer, p, dϕ=dΘ_buffer)
                return dΘ_buffer
            end

            J_numeric = Calculus.finite_difference_jacobian(er, Θ)

            @test er(Θ) ≈ zeros(93*2)
            @test jac(Θ) ≈ J_numeric
        end

        @testset "Test ExtendedParamsError" begin
            e = ExtendedParamsError(pred, iddata)

            function er(p)
                e_buffer = Vector{Float64}(93*2)
                error_function!(e, e_buffer, p)
                return e_buffer
            end

            function jac(p)
                e_buffer = Vector{Float64}(93*2)
                dΘ_buffer = Matrix{Float64}(93*2, 12*2+2*5)
                error_function!(e, e_buffer, p, dϕ=dΘ_buffer)
                return dΘ_buffer
            end

            J_numeric = Calculus.finite_difference_jacobian(er, ϕ)

            @test er(ϕ) ≈ zeros(93*2)
            @test jac(ϕ) ≈ J_numeric
        end
    end
end
