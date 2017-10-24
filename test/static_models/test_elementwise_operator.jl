@testset "Test LogisticFunction"  begin
    L = LogisticFunction(3)
    
    x = [1, 2, 3]
    z = z_buffer(L)
    dx = dx_buffer(L)

    evaluate!(L, x, z, dx=dx)
    
    @test z ≈ [0.7310585786, 0.8807970779, 0.9525741268]
    @test dx ≈ [0.1966119332 0 0;
                0 0.1049935854 0;
                0 0 0.0451766597]
end

@testset "Test HyperbolicTangent"  begin
    H = HyperbolicTangent(3)
    
    x = [1, 2, 3]
    z = z_buffer(H)
    dx = dx_buffer(H)

    evaluate!(H, x, z, dx=dx)
    
    @test z ≈ [0.7615941559, 0.9640275801, 0.9950547537]
    @test dx ≈ [0.4199743416 0 0;
                0 0.0706508249 0;
                0 0 0.0098660372]
end

@testset "Test AffineMap"  begin
    @testset "Full Input"  begin
        A = AffineMap([2, 2, 4], [1, 2, 3])
        
        x = [1, 2, 3]
        z = z_buffer(A)
        dx = dx_buffer(A)

        evaluate!(A, x, z, dx=dx)
        
        @test z ≈ [3.0, 6.0, 15.0]
        @test dx ≈ [2.0 0 0;
                    0 2.0 0;
                    0 0 4.0]
    end
    @testset "Default Bias"  begin
        A = AffineMap([2, 2, 4])
        
        x = [1, 2, 3]
        z = z_buffer(A)
        dx = dx_buffer(A)

        evaluate!(A, x, z, dx=dx)
        
        @test z ≈ [2.0, 4.0, 12.0]
        @test dx ≈ [2.0 0 0;
                    0 2.0 0;
                    0 0 4.0]
    end

    @testset "Argument Error"  begin
        @test_throws ArgumentError AffineMap([2, 2, 4],
                                             [1, 1, 1, 1])
    end
end

@testset "Test Identity"  begin
    H = Identity(3)
    
    x = [1, 2, 3]
    z = z_buffer(H)
    dx = dx_buffer(H)

    evaluate!(H, x, z, dx=dx)
    
    @test z ≈ [1.0, 2.0, 3.0]
    @test dx ≈ [1.0 0 0;
                0 1.0 0;
                0 0 1.0]
end
