@testset "Test Bias Evaluate" begin
    b = Bias(5)

    x = [1, 2, 3, 4, 5]
    Θ = [1, 2, 3, 4, 5]
    z = z_buffer(b)
    dx = dx_buffer(b)
    dΘ = dΘ_buffer(b)

    evaluate!(b, x, Θ, z, dx=dx, dΘ=dΘ)

    @test z == [2, 4, 6, 8, 10]
    @test dx == Diagonal(ones(5))
    @test dΘ == Diagonal(ones(5))
end
