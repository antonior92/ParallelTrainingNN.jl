@testset "Test Learn Offset" begin
    y = repmat([1; 2; 3; 4], 1, 10)
    u = repmat([5; 6; 7], 1, 10)

    yterms = [[1, 2, 3], [1, 2], [2], [3, 4]]
    uterms = [[1], [2, 3], [4, 5]]

    mdl = Linear(13, 4)

    new_mdl = learn_offset(mdl, yterms, uterms, IdData(y, u))

    @test new_mdl.mdls[1].a == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    @test new_mdl.mdls[1].b == -[1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7]
    @test new_mdl.mdls[3].a == [1, 1, 1, 1]
    @test new_mdl.mdls[3].b == [1, 2, 3, 4]
end

@testset "Test Learn Normalization" begin
    @testset "Basic Usage" begin
        y = repmat([1 3; 1 3], 1, 1000)
        u = repmat([-1 1], 1, 1000)

        yterms = [[1, 2], [1, 2]]
        uterms = [[1, 2]]

        mdl = Linear(6, 2)

        new_mdl = learn_normalization(mdl, yterms, uterms, IdData(y, u); nσ=1)

        @test new_mdl.mdls[1].a ≈ 1./[1, 1, 1, 1, 1, 1] atol=1e-1
        @test new_mdl.mdls[1].b ≈  -[2, 2, 2, 2, 0, 0]./[1, 1, 1, 1, 1, 1] atol=1e-1
        @test new_mdl.mdls[3].a ≈ [1, 1] atol=1e-1
        @test new_mdl.mdls[3].b ≈ [2, 2] atol=1e-1
    end

    @testset "Zero Standard Deviation" begin
        y = repmat([1 1; -2 6], 1, 1000)
        u = repmat([-1 1], 1, 1000)

        yterms = [[1, 2], [1, 2]]
        uterms = [[1, 2]]

        mdl = Linear(6, 2)

        new_mdl = learn_normalization(mdl, yterms, uterms, IdData(y, u); nσ=1)

        @test new_mdl.mdls[1].a ≈ 1./[1, 1, 4, 4, 1, 1]  atol=1e-1
        @test new_mdl.mdls[1].b ≈ -[1, 1, 2, 2, 0, 0]./[1, 1, 4, 4, 1, 1]  atol=1e-1
        @test new_mdl.mdls[3].a ≈ [1, 4]  atol=1e-1
        @test new_mdl.mdls[3].b ≈ [1, 2]  atol=1e-1
    end
end
