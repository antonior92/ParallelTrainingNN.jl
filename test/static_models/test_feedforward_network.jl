@testset "Test Contructor I" begin
    mdl1 = LogisticFunction(2)
    mdl2 = HyperbolicTangent(3)
    mdl3 = LogisticFunction(4)
    mdl4 = Identity(2)
    mdl = FeedforwardNetwork(3, [mdl1, mdl2, mdl3, mdl4])

    @test mdl.ninputs == 3
    @test mdl.noutputs == 2
    @test mdl.mdls[1] == Linear(3, 2)
    @test mdl.mdls[2] == Bias(2)
    @test mdl.mdls[3] == LogisticFunction(2)
    @test mdl.mdls[4] == Linear(2, 3)
    @test mdl.mdls[5] == Bias(3)
    @test mdl.mdls[6] == HyperbolicTangent(3)
    @test mdl.mdls[7] == Linear(3, 4)
    @test mdl.mdls[8] == Bias(4)
    @test mdl.mdls[9] == LogisticFunction(4)
    @test mdl.mdls[10] == Linear(4, 2)
    @test mdl.mdls[11] == Bias(2)
    @test mdl.mdls[12] == Identity(2)
end

@testset "Test Contructor II" begin
    mdl = FeedforwardNetwork(3, 2, [2, 3, 4])

    @test mdl.ninputs == 3
    @test mdl.noutputs == 2
    @test mdl.mdls[1] == Linear(3, 2)
    @test mdl.mdls[2] == Bias(2)
    @test mdl.mdls[3] == HyperbolicTangent(2)
    @test mdl.mdls[4] == Linear(2, 3)
    @test mdl.mdls[5] == Bias(3)
    @test mdl.mdls[6] == HyperbolicTangent(3)
    @test mdl.mdls[7] == Linear(3, 4)
    @test mdl.mdls[8] == Bias(4)
    @test mdl.mdls[9] == HyperbolicTangent(4)
    @test mdl.mdls[10] == Linear(4, 2)
    @test mdl.mdls[11] == Bias(2)
    @test mdl.mdls[12] == Identity(2)
end
