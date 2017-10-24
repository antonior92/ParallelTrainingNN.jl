@testset "Test Constructor for MIMO" begin
    y = rand(2, 100)
    u = rand(3, 100)
    ts = 0.1

    iddata = IdData(y, u, ts)

    @test iddata.t_start == 0
    @test iddata.time_unit == ""
    @test iddata.input_unit == ["", "", ""]
    @test iddata.output_unit == ["", ""]
    @test iddata.input_name == ["u1", "u2", "u3"]
    @test iddata.output_name == ["y1", "y2"]

    @test_throws DimensionMismatch IdData(y[:, 1:end-1], u, ts)
end

@testset "Test Constructor for SISO" begin
    y = rand(100)
    u = rand(100)
    ts = 0.1

    iddata = IdData(y, u, ts)

    @test iddata.t_start == 0
    @test iddata.time_unit == ""
    @test iddata.input_unit == [""]
    @test iddata.output_unit == [""]
    @test iddata.input_name == ["u"]
    @test iddata.output_name == ["y"]

    @test_throws DimensionMismatch IdData(y[1:end-1], u, ts)
end

@testset "Test Constructor for MISO" begin
    y = rand(100)
    u = rand(2, 100)
    ts = 0.1

    iddata = IdData(y, u, ts)

    @test iddata.t_start == 0
    @test iddata.time_unit == ""
    @test iddata.input_unit == ["", ""]
    @test iddata.output_unit == [""]
    @test iddata.input_name == ["u1", "u2"]
    @test iddata.output_name == ["y"]

    @test_throws DimensionMismatch IdData(y[1:end-1], u, ts)
end

@testset "Test Constructor for SIMO" begin
    y = rand(2, 100)
    u = rand(100)
    ts = 0.1

    iddata = IdData(y, u, ts)

    @test iddata.t_start == 0
    @test iddata.time_unit == ""
    @test iddata.input_unit == [""]
    @test iddata.output_unit == ["", ""]
    @test iddata.input_name == ["u"]
    @test iddata.output_name == ["y1", "y2"]

    @test_throws DimensionMismatch IdData(y[:, 1:end-1], u, ts)
end

@testset "Test Get Methods" begin
    y = rand(100)
    u = rand(100)
    ts = 0.1
    t_start = 10

    iddata = IdData(y, u, ts, t_start=t_start)

    @test get_time_vector(iddata) ≈ Vector(10.0:0.1:19.9)
    @test get_output(iddata) ≈ reshape(y, 1, 100)
    @test get_input(iddata) ≈ reshape(u, 1, 100)
end
