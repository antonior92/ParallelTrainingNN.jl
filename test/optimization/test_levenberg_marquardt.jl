# Copyright (c) 2012: John Myles White and other contributors.
# 
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


@testset "Test simple function" begin

    function f_lm!(x, r)
        r[1] = x[1]
        r[2] = 2.0 - x[2]
    end

    function g_lm!(x, j)
        j[1, 1] = 1.0
        j[1, 2] = 0.0
        j[2, 1] = 0.0
        j[2, 2] = -1.0
    end

    initial_x = [100.0, 100.0]

    m = 2

    results = ParallelTrainingNN.levenberg_marquardt(f_lm!, g_lm!, initial_x, m)
    @test norm(Optim.minimizer(results) - [0.0, 2.0]) < 0.01
end

@testset "Test Rosenbrock" begin
    
    function rosenbrock_res!(x, r)
        r[1] = 10.0 * (x[2] - x[1]^2 )
        r[2] =  1.0 - x[1]
    end

    function rosenbrock_jac!(x, j)
        j[1, 1] = -20.0 * x[1]
        j[1, 2] =  10.0
        j[2, 1] =  -1.0
        j[2, 2] =   0.0
    end

    m = 2

    initial_xrb = [-1.2, 1.0]

    results = ParallelTrainingNN.levenberg_marquardt(rosenbrock_res!,
                                        rosenbrock_jac!,
                                        initial_xrb, m)

    @test norm(Optim.minimizer(results) - [1.0, 1.0]) < 0.01

    # check estimate is within the bound PR #278
    result = ParallelTrainingNN.levenberg_marquardt(rosenbrock_res!,
                                       rosenbrock_jac!,
                                       [150.0, 150.0], m;
                                       lower = [10.0, 10.0],
                                       upper = [200.0, 200.0])
    @test Optim.minimizer(result)[1] >= 10.0
    @test Optim.minimizer(result)[2] >= 10.0
end
