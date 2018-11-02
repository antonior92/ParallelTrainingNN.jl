#*************************************************************************#
#
# IdData type
#
#*************************************************************************#
"""
    IdData(y, [u, ts])

Create type used to store sampled data used in the identification
procedure.

Constructor Inputs:
------------------

* `y` is an output signal. A vector of length `nd` for a single-output system,
 where `nd` is the number of observations and, for a multiple-output
 system with `ny` output chanels, a `ny`-by-`nd` matrix.
* `u` is an input signal. A vector of length `nd` for a single-input system,
 where `nd` is the number of observations and, for a multiple-input
 system with `ny` output chanels, a `nu`-by-`nd` matrix.
* Time interval between two successive data samples.

Constructor Optional Keywords:
-----------------------------

* `t_start` is a Float64 number specifying the start time of
 the time vector. By default `t_start=0`.
* `time_unit` is a string describing the time unit.
* `input_unit` is, for single-input system, a string describing
 the input unit. For a multiple-input system with `ny` output chanels,
 a vector of length `ny` containing one string per input.
* `output_unit` is, for single-output system, a string describing
 the output unit. For a multiple-output system with `ny` output chanels,
 a vector of length `ny` containing one string per output.
* `input_name` is, for single-input system, a string describing
 the input name. For a multiple-input system with `ny` output chanels,
 a vector of length `ny` containing one string per input.
* `output_name` is, for single-output system, a string describing
 the output name. For a multiple-output system with `ny` output chanels,
 a vector of length `ny` containing one string per output.
* `annotation` is a string containing a description of the data.

Basic Functions:
---------------

| Function                  | Description                                          |
|:------------------------- |:---------------------------------------------------- |
| `get_time_vector(iddata)` | Return time vector corresponding to the observations |
| `get_input(iddata)`       | Return a matrix containing input samples rowwise     |
| `get_output(iddata)`      | Return a matrix containing output samples rowwise    |
| `plot_input(iddata)`      | Plot inputs                                          |
| `plot_output(iddata)`     | Plot outputs                                         |
"""
mutable struct IdData
    y::Matrix{Float64}
    u::Matrix{Float64}
    ts::Float64
    annotation::String
    t_start::Float64
    time_unit::String
    input_unit::Vector{String}
    output_unit::Vector{String}
    input_name::Vector{String}
    output_name::Vector{String}
end

# Constructor for MIMO data
function IdData(y::Matrix,u::Matrix=Matrix{Float64}(0, 0),
                ts=1.0;
                annotation::String="",
                t_start=0.0,
                time_unit::String="",
                input_unit::Vector{String}=fill("", size(u, 1)),
                output_unit::Vector{String}=fill("", size(y, 1)),
                input_name::Vector{String}=[string("u", i)
                                            for i=1:size(u, 1)],
                output_name::Vector{String}=[string("y", i)
                                             for i=1:size(y, 1)])

    # Check Arguments
    ndy = size(y, 2)
    ndu = size(u, 2)
    if ndy != ndu
        throw(DimensionMismatch("y has $ndy while u has $ndu samples."))
    end

    IdData(y, u, ts, annotation, t_start, time_unit, input_unit, output_unit,
           input_name, output_name)
end

# Constructor for SISO data
function IdData(y::Vector, u::Vector=Vector{Float64}(0, 0),
              ts=1.0;
              annotation::String="",
              t_start=0.0,
              time_unit::String="",
              input_unit::String="",
              output_unit::String="",
              input_name::String="u",
              output_name::String="y")

    # Check Arguments
    ndy = length(y)
    ndu = length(u)
    if ndy != ndu
        throw(DimensionMismatch("y has $ndy while u has $ndu samples."))
    end

    y = reshape(y, 1, ndy)
    u = reshape(u, 1, ndu)
    IdData(y, u, ts, annotation, t_start, time_unit, [input_unit], [output_unit],
           [input_name], [output_name])
end

# Constructor for MISO data
function IdData(y::Vector, u::Matrix,
              ts=1.0;
              annotation::String="",
              t_start=0.0,
              time_unit::String="",
              input_unit::Vector{String}=fill("", size(u, 1)),
              output_unit::String="",
              input_name::Vector{String}=[string("u", i)
                                          for i=1:size(u, 1)],
              output_name::String="y")

    # Check Arguments
    ndy = length(y)
    ndu = size(u, 2)
    if ndy != ndu
        throw(DimensionMismatch("y has $ndy while u has $ndu samples."))
    end

    y = reshape(y, 1, ndy)
    IdData(y, u, ts, annotation, t_start, time_unit, input_unit, [output_unit],
           input_name, [output_name])
end

# Constructor for SIMO data
function IdData(y::Matrix, u::Vector,
              ts=1.0;
              annotation::String="",
              t_start=0.0,
              time_unit::String="",
              input_unit::String="",
              output_unit::Vector{String}=fill("", size(y, 1)),
              input_name::String="u",
              output_name::Vector{String}=[string("y", i)
                                           for i=1:size(y, 1)])

    # Check Arguments
    ndy = size(y, 2)
    ndu = length(u)
    if ndy != ndu
        throw(DimensionMismatch("y has $ndy while u has $ndu samples."))
    end

    u = reshape(u, 1, ndu)
    IdData(y, u, ts, annotation, t_start, time_unit, [input_unit], output_unit,
           [input_name], output_name)
end


"""
    get_time_vector(iddata::IdData) -> t

Return a time vector `t` containing the moments
correspondents to each sample.
"""
function get_time_vector(iddata::IdData)
    nd = size(iddata.y, 2)
    t = iddata.t_start + iddata.ts*(0:nd-1)

    return Vector(t)
end


"""
    get_input(iddata::IdData) -> u

Return a `u` matrix containing the inputs. Each
matrix row contains a different input.
"""
function get_input(iddata::IdData)
    return iddata.u
end


"""
    get_output(iddata::IdData) -> y

Return a `y` matrix containig the outputs. Each
matrix row contains a different output.
"""
function get_output(iddata::IdData)
    return iddata.y
end
