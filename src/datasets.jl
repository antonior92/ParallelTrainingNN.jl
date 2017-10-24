#*************************************************************************#
#
# Datasets 
#
#*************************************************************************#
"""
    pilot_plant() -> identification_data, validation_data

Return pilot plant data.
"""
function pilot_plant()
    path = joinpath(Pkg.dir(), "SysId", "datasets", "pilot_plant.jld")
    JLD.load(path)
end
