using Pkg

packages = [
    # [deps] MOMDPs.jl
    PackageSpec(url=joinpath(@__DIR__, "MOMDPs.jl")),

    # [deps] DirichletBeliefs.jl
    PackageSpec(url=joinpath(@__DIR__, "DirichletBeliefs.jl")),

    # [deps] EvacuationPOMDP.jl
    # PackageSpec(url=joinpath(@__DIR__)),
]

# Run dev altogether
# This is important that it's run together so there
# are no "expected pacakge X to be registered" errors.
Pkg.develop(packages)
