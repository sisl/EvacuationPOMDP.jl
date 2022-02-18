module DirichletBeliefs

using BeliefUpdaters
using POMDPs
import POMDPs: Updater, update, initialize_belief, pdf, mode, updater, support
using POMDPModelTools
using LinearAlgebra
using Distributions
using Parameters
using Random
using StatsBase
using Statistics
using MOMDPs

export
    DirichletBelief,
    uniform_belief,
    pdf,
    support,
    DirichletUpdater,
    initialize_belief,
    update,
    DirichletSubspaceBelief,
    DirichletSubspaceUpdater

include("dirichlet.jl")

export
    DiscreteSubspaceBelief,
    uniform_belief,
    pdf,
    support,
    DiscreteSubspaceUpdater,
    initialize_belief,
    update

include("discrete.jl")

end # module
