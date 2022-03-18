module EvacuationPOMDP

using Revise

using BasicPOMCP
using BSON
using ColorSchemes
using D3Trees
using DataStructures
using DirichletBeliefs
using DiscreteValueIteration
using Distributions 
using Graphs
using JSON
using LaTeXStrings
using LinearAlgebra
using Measures
using MOMDPs
using Parameters
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using POMDPModelTools
using POMDPPolicies
using POMDPs
using POMDPSimulators
using QuickPOMDPs
using Random
using StatsBase
import TikzGraphs

export
    VisaStatus,
    ISIS,
    VulAfghan,
    P1P2Afghan,
    SIV,
    AMCIT,
    NULL,
    MDPState,
    VisibleState,
    HiddenState,
    POMDPState,
    Action,
    REJECT,
    ACCEPT,
    VisaDocument,
    ISIS_indicator,
    VulAfghan_document,
    P1P2Afghan_document,
    SIV_document,
    AMCIT_document,
    NULL_document,
    Observation,
    documentation, # TODO: EvacuationParameters or in EvacuationPOMDPType
    EvacuationParameters,
    ClaimModel,
    EvacuationMDP,
    EvacuationPOMDPType, # TODO: Rename (module too)
    validtime,
    validcapacity,
    R
include("common.jl")


include("mdp.jl")

export
    likelihood,
    reset_population_belief!
include("pomdp.jl")

export
    AcceptAllPolicy,
    AMCITsPolicy,
    SIVAMCITsPolicy,
    AfterThresholdAMCITsPolicy,
    BeforeThresholdAMCITsPolicy,
    RandomBaselinePolicy,
    MDPRolloutPolicy,
    SIVAMCITsP1P2Policy
include("policies.jl")


export
    get_metrics,
    manual_simulate,
    simulation,
    mean_std,
    simulations,
    experiments
include("simulation.jl")


export
    simulate_trajectory,
    plot_trajectory
include("trajectory.jl")

export
    plot_claims,
    plot_claims_tiny,
    plot_all_claims,
    vis_all
include("visualization.jl")


export
    create_family_size_distribution,
    plot_family_size_distribution
include("familysize.jl")

end # module
