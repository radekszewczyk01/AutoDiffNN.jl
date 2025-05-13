module MiniFlux

using LinearAlgebra

using ..AutoDiff
const AD = AutoDiff

include("layers.jl")
include("losses.jl")
include("optimizers.jl")
include("training.jl")

export Dense, mse_loss, sgd!, train!, Model, relu, swish, linear, layer_vars, create_batches,
       Embedding, PermuteDims, Conv1D, MaxPool1D, Flatten, Model, train!, binary_cross_entropy_loss,
       accuracy_fn, Conv


end 
