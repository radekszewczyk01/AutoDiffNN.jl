module MiniFlux

using LinearAlgebra

using ..AutoDiff
const AD = AutoDiff
using NNlib: conv

include("cnn_layers.jl")
include("layers.jl")
include("losses.jl")
include("optimizers.jl")
include("training.jl")

export  Dense, mse_loss, sgd!, train!, Model, relu, swish, linear, layer_vars, create_batches, 
        Model, train!, binary_cross_entropy_loss, categorical_cross_entropy, softmax, Adam,
        accuracy, Embedding, Conv, Flatten, MaxPool


end 
