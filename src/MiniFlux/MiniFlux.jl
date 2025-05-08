module MiniFlux

using LinearAlgebra

using ..AutoDiff
const AD = AutoDiff

include("layers.jl")
include("conv_layers.jl")
include("losses.jl")
include("optimizers.jl")
include("training.jl")

export Dense, Conv2D, MaxPool2D, Flatten, mse_loss, sgd!, train!, Model, relu, swish, linear

struct Model
    layers::Vector
    params::Vector{AD.Variable}
end

function (m::Model)(x)
    a = x
    for layer in m.layers
        a = layer(a)
    end
    return a
end

end 
