module MiniFlux

using LinearAlgebra

using ..AutoDiff
const AD = AutoDiff

include("layers.jl")
include("losses.jl")
include("optimizers.jl")
include("training.jl")

export Dense, mse_loss, sgd!, train!, Model, σ, relu, tanh

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
