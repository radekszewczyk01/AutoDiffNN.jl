using LinearAlgebra

σ(x) = 1 ./ (1 .+ exp.(-x))
relu(x) = max.(0, x)
tanh(x) = Base.tanh.(x)

struct Dense
    W::AD.Variable
    b::Union{AD.Variable, Nothing}
    activation::Function
end

function Dense(in_dim::Int, out_dim::Int, activation::Function=identity; bias::Bool=true)
    W = AD.Variable(randn(out_dim, in_dim), name="W")
    b = bias ? AD.Variable(randn(out_dim), name="b") : nothing
    return Dense(W, b, activation)
end

function (layer::Dense)(x::AD.Variable)
    lin = layer.W * x
    z   = layer.b === nothing ? lin : lin .+ layer.b
    return layer.activation === identity ? z : layer.activation(z)
end
