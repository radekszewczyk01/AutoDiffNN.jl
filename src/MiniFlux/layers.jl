struct Dense
    W::AD.Variable
    b::Union{AD.Variable, Nothing}
    activation::Function  # <- To może zostać
end

function Dense(in_dim::Int, out_dim::Int, activation::Function=AD.linear; bias::Bool=true)
    W = AD.Variable(randn(out_dim, in_dim), name="W")
    b = bias ? AD.Variable(randn(out_dim), name="b") : nothing
    return Dense(W, b, activation)
end

function (layer::Dense)(x::AD.Variable)
    lin = layer.W * x
    z   = layer.b === nothing ? lin : lin .+ layer.b
    return layer.activation(z)
end
