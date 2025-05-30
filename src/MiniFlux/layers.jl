struct Model
    layers::Vector
    params::Vector{AD.Variable}
end

function Model(layers::Vector)
    ps = reduce(vcat, layer_vars.(layers))  # spłaszczamy listę wektorów
    return Model(layers, ps)
end

function (m::Model)(x)
    a = x
    for layer in m.layers
        a = layer(a)
    end
    return a
end

function chain(layers...)
    return Model(collect(layers))
end

function layer_vars(layer)
    vars = AD.Variable[]
    if layer isa Embedding || layer isa Conv
        push!(vars, layer.weight)
        if isdefined(layer, :bias) && layer.bias !== nothing
            push!(vars, layer.bias)
        end
    elseif layer isa Dense
        push!(vars, layer.W)
        layer.b !== nothing && push!(vars, layer.b)
    end
    return vars
end


struct Dense
    W::AD.Variable
    b::Union{AD.Variable, Nothing}
    activation::Function
end

function Dense(in_dim::Int, out_dim::Int, activation::Function=AD.linear; bias::Bool=true)
    W = AD.Variable(randn(out_dim, in_dim) * 0.01, name="W")
    # Store bias as column vector (out_dim, 1)
    b = bias ? AD.Variable(zeros(out_dim, 1), name="b") : nothing
    return Dense(W, b, activation)
end

function (layer::Dense)(x::AD.GraphNode)
    lin = layer.W * x
    z = layer.b === nothing ? lin : lin .+ layer.b
    return layer.activation(z)
end