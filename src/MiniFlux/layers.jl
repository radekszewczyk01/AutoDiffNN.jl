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
    for f in fieldnames(typeof(layer))
        val = getfield(layer, f)
        if val isa AD.Variable
            push!(vars, val)
        elseif val isa Union{Nothing, AD.Variable} && val !== nothing
            push!(vars, val)
        elseif val isa AbstractArray
            append!(vars, filter(x -> x isa AD.Variable, val))
        end
    end
    return vars
end


struct Dense
    W::AD.Variable
    b::Union{AD.Variable, Nothing}
    activation::Function
end

function Dense(in_dim::Int, out_dim::Int, activation::Function=AD.linear; bias::Bool=true)
    W = AD.Variable(randn(out_dim, in_dim), name="W")
    b = bias ? AD.Variable(randn(out_dim), name="b") : nothing
    return Dense(W, b, activation)
end

function (layer::Dense)(x::AD.GraphNode)
    lin = layer.W * x
    z = layer.b === nothing ? lin : lin .+ layer.b
    return layer.activation(z)
end