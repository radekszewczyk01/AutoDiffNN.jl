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

struct Embedding
    W::AD.Variable
end

function Embedding(vocab_size::Int, embedding_dim::Int)
    W = AD.Variable(randn(embedding_dim, vocab_size), name="W_embed")
    return Embedding(W)
end

function (layer::Embedding)(x::AD.GraphNode)
    return AD.Embedding(layer.W, x)
end

struct Conv
    W::AD.Variable  # (out_channels, in_channels, kernel_size...)
    b::Union{AD.Variable, Nothing}
    activation::Function
    stride::Int
    pad::Int
end

function Conv(kernel_size::Tuple, dims::Pair{Int, Int}, activation::Function=AD.linear; stride::Int=1, pad::Int=0)
    in_channels, out_channels = dims
    k = kernel_size[1]
    # filtr y[oc, i, c] gdzie i w kernel_size, c in_channels
    W = AD.Variable(randn(out_channels, k, in_channels), name="W_conv")
    b = AD.Variable(zeros(out_channels), name="b_conv")
    return Conv(W, b, activation, stride, pad)
end

function (layer::Conv)(x::AD.GraphNode)
    conv_op = AD.ConvOperator((x, layer.W, layer.b), nothing, nothing, layer.stride, layer.pad, "Conv1D")
    return layer.activation(conv_op)
end

struct PermuteDims
    perm::Tuple{Vararg{Int}}
end

function (layer::PermuteDims)(x::AD.GraphNode)
    return AD.PermuteDims(x, layer.perm)
end

"""  
    MaxPool1D(pool_size::Int)

Warstwa max‐poolingu 1D (bez overlapu, stride = pool_size).
"""
struct MaxPool1D
    pool_size::Int
end

function (layer::MaxPool1D)(x::GraphNode)
    return AD.MaxPool1D(x, layer.pool_size)
end

"""
    Flatten()

Warstwa spłaszczająca tensor `(batch, d1, d2, …, dn)` → `(batch, d1*d2*…*dn)`.
"""
struct Flatten end

function (layer::Flatten)(x::GraphNode)
    return AD.Flatten(x)
end


function layer_vars(layer)
    vs = AD.Variable[]
    for f in fieldnames(typeof(layer))
        val = getfield(layer, f)
        if val isa AD.Variable
            push!(vs, val)
        elseif val isa Union{AD.Variable, Nothing} && val !== nothing
            push!(vs, val)
        end
    end
    return vs
end
# Train: accepts Flux-like optimiser:
function train_cnn!(model::Model, loss_fn, data, optimiser, epochs::Int)
    # optimiser must implement update!(opt, ps, gs)
    opt_state = Optimisers.setup(optimiser, model.params)
    for epoch in 1:epochs
        for (x_batch, y_batch) in data
            # wrap inputs
            x = AD.Constant(x_batch)
            y = AD.Constant(y_batch)
            # forward
            ŷ = model(x)
            loss = loss_fn(y, ŷ)
            # backprop
            graph = AD.topological_sort(loss)
            AD.forward!(graph)
            AD.backward!(graph)
            # collect grads
            grads = map(p->p.gradient, model.params)
            # update
            Optimisers.update!(opt_state, model.params, grads)
            # zero grads
            for p in model.params
                p.gradient .= zero(p.gradient)
            end
        end
    end
end
