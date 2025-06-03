struct Model
    layers::Vector
    params::Vector{AD.Variable}
end

function Model(layers::Vector)
    ps = reduce(vcat, layer_vars.(layers))
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
    W_val = randn(Float32, out_dim, in_dim)
    W = AD.Variable(W_val, name="W")
    b_node = if bias
        bias_val = zeros(Float32, out_dim)
        AD.Variable(bias_val, name="b")
    else
        nothing
    end
    return Dense(W, b_node, activation)
end

function (layer::Dense)(x::AD.GraphNode)
    lin = layer.W * x
    z = layer.b === nothing ? lin : lin .+ layer.b
    return layer.activation(z)
end

struct Embedding
    W::AD.Variable
    vocab_size::Int
    embedding_dim::Int
end

function Embedding(vocab_size::Int, embedding_dim::Int)
    W_val = randn(Float32, embedding_dim, vocab_size) .* sqrt(1.0f0 / Float32(vocab_size))
    W = AD.Variable(W_val, name="W_embed")
    return Embedding(W, vocab_size, embedding_dim)
end

function (layer::Embedding)(x_indices_node::AD.GraphNode)
    return AD.EmbeddingOp(layer.W, x_indices_node)
end

struct Permute
    dims_order::Tuple
end
Permute(dims...) = Permute(tuple(dims...))

function (p::Permute)(x::AD.GraphNode)
    return AD.PermuteDimsOp(x, p.dims_order)
end

MiniFlux.layer_vars(layer::Permute) = AD.Variable[]


struct Conv1D
    kernel::AD.Variable
    bias::Union{AD.Variable, Nothing}
    activation_fn::Function
end

function Conv1D(kernel_size::Tuple{Int}, channels_in_out::Pair{Int,Int}; activation::Function=AD.linear, use_bias::Bool=true)
    KW = kernel_size[1]
    Cin, Cout = channels_in_out.first, channels_in_out.second
    
    kernel_val = randn(Float32, KW, Cin, Cout) .* sqrt(2.0f0 / Float32(KW * Cin))
    K = AD.Variable(kernel_val, name="K_conv1d")
    
    b = if use_bias
        bias_val = zeros(Float32, 1, Cout, 1)
        AD.Variable(bias_val, name="b_conv1d")
    else
        nothing
    end
    return Conv1D(K, b, activation)
end

function (layer::Conv1D)(x::AD.GraphNode)
    conv_out = AD.Conv1DOp(x, layer.kernel)
    z = if layer.bias === nothing
        conv_out
    else
        conv_out .+ layer.bias
    end
    return layer.activation_fn(z)
end

struct MaxPool1D
    pool_size::Tuple{Int}
end
MaxPool1D(pool_size::Int) = MaxPool1D((pool_size,))

function (layer::MaxPool1D)(x::AD.GraphNode)
    return AD.MaxPool1DOp(x, layer.pool_size)
end
MiniFlux.layer_vars(layer::MaxPool1D) = AD.Variable[]

struct Flatten end

function (f::Flatten)(x::AD.GraphNode)
    return AD.FlattenOp(x)
end
MiniFlux.layer_vars(layer::Flatten) = AD.Variable[]