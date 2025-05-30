# Embedding layer
struct Embedding
    weight::AD.Variable
    name::String
end

function Embedding(vocab_size::Int, embed_size::Int; name="embed")
    weight = AD.Variable(randn(embed_size, vocab_size), name=name)
    return Embedding(weight, name)
end

function (layer::Embedding)(x::AD.GraphNode)
    return AD.EmbeddingOperator(layer.weight, x)
end

# Conv layer
struct Conv
    weight::AD.Variable
    bias::Union{AD.Variable, Nothing}
    activation::Function
    kernel_size::Tuple
    stride::Tuple
    padding::Tuple
    dilation::Tuple
end

# In cnn_layers.jl
# In cnn_layers.jl
function Conv(kernel_size::Int, in_chs::Int, out_chs::Int; 
              activation=AD.relu, stride=1, padding=1, dilation=1, bias=true)
    weight = AD.Variable(randn(kernel_size, in_chs, out_chs), name="conv_weight")
    # Correct bias dimensions: (1, out_chs, 1)
    b = bias ? AD.Variable(randn(1, out_chs, 1), name="conv_bias") : nothing
    return Conv(weight, b, activation, (kernel_size,), (stride,), (padding,), (dilation,))
end

# In cnn_layers.jl
# In cnn_layers.jl
function (layer::Conv)(x::AD.GraphNode)
    conv_out = AD.ConvOperator(x, layer.weight;
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation)
    
    if layer.bias !== nothing
        # Use BroadcastedOperator for addition instead of direct +
        conv_out = AD.BroadcastedOperator(+, conv_out, layer.bias)
    end
    
    return layer.activation(conv_out)
end
# MaxPool layer
struct MaxPool
    k::Tuple
end

function (layer::MaxPool)(x::AD.GraphNode)
    return AD.MaxPoolOperator(layer.k, x)
end

# Flatten layer
struct Flatten end

function (layer::Flatten)(x::AD.GraphNode)
    return AD.FlattenOperator(x)
end
