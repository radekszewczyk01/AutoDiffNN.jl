# Conv operator
mutable struct ConvOperator{D} <: Operator
    inputs::Tuple
    output::Any
    gradient::Any
    name::String
    kernel_size::Tuple
    stride::Tuple
    padding::Tuple
    dilation::Tuple
    ConvOperator(inputs...; kernel_size, stride=1, padding=0, dilation=1, name="?") = 
        new{length(kernel_size)}(inputs, nothing, nothing, name, kernel_size, stride, padding, dilation)
end

# MaxPool operator
mutable struct MaxPoolOperator{D} <: Operator
    inputs::Tuple
    output::Any
    gradient::Any
    name::String
    k::Tuple
    MaxPoolOperator(k, inputs...; name="?") = new{length(k)}(inputs, nothing, nothing, name, k)
end

# Embedding operator
mutable struct EmbeddingOperator <: Operator
    inputs::Tuple
    output::Any
    gradient::Any
    name::String
    EmbeddingOperator(inputs...; name="?") = new(inputs, nothing, nothing, name)
end

# Flatten operator
mutable struct FlattenOperator <: Operator
    inputs::Tuple
    output::Any
    gradient::Any
    name::String
    FlattenOperator(inputs...; name="?") = new(inputs, nothing, nothing, name)
end 

# Conv forward/backward

function forward(op::ConvOperator, x, w)
    return conv(x, w; 
        stride=op.stride,
        pad=op.padding,
        dilation=op.dilation)
end

function backward(op::ConvOperator, x, w, g)
    # println(size(x), size(w), size(g))
    
    # Create ConvDims object
    cdims = NNlib.DenseConvDims(
        size(x), size(w);
        stride=op.stride,
        padding=op.padding,
        dilation=op.dilation
    )
    
    ∇x = NNlib.∇conv_data(g, w, cdims)
    ∇w = NNlib.∇conv_filter(x, g, cdims)
    
    return (∇x, ∇w)
end

# MaxPool forward/backward
function forward(op::MaxPoolOperator, x)
    return maxpool(x, op.k)
end

function backward(op::MaxPoolOperator, x, g)
    return ∇maxpool(g, op.output, x, op.k)
end

# Embedding forward/backward
function forward(op::EmbeddingOperator, weight, indices)
    return weight[:, indices]
end

function backward(op::EmbeddingOperator, weight, indices, g)
    ∇weight = zeros(size(weight))
    
    # Handle both 2D and 3D gradient cases
    if ndims(g) == 3
        # 3D gradient: (embed_dim, seq_len, batch_size)
        for b in 1:size(g, 3)
            for j in 1:size(g, 2)
                idx = indices[j, b]
                ∇weight[:, idx] += g[:, j, b]
            end
        end
    else
        # 2D gradient: (embed_dim, seq_len * batch_size)
        for (i, idx) in enumerate(indices)
            ∇weight[:, idx] += g[:, i]
        end
    end
    
    return (∇weight, nothing)
end

# Flatten forward/backward
function forward(op::FlattenOperator, x)
    return reshape(x, :, size(x, ndims(x)))
end

function backward(op::FlattenOperator, x, g)
    return reshape(g, size(x))
end

# MaxPool helper
# MaxPool helper - 1D pooling
function maxpool(x, k::Tuple{Int})
    k1 = k[1]
    out_width = div(size(x, 1), k1)
    out = similar(x, (out_width, size(x, 2), size(x, 3)))
    
    for i in 1:size(x, 3)    # Batch dimension
        for j in 1:size(x, 2) # Channel dimension
            for l in 1:out_width
                start_idx = (l-1)*k1 + 1
                end_idx = min(l*k1, size(x, 1))
                window = view(x, start_idx:end_idx, j, i)
                out[l, j, i] = maximum(window)
            end
        end
    end
    return out
end

function ∇maxpool(g, y, x, k)
    # Convert scalar gradient to array with same size as maxpool output
    if isa(g, Number)
        g = fill(g, size(y))
    end
    
    ∇x = zeros(size(x))
    out_height, out_channels, batch_size = size(y)
    k1 = k[1]
    
    for b in 1:batch_size
        for c in 1:out_channels
            for h in 1:out_height
                start_idx = (h-1)*k1 + 1
                end_idx = min(h*k1, size(x, 1))
                window = start_idx:end_idx
                
                # Find index of max value in window
                window_vals = view(x, window, c, b)
                max_idx = argmax(window_vals)
                global_idx = start_idx + max_idx - 1
                
                # Accumulate gradient
                ∇x[global_idx, c, b] += g[h, c, b]
            end
        end
    end
    return ∇x
end

mutable struct PermuteDims <: Operator
    inputs::Tuple
    output::Any
    gradient::Any
    perm::Tuple
    name::String
    PermuteDims(input, perm; name="permutedims") = new((input,), nothing, nothing, perm, name)
end

function forward(node::PermuteDims, x)
    return permutedims(x, node.perm)
end

function backward(node::PermuteDims, x, g)
    inv_perm = sortperm(collect(node.perm))
    return (permutedims(g, inv_perm),)
end