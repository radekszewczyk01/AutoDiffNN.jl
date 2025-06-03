import Base: ^, sin, *, sum, max
import LinearAlgebra: mul!
import Base.Broadcast: broadcasted

function unbroadcast_gradient(grad_output::AbstractArray, input_shape::Tuple)
    if isempty(input_shape)
        return sum(grad_output)
    end

    if size(grad_output) == input_shape
        return grad_output
    end

    sum_dims = Int[]
    for i in 1:ndims(grad_output)
        s_out = size(grad_output, i)
        s_in = (i <= length(input_shape)) ? input_shape[i] : 1
        
        if s_out > 1 && s_in == 1
            push!(sum_dims, i)
        end
    end
    
    grad_in = grad_output
    if !isempty(sum_dims)
        grad_in = sum(grad_in, dims=Tuple(sum_dims))
    end
    
    return reshape(grad_in, input_shape)
end

function unbroadcast_gradient(grad_output::Number, input_shape::Tuple)
    if isempty(input_shape)
        return grad_output
    end
    return grad_output
end

^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = begin
    res = A * x
    return res
end
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x_val, y_val) = x_val .* y_val
backward(node::BroadcastedOperator{typeof(*)}, x_val, y_val, g) = (
    unbroadcast_gradient(g .* y_val, size(x_val)),
    unbroadcast_gradient(g .* x_val, size(y_val))
)

broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x_val, y_val) = x_val .- y_val
backward(::BroadcastedOperator{typeof(-)}, x_val, y_val, g) = (
    unbroadcast_gradient(g, size(x_val)),
    unbroadcast_gradient(-g, size(y_val))
)

broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x_val, y_val) = begin
    res = x_val .+ y_val
    return res
end
backward(::BroadcastedOperator{typeof(+)}, x_val, y_val, g) = (
    unbroadcast_gradient(g, size(x_val)),
    unbroadcast_gradient(g, size(y_val))
)

broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x_val, y_val) = x_val ./ y_val
backward(node::BroadcastedOperator{typeof(/)}, x_val, y_val, g) = (
  unbroadcast_gradient(g ./ y_val, size(x_val)),
  unbroadcast_gradient(-g .* x_val ./ (y_val .^ 2), size(y_val))
)

σ(x::GraphNode) = BroadcastedOperator(σ, x)
forward(::BroadcastedOperator{typeof(σ)}, x_val) = return 1.0 ./ (1.0 .+ exp.(-x_val))
backward(node::BroadcastedOperator{typeof(σ)}, x_val, g) = begin
    y = node.output
    one_val = eltype(y)(1)
    
    grad_x = g .* y .* (one_val .- y)
    
    final_grad = unbroadcast_gradient(grad_x, size(x_val))
    
    return (final_grad,)
end
Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x_val, y_val) = return x_val .^ y_val
backward(node::BroadcastedOperator{typeof(^)}, x_val, y_val, g) = let
    gx_raw = g .* y_val .* x_val .^ (y_val .- 1)
    gy_raw = g .* log.(abs.(x_val)) .* (x_val .^ y_val)
    return (
        unbroadcast_gradient(gx_raw, size(x_val)),
        unbroadcast_gradient(gy_raw, size(y_val))
    )
end

Base.Broadcast.broadcasted(exp, x::GraphNode) = BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x_val) = return exp.(x_val)
backward(node::BroadcastedOperator{typeof(exp)}, x_val, g) = let
    y = node.output
    grad_x = g .* y
    return (unbroadcast_gradient(grad_x, size(x_val)),)
end

Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x_val) = return log.(x_val)
backward(::BroadcastedOperator{typeof(log)}, x_val, g) = (
    unbroadcast_gradient(g .* (1.0 ./ x_val), size(x_val)),
)

sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x_val) = sum(x_val)
backward(node::BroadcastedOperator{typeof(sum)}, x_val, g) = (
  fill(g, size(x_val)),
)

broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x_val, y_val) = max.(x_val, y_val)
backward(node::BroadcastedOperator{typeof(max)}, x_val, y_val, g) = (
  unbroadcast_gradient(g .* (x_val .>= y_val), size(x_val)),
  unbroadcast_gradient(g .* (y_val .>  x_val), size(y_val)) 
)

linear(x::GraphNode) = BroadcastedOperator(linear, x)
forward(::BroadcastedOperator{typeof(linear)}, x_val) = x_val
backward(::BroadcastedOperator{typeof(linear)}, x_val, g) = (
  unbroadcast_gradient(g, size(x_val)),
)

relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x_val) = max.(x_val, 0.0f0)
backward(node::BroadcastedOperator{typeof(relu)}, x_val, g) = (
  unbroadcast_gradient(g .* (x_val .> 0.0f0), size(x_val)),
)

swish(x::GraphNode) = BroadcastedOperator(swish, x)
forward(::BroadcastedOperator{typeof(swish)}, x_val) = x_val ./ (1 .+ exp.(-x_val))
backward(node::BroadcastedOperator{typeof(swish)}, x_val, g) = let
    σx_val = 1.0 ./ (1.0 .+ exp.(-x_val))
    deriv = σx_val .+ x_val .* σx_val .* (1.0 .- σx_val)
    grad_x = g .* deriv
    return (unbroadcast_gradient(grad_x, size(x_val)),)
end

softmax(x::GraphNode) = BroadcastedOperator(softmax, x; name="softmax")
forward(::BroadcastedOperator{typeof(softmax)}, x_val) = begin
    exp_x = exp.(x_val .- maximum(x_val, dims=1))
    sum_exp_x = sum(exp_x; dims=1)
    return exp_x ./ sum_exp_x
end
backward(actual_node_object::BroadcastedOperator{typeof(softmax)}, x_val_from_fwd, g) = begin
    y = actual_node_object.output
    s = sum(g .* y; dims=1)
    grad_x = y .* (g .- s)
    return (grad_x,)
end

EmbeddingOp(W::GraphNode, x_indices::GraphNode) = EmbeddingOp(W, x_indices, name="EmbeddingOp")
function forward(op::EmbeddingOp, W_matrix, x_indices_matrix)
    embed_dim, vocab_size = size(W_matrix)
    if !(eltype(x_indices_matrix) <: Integer)
        error("Embedding input indices must be integers.")
    end

    op.indices_matrix_cache = x_indices_matrix

    if ndims(x_indices_matrix) == 1
        seq_len = length(x_indices_matrix)
        batch_size = 1
        _indices = reshape(x_indices_matrix, seq_len, 1)
    elseif ndims(x_indices_matrix) == 2
        seq_len, batch_size = size(x_indices_matrix)
        _indices = x_indices_matrix
    else
        error("Embedding input must be 1D (seq_len) or 2D (seq_len, batch_size). Got $(ndims(x_indices_matrix))D")
    end
    
    out_tensor = similar(W_matrix, embed_dim, seq_len, batch_size)

    for b in 1:batch_size
        for s in 1:seq_len
            idx = _indices[s, b]
            if !(1 <= idx <= vocab_size)
                error("Index $idx out of vocab range 1:$vocab_size at seq pos $s, batch $b")
            end
            out_tensor[:, s, b] = W_matrix[:, idx]
        end
    end
    return ndims(x_indices_matrix) == 1 ? out_tensor[:,:,1] : out_tensor
end
function backward(op::EmbeddingOp, W_matrix, _, g)
    x_indices_matrix = op.indices_matrix_cache 
    embed_dim, vocab_size = size(W_matrix)
    
    if ndims(x_indices_matrix) == 1
        seq_len = length(x_indices_matrix)
        batch_size = 1
        _indices = reshape(x_indices_matrix, seq_len, 1)
        _g = ndims(g) == 2 ? reshape(g, embed_dim, seq_len, 1) : g
    elseif ndims(x_indices_matrix) == 2
        seq_len, batch_size = size(x_indices_matrix)
        _indices = x_indices_matrix
        _g = g
    else
        error("Unexpected shape for x_indices_matrix in backward EmbeddingOp")
    end

    grad_W = zeros(eltype(W_matrix), size(W_matrix))

    for b in 1:batch_size
        for s in 1:seq_len
            idx = _indices[s, b]
            if 1 <= idx <= vocab_size
                grad_W[:, idx] .+= _g[:, s, b]
            end
        end
    end
    return (grad_W, nothing)
end
Base.show(io::IO, x::EmbeddingOp) = print(io, "op Embedding")

PermuteDimsOp(x::GraphNode, dims::Tuple) = PermuteDimsOp(x, dims, name="PermuteDimsOp")
function forward(op::PermuteDimsOp, x_val)
    return permutedims(x_val, op.dims_order)
end
function backward(op::PermuteDimsOp, x_val, g)
    return (permutedims(g, op.inv_dims_order),)
end
Base.show(io::IO, x::PermuteDimsOp) = print(io, "op PermuteDims($(x.dims_order))")

ReshapeOp(x::GraphNode, ts::Tuple) = ReshapeOp(x, ts..., name="ReshapeOpToTuple")
function forward(op::ReshapeOp, x_val)
    op.original_shape_cache = size(x_val)
    return reshape(x_val, op.target_shape_dims...)
end
function backward(op::ReshapeOp, x_val, g)
    return (reshape(g, op.original_shape_cache),)
end
Base.show(io::IO, x::ReshapeOp) = print(io, "op Reshape($(x.target_shape_dims))")

function forward(op::FlattenOp, x_val)
    op.original_shape_cache = size(x_val)
    if ndims(x_val) == 1
        return reshape(x_val, (length(x_val), 1))
    elseif ndims(x_val) == 2 && size(x_val,2)==1
        return x_val
    end
    
    batch_size = size(x_val)[end]
    num_features = div(length(x_val), batch_size)
    return reshape(x_val, (num_features, batch_size))
end
function backward(op::FlattenOp, x_val, g)
    return (reshape(g, op.original_shape_cache),)
end
Base.show(io::IO, x::FlattenOp) = print(io, "op Flatten")


Conv1DOp(x::GraphNode, k::GraphNode) = Conv1DOp(x, k, name="Conv1DOp")
function forward(op::Conv1DOp, X_val::AbstractArray{T,3}, K_val::AbstractArray{T,3}) where T
    op.X_val_cache = X_val
    op.K_val_cache = K_val

    cdims = DenseConvDims(X_val, K_val; stride=1, padding=0, dilation=1)
    
    return NNlib.conv(X_val, K_val, cdims)
end
function backward(op::Conv1DOp, X_val_z_fwd::AbstractArray{Tx,3}, K_val_z_fwd::AbstractArray{Tk,3}, g_incoming::AbstractArray{Tg,3}) where {Tx, Tk, Tg}
    
    X_val_f32 = op.X_val_cache::Array{Float32,3}
    K_val_f32 = op.K_val_cache::Array{Float32,3}

    G_val_f32 = g_incoming
    if eltype(g_incoming) != Float32
        G_val_f32 = convert(Array{Float32,3}, g_incoming)
    end

    cdims = DenseConvDims(X_val_f32, K_val_f32; stride=1, padding=0, dilation=1)
    dX = NNlib.∇conv_data(G_val_f32, K_val_f32, cdims)
    dK = NNlib.∇conv_filter(X_val_f32, G_val_f32, cdims)

    return (dX, dK)
end

MaxPool1DOp(x::GraphNode, ps::Tuple{Int}) = MaxPool1DOp(x, ps, name="MaxPool1DOp")
function forward(op::MaxPool1DOp, X_val::AbstractArray{T,3}) where T
    op.X_val_cache = X_val

    pool_width = op.pool_size[1]
    
    pdims = NNlib.PoolDims(X_val, (pool_width,); stride=(pool_width,), padding=(0,))
    
    Y = NNlib.maxpool(X_val, pdims)
    op.Y_val_cache = Y 
    return Y
end
function backward(op::MaxPool1DOp, x_val_z_forward::AbstractArray{Tx,3}, g_incoming::AbstractArray{Tg,3}) where {Tx, Tg}

    X_val_f32 = op.X_val_cache::Array{Float32,3}
    Y_val_f32 = op.Y_val_cache::Array{Float32,3}

    G_val_f32 = g_incoming
    if eltype(g_incoming) != Float32
        G_val_f32 = convert(Array{Float32,3}, g_incoming)
    end

    pool_width = op.pool_size[1]
    pdims = NNlib.PoolDims(X_val_f32, (pool_width,); stride=(pool_width,), padding=(0,))
    
    dX = NNlib.∇maxpool(G_val_f32, Y_val_f32, X_val_f32, pdims)
    
    return (dX,)
end