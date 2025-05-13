import Base: ^, sin, *, sum, max
import LinearAlgebra: mul!
import Base.Broadcast: broadcasted

^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = (
    g .* y,
    g .* x
)

broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)

broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y, g) = (
  g ./ y,               # ∂L/∂x = g / y
  -g .* x ./ (y .^ 2)   # ∂L/∂y = -g * x / y²
)

broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x, y) = x .^ y
backward(node::BroadcastedOperator{typeof(^)}, x, y, g) = (
  g .* (y .* x .^ (y .- 1)),          # ∂L/∂x = g * y * x^(y-1)
  g .* (log.(abs.(x)) .* x .^ y)      # ∂L/∂y = g * ln(x) * x^y
)

broadcasted(exp, x::GraphNode) = BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x) = exp.(x)
backward(node::BroadcastedOperator{typeof(exp)}, x, g) = (
  g .* node.output,   # ∂L/∂x = g * exp(x)  (node.output == exp(x))
)

broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = log.(x)
backward(node::BroadcastedOperator{typeof(log)}, x, g) = (
  g ./ x,             # ∂L/∂x = g / x
)

sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = sum(x)
backward(node::BroadcastedOperator{typeof(sum)}, x, g) = (
  fill(g, size(x)),   # ∂L/∂xᵢ = g  dla każdego elementu x
)
broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = max.(x, y)
backward(node::BroadcastedOperator{typeof(max)}, x, y, g) = (
  g .* (x .>= y),     # gradient tylko tam, gdzie x ≥ y
  g .* (y .>  x)      # gradient tylko tam, gdzie y >  x
)
# Sigmoid σ(x) = 1/(1+exp(−x))
σ(x::GraphNode) = BroadcastedOperator(σ, x)
forward(::BroadcastedOperator{typeof(σ)}, x) = 1.0 ./ (1.0 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(σ)}, x, g) = (
  # ∂σ/∂x = σ(x)*(1−σ(x)), gradient = g .* ∂σ/∂x
  g .* (node.output .* (1 .- node.output))
)

# Linear: f(x)=x
linear(x::GraphNode) = BroadcastedOperator(linear, x)
forward(::BroadcastedOperator{typeof(linear)}, x) = x
backward(::BroadcastedOperator{typeof(linear)}, x, g) = (
  # ∂x/∂x = 1, gradient = g
  g,
)

# ReLU: f(x)=max(x,0)
relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = max.(x, 0.0)
backward(node::BroadcastedOperator{typeof(relu)}, x, g) = (
  # ∂ReLU/∂x = x .> 0
  g .* (x .> 0.0),
)

# Swish: f(x)=x/(1+exp(-x))
swish(x::GraphNode) = BroadcastedOperator(swish, x)
forward(::BroadcastedOperator{typeof(swish)}, x) = x ./ (1 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(swish)}, x, g) = (
  # ∂swish/∂x = σ(x) + x*σ(x)*(1−σ(x))
  begin
    σx = 1.0 ./(1.0 .+ exp.(-x))
    deriv = σx .+ x.* σx .* (1 .- σx)
    g .* deriv
  end,
)

Embedding(W::Variable, x::GraphNode) = EmbeddingOperator((W, x), nothing, nothing, "Embedding")
function forward(::EmbeddingOperator, W_output::AbstractMatrix, x_output::AbstractVector{<:Integer})
    return W_output[:, x_output]
end
function backward(::EmbeddingOperator, W_output::AbstractMatrix, x_output::AbstractVector{<:Integer}, g::AbstractMatrix)
    gradW = zeros(size(W_output))
    for (i, idx) in enumerate(x_output)
        gradW[:, idx] .+= g[:, i]
    end
    return (gradW, nothing)
end

function forward(::EmbeddingOperator, W_output::AbstractMatrix, x_output::AbstractMatrix{<:Integer})
    embed_dim, _ = size(W_output)
    seq_len, batch_size = size(x_output)
    out = zeros(eltype(W_output), embed_dim, seq_len, batch_size)
    @inbounds for i in 1:seq_len, j in 1:batch_size
        out[:, i, j] = @view W_output[:, x_output[i, j]]
    end
    return out
end


function backward(::EmbeddingOperator,
                  W_output::AbstractMatrix,
                  x_output::AbstractMatrix{<:Integer},
                  g::AbstractArray)
    gradW = zeros(eltype(W_output), size(W_output))
    seq_len, batch_size = size(x_output)
    @inbounds for i in 1:seq_len, j in 1:batch_size
        idx = x_output[i, j]
        gradW[:, idx] .+= g[:, i, j]
    end
    return (gradW, nothing)  # tylko W ma gradient
end

function PermuteDims(x::GraphNode, perm::Tuple{Vararg{Int}}; name="PermuteDims")
    return PermuteDimsOperator((x,), perm, nothing, nothing, name)
end
function forward(op::PermuteDimsOperator, x)
    return permutedims(x, op.perm)
end
function backward(op::PermuteDimsOperator, x, grad)
    inv_perm = invperm(op.perm)
    return (permutedims(grad, inv_perm),)
end

Conv(W::Variable, x::GraphNode, b::Union{Variable,Nothing}, stride::Int, pad::Int; name::String="Conv") =
 ConvOperator((x, W, b), nothing, nothing, stride, pad, name)

function pad_array(x, pad::Int)
    if pad == 0
        return x
    end
    padded = zeros(size(x, 1), size(x, 2)+2pad, size(x, 3))
    padded[:, pad+1:end-pad, :] = x
    return padded
end

function unpad_array(x, pad::Int)
    if pad == 0
        return x
    end
    return x[:, pad+1:end-pad, :]
end

function forward(node::ConvOperator, x, W, b)
    batch_size, in_len, in_channels = size(x)
    out_channels, _, kernel_size = size(W)
    
    out_len = div(in_len + 2*node.pad - kernel_size, node.stride) + 1
    
    output = zeros(batch_size, out_len, out_channels)
    
    x_padded = pad_array(x, node.pad)
    
    for b in 1:batch_size, t in 1:out_len, oc in 1:out_channels
        start = (t-1)*node.stride + 1
        window = @view x_padded[b, start:start+kernel_size-1, :]
        output[b, t, oc] = sum(W[oc, :, :] .* window)
    end
    
    if !isnothing(b)
        output .+= reshape(b, 1, 1, :)
    end

    return output
end

function backward(node::ConvOperator, x, W, b, g)
    x_grad = zeros(size(x))
    W_grad = zeros(size(W))
    b_grad = isnothing(b) ? nothing : zeros(size(b))
    
    x_padded = pad_array(x, node.pad)
    x_grad_padded = pad_array(x_grad, node.pad)
    
    for b in 1:size(g, 1), t in 1:size(g, 2), oc in 1:size(g, 3)
        start = (t-1)*node.stride + 1
        window = @view x_padded[b, start:start+size(W, 3)-1, :]
        
        W_grad[oc, :, :] .+= window .* g[b, t, oc]
        
        x_grad_padded[b, start:start+size(W, 3)-1, :] .+= W[oc, :, :] .* g[b, t, oc]
        
        if !isnothing(b_grad)
            b_grad[oc] += g[b, t, oc]
        end
    end
    
    x_grad = unpad_array(x_grad_padded, node.pad)
    
    return (x_grad, W_grad, b_grad)
end


function MaxPool1D(x::GraphNode, pool_size::Int; name::String="MaxPool1D")
    return MaxPool1DOperator((x,), pool_size, nothing, nothing, name)
end

function forward(op::MaxPool1DOperator, x::Array)
    batch, seq_len, channels = size(x)
    out_len = div(seq_len, op.pool_size)
    y = zeros(eltype(x), batch, out_len, channels)
    @inbounds for b in 1:batch, c in 1:channels, i in 1:out_len
        window = @view x[b, (i-1)*op.pool_size+1:i*op.pool_size, c]
        y[b,i,c] = maximum(window)
    end
    return y
end

function backward(op::MaxPool1DOperator, x::Array, grad::Array)
    batch, seq_len, channels = size(x)
    out_len = div(seq_len, op.pool_size)
    x_grad = zeros(eltype(x), size(x))
    @inbounds for b in 1:batch, c in 1:channels, i in 1:out_len
        start = (i-1)*op.pool_size + 1
        window = @view x[b, start:start+op.pool_size-1, c]
        maxval = maximum(window)
        for j in 1:op.pool_size
            if window[j] == maxval
                x_grad[b, start+j-1, c] += grad[b, i, c]
                break
            end
        end
    end
    return (x_grad,)
end


function Flatten(x::GraphNode; name::String="Flatten")
    return FlattenOperator((x,), nothing, nothing, (), name)
end

function forward(op::FlattenOperator, x::Array)
    op.input_shape = size(x)
    batch = size(x, 1)
    new_dim = prod(size(x)[2:end])
    return reshape(x, batch, new_dim)
end

function backward(op::FlattenOperator, x::Array, grad::Array)
    return (reshape(grad, op.input_shape),)
end