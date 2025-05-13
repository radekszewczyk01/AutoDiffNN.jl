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
    return W_output[:, x_output]  # Embedding lookup
end
function backward(::EmbeddingOperator, W_output::AbstractMatrix, x_output::AbstractVector{<:Integer}, g::AbstractMatrix)
    gradW = zeros(size(W_output))
    for (i, idx) in enumerate(x_output)
        gradW[:, idx] .+= g[:, i]  # Accumulate gradients for embedding matrix
    end
    return (gradW, nothing)  # No gradient for indices (x is treated as integer input)
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
    # x shape: (batch, length, in_channels)
    # W shape: (out_channels, in_channels, kernel_size)
    batch_size, in_len, in_channels = size(x)
    out_channels, _, kernel_size = size(W)
    
    # Calculate output length
    out_len = div(in_len + 2*node.pad - kernel_size, node.stride) + 1
    
    # Initialize output
    output = zeros(batch_size, out_len, out_channels)
    
    # Add padding
    x_padded = pad_array(x, node.pad)
    
    # Perform convolution
    for b in 1:batch_size, t in 1:out_len, oc in 1:out_channels
        start = (t-1)*node.stride + 1
        window = @view x_padded[b, start:start+kernel_size-1, :]
        output[b, t, oc] = sum(W[oc, :, :] .* window)
    end
    
    # Add bias if present
    if !isnothing(b)
        output .+= reshape(b, 1, 1, :)
    end

    #node.output = output
    return output
end

function backward(node::ConvOperator, x, W, b, g)
    # Gradient calculations
    x_grad = zeros(size(x))
    W_grad = zeros(size(W))
    b_grad = isnothing(b) ? nothing : zeros(size(b))
    
    # Pad input gradient
    x_padded = pad_array(x, node.pad)
    x_grad_padded = pad_array(x_grad, node.pad)
    
    # Backward pass
    for b in 1:size(g, 1), t in 1:size(g, 2), oc in 1:size(g, 3)
        start = (t-1)*node.stride + 1
        window = @view x_padded[b, start:start+size(W, 3)-1, :]
        
        # Weight gradient
        W_grad[oc, :, :] .+= window .* g[b, t, oc]
        
        # Input gradient
        x_grad_padded[b, start:start+size(W, 3)-1, :] .+= W[oc, :, :] .* g[b, t, oc]
        
        # Bias gradient
        if !isnothing(b_grad)
            b_grad[oc] += g[b, t, oc]
        end
    end
    
    # Remove padding from input gradient
    x_grad = unpad_array(x_grad_padded, node.pad)
    
    return (x_grad, W_grad, b_grad)
end

mutable struct PermuteDimsOperator <: Operator
    inputs::Tuple{GraphNode}
    perm::Tuple{Vararg{Int}}
    output::Any
    gradient::Any
    name::String
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

relu(x::AbstractArray) = max.(x, 0.0)

σ(x::AbstractArray) = 1.0 ./ (1.0 .+ exp.(-x))

linear(x::AbstractArray) = x

swish(x::AbstractArray) = x ./ (1 .+ exp.(-x))