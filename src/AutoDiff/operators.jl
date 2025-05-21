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
  g ./ y,
  -g .* x ./ (y .^ 2)
)

σ(x) = BroadcastedOperator(σ, x)
forward(::BroadcastedOperator{typeof(σ)}, x) = return 1.0 ./ (1.0 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(σ)}, x, g) = let
    y = node.output
    𝟏 = ones(length(y))
    return (g .* y .* (1 .- y),)
end

Base.Broadcast.broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x, y) = return x .^ y
backward(node::BroadcastedOperator{typeof(^)}, x, y, g) = let
    gx = g .* y .* x .^ (y .- 1)
    gy = g .* log.(abs.(x)) .* x .^ y
    return (gx, gy)
end

Base.Broadcast.broadcasted(exp, x::GraphNode) = BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x) = return exp.(x)
backward(node::BroadcastedOperator{typeof(exp)}, x, g) = let
    y = node.output
    return (g .* y,)
end

Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = return log.(x)
backward(::BroadcastedOperator{typeof(log)}, x, g) = tuple(g .* (1.0 ./ x))


sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = sum(x)
backward(node::BroadcastedOperator{typeof(sum)}, x, g) = (
  fill(g, size(x)),
)
broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = max.(x, y)
backward(node::BroadcastedOperator{typeof(max)}, x, y, g) = (
  g .* (x .>= y),
  g .* (y .>  x)
)

linear(x::GraphNode) = BroadcastedOperator(linear, x)
forward(::BroadcastedOperator{typeof(linear)}, x) = x
backward(::BroadcastedOperator{typeof(linear)}, x, g) = (
  g,
)

relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = max.(x, 0.0)
backward(node::BroadcastedOperator{typeof(relu)}, x, g) = (
  g .* (x .> 0.0),
)

swish(x::GraphNode) = BroadcastedOperator(swish, x)
forward(::BroadcastedOperator{typeof(swish)}, x) = x ./ (1 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(swish)}, x, g) = (
  begin
    σx = 1.0 ./(1.0 .+ exp.(-x))
    deriv = σx .+ x.* σx .* (1 .- σx)
    g .* deriv
  end,
)

softmax(x::GraphNode) = BroadcastedOperator(softmax, x; name="softmax")
forward(::BroadcastedOperator{typeof(softmax)}, x) = begin
    ex = exp.(x)
    sum_ex = sum(ex; dims=1)
    ex ./ sum_ex
end
backward(::BroadcastedOperator{typeof(softmax)}, x, g) = begin
    y = forward(BroadcastedOperator(softmax, x), x)
    s = sum(g .* y; dims=1)
    (y .* (g .- s),)
end

