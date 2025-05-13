abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name :: String
    Variable(output; name="?") = new(output, nothing, name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct EmbeddingOperator <: Operator
    inputs::Tuple{Variable, GraphNode}
    output::Any
    gradient::Any
    name::String
end

mutable struct ConvOperator <: Operator
    inputs::Tuple{GraphNode, Variable, Union{Variable, Nothing}}
    output::Any
    gradient::Any
    stride::Int
    pad::Int
    name::String
end

show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")")
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")")
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name)
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ "); summary(io, x.gradient)
end
show(io::IO, op::EmbeddingOperator) = begin
    print(io, "op.embedding(", op.name)
    print(io, "\n ┣━ W: ", op.inputs[1].name)
    print(io, "\n ┣━ x: "); summary(io, op.inputs[2].output)
    print(io, "\n ┣━ output: "); summary(io, op.output)
    print(io, "\n ┗━ gradient: "); summary(io, op.gradient)
end


