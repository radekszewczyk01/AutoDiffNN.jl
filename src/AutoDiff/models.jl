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

show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")")
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")")
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name)
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ "); summary(io, x.gradient)
end

mutable struct EmbeddingOp <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    indices_matrix_cache::Any
    EmbeddingOp(W_node::GraphNode, x_indices_node::GraphNode; name="embedding") = 
        new((W_node, x_indices_node), nothing, nothing, name, nothing)
end

mutable struct PermuteDimsOp <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    dims_order::Tuple
    inv_dims_order::Tuple
    function PermuteDimsOp(x_node::GraphNode, dims_order::Tuple; name="permutedims")
        # ido =- map(x -> findfirst(==(x), dims_order), 1:length(dims_order)) # szybka invperm
        # # lub:
        inv_order = Vector{Int}(undef, length(dims_order))
        for (i, val) in enumerate(dims_order)
            inv_order[val] = i
        end
        new((x_node,), nothing, nothing, name, dims_order, tuple(inv_order...))
    end
end


mutable struct ReshapeOp <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    target_shape_dims::Union{Tuple, Nothing} # Może być NTuple{N, Int} lub Int...
    original_shape_cache::Any
    function ReshapeOp(x_node::GraphNode, target_shape_dims...; name="reshape") # akceptuje (Int,Int) lub Int,Int...
        new((x_node,), nothing, nothing, name, tuple(target_shape_dims...), nothing)
    end
end

mutable struct Conv1DOp <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    X_val_cache::Any
    K_val_cache::Any
    Conv1DOp(x_node::GraphNode, kernel_node::GraphNode; name="conv1d") = 
        new((x_node, kernel_node), nothing, nothing, name, nothing, nothing)
end

mutable struct MaxPool1DOp <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    pool_size::Tuple{Int}

    X_val_cache::Any 
    Y_val_cache::Any

    MaxPool1DOp(x_node::GraphNode, pool_size::Tuple{Int}; name="maxpool1d") = 
        new((x_node,), nothing, nothing, name, pool_size, nothing, nothing)
end

mutable struct FlattenOp <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    original_shape_cache::Any
    function FlattenOp(x_node::GraphNode; name="flatten")
        new((x_node,), nothing, nothing, name, nothing)
    end
end
