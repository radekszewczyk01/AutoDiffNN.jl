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

import Base: show, summary, size, length, ndims, eltype, reshape, iterate, getindex

show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")")
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")")
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name)
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ "); summary(io, x.gradient)
end

size(x::GraphNode) = size(x.output)
size(x::GraphNode, dim::Integer) = size(x.output, dim)
length(x::GraphNode) = length(x.output)
ndims(x::GraphNode) = ndims(x.output)
eltype(x::GraphNode) = eltype(x.output)

# Add reshape method
reshape(x::GraphNode, dims...) = BroadcastedOperator(reshape, x, dims...)

function forward(::BroadcastedOperator{typeof(reshape)}, x, dims...)
    # If dims is a tuple of integers or Colon, use it directly
    if all(d -> isa(d, Union{Integer, Colon}), dims)
        return reshape(x, dims...)
    # If dims is a tuple containing a tuple of integers or Colon, use the inner tuple
    elseif length(dims) == 1 && isa(dims[1], Tuple)
        return reshape(x, dims[1]...)
    # If we have multiple arguments and the first is a tuple, use that for reshaping
    elseif length(dims) > 1 && isa(dims[1], Tuple)
        return reshape(x, dims[1]...)
    else
        error("Invalid dimensions for reshape: $dims")
    end
end

function backward(node::BroadcastedOperator{typeof(reshape)}, x, dims...)
    # Get the original shape from the input
    original_shape = size(x)
    
    # If we have an extra argument for the original shape, use it
    if length(dims) > 1 && dims[2] isa Tuple
        original_shape = dims[2]
    end
    
    # Reshape gradient back to original shape
    if node.gradient isa Array
        # If gradient is an array, reshape it to match the original shape
        if length(node.gradient) == prod(original_shape)
            grad = reshape(node.gradient, original_shape)
        else
            # If sizes don't match, create a zero array and copy what we can
            grad = zeros(original_shape)
            min_size = min(length(node.gradient), length(grad))
            grad[1:min_size] = node.gradient[1:min_size]
        end
    else
        # If gradient is a scalar, create a zero array of the original shape
        grad = zeros(original_shape)
        grad[1] = node.gradient
    end
    
    tuple(grad)
end

# For single values
iterate(x::GraphNode) = iterate(x.output)
iterate(x::GraphNode, state) = iterate(x.output, state)

# For tuples
getindex(x::GraphNode, i::Integer) = begin
    if x isa BroadcastedOperator
        # Compute the output if it hasn't been computed yet
        if x.output === nothing
            x.output = forward(x, [input.output for input in x.inputs]...)
        end
    end
    getindex(x.output, i)
end

# Add method for Tuple types
Base.getindex(x::Tuple, i::Integer) = x[i]

# Add size methods for getindex operator
function forward(::BroadcastedOperator{typeof(getindex)}, x, i)
    return getindex(x, i)
end

function backward(node::BroadcastedOperator{typeof(getindex)}, x, i, g)
    # Handle tuple inputs (like from maxpool2d)
    if x isa Tuple
        x = x[1]  # Use the first element of the tuple (the output array)
    end
    
    grad = zeros(size(x))
    
    if g isa Array
        if ndims(g) == 4
            h, w, c, b = size(g)
            
            slice = g[:, :, :, 1]
            
            for h_idx in 1:h, w_idx in 1:w, ch in 1:c
                try
                    grad[h_idx, w_idx, ch, 1] = slice[h_idx, w_idx, ch]
                catch e
                    println("getindex backward - Error at indices: h=$h_idx, w=$w_idx, ch=$ch")
                    println("getindex backward - grad shape: ", size(grad))
                    println("getindex backward - slice shape: ", size(slice))
                    rethrow(e)
                end
            end
        else
            grad[i] = g
        end
    else
        grad[i] = g
    end
    
    tuple(grad)
end

function size(x::BroadcastedOperator{typeof(getindex)})
    if x.output === nothing
        input = x.inputs[1]
        index = x.inputs[2]
        x.output = forward(x, input.output, index)
    end
    return size(x.output)
end

function size(x::BroadcastedOperator{typeof(getindex)}, dim::Integer)
    if x.output === nothing
        # If output hasn't been computed yet, compute it
        input = x.inputs[1]
        index = x.inputs[2]
        x.output = forward(x, input.output, index)
    end
    return size(x.output, dim)
end
