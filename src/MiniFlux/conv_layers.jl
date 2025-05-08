using ..AutoDiff
const AD = AutoDiff

struct Conv2D
    W::AD.Variable  # Kernel weights
    b::Union{AD.Variable, Nothing}  # Bias
    activation::Function
    padding::Tuple{Int,Int}
    stride::Tuple{Int,Int}
end

function Conv2D(in_channels::Int, out_channels::Int, kernel_size::Tuple{Int,Int};
               activation::Function=AD.linear,
               padding::Tuple{Int,Int}=(0,0),
               stride::Tuple{Int,Int}=(1,1),
               bias::Bool=true)
    
    # Initialize kernel weights with Xavier/Glorot initialization
    fan_in = in_channels * prod(kernel_size)
    fan_out = out_channels * prod(kernel_size)
    scale = sqrt(2.0 / (fan_in + fan_out))
    
    W = AD.Variable(randn(out_channels, in_channels, kernel_size...) .* scale, name="W")
    b = bias ? AD.Variable(zeros(out_channels), name="b") : nothing
    
    return Conv2D(W, b, activation, padding, stride)
end

function (layer::Conv2D)(x::AD.GraphNode)
    # Apply padding if needed
    if layer.padding != (0,0)
        x = AD.pad2d(x, layer.padding)
    end
    
    # Perform convolution
    output = AD.conv2d(x, layer.W)
    
    # Add bias if present
    if layer.b !== nothing
        # Reshape bias to match output dimensions
        bias_reshaped = AD.reshape(layer.b, 1, 1, :, 1)
        output = output .+ bias_reshaped
    end
    
    # Apply activation function
    return layer.activation(output)
end

# MaxPooling layer
struct MaxPool2D
    kernel_size::Tuple{Int,Int}
    stride::Tuple{Int,Int}
end

function MaxPool2D(kernel_size::Tuple{Int,Int}; stride::Tuple{Int,Int}=(1,1))
    return MaxPool2D(kernel_size, stride)
end

function compute_output!(node::AD.GraphNode)
    if node.output === nothing
        if node isa AD.BroadcastedOperator || node isa AD.ScalarOperator
            # First compute outputs of all inputs that are GraphNodes
            for input in node.inputs
                if input isa AD.GraphNode
                    compute_output!(input)
                end
            end
            # Then compute this node's output
            node.output = AD.forward(node, [input isa AD.GraphNode ? input.output : input for input in node.inputs]...)
        end
    end
    return node.output
end

function (layer::MaxPool2D)(x::AD.GraphNode)
    # Compute input's output if needed
    if x.output === nothing
        compute_output!(x)
    end
    
    # Create the maxpool2d operator and compute its output
    result = AD.maxpool2d(x, layer.kernel_size)
    
    # Force computation of maxpool2d output
    if result.output === nothing
        result.output = AD.forward(result, x.output, layer.kernel_size)
    end
    
    # Create a new operator that extracts just the output part
    output = AD.BroadcastedOperator(getindex, result, 1)
    
    # Force computation of the output
    if output.output === nothing
        output.output = AD.forward(output, result.output, 1)
    end
    return output
end

# Flatten layer for transitioning from conv to dense layers
struct Flatten end

function (layer::Flatten)(x::AD.GraphNode)
    println("Flatten input shape: ", size(x.output))
    # Reshape to (features, batch_size)
    features = prod(size(x.output)[1:end-1])
    batch_size = size(x.output, ndims(x.output))
    println("Flatten output shape: (", features, ", ", batch_size, ")")
    # Store the original shape for backward pass
    original_shape = size(x.output)
    # Create a reshape operator with the original shape stored
    return AD.BroadcastedOperator(reshape, x, (features, batch_size), original_shape)
end 