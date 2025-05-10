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
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(y .* 𝟏)
    Jy = diagm(x .* 𝟏)
    tuple(Jx' * g, Jy' * g)
end

broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)

broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(𝟏 ./ y)
    Jy = (-x ./ y .^ 2)
    tuple(Jx' * g, Jy' * g)
end

broadcasted(^, x::GraphNode, y::GraphNode) = BroadcastedOperator(^, x, y)
forward(::BroadcastedOperator{typeof(^)}, x, y) = x .^ y
backward(node::BroadcastedOperator{typeof(^)}, x, y, g) = let
    # Handle scalar inputs
    if isa(x, Number)
        x_vec = [x]
    else
        x_vec = vec(x)
    end
    
    if isa(y, Number)
        y_vec = [y]
    else
        y_vec = vec(y)
    end
    
    g_vec = vec(g)
    
    Jx = diagm(y_vec .* x_vec .^ (y_vec .- 1.0))
    Jy = diagm(log.(abs.(x_vec)) .* x_vec .^ y_vec)
    
    # Reshape back to original dimensions
    if isa(x, Number)
        grad_x = Jx' * g_vec
    else
        grad_x = reshape(Jx' * g_vec, size(x))
    end
    
    if isa(y, Number)
        grad_y = Jy' * g_vec
    else
        grad_y = reshape(Jy' * g_vec, size(y))
    end
    
    tuple(grad_x, grad_y)
end

broadcasted(exp, x::GraphNode) = BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x) = exp.(x)
backward(node::BroadcastedOperator{typeof(exp)}, x, g) = let
    y = node.output
    J = diagm(y)
    tuple(J' * g)
end

broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = log.(x)
backward(::BroadcastedOperator{typeof(log)}, x, g) = tuple(diagm(1.0 ./ x)' * g)

sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x))
    J = 𝟏'
    tuple(J' * g)
end

broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
    Jx = diagm(isless.(y, x))
    Jy = diagm(isless.(x, y))
    tuple(Jx' * g, Jy' * g)
end

σ(x) = BroadcastedOperator(σ, x)
forward(::BroadcastedOperator{typeof(σ)}, x) = 1.0 ./ (1.0 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(σ)}, x, g) = let
    if ndims(x) == 1
        # For vector inputs
        s = σ.(x)
        return g .* s .* (1 .- s)
    else
        # For batch inputs
        s = σ.(x)
        return reshape(g, size(x)) .* s .* (1 .- s)
    end
end

linear(x) = BroadcastedOperator(linear, x)
forward(::BroadcastedOperator{typeof(linear)}, x) = x
backward(::BroadcastedOperator{typeof(linear)}, x, g) = begin
    # Handle array gradients
    if !isempty(x)
        if isa(x, AbstractArray)
            # For matrix multiplication, we need to handle the gradient properly
            if ndims(x) == 2 && ndims(g) == 2
                # Reshape gradient to match input dimensions
                g_reshaped = reshape(g, size(x))
                return tuple(g_reshaped)
            else
                return tuple(g)
            end
        else
            return tuple(g)
        end
    end
    return tuple(g)
end

relu(x) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = max.(x, 0.0)
backward(node::BroadcastedOperator{typeof(relu)}, x, g) = begin
    if g isa Number
        g = fill(g, size(x))
    end
    
    mask = Float64.(x .> 0.0)
    grad = mask .* g
    
    tuple(grad)
end

swish(x) = BroadcastedOperator(swish, x)
forward(::BroadcastedOperator{typeof(swish)}, x) = x ./ (1 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(swish)}, x, g) = begin
    σ = 1.0 ./ (1 .+ exp.(-x))
    y = node.output
    J = diagm(σ .+ x .* σ .* (1 .- σ))
    tuple(J' * g)
end

function conv2d(x::GraphNode, kernel::GraphNode)
    return BroadcastedOperator(conv2d, x, kernel)
end

function forward(::BroadcastedOperator{typeof(conv2d)}, x, kernel)
    if ndims(x) == 4
        batch_size = size(x, 4)
        in_channels = size(x, 3)
        out_channels = size(kernel, 1)
        out_h = size(x, 1) - size(kernel, 3) + 1
        out_w = size(x, 2) - size(kernel, 4) + 1
        
        output = zeros(out_h, out_w, out_channels, batch_size)
        
        for b in 1:batch_size
            for c_out in 1:out_channels
                for i in 1:out_h, j in 1:out_w
                    patch = x[i:i+size(kernel,3)-1, j:j+size(kernel,4)-1, :, b]
                    patch_reshaped = reshape(patch, :, in_channels)
                    kernel_reshaped = reshape(kernel[c_out,:,:,:], :, in_channels)
                    output[i,j,c_out,b] = sum(patch_reshaped .* kernel_reshaped)
                end
            end
        end
        return output
    else
        n, m = size(x) .- size(kernel) .+ 1
        J = zeros(n, m)
        for i in 1:n, j in 1:m
            patch = x[i:i+size(kernel,1)-1, j:j+size(kernel,2)-1]
            J[i, j] = sum(patch .* kernel)
        end
        return J
    end
end

function backward(node::BroadcastedOperator{typeof(conv2d)}, x, kernel, g)
    if ndims(x) == 4
        batch_size = size(x, 4)
        in_channels = size(x, 3)
        out_channels = size(kernel, 1)
        out_h = size(x, 1) - size(kernel, 3) + 1
        out_w = size(x, 2) - size(kernel, 4) + 1
        
        grad_x = zeros(size(x))
        grad_k = zeros(size(kernel))
        
        for b in 1:batch_size
            for c_out in 1:out_channels
                for i in 1:out_h, j in 1:out_w
                    patch = x[i:i+size(kernel,3)-1, j:j+size(kernel,4)-1, :, b]
                    patch_reshaped = reshape(patch, :, in_channels)
                    kernel_reshaped = reshape(kernel[c_out,:,:,:], :, in_channels)
                    
                    grad_patch = g[i,j,c_out,b] .* kernel_reshaped
                    grad_kernel = g[i,j,c_out,b] .* patch_reshaped
                    
                    grad_patch = reshape(grad_patch, size(patch))
                    grad_kernel = reshape(grad_kernel, size(kernel[c_out,:,:,:]))
                    
                    grad_x[i:i+size(kernel,3)-1, j:j+size(kernel,4)-1, :, b] .+= grad_patch
                    grad_k[c_out,:,:,:] .+= grad_kernel
                end
            end
        end
        
        return tuple(grad_x, grad_k)
    else
        n, m = size(x) .- size(kernel) .+ 1
        grad_x = zeros(size(x))
        grad_k = zeros(size(kernel))
        
        for i in 1:n, j in 1:m
            patch = x[i:i+size(kernel,1)-1, j:j+size(kernel,2)-1]
            grad_patch = g[i,j] .* kernel
            grad_kernel = g[i,j] .* patch
            
            grad_x[i:i+size(kernel,1)-1, j:j+size(kernel,2)-1] .+= grad_patch
            grad_k .+= grad_kernel
        end
        
        return tuple(grad_x, grad_k)
    end
end

function pad2d(x::GraphNode, padding::Tuple{Int,Int})
    return BroadcastedOperator(pad2d, x, padding)
end

function forward(::BroadcastedOperator{typeof(pad2d)}, x, padding)
    if ndims(x) == 4
        h, w, c, b = size(x)
        ph, pw = padding
        padded = zeros(h + 2*ph, w + 2*pw, c, b)
        padded[ph+1:ph+h, pw+1:pw+w, :, :] = x
        return padded
    else
        h, w = size(x)
        ph, pw = padding
        padded = zeros(h + 2*ph, w + 2*pw)
        padded[ph+1:ph+h, pw+1:pw+w] = x
        return padded
    end
end

function backward(node::BroadcastedOperator{typeof(pad2d)}, x, padding, g)
    if ndims(x) == 4
        h, w, c, b = size(x)
        ph, pw = padding
        return g[ph+1:ph+h, pw+1:pw+w, :, :]
    else
        h, w = size(x)
        ph, pw = padding
        return g[ph+1:ph+h, pw+1:pw+w]
    end
end

function maxpool2d(x::GraphNode, kernel_size::Tuple{Int,Int})
    return BroadcastedOperator(maxpool2d, x, kernel_size)
end

function forward(::BroadcastedOperator{typeof(maxpool2d)}, x, kernel_size)
    if ndims(x) == 4
        kh, kw = kernel_size
        h, w, c, b = size(x)
        out_h = div(h, kh)
        out_w = div(w, kw)
        output = zeros(out_h, out_w, c, b)
        indices = zeros(Int, out_h, out_w, c, b)
        
        for i in 1:out_h, j in 1:out_w, ch in 1:c, batch in 1:b
            patch = x[(i-1)*kh+1:i*kh, (j-1)*kw+1:j*kw, ch, batch]
            val, idx = findmax(patch)
            output[i,j,ch,batch] = val
            indices[i,j,ch,batch] = (idx[1]-1)*kw + idx[2]
        end
        
        return output, indices
    else
        kh, kw = kernel_size
        h, w = size(x)
        out_h = div(h, kh)
        out_w = div(w, kw)
        output = zeros(out_h, out_w)
        indices = zeros(Int, out_h, out_w)
        
        for i in 1:out_h, j in 1:out_w
            patch = x[(i-1)*kh+1:i*kh, (j-1)*kw+1:j*kw]
            val, idx = findmax(patch)
            output[i,j] = val
            indices[i,j] = (idx[1]-1)*kw + idx[2]
        end
        
        return output, indices
    end
end

function backward(node::BroadcastedOperator{typeof(maxpool2d)}, x, kernel_size, g)
    if ndims(x) == 4
        kh, kw = kernel_size
        h, w, c, b = size(x)
        out_h = div(h, kh)
        out_w = div(w, kw)
        grad_x = zeros(size(x))
        indices = node.output[2]
        
        if g isa Number
            g = fill(g, out_h, out_w, c, b)
        end
        
        for i in 1:out_h, j in 1:out_w, ch in 1:c, batch in 1:b
            idx = indices[i,j,ch,batch]
            row = div(idx-1, kw) + 1
            col = mod(idx-1, kw) + 1
            grad_x[(i-1)*kh+row, (j-1)*kw+col, ch, batch] = g[i,j,ch,batch]
        end
        
        return grad_x
    else
        kh, kw = kernel_size
        h, w = size(x)
        out_h = div(h, kh)
        out_w = div(w, kw)
        grad_x = zeros(size(x))
        indices = node.output[2]
        
        if g isa Number
            g = fill(g, out_h, out_w)
        end
        
        for i in 1:out_h, j in 1:out_w
            idx = indices[i,j]
            row = div(idx-1, kw) + 1
            col = mod(idx-1, kw) + 1
            grad_x[(i-1)*kh+row, (j-1)*kw+col] = g[i,j]
        end
        
        return grad_x
    end
end

function forward!(node::BroadcastedOperator{typeof(maxpool2d)})
    x = node.inputs[1].output
    kernel_size = node.inputs[2]
    output, indices = forward(node, x, kernel_size)
    node.output = (output, indices)
    return node.output
end

function backward!(node::BroadcastedOperator{typeof(maxpool2d)})
    inputs = node.inputs
    graph_inputs = [input for input in inputs if input isa GraphNode]
    graph_outputs = [input.output for input in graph_inputs]
    
    all_inputs = [input isa GraphNode ? input.output : input for input in inputs]
    
    gradients = backward(node, all_inputs..., node.gradient)
    
    for (input, gradient) in zip(graph_inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end

broadcasted(reshape, x::GraphNode, dims::Tuple) = BroadcastedOperator(reshape, x, dims)
forward(::BroadcastedOperator{typeof(reshape)}, x, dims) = reshape(x, dims)
backward(node::BroadcastedOperator{typeof(reshape)}, x, dims, g) = tuple(reshape(g, size(x)))

function forward(::BroadcastedOperator{typeof(mul!)}, A, x)
    if ndims(A) == 2 && ndims(x) == 2
        return A * x
    else
        return A .* x
    end
end

function backward(::BroadcastedOperator{typeof(mul!)}, A, x, g)
    if ndims(A) == 2 && ndims(x) == 2 && ndims(g) == 2
        return tuple(g * x', A' * g)
    else
        return tuple(g, g)
    end
end

function forward(op::BroadcastedOperator{typeof(linear)}, x::AbstractArray)
    if isempty(x)
        return x
    end
    
    if isa(x, AbstractArray)
        return x
    end
    return x
end

function backward(node::BroadcastedOperator{typeof(linear)}, x, g)
    if ndims(x) == 1
        return g
    else
        return reshape(g, size(x))
    end
end
