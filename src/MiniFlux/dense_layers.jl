using ..AutoDiff

struct Dense
    W::AutoDiff.Variable
    b::Union{AutoDiff.Variable, Nothing}
    activation::Function
    in_features::Int
    out_features::Int
    
    function Dense(in_features::Int, out_features::Int, activation::Function=AutoDiff.linear; bias::Bool=true)
        # He initialization for better gradient flow
        scale = sqrt(2.0 / in_features)
        W = AutoDiff.Variable(randn(out_features, in_features) .* scale, name="W")
        b = bias ? AutoDiff.Variable(zeros(out_features), name="b") : nothing
        new(W, b, activation, in_features, out_features)
    end
end

function (layer::Dense)(x::AutoDiff.Variable)
    # Handle both vector and matrix inputs
    if ndims(x.output) == 1
        x = reshape(x, :, 1)
    end
    
    # Linear transformation
    y = layer.W * x
    if layer.b !== nothing
        y = y .+ layer.b
    end
    
    # Apply activation
    return layer.activation(y)
end

# Add support for batch normalization
struct BatchNorm
    gamma::AutoDiff.Variable
    beta::AutoDiff.Variable
    running_mean::Vector{Float64}
    running_var::Vector{Float64}
    momentum::Float64
    eps::Float64
    num_features::Int
    
    function BatchNorm(num_features::Int; momentum::Float64=0.1, eps::Float64=1e-5)
        gamma = AutoDiff.Variable(ones(num_features), name="gamma")
        beta = AutoDiff.Variable(zeros(num_features), name="beta")
        running_mean = zeros(num_features)
        running_var = ones(num_features)
        new(gamma, beta, running_mean, running_var, momentum, eps, num_features)
    end
end

function (bn::BatchNorm)(x::AutoDiff.Variable)
    # Handle both vector and matrix inputs
    if ndims(x.output) == 1
        x = reshape(x, :, 1)
    end
    
    # Compute mean and variance
    mean_x = mean(x.output, dims=2)
    var_x = var(x.output, dims=2, corrected=false)
    
    # Update running statistics
    bn.running_mean .= (1 - bn.momentum) .* bn.running_mean .+ bn.momentum .* vec(mean_x)
    bn.running_var .= (1 - bn.momentum) .* bn.running_var .+ bn.momentum .* vec(var_x)
    
    # Normalize
    x_norm = (x.output .- mean_x) ./ sqrt.(var_x .+ bn.eps)
    
    # Scale and shift
    y = bn.gamma .* x_norm .+ bn.beta
    return AutoDiff.Variable(y)
end

# Add dropout for regularization
struct Dropout
    p::Float64
    training::Bool
    
    function Dropout(p::Float64=0.5)
        new(p, true)
    end
end

function (dropout::Dropout)(x::AutoDiff.Variable)
    if !dropout.training
        return x
    end
    
    mask = rand(size(x.output)) .> dropout.p
    y = x.output .* mask ./ (1 - dropout.p)
    return AutoDiff.Variable(y)
end

# Add a sequential container for MLPs
struct Sequential
    layers::Vector{Any}
    
    function Sequential(layers...)
        new(collect(layers))
    end
end

function (seq::Sequential)(x::AutoDiff.Variable)
    for layer in seq.layers
        x = layer(x)
    end
    return x
end

# Add a function to get all parameters from a model
function get_parameters(model::Union{Dense, BatchNorm, Dropout, Sequential})
    params = AutoDiff.Variable[]
    if model isa Dense
        push!(params, model.W)
        if model.b !== nothing
            push!(params, model.b)
        end
    elseif model isa BatchNorm
        push!(params, model.gamma, model.beta)
    elseif model isa Sequential
        for layer in model.layers
            append!(params, get_parameters(layer))
        end
    end
    return params
end 