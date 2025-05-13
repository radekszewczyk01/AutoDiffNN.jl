function train!(model, loss_fn, data, opt, epochs::Int; lr=0.01)
    
    for epoch in 1:epochs

        for (i, (x_val, y_val)) in enumerate(data)

            for p in model.params
                p.gradient = zeros(size(p.output))
            end         

            x = AD.Variable(x_val, name="x")
            y = AD.Variable(y_val, name="y")

            ŷ = model(x)

            loss = loss_fn(y, ŷ)

            graph = AD.topological_sort(loss)

            AD.forward!(graph)
            AD.backward!(graph)

            opt(model.params, lr)
        end
    end
end

function create_batches(X, y, batchsize)
    n = size(X, 2)
    batches = []
    for i in 1:batchsize:n
        batch_end = min(i + batchsize - 1, n)
        xs = X[:, i:batch_end]
        ys = y[:, i:batch_end]
        push!(batches, (xs, ys))
    end
    return batches
end

function accuracy_fn(y::AD.GraphNode, ŷ::AD.GraphNode)
    # Ensure we've run a forward pass
    graph = AD.topological_sort(ŷ)
    AD.forward!(graph)
    
    # Get the actual numerical values
    y_val = y.output
    ŷ_val = ŷ.output
    
    # Handle numerical stability for very small values
    ŷ_val = clamp.(ŷ_val, 1e-10, 1 - 1e-10)  # Prevent exact 0 or 1
    
    # Convert to binary predictions (use 0.5 threshold)
    binary_preds = ŷ_val .> 0.5f0
    binary_truth = y_val .> 0.5f0
    
    # Calculate accuracy
    correct = sum(binary_preds .== binary_truth)
    total = length(y_val)
    
    return correct / total
end