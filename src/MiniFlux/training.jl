function train!(model, loss_fn, data, opt, epochs::Int; lr=0.01)
    for epoch in 1:epochs

        for (i, (x_val, y_val)) in enumerate(data)

            # Reset gradients for all parameters
            for p in model.params
                p.gradient = zeros(size(p.output))
            end         
            
            # Create input and output variables
            x = AD.Variable(x_val, name="x")
            y = AD.Variable(y_val, name="y")

            # Materialize if x is a broadcasted operation
            if isa(x, AD.BroadcastedOperator)
                x = AD.materialize(x)
            end

            # Forward pass: compute predictions
            ŷ = model(x)
            #println("Predicted output (ŷ): ", ŷ)

            # Calculate loss
            loss = loss_fn(y, ŷ)
            #println("Loss: ", loss)

            # Topological sorting of the computation graph
            graph = AD.topological_sort(loss)

            # Forward pass through the graph to compute outputs
            AD.forward!(graph)
            #println("Forward pass completed.")

            # Backward pass to compute gradients
            AD.backward!(graph)

            # Update parameters using the optimizer
            opt(model.params, lr)
            #println("Parameters updated.")
        end
    end
end
