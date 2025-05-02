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
