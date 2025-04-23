

using LinearAlgebra


# Aktualizacja parametrów przy użyciu SGD
function train!(model, loss_fn, data, opt, epochs::Int; lr=0.01)
    for epoch in 1:epochs
        for (x_val, y_val) in data
            # Resetujemy gradienty dla wszystkich parametrów
            for p in model.params
                p.gradient = zeros(size(p.output))
            end

            # Tworzymy zmienne AutoDiff dla wejścia i etykiety
            x = AD.Variable(x_val, name="x")
            y = AD.Variable(y_val, name="y")
            # Propagacja w przód: obliczamy predykcję
            ŷ = model(x)
            # Obliczamy stratę
            loss = loss_fn(y, ŷ)
            # Budujemy graf obliczeniowy (topologicznie posortowany)
            graph = AD.topological_sort(loss)
            # Propagacja w przód
            AD.forward!(graph)
            # Backpropagacja – obliczenie gradientów
            AD.backward!(graph)
            #println("∇W = ", model.params[1].gradient, "\n∇b = ", model.params[2].gradient)
            # Aktualizacja parametrów przy użyciu optymalizatora
            opt(model.params, lr)
        end
        println("Epoch $epoch ukończona.")
    end
end